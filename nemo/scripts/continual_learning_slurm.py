from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from megatron.core.optimizer import OptimizerConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule, PytorchOptimizerModule
from lightning.pytorch.loggers import WandbLogger


load_dotenv()

PROJECT_NAME = "BFS"
RUN_NAME = "qwen25_7b_continual"

# SLURM RELATED
USER = None # for SSH tunnel
HOST = None # for SSH tunnel
JOB_DIR = "/mnt/home/jinheo/.nemo_run/experiments/"
ACCOUNT = "BFS" # SLURM account
PARTITION = "h100"
CONTAINER_IMAGE = "/mnt/cephfs/scratch/enroot-images/nvidia-nemo:25.04.rc2.sqsh"
CUSTOM_MOUNTS = [
    "/mnt/cephfs:/cephfs",
    "/dev/infiniband/:/dev/infiniband",  # infiniband might not be needed
]
NODELIST = "DGX-H100-[1,3,4,5]" # replace with None
EXCLUDE = "DGX-H100-[10-12]" # replace with None

# Logging and Checkpointing Related
MAX_EPOCHS = 1
MAX_STEPS = 21000
LIMIT_VAL_BATCHES = 0 # 0 means no validation
VAL_CHECK_INTERVAL = 100
LOG_EVERY_N_STEPS = 10
CHECKPOINT_DIR_BASE_PATH = f"/cephfs/scratch/{PROJECT_NAME}/{RUN_NAME}"
CHECKPOINT_EVERY_N_STEPS = 1000

# Learning Rate Related
WARMUP_STEPS = MAX_STEPS * 0.1
MIN_LR = 1.5e-06
MAX_LR = 1.5e-05

# GPU and Node Related
NODES = 4
GPUS_PER_NODE = 8

TENSOR_PARALLEL = 2
PIPELINE_PARALLEL = 1
CONTEXT_PARALLEL = 1

# Data and Model Related
SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 1024
PER_GPU_BATCH_SIZE = 1

PRETRAINED_MODEL_NAME = "Qwen/Qwen2.5-7B"
DATA_PATH = [
    0.25, "/cephfs/scratch/BFS/data/fineweb-edu-dedup-even/fineweb-edu-dedup-even_text_document",
    0.25, "/cephfs/scratch/BFS/data/fineweb-edu-dedup-odd/fineweb-edu-dedup-odd_text_document",
    0.5, "/cephfs/scratch/BFS/data/maumweb-edu/maumweb-edu_text_document",
]


ENV_VARS = {
    'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
    'NCCL_SOCKET_IFNAME': 'ibp64s0', # Infiniband
    'NEMO_HOME': '/cephfs/scratch/BFS/nemo_home',
    'NEMO_MODELS_CACHE': '/cephfs/scratch/BFS/nemo_home/.cache/nemo/models', # import_ckpt saves to this path
}


# ────────────────────────────────────────────────────────────
# 1.  Core recipe (unchanged from your local script)
# ────────────────────────────────────────────────────────────
def configure_recipe() -> run.Partial:
    # 1‑a  Convert HF checkpoint once (only rank‑0 actually runs it)
    # llm.import_ckpt(
    #     model=llm.Qwen2Model(llm.Qwen25Config1P5B()),
    #     source="hf://Qwen/Qwen2.5-1.5B",
    # )

    recipe = llm.qwen25_7b.pretrain_recipe(
        dir=CHECKPOINT_DIR_BASE_PATH,
        name=RUN_NAME,
        num_nodes=NODES,
        num_gpus_per_node=GPUS_PER_NODE,
    )
    recipe.model.config.seq_length = SEQUENCE_LENGTH

    # Continual‑learning resume logic
    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig, path=f"nemo://{PRETRAINED_MODEL_NAME}"),
        resume_if_exists=True, # a must for resuming from a pretrained checkpoint
        # resume_ignore_no_checkpoint=True,
    )

    # Parallelism overrides
    recipe.trainer.strategy.tensor_model_parallel_size = TENSOR_PARALLEL
    recipe.trainer.strategy.pipeline_model_parallel_size = PIPELINE_PARALLEL
    recipe.trainer.strategy.context_parallel_size = CONTEXT_PARALLEL
    
    recipe.trainer.max_epochs = MAX_EPOCHS
    recipe.trainer.max_steps = MAX_STEPS
    recipe.trainer.log_every_n_steps = LOG_EVERY_N_STEPS
    recipe.trainer.limit_val_batches = LIMIT_VAL_BATCHES
    recipe.trainer.val_check_interval = VAL_CHECK_INTERVAL

    recipe.optim.config.lr = MAX_LR
    recipe.optim.lr_scheduler.min_lr = MIN_LR
    recipe.optim.lr_scheduler.warmup_steps = WARMUP_STEPS
    
    # For testing, comment following data reinitialization. MockDataModule is used with recipe as default. 
    recipe.data = run.Config(
        llm.PreTrainingDataModule,
        paths=DATA_PATH,
        seq_length=SEQUENCE_LENGTH,
        global_batch_size=GLOBAL_BATCH_SIZE,
        micro_batch_size=PER_GPU_BATCH_SIZE,
        tokenizer=run.Config(
            AutoTokenizer,
            pretrained_model_name=PRETRAINED_MODEL_NAME,
        ),
        split="100,0,0",
        # num_workers=DATA_LOADER_WORKERS,
        # pin_memory=DATA_LOADER_PIN_MEMORY,
    )
    
    # Checkpointing
    recipe.log.ckpt = run.Config(
        nl.ModelCheckpoint,
        # save_last=True, # save last checkpoint only
        save_top_k=1,
        every_n_train_steps=CHECKPOINT_EVERY_N_STEPS,
        filename="{PRETRAINED_MODEL_NAME}--{step}-{consumed_samples}"
    )

    # 🔑 1‑b  **Add W&B logger**
    wandb_logger_cfg = run.Config(
        WandbLogger,
        project=PROJECT_NAME,
        name=RUN_NAME,
    )
    recipe.log.wandb = wandb_logger_cfg      # attach to NeMo logger

    return recipe


# ────────────────────────────────────────────────────────────
# 2.  Slurm executor (Pyxis)
# ────────────────────────────────────────────────────────────
def slurm_executor(
    *,
    account: str,
    nodes: int,
    ntasks_per_node: int,
    node_job_dir: str,
    # optional – provide both or neither
    host: Optional[str] = None,
    user: Optional[str] = None,
    # common Slurm options
    partition: Optional[str] = None,
    gpus_per_node: Optional[int] = None,
    container_image: Optional[str] = None,
    time: str = None,
    gres: Optional[str] = None,
    mem: str = "0",
    exclusive: bool = True,
    container_mounts: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    retries: int = 0,
    nodelist:  Optional[str] = None,
    exclude:   Optional[str] = None,
) -> run.SlurmExecutor:
    # 1) Pick tunnel
    if host and user:
        tunnel = run.SSHTunnel(user=user, host=host, job_dir=node_job_dir)
    else:
        tunnel = run.LocalTunnel(job_dir=node_job_dir)

    # 2) Decide which packager to use
    packager = run.Packager()                    # rsync‑style copy

    # 3) Build executor
    executor = run.SlurmExecutor(
        account=account,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        tunnel=tunnel,
        partition=partition,
        gpus_per_node=gpus_per_node,
        container_image=container_image,
        time=time,
        gres=gres,
        mem=mem,
        exclusive=exclusive,
        container_mounts=container_mounts or [],
        env_vars=env_vars or {},
        packager=packager,
        # pass through only if supplied
        **(
            {"additional_parameters": {
                 **({"nodelist": nodelist} if nodelist else {}),
                 **({"exclude":  exclude}  if exclude  else {}),
            }}
            if nodelist or exclude else {}
        ),
    )
    executor.retries = retries
    return executor


# ────────────────────────────────────────────────────────────
# 3.  Entrypoint
# ────────────────────────────────────────────────────────────
def main() -> None:
    recipe = configure_recipe()
    executor = slurm_executor(
        account          = ACCOUNT,
        partition        = PARTITION,
        nodes            = NODES,
        ntasks_per_node  = GPUS_PER_NODE,
        gpus_per_node    = GPUS_PER_NODE,
        node_job_dir     = JOB_DIR,
        container_mounts = CUSTOM_MOUNTS,
        container_image  = CONTAINER_IMAGE,
        nodelist         = NODELIST,
        exclude          = EXCLUDE,
        env_vars         = ENV_VARS,
    )

    # detach=False keeps stdout/stderr streaming in your terminal
    run.run(recipe, executor=executor, name=RUN_NAME, detach=True)


if __name__ == "__main__":
    main()
