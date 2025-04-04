from nemo import lightning as nl
import nemo_run as run
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer

from dotenv import load_dotenv
import os


def configure_recipe(nodes: int = 1, gpus_per_node: int = 4):
    # Convert HF model to NeMo format
    llm.import_ckpt(
        model=llm.Qwen2Model(llm.Qwen25Config1P5B()),
        source="hf://Qwen/Qwen2.5-1.5B",
        # output_path=Path('./adfadfadf'),
        overwrite=True,
    )

    recipe = llm.qwen25_1p5b.pretrain_recipe(
        dir="./checkpoints/qwen",  # Path to store checkpoints
        name="qwen_continual_learning",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        max_steps=96,  # ✅ Setting a small value for the quickstart
    )
    recipe.model.config.seq_length = 8192

    # Configure continual learning
    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig, path="nemo://Qwen/Qwen2.5-1.5B"),
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory="./checkpoints/qwen",
    )

    # Add overrides here
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.pipeline_model_parallel_size = 1
    recipe.trainer.strategy.context_parallel_size = 1
    # Modify Data Blend if needed
    # new_paths = [.3, "path/to/data1", 1.5, "path/to/data2"]
    new_paths = [1, "/root/llm-training-framework-comparision/nemo/scripts/data/my-qwen_text_document",]

    recipe.data = run.Config(
        llm.PreTrainingDataModule,
        paths=new_paths,
        seq_length=8192,
        tokenizer=run.Config(
            AutoTokenizer,
            pretrained_model_name="Qwen/Qwen2.5-1.5B",
        ),
        global_batch_size=8,
        micro_batch_size=1,
        # seed = 42,
        # split = "100,0,0",
    )

    # Modify Learning Rate Scheduler if needed
    # recipe.optim.lr_scheduler.warmup_steps = 20000
    # recipe.optim.lr_scheduler.min_lr = min_lr
    # recipe.optim.config.lr = max_lr

    # recipe.trainer.val_check_interval = 5 # Setting a small value for the quickstart

    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(
        ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars
    )

    return executor


def run_pretraining():
    recipe = configure_recipe()
    executor = local_executor_torchrun(
        nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices
    )

    executor.env_vars["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
    # executor.env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]

    run.run(recipe, executor=executor, name="qwen25_1p5b_pretraining")


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    load_dotenv()
    run_pretraining()
