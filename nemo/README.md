# NeMo for Qwen2.5 Continual Learning

[Notion Page](https://www.notion.so/NeMo-1c81cbf802fa80abba21cc1fc1d29580?pvs=4)

## Install NeMo
> https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html#install-nemo-framework

I recommend using nemo image.
``` shell
IMAGE="nvcr.io/nvidia/nemo:25.02"
CONT_NAME="nemo_test"
# GPUS="4,5"

docker run -it -d --name "$CONT_NAME" \
  --gpus "\"device=${GPUS}"\" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=512g \
  -v /NAS_NAENGJANGO:/NAS_NAENGJANGO \
  $IMAGE
``` 


## Download and Prepare Data
Run scripts/download_hf_dataset.py to download and convert the dataset to json format.

Modify `scripts/download_hf_dataset.py` for your use case.


## Preprocess Data for NeMo
https://docs.nvidia.com/nemo-framework/user-guide/latest/data/pretrain_data.html

Run `scripts/preprocess_data_for_megatron.py` to preprocess the data for NeMo.


> [!NOTE] 
> You need to specify the `--tokenizer-type` to be `Qwen/Qwen2.5-1.5B` for Qwen2.5-1.5B.

``` Python
INPUT_DIR=/NAS_NAENGJANGO/BFS/pretrain/maumweb-edu/data/filtered_jsonl

python preprocess_data_for_megatron.py \
	--preproc-folder \
    --input=$INPUT_DIR \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type=Qwen/Qwen2.5-1.5B \
    --dataset-impl=mmap \
    --output-prefix=my-qwen \
    --append-eod \
    --workers=64
```

After running the script, you should see the files in the working directory:

    my-qwen_text_document/qwen_text_document.bin
    my-qwen_text_document/qwen_text_document.idx

## Train with NeMo
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html
https://docs.nvidia.com/nemo-framework/user-guide/latest/continuetraining.html

Run `scripts/qwen_continual_learning.py` to train the model. Modify the script for your use case.

> [!NOTE] 
> To estimate the optimal configuation, refer to the https://docs.nvidia.com/nemo-framework/user-guide/latest/usingautoconfigurator.html

> [!IMPORTANT] 
> For slurm, you need to define slrum executor in the script.

