from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import polars as pl
from tqdm import tqdm
import os

def _convert_parquet_to_jsonl(parquet_path, jsonl_path):
    try:
        df = pl.read_parquet(parquet_path)
        df = df.select(['text'])
        
        df.write_ndjson(jsonl_path)
        
        print(f"Conversion completed. JSONL file saved to: {jsonl_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def convert_parquet_to_jsonl(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    parquet_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.parquet')]

    jsonl_files = [os.path.join(output_dir, file.replace('.parquet', '.jsonl')) for file in os.listdir(input_dir) if file.endswith('.parquet')]

    print("Converting parquet files to jsonl format... This may take a while.")
    for parquet_file, jsonl_file in tqdm(zip(parquet_files, jsonl_files), total=len(parquet_files)):
        _convert_parquet_to_jsonl(parquet_file, jsonl_file)


if __name__ == "__main__":
    load_dotenv()
       
    folder = snapshot_download(
                    "maum-ai/fineweb-edu-korean", 
                    repo_type="dataset",
                    local_dir="/NAS_NAENGJANGO/BFS/pretrain/maumweb-edu/",
                    allow_patterns=["data/filtered/*"])
    
    
    convert_parquet_to_jsonl("/NAS_NAENGJANGO/BFS/pretrain/maumweb-edu/data/filtered", "/NAS_NAENGJANGO/BFS/pretrain/maumweb-edu/data/filtered_jsonl")