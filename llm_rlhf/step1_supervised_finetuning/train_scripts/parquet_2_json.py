import pandas as pd

parquet_file = '/share/project/weiyifan/KG_RAG/data/benchmark_data/train_data/train-00000-of-00001.parquet'
json_file = '/share/project/weiyifan/KG_RAG/data/benchmark_data/train_data/PubMedQA_train.json'


def parquet_to_json(parquet_file: str, json_file: str, orient: str = "records"):
    """
    Convert a Parquet file to JSON format.

    :param parquet_file: Path to the input Parquet file.
    :param json_file: Path to save the output JSON file.
    :param orient: The JSON format (default is "records", suitable for most cases).
    """
    df = pd.read_parquet(parquet_file)
    df.to_json(json_file, orient=orient, indent=4, force_ascii=False)
    print(f"Conversion complete: {parquet_file} -> {json_file}")


parquet_to_json(parquet_file, json_file)