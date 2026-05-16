
import pandas as pd
import os
import requests
import math

from datasets import Dataset, DatasetDict


def load_dataset(cache_dir: str = "./data/complex") -> DatasetDict:
    """
    Load the CompLex lexical complexity prediction dataset as a Hugging Face DatasetDict.

    Downloads from https://github.com/MMU-TDMLab/CompLex if not already cached.

    Params:
        cache_dir: Directory to save/load the dataset files

    Returns:
        DatasetDict with keys 'train' and 'test', each combining single-word and MWE
        data with an additional 'task' column ('single' or 'multi')
    """
    
    base_url = "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master"
    splits = ['train', 'trial'] # test has no labels
    tasks = ['single', 'multi']
    
    os.makedirs(cache_dir, exist_ok=True)
    
    
    # download and save data first 
    for split in splits:
        for task in tasks:
            filename = f"lcp_{task}_{split}.tsv"
            local_path = os.path.join(cache_dir, filename)

            if not os.path.exists(local_path):
                print(f"Downloading {filename}")
                url = f"{base_url}/{split}/{filename}"
                response = requests.get(url)
                if not response.ok:
                    raise RuntimeError(f"Failed to download {filename}")
                
                # save data
                with open(local_path, "w") as f:
                    text = response.text.replace('"', '').replace("'", '') # quotation marks break tsv
                    f.write(text)

    # now try to load it from the downloaded files
    dataset_dict = {}
    for split in splits:
        dfs = []
        for task in tasks:
            filename = f"lcp_{task}_{split}.tsv"
            local_path = os.path.join(cache_dir, filename)

            try:
                df = pd.read_csv(local_path, sep="\t")
            except Exception as e:
                print(f"Failed to load {filename}: {e} ")
            
            df.columns = [col.lower().strip() for col in df.columns]
            df['task'] = task
            dfs.append(df)

        
        # creae
        combined_df = pd.concat(dfs, ignore_index=True)

        dataset_dict[split] = Dataset.from_pandas(combined_df, preserve_index=False)

        if split == "trial":
            dataset_dict["trial"] = dataset_dict["trial"].rename_column("subcorpus", "corpus")
            # dirty fix to load trial as test
            dataset_dict["test"] = dataset_dict.pop("trial")
    
    return DatasetDict(dataset_dict)

def preprocess_data(dataset: DatasetDict):
    """
    Filter out rows with missing values or invalid complexity labels.

    Params:
        dataset: DatasetDict with 'train' and 'test' splits

    Returns:
        Filtered DatasetDict with only valid rows
    """
    def no_missing(row):
        for v in row.values():
            if v is None:
                return False
            if isinstance(v, float) and math.isnan(v):
                return False
            if isinstance(v, str) and v.strip() == "":
                return False
        return True

    # filter out rows that contain any empty value
    train = dataset["train"].filter(
        lambda row: no_missing(row) 
    )
    # filter out invalid labels
    train = train.filter(
        lambda row: 0.0 <= row["complexity"] <= 1.0 
    )
    # filter out rows that contain any empty value
    test = dataset["test"].filter(
        lambda row: no_missing(row) 
    )
    return DatasetDict(dict(train=train, test=test)) 

def tokenize_complex_dataset(
        dataset: DatasetDict,
        tokenizer,
        max_length: int = 128,
        ):
    """
    Tokenize the CompLex dataset for sequence classification.

    Encodes each example as [CLS] sentence [SEP] token [SEP].

    Params:
        dataset: DatasetDict with 'train' and 'test' splits
        tokenizer: Tokenizer to use
        max_length: Maximum token sequence length

    Returns:
        Tokenized DatasetDict formatted as torch tensors
    """

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            batch["token"],
            padding="max_length",
            truncation="only_first", # 128 tokens should be more than enough
            max_length=max_length,
            return_token_type_ids=True,
        )
    
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["id", "corpus", "sentence", "token", "task"])
    dataset["train"] = dataset["train"].rename_column("complexity", "labels") # the trainer expects labels
    dataset["test"] = dataset["test"].rename_column("complexity", "labels") # the trainer expects labels
    dataset.set_format("torch")
    
    return dataset 

