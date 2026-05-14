import pandas as pd
import os
import requests
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from CompLexPerAnnotator.retriever import Retriever

KEEP = {
    "HITId": "task_id",
    "WorkerId": "annotator_id",
    "Input.corpus_id": "corpus",
    "Input.sentence": "sentence",
    "Input.token": "token",
    "Answer.sentiment.label": "complexity",
}

LABEL_MAP = {
    "Very Easy": 0,
    "Easy": 1,
    "Neutral": 2,
    "Difficult": 3,
    "Very Difficult": 4,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()} # inverted

COLUMNS = ["HITId","HITTypeId","Title","Description","Keywords","Reward","CreationTime","MaxAssignments","RequesterAnnotation","AssignmentDurationInSeconds","AutoApprovalDelayInSeconds","Expiration","NumberOfSimilarHITs","LifetimeInSeconds","AssignmentId","WorkerId","AssignmentStatus","AcceptTime","SubmitTime","AutoApprovalTime","ApprovalTime","RejectionTime","RequesterFeedback","WorkTimeInSeconds","LifetimeApprovalRate","Last30DaysApprovalRate","Last7DaysApprovalRate","Input.corpus_id","Input.file_id","Input.token_id","Input.sentence","Input.token","Input.begin","Input.end","Answer.sentiment.label","Approve","Reject"]


def load(path):
    raw = pd.read_csv(path, names=COLUMNS, low_memory=False)
    print(f"Loaded {len(raw):,} rows from {path}")
    return raw


def select_columns(raw):
    df = raw[list(KEEP.keys())].rename(columns=KEEP).copy()
    return df

def filter_mwes(df):
    print("Filtering out MWEs")
    before = len(df)
    df = df[~df["token"].str.contains(r"\s", regex=True, na=False)].reset_index(drop=True)
    print(f"Rows after filtering MWEs: {len(df):,} ({before - len(df):,} dropped)")
    return df


def filter_missing(df):
    print("Filtering out missing values")
    before = len(df)
    df = df.dropna()
    for col in df.select_dtypes(include="object").columns:
        df = df[df[col].str.strip() != ""]
    df = df.reset_index(drop=True)
    print(f"Rows after filtering missing values: {len(df):,} ({before - len(df):,} dropped)")
    return df


def map_labels(df):
    print("Mapping labels")
    before = len(df)
    df["complexity"] = df["complexity"].map(LABEL_MAP)
    unmapped = df["complexity"].isna().sum()
    if unmapped:
        print(f"{unmapped:,} rows had unmapped labels and will be dropped")
    df = df.dropna().reset_index(drop=True)
    df["complexity"] = df["complexity"].astype(int) / 4.0
    print(f"Rows after label mapping: {len(df):,} ({before - len(df):,} dropped)")
    return df

def split_dataset(df, test_size: float, seed: int, n_buckets: int = 5) -> DatasetDict:
    """
    Splits the dataset into train and test users

    Bucket users by history length and split proportionally within each bucket

    Params:
        df: pandas Dataframe
        test_size: ratio of users used for testing
        seed: Random seed for reproduceability
        n_buckets: number of history-length quantile buckets

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    user_counts = df.groupby("annotator_id").size().reset_index(name="count")
    user_counts["bucket"] = pd.qcut(
        user_counts["count"], q=n_buckets, duplicates="drop", labels=False
    )

    test_users = (
        user_counts
        .groupby("bucket", group_keys=False)["annotator_id"]
        .apply(lambda s: s.sample(frac=test_size, random_state=seed))
    )
    test_user_ids = set(test_users)

    train_df = df[~df["annotator_id"].isin(test_user_ids)].reset_index(drop=True)
    test_df = df[df["annotator_id"].isin(test_user_ids)].reset_index(drop=True)

    print(
        f"Train: {len(train_df):,} rows from {train_df['annotator_id'].nunique()} users | "
        f"Test: {len(test_df):,} rows from {test_df['annotator_id'].nunique()} users"
    )

    return DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })

def download_dataset(cache_dir: str) -> str:
    """
    Download the per-annotator lexical complexity dataset if not already cached.

    Downloads from https://github.com/MMU-TDMLab/LCP_Subjectivity.

    Params:
        cache_dir: Directory to save/load the dataset file

    Returns:
        Local path to the downloaded CSV file
    """
    url = "https://raw.githubusercontent.com/MMU-TDMLab/LCP_Subjectivity/master/LCP_2021/batchResults/all.csv"
    local_path = os.path.join(cache_dir, "all.csv")

    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(local_path):
        print("Downloading per-annotator data")
        response = requests.get(url)
        if not response.ok:
            raise RuntimeError(f"Failed to download per-annotator data: {response.status_code}")
        with open(local_path, "w") as f:
            f.write(response.text)

    return local_path


def load_dataset(cache_dir: str = "./data/per_annotator", test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the per-annotator lexical complexity dataset as a Hugging Face DatasetDict.

    Downloads from https://github.com/MMU-TDMLab/LCP_Subjectivity if not already cached.

    Params:
        cache_dir: Directory to save/load the dataset file
        test_size: Fraction of data to use as test set
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with keys 'train' and 'test'
    """
    local_path = download_dataset(cache_dir)

    raw = load(local_path)
    df = select_columns(raw)
    df = map_labels(df)
    df = filter_mwes(df)
    df = filter_missing(df)

    return split_dataset(df, test_size=test_size, seed=seed)

def tokenize_per_annotator_dataset(
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        retriever_map: dict[str, Retriever],
        user_history_length: int = 5
    ):
    """
    Tokenize the CompLex dataset for sequence classification.

    Encodes each example as [CLS] user history [SEP] sentence [SEP] token [SEP].

    Note: the user history is currently encoded as follows <token>: <score>, where score is one of
    Very Easy, Easy, ..., Very Difficult

    Params:
        dataset: DatasetDict with 'train' and 'test' splits
        tokenizer: Tokenizer to use
        retriever_map: mapping of annotator_id -> Retriever. The retriever is used to select the relevant items from the user history
        user_history_length: number of items of the user history that should be inside the prompt
    Returns:
        Tokenized DatasetDict formatted as torch tensors
    """

    def tokenize(row):
        retriever = retriever_map[row["annotator_id"]]
        user_history = retriever(sample=row, n=user_history_length)
        user_history_str = ""
        for item in user_history:
            score_str = LABEL_NAMES[round(item["complexity"] * 4)]
            user_history_str += f"{item['token']}: {score_str}, "
        user_history_str = user_history_str.removesuffix(", ")

        context = row["sentence"]
        token = row["token"]

        return tokenizer(
            user_history_str,
            f"{context} {token}",
            padding="max_length",
            truncation="only_first", 
            max_length=512,
            return_token_type_ids=True,
        )
    
    dataset = dataset.map(tokenize, batched=False)
    dataset = dataset.remove_columns(["annotator_id", "task_id", "corpus", "sentence", "token"])
    dataset["train"] = dataset["train"].rename_column("complexity", "labels") # the trainer expects labels
    dataset["test"] = dataset["test"].rename_column("complexity", "labels") # the trainer expects labels
    dataset.set_format("torch")
    
    return dataset 


def get_user_histories(dataset: DatasetDict) -> dict[str, list[dict]]:
    """
    Build a per-annotator history from the dataset.

    Params:
        dataset: DatasetDict with 'train' and 'test' splits

    Returns:
        Dict mapping annotator_id -> list of row dicts (keys: task_id, annotator_id, corpus, sentence, token, complexity)
    """
    history: dict[str, list[dict]] = {}
    for key, split in dataset.items():
        for row in split:
            aid = row["annotator_id"]
            if aid not in history:
                history[aid] = []
            history[aid].append(dict(row))
    return history
