
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="sulabhkatiyar/indic-tts-mini-24k",
    filename="index/parts/hi_M.parquet",
    repo_type="dataset"
)

df = pd.read_parquet(path)

print("\n1. AUDIO SHARDS")
print("Number of unique shards:", df["shard"].nunique())
print(df["shard"].value_counts().head(10))

print("\n2. RECORDING DURATION")
print(df["duration_s"].describe())

print("\n3. MISSING VALUES")
print(df.isnull().sum())

print("\n4. DUPLICATE IDS")
print("Duplicate IDs:", df["id"].duplicated().sum())

print("\n5. SPLIT DISTRIBUTION")
print(df.groupby("split").agg(
    recordings=("id", "count"),
    total_duration=("duration_s", "sum"),
    avg_duration=("duration_s", "mean")
))

print("\n6. SAMPLE AUDIO FILE REFERENCES")
print(df[[
    "id",
    "shard",
    "member_offset",
    "member_size",
    "duration_s",
    "split",
    "text"
]].head(10).to_string(index=False))
