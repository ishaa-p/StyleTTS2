from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO = "sulabhkatiyar/indic-tts-mini-24k"

# Download metadata for the male Hindi speaker only
path = hf_hub_download(
    repo_id=REPO,
    filename="index/parts/hi_M.parquet",
    repo_type="dataset"
)

table = pq.read_table(path)
df = table.to_pandas()

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nSPLIT COUNTS:")
print(df["split"].value_counts())

print("\nDURATION BY SPLIT (hours):")
print(
    df.groupby("split")["duration_s"].sum() / 3600
)

print("\nSAMPLE TRANSCRIPTS:")
for _, row in df.head(10).iterrows():
    print(
        row["split"],
        "|",
        round(row["duration_s"], 2),
        "seconds |",
        row["text"]
    )

print("\nSPEAKER IDs:")
print(df["speaker"].unique())
