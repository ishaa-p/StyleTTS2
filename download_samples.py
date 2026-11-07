
import os
import io
import numpy as np
import pandas as pd
import requests
import soundfile as sf

from huggingface_hub import hf_hub_download, hf_hub_url

REPO = "sulabhkatiyar/indic-tts-mini-24k"

OUTPUT_DIR = "data/hi_M_inspection"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

os.makedirs(AUDIO_DIR, exist_ok=True)

# Load the metadata
index_path = hf_hub_download(
    repo_id=REPO,
    filename="index/parts/hi_M.parquet",
    repo_type="dataset"
)

df = pd.read_parquet(index_path)

# Select validation recordings only
val_df = df[df["split"] == "val"].copy()

# Select 10 recordings spread across different durations
val_df = val_df.sort_values("duration_s")

positions = np.linspace(
    0, len(val_df) - 1, 10, dtype=int
)

samples = val_df.iloc[positions]

transcripts = []

for i, (_, row) in enumerate(samples.iterrows(), start=1):

    print(f"\nDownloading sample {i}/10")

    # Calculate exact byte range inside the TAR archive
    start = int(row["member_offset"])
    end = start + int(row["member_size"]) - 1

    url = hf_hub_url(
        REPO,
        row["shard"],
        repo_type="dataset"
    )

    response = requests.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=60
    )

    # Ensure the server returned only the requested bytes
    if response.status_code != 206:
        raise RuntimeError(
            f"Expected HTTP 206, got {response.status_code}"
        )

    if len(response.content) != int(row["member_size"]):
        raise RuntimeError("Downloaded audio size does not match metadata")

    # Decode FLAC
    audio, sample_rate = sf.read(
        io.BytesIO(response.content)
    )

    if sample_rate != 24000:
        raise ValueError(f"Unexpected sample rate: {sample_rate}")

    filename = f"sample_{i:02d}_{row['id'][:8]}.wav"
    output_path = os.path.join(AUDIO_DIR, filename)

    # Save as 16-bit PCM WAV
    sf.write(
        output_path,
        audio,
        sample_rate,
        subtype="PCM_16"
    )

    transcripts.append({
        "filename": filename,
        "text": row["text"],
        "duration_s": row["duration_s"],
        "split": row["split"],
        "sample_rate": sample_rate,
        "channels": 1 if audio.ndim == 1 else audio.shape[1]
    })

    print("Saved:", output_path)
    print("Transcript:", row["text"])
    print("Audio shape:", audio.shape)
    print("Sample rate:", sample_rate)

# Save matching transcripts
transcript_df = pd.DataFrame(transcripts)

transcript_df.to_csv(
    os.path.join(OUTPUT_DIR, "transcripts.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nAll 10 samples downloaded successfully!")
print("Output directory:", OUTPUT_DIR)
