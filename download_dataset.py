from huggingface_hub import snapshot_download

DATASET_REPO_ID = "SoccerNet/SN-GSR-2025"
LOCAL_DIR = "SoccerNet/SN-GSR-2025"

# Download only the training and validation archives/directories
snapshot_download(
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    revision="main",
    local_dir=LOCAL_DIR,
    allow_patterns=[
        # Only the zipped splits (including nested and case variants)
        "train.zip",
        "valid.zip",
        "**/train.zip",
        "**/valid.zip",
        "Train.zip",
        "Valid.zip",
        "**/Train.zip",
        "**/Valid.zip",
    ],
)
print("Downloaded only 'train' and 'valid' to SoccerNet/SN-GSR-2025")