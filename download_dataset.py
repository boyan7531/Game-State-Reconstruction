from huggingface_hub import snapshot_download
snapshot_download(repo_id="SoccerNet/SN-GSR-2025",
                  repo_type="dataset", revision="main",
                  local_dir="SoccerNet/SN-GSR-2025")
print("Dataset downloaded successfully to SoccerNet/SN-GSR-2025")