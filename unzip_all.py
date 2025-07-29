import os
import zipfile

base_dir = "SoccerNet/SN-GSR-2025"
for filename in os.listdir(base_dir):
    if filename.endswith(".zip"):
        zip_path = os.path.join(base_dir, filename)
        extract_dir = os.path.join(base_dir, filename.replace(".zip", ""))
        print(f"Extracting {zip_path} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
print("All zip files extracted.")