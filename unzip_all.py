import os
import zipfile

base_dir = "SoccerNet/SN-GSR-2025"
for filename in os.listdir(base_dir):
    if filename.endswith(".zip"):
        zip_path = os.path.join(base_dir, filename)
        extract_dir = os.path.join(base_dir, filename.replace(".zip", ""))
        print(f"Extracting {zip_path} to {extract_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            os.remove(zip_path)
            print(f"Deleted {zip_path}")
        except zipfile.BadZipFile:
            print(f"Skipping {zip_path}: not a valid zip file.")
        except Exception as e:
            print(f"Error processing {zip_path}: {e}")
print("All zip files extracted and original .zip files deleted.")