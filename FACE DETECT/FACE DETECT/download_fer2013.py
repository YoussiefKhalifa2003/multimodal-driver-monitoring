import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")
print("Path to dataset files:", path)

# Search for fer2013.csv in all subdirectories
csv_found = False
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower() == "fer2013.csv":
            csv_path = os.path.join(root, file)
            shutil.copy(csv_path, "./fer2013.csv")
            print(f"{file} copied to current directory from {csv_path}.")
            csv_found = True
            break
    if csv_found:
        break
if not csv_found:
    print("fer2013.csv not found in the downloaded dataset.") 