import os

path = "/Users/yousef/.cache/kagglehub/datasets/msambare/fer2013/versions/1"
for root, dirs, files in os.walk(path):
    print("Directory:", root)
    for file in files:
        print("  -", file)