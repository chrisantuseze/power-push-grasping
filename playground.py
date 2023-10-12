

import os


directory = "/ppg-dataset/"
for root, dirs, files in os.walk(directory, topdown=False):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")