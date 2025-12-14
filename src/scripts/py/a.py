import json
import os
import shutil

files = os.listdir("datamining/CNSE/metadata/annotations/")

for file in files:
    path: str = os.path.join("datamining/CNSE/metadata/annotations/", file)
    if os.path.isdir(path):
        continue
    with open(os.path.join("datamining/CNSE/metadata/annotations/", file), "r") as f:
        data = json.load(f)
        if data["num_signers"] == 0:
            shutil.move(path, "datamining/CNSE/metadata/annotations/failed")
