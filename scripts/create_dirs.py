import os

base_dir = "d:/projects/lunwen2/reproduction"
subdirs = ["scripts", "configs", "data", "models", "results"]

os.makedirs(base_dir, exist_ok=True)
for subdir in subdirs:
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")

print("Directory structure created successfully.")