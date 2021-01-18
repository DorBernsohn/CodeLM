import torch
import random
import subprocess
import numpy as np
from tqdm.notebook import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def install_dependencies(package_names: list) -> None:
  for i, package_name in enumerate(tqdm(package_names)):
    print(f"{i + 1}. pip install {package_name}", flush=True)
    subprocess.call(["pip", "install", package_name])

def print_items_from_dict(d:dict) -> None:
    print("Items held:")
    for item, amount in d.items():
        print(f"{item} ({amount})")