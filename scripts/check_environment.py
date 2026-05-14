import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import cv2
from PIL import Image
import matplotlib
import yaml


def main():
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))
        print("torch cuda runtime:", torch.version.cuda)
        print(
            "gpu memory GB:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        )

    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("sklearn:", sklearn.__version__)
    print("opencv:", cv2.__version__)
    print("PIL:", Image.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("yaml imported successfully")
    print("Environment sanity check passed.")


if __name__ == "__main__":
    main()
