import gdown
import os
from pathlib import Path
import shutil

def load_data():
    """
    Download the required data used for RTEX classification (if they are not already present)
    """
    Path("data").mkdir(parents=True, exist_ok=True)
    if not os.path.isdir("data/tags"):
        print("Downloading image tags ...")
        gdown.download("https://drive.google.com/uc?id=1nubphDVrKpB3Ss9uNaxHLUf2DWpqWRzq", "data/tags.zip")
        shutil.unpack_archive("data/tags.zip", "data/tags")
        print("Finished downloading image tags!")
    if not os.path.isdir("data/images"):
        print("Downloading images (this may take a while) ...")
        gdown.download("https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz", "data/images.tgz")
        shutil.unpack_archive("data/images.tgz", "data/images/iu_xray", format="tar")
        print("Finished downloading images!")
    if not os.path.isdir("data/models"):
        print("Downloading models ...")
        gdown.download("https://drive.google.com/uc?id=1D8oHHSib1k8QHnDD_LLzf6lLbyjrtfOm", "data/models.zip")
        shutil.unpack_archive("data/models.zip", "data/models")
        print("Finished downloading models!")

