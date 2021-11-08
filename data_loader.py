import dload
import os


def load_data():
    """
    Download the required data used for RTEX classification (if they are not already present)
    """
    if not os.path.isdir("data/tags"):
        print("Downloading image tags ...")
        dload.save_unzip("https://drive.google.com/uc?id=1nubphDVrKpB3Ss9uNaxHLUf2DWpqWRzq", "data/tags")
        print("Finished downloading image tags!")
        os.remove("uc")
    if not os.path.isdir("data/images"):
        print("Downloading images (this may take a while) ...")
        dload.save_unzip("https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz", "data/images")
        print("Finished downloading images!")
