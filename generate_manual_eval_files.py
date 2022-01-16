import os.path
import random

import pandas as pd
import matplotlib.pyplot as plt
from natsort import os_sorted

import data_loader
from data_handler import load_image
from utils import stitchImages

num_groups = 4
group_size = 100
output_folder = "manual_classification"
input_folder = "data/images/iu_xray"

data_loader.load_data()
test = pd.read_csv("data/tags/iu_xray_all_test.tsv", sep="\t")
test_cases_images = dict(zip(test.reports, test.images))

patient_ids = list(test_cases_images.keys())
random.shuffle(patient_ids)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

pat_idx = 0

for group_nr in range(num_groups):
    group_pats = patient_ids[pat_idx:pat_idx + group_size]
    print(f"IDs for group {group_nr}")
    print("\n".join(os_sorted(group_pats)))
    pat_idx += group_size

    group_output_folder = f"{output_folder}/group_{group_nr}"

    if not os.path.exists(group_output_folder):
        os.mkdir(group_output_folder)

    i = 0
    for pat in group_pats:
        file_name = f"{group_output_folder}/{pat}.jpg"

        images = test_cases_images[pat].split(";")

        img1 = load_image(os.path.join(input_folder, images[0]))
        img2 = load_image(os.path.join(input_folder, images[1]))
        stitched_images = stitchImages(img1, img2)
        plt.imshow(stitched_images)
        plt.title(f"Patient: {pat}")
        plt.savefig(file_name)
        i += 1
        print(f"Saved {pat} for group {group_nr}. {i}/{group_size}")
