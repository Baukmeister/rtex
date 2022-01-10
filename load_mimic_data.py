"""
This script loads a selection of patient reports from the MIMIC-CXR dataset and converts the DICOM images to png.
"""
import json
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image


TMP_DIR = ".tmp"
DATA_DIR = "./data/images/mimic"
RE_DOWNLOAD = False

if not os.path.isdir(TMP_DIR):
    os.mkdir(TMP_DIR)

with open("./mimic_config.json", "rb") as f:
    mimic_config = json.load(f);

pat_group = mimic_config["pat_group"]
patient_ids = mimic_config["patient_ids"]

# download the patient reports
for patient_id in patient_ids:
    if RE_DOWNLOAD:
        url = f"{mimic_config['physionet_url']}/{pat_group}/{patient_id}/"
        os.system(f"wget -r -N -c -np "
                  f"--user={mimic_config['physionet_user']} "
                  f"--password={mimic_config['physionet_pass']} "
                  f"--directory-prefix=./{TMP_DIR} "
                  f"{url}")

# Convert the dicom files for the reports to png files and store them in a flat list

download_path = f"{TMP_DIR}/physionet.org/files/mimic-cxr/2.0.0/files/{pat_group}"
patient_folders = os.listdir(download_path)

for patient_folder in patient_folders:
    report_folders = [elem for elem in os.listdir(f"{download_path}/{patient_folder}") if elem.startswith("s") and not "." in elem]

    for report_folder in report_folders:
        report_scans = [elem for elem in os.listdir(f"{download_path}/{patient_folder}/{report_folder}") if elem.endswith(".dcm")]

        for report_scan in report_scans:
            ds = pydicom.read_file(f"{download_path}/{patient_folder}/{report_folder}/{report_scan}")  # read dicom image
            img_array = ds.pixel_array  # get image array

            arr = img_array - img_array.mean()
            scaled_arr = (arr / arr.max()) * 255

            im = Image.fromarray(scaled_arr).convert('RGB')

            out_dir = f"{DATA_DIR}"

            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            out_path = f"{out_dir}/{patient_folder}_{report_folder}_{report_scan[0:10]}.png"

            im.save(out_path)
            print(f"Saved {report_scan} for {patient_id}")

