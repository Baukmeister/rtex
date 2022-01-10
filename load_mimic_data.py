"""
This script loads a selection of patient reports from the MIMIC-CXR dataset and converts the DICOM images to png.
"""
import json
import os
import pydicom
from PIL import Image

TMP_DIR = ".tmp"
DATA_DIR = "./data/images/mimic"
RE_DOWNLOAD = False
PAT_NUM = None

if not os.path.isdir(TMP_DIR):
    os.mkdir(TMP_DIR)

with open("./mimic_config.json", "rb") as f:
    mimic_config = json.load(f)

pat_group = mimic_config["pat_group"]
patient_ids = mimic_config["patient_ids"]

if PAT_NUM is not None:
    patient_ids = patient_ids[0:PAT_NUM]

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

if PAT_NUM is not None:
    patient_folders = patient_folders[0:PAT_NUM]

report_images_dict = {}

for patient_folder in patient_folders:
    report_folders = [elem for elem in os.listdir(f"{download_path}/{patient_folder}") if
                      elem.startswith("s") and not "." in elem]

    for report_folder in report_folders:
        report_scans = [elem for elem in os.listdir(f"{download_path}/{patient_folder}/{report_folder}") if
                        elem.endswith(".dcm")]

        # only choose patient reports with exactly two cans
        if len(report_scans) == 2:

            report_files = []
            for report_scan in report_scans:
                ds = pydicom.read_file(
                    f"{download_path}/{patient_folder}/{report_folder}/{report_scan}")  # read dicom image
                img_array = ds.pixel_array  # get image array

                arr = img_array - img_array.mean()
                scaled_arr = (arr / arr.max()) * 255

                im = Image.fromarray(scaled_arr).convert('RGB')

                out_dir = f"{DATA_DIR}"

                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)

                file_name = f"{patient_folder}_{report_folder}_{report_scan[0:10]}.png"
                out_path = f"{out_dir}/{file_name}"

                im.save(out_path)
                print(f"Saved {report_scan} for {patient_id}")
                report_files.append(file_name)

            report_images_dict[f"{patient_id}_{report_folder}"] = ";".join(report_files)

with open(f"{DATA_DIR}/report_images.json", "w") as f:
    json.dump(report_images_dict, f)
