"""
This script loads the corresponding to repot text for certain patient IDs
"""
import json
import os
import pydicom
from PIL import Image
import pandas as pd

DATA_DIR = "./data/tags"
OUTPUT_DIR = "./manual_classification"

groups = pd.read_csv(f"{DATA_DIR}/groups.csv", sep=";")
tags = pd.read_csv(f"{DATA_DIR}/iu_xray_all_test.tsv", sep="\t")
result = pd.DataFrame(columns=["pat_id", "sex", "normal"])

group_pat_id_list = groups.groupby("group")

for _, relevant_group in group_pat_id_list:
    relevant_group_name = f"group_{int(relevant_group['group'].iloc[0])}"

    for _, row in relevant_group.iterrows():
        pat_id = row["pat_id"]
        pat_tags = tags[tags["reports"] == pat_id]["mti_tags"]
        is_normal = "Normal" if (pat_tags == "none").iloc[0] else "Abnormal"
        result = result.append(
            {'pat_id': pat_id, 'sex': row['sex'], 'normal': is_normal},
            ignore_index=True,
        )

result.to_csv(f"{OUTPUT_DIR}/sex_matched_normality.csv")