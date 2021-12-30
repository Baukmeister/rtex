import os

import pandas as pd

import data_handler
import data_loader
import rtex_r
import rtex_t
from lime_rtex_r import plot_explainability_rtex_r
from lime_rtex_t import plot_explainability_rtex_t

eval_modules = [
    "rtex_r"
 #   ,"rtex_t"
]

# preparing the test data
data_loader.load_data()

tags_df = pd.read_csv("data/tags/mti_tags.csv", header=None)
tags_list = tags_df[0].to_list()

train = pd.read_csv("data/tags/iu_xray_abnormal_train.tsv", sep="\t")
train_cases_images = dict(zip(train.reports, train.images))
train_cases_tags = dict(zip(train.reports, train.mti_tags))
train_cases_captions = dict(zip(train.reports, train.captions))

test = pd.read_csv("data/tags/iu_xray_all_test.tsv", sep="\t")
test_cases_images = dict(zip(test.reports, test.images))
test_case_ids = list(test_cases_images.keys())

x_test = data_handler.encode_images(test_cases_images, "data/images/iu_xray")

if not os.path.exists("plots"):
    os.mkdir("plots")

for module in eval_modules:

    if module == "rtex_r":
        # running RTEX@R
        use_abnormal_images = True

        abnormal_images, rtex_r_model = rtex_r.rate_images(
            test_case_ids,
            x_test,
            test_cases_images,
            clean=False,
            abnormal=use_abnormal_images)

        abnormal_test_case_images = {k: test_cases_images[k] for k in abnormal_images.keys()}

        plot_explainability_rtex_r(
            image_paths=abnormal_test_case_images,
            num=20,
            model=rtex_r_model,
            method="lime",
            lime_samples=1000,
            lime_features=10,
            save_figs=True,
            abnormal=use_abnormal_images
        )

    elif module == "rtex_t":
        image_tags, all_tags, rtex_t_model = rtex_t.tag_images(
            x_test,
            test_cases_images,
            "data/tags/mti_tags.csv"
        )

        plot_explainability_rtex_t(
            image_tags,
            all_tags=all_tags,
            image_paths=test_cases_images,
            num=20,
            model=rtex_t_model,
            method="lime",
            lime_samples=1000,
            num_top_labels=4,
            lime_features=10,
            save_figs=True
        )
