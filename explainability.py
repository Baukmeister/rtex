import pandas as pd
import data_handler
import data_loader
import rtex_r
from data_visualizer import show_images

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

# running RTEX@R
abnormal_images, rtex_model = rtex_r.rate_images(test_case_ids, x_test, test_cases_images, clean=False)
abnormal_test_case_images = {k: test_cases_images[k] for k in abnormal_images.keys()}
show_images(
    image_paths=abnormal_test_case_images,
    num=5,
    explainability=True,
    model=rtex_model,
    lime_samples=50
)
