import pandas as pd
from keras.models import load_model

import data_handler
import data_loader

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

bi_cxn = load_model("data/models/iu_xray_bi_cxn.hdf5")

test_abn_probs = bi_cxn.predict(x_test, batch_size=16, verbose=1).flatten()

cases_probs = dict(zip(test_case_ids, test_abn_probs))
# Sort all exams (a.k.a. cases)
sorted_cases_probs = {k: v for k, v in sorted(cases_probs.items(), key=lambda item: item[1], reverse=True)}
sorted_cases = list(sorted_cases_probs.keys())
# Get the top 100 abnormal exams
abnormal_cases_images = {case: test_cases_images[case] for case in sorted_cases[:100]}
print(abnormal_cases_images)
