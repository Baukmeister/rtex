import numpy as np
import pandas as pd
from keras.models import load_model
import json
import os


def tag_images(
        test_cases_images,
        tag_list_file,
        clean=False
):
    """
    Performs the tagging of abnormal images using RTEX@R
    :param test_case_ids:
    :param x_test:
    :param test_cases_images:
    :param tag_list_file:
    :param clean: if True the prediction is performed in any case if False a dump file is loaded if it exists
    :return: a dict containing the abnormal image paths
    :param num: number of images that should be returned
    :param abnormal: whether to return abnormal cases or not
    """

    rtex_t_model = load_model("data/models/rtex_t/iu_xray_tag_cxn.hdf5")
    dump_file_name = "data/rtex_t_tags_pre_calc.txt"

    if not clean and os.path.isfile(dump_file_name):
        print("Using pre-stored RTEX@T results from dump file!")
        predicted_tags_file = open(dump_file_name, "r")
        test_tag_probs = np.loadtxt(predicted_tags_file)
    else:
        print("Performing RTEX@T predictions!")
        test_tag_probs = rtex_t_model.predict(test_cases_images, batch_size=16, verbose=1)
        predicted_tags_file = open(dump_file_name, "w")
        np.savetxt(predicted_tags_file, test_tag_probs)

    tag_df = pd.read_csv(tag_list_file, header=None)
    tag_list = tag_df[0].to_list()

    best_threshold = 0.097

    tagging_results = {}
    # for each exam, assign all tags above threshold
    for i in range(len(test_tag_probs)):
        predicted_tags = []
        for j in range(len(tag_list)):
            if test_tag_probs[i, j] >= best_threshold:
                predicted_tags.append(tag_list[j])
        tagging_results[list(test_cases_images.keys())[i]] = ";".join(predicted_tags)

    results = list(tagging_results.items())
    return results, rtex_t_model
