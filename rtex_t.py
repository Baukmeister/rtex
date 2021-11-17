from keras.models import load_model
import json
import os

import data_handler


def tag_images(
        test_case_ids,
        x_test,
        test_cases_images,
        tag_list,
        clean=True,
        num=100,
        abnormal=True
):
    """
    Performs the tagging of abnormal images using RTEX@R
    :param test_case_ids:
    :param x_test:
    :param test_cases_images:
    :param tag_list:
    :param clean: if True the prediction is performed in any case if False a dump file is loaded if it exists
    :return: a dict containing the abnormal image paths
    :param num: number of images that should be returned
    :param abnormal: whether to return abnormal cases or not
    """
    rtex_t_model = load_model("data/models/rtex_r/iu_xray_bi_cxn.hdf5")

    abnormal_x_test = data_handler.encode_images(test_cases_images, "iu_xray")

    # Get predictions for test set
    test_tag_probs = rtex_t_model.predict(abnormal_x_test, batch_size=16, verbose=1)

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
    return results
