from keras.models import load_model
import json
import os


def rate_images(test_case_ids, x_test, test_cases_images, clean=True, dump_file_name="data/abnormal_cases_images.json"):
    """
    Performs the ranking of abnormal images using RTEX@R
    :param test_case_ids:
    :param x_test:
    :param test_cases_images:
    :param clean: if True the prediction is performed in any case if False a dump file is loaded if it exists
    :param dump_file_name: the name of the dump file used for storing the abnormal image paths
    :return: a dict containing the abnormal image paths
    """
    rtex_r_model = load_model("data/models/iu_xray_bi_cxn.hdf5")

    if not clean and os.path.isfile(dump_file_name):
        print("Using pre-stored RTEX@R results from dump file!")
        abnormal_cases_file = open("data/abnormal_cases_images.json", "r")
        abnormal_cases_images = json.load(abnormal_cases_file)
    else:
        test_abn_probs = rtex_r_model.predict(x_test, batch_size=16, verbose=1).flatten()

        cases_probs = dict(zip(test_case_ids, test_abn_probs))
        # Sort all exams (a.k.a. cases)
        sorted_cases_probs = {k: v for k, v in sorted(cases_probs.items(), key=lambda item: item[1], reverse=True)}
        sorted_cases = list(sorted_cases_probs.keys())
        # Get the top 100 abnormal exams
        abnormal_cases_images = {case: test_cases_images[case] for case in sorted_cases[:100]}

        abnormal_cases_file = open(dump_file_name, "w")
        json.dump(abnormal_cases_images, abnormal_cases_file)

    return abnormal_cases_images, rtex_r_model
