from keras.models import load_model
import json
import os


def rate_images(
        test_case_ids,
        x_test,
        test_cases_images,
        clean=True,
        num=100,
        abnormal=True
):
    """
    Performs the ranking of abnormal images using RTEX@R
    :param test_case_ids:
    :param x_test:
    :param test_cases_images:
    :param clean: if True the prediction is performed in any case if False a dump file is loaded if it exists
    :return: a dict containing the abnormal image paths
    :param num: number of images that should be returned
    :param abnormal: whether to return abnormal cases or not
    """
    rtex_r_model = load_model("data/models/iu_xray_bi_cxn.hdf5")

    dump_file_name = "data/case_probs_pre_calc.json"

    if not clean and os.path.isfile(dump_file_name):
        print("Using pre-stored RTEX@R results from dump file!")
        cases_probs_file = open(dump_file_name, "r")
        cases_probs = json.load(cases_probs_file)
    else:
        print("Performing RTEX@R predictions!")
        test_abn_probs = rtex_r_model.predict(x_test, batch_size=16, verbose=1).flatten()
        cases_probs_float_32 = dict(zip(test_case_ids, test_abn_probs))
        cases_probs = dict(map(lambda x: (x[0], float(x[1])), cases_probs_float_32.items()))
        cases_probs_file = open(dump_file_name, "w")
        json.dump(cases_probs, cases_probs_file)

    # Sort all exams (a.k.a. cases)
    sorted_cases_probs = {k: v for k, v in sorted(cases_probs.items(), key=lambda item: item[1], reverse=True)}
    sorted_cases = list(sorted_cases_probs.keys())

    if abnormal:
        # Get the top abnormal exams
        case_images = {case: test_cases_images[case] for case in sorted_cases[:num]}
    else:
        # Get the top normal exams
        case_images = {case: test_cases_images[case] for case in sorted_cases[-num:]}

    return case_images, rtex_r_model
