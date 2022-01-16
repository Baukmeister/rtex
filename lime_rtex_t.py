import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime import lime_image
from skimage.segmentation import mark_boundaries

import data_handler
from utils import stitchImages


def plot_explainability_rtex_t(
        image_tags,
        image_paths,
        num,
        all_tags,
        path_prefix="data/images/iu_xray/",
        num_top_labels=5,
        model=None,
        img_width=224,
        method=None,
        lime_samples=1000,
        lime_features=10,
        save_figs=False,
):
    def _predict_rtex_t(pasted_image):
        x1_data, x2_data = [], []

        for image in pasted_image:
            img1 = image[:, 0:img_width, :]
            img2 = image[:, img_width:, :]

            x1_data.append(img1)
            x2_data.append(img2)

        encoded_images = [np.array(x1_data), np.array(x2_data)]
        model_predictions = model.predict(encoded_images)

        return model_predictions

    plots_folder = "plots/rtex_t"

    all_tags_series = pd.Series(all_tags)

    if save_figs:
        if os.path.exists(plots_folder):
            shutil.rmtree(plots_folder)

        os.mkdir(plots_folder)

    for i in range(num):
        image_path_key = list(image_tags.keys())[i]

        encoded_images, img1, img2 = data_handler.encode_images(
            {image_path_key: image_paths[image_path_key]},
            path_prefix,
            return_images=True
        )
        stitched_images = stitchImages(img1, img2)

        if method == "lime":

            # Lime explainer
            lime_explainer = lime_image.LimeImageExplainer()
            explanation = lime_explainer.explain_instance(
                stitched_images,
                _predict_rtex_t,
                hide_color=0,
                num_samples=lime_samples)

            for label_idx in range(num_top_labels):

                # overlay
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[label_idx],
                                                            positive_only=False,
                                                            num_features=lime_features,
                                                            hide_rest=False)

                current_label = all_tags_series[explanation.top_labels[label_idx]]

                fig, axs = plt.subplots(2, 2)

                fig.suptitle("File:{} - Tagged as: {}".format(image_path_key, current_label))

                top_left = axs[0][0]
                top_right = axs[0][1]
                low_left = axs[1][0]
                low_right = axs[1][1]

                for figure in [top_left, top_right, low_left, low_right]:
                    figure.axes.get_yaxis().set_ticks([])
                    figure.axes.get_xaxis().set_ticks([])

                top_left.imshow(stitched_images / 2 + 0.5)
                top_left.set_xlabel("Input image")

                top_right.imshow(mark_boundaries(temp / 2 + 0.5, mask))
                top_right.set_xlabel("LIME explanation", fontsize=10)

                # heatmap
                ind = explanation.top_labels[label_idx]
                dict_heatmap = dict(explanation.local_exp[ind])
                heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
                mappable = low_left.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
                plt.colorbar(mappable, ax=low_left)
                low_left.set_xlabel("Importance heatmap", fontsize=10)

                # mask
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[label_idx],
                                                            positive_only=True,
                                                            num_features=lime_features,
                                                            hide_rest=True)
                low_right.imshow(mark_boundaries(temp / 2 + 0.5, mask))
                low_right.set_xlabel("Importance mask", fontsize=10)

                fig.show()

                if save_figs:
                    current_label_image_key = "{}-{}".format(image_path_key, current_label)
                    fig.savefig("{}/{}_output.png".format(plots_folder, current_label_image_key), dpi=600)

