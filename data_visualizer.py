import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models, layers
from lime import lime_image
from skimage.segmentation import mark_boundaries

import data_handler


def plot_explainability_rtex_r(
        image_paths,
        num,
        path_prefix="data/images/iu_xray/",
        model=None,
        img_width=224,
        method=None,
        lime_samples=1000,
        lime_features=10,
        save_figs=False,
        abnormal=True
):
    def _predict_rtex_r(pasted_image):
        x1_data, x2_data = [], []

        for image in pasted_image:
            img1 = image[:, 0:img_width, :]
            img2 = image[:, img_width:, :]

            x1_data.append(img1)
            x2_data.append(img2)

        encoded_images = [np.array(x1_data), np.array(x2_data)]
        model_predictions = model.predict(encoded_images)
        if abnormal:
            return model_predictions
        else:
            return 1 - model_predictions

    plots_folder = "plots/rtex_r"
    if save_figs:
        if os.path.exists(plots_folder):
            shutil.rmtree(plots_folder)

        os.mkdir(plots_folder)

    for i in range(num):
        image_path_key = list(image_paths.keys())[i]

        encoded_images, img1, img2 = data_handler.encode_images(
            {image_path_key: image_paths[image_path_key]},
            path_prefix,
            return_images=True
        )
        stitched_images = _stitchImages(img1, img2)

        fig, axs = plt.subplots(2, 2)
        if abnormal:
            fig.suptitle("File: {} - Classified as abnormal".format(image_path_key))
        else:
            fig.suptitle("File: {} - Classified as normal".format(image_path_key))

        top_left = axs[0][0]
        top_right = axs[0][1]
        low_left = axs[1][0]
        low_right = axs[1][1]

        top_left.imshow(stitched_images / 2 + 0.5)
        if abnormal:
            top_left.set_xlabel("Abnormal image")
        else:
            top_left.set_xlabel("Normal image")

        if method == "lime":
            # Lime explainer
            lime_explainer = lime_image.LimeImageExplainer()
            explanation = lime_explainer.explain_instance(
                stitched_images,
                _predict_rtex_r,
                top_labels=1,
                hide_color=0,
                num_samples=lime_samples)

            # overlay
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                        positive_only=False,
                                                        num_features=lime_features,
                                                        hide_rest=False)
            top_right.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            top_right.set_xlabel("LIME explanation", fontsize=10)

            # heatmap
            ind = explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
            mappable = low_left.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
            plt.colorbar(mappable, ax=low_left)
            low_left.set_xlabel("Importance heatmap", fontsize=10)

            # mask
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                        positive_only=True,
                                                        num_features=lime_features,
                                                        hide_rest=True)
            low_right.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            low_right.set_xlabel("Importance mask", fontsize=10)

            fig.show()

            if save_figs:
                fig.savefig("{}/{}_output.png".format(plots_folder, image_path_key), dpi=600)

        elif method == "grad":
            # Grad CAM explainer

            # variable configuration
            img_size = (224, 224)

            inner_layers = model.layers[2]
            inner_model = tf.keras.models.Model(inner_layers.inputs, inner_layers.get_layer("avg_pool").output)

            s_i_model = models.Sequential()
            s_i_model.add(inner_model)

            s_i_model.add(layers.RepeatVector(2))
            s_i_model.add(layers.Reshape((1, 2048)))
            s_i_model.add(model.layers[4])

            last_conv_layer = s_i_model.layers[0].get_layer('relu')

            grad_model = models.Model(
                [inner_model.inputs, s_i_model.inputs], [last_conv_layer.output, s_i_model.output]
            )

            pred_index = 0

            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(encoded_images)
                class_channel = preds[:, 0]

            # This is the gradient of the output neuron (top predicted or chosen)
            # with regard to the output feature map of the last conv layer
            # TODO: figure out why this returns None
            grads = tape.gradient(class_channel, last_conv_layer_output)

            if grads is None:
                print("There are currently unsolved issues regarding GradCam that prevent this option from being used!")
                return

            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            # then sum all the channels to obtain the heatmap class activation
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # For visualization purpose, we will also normalize the heatmap between 0 & 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

            # Display heatmap
            plt.matshow(heatmap)
            plt.show()


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
        stitched_images = _stitchImages(img1, img2)

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


def _stitchImages(im1, im2):
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)

    return np.concatenate((im1, im2), axis=1).astype("double")


def _rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
