import shutil

import keras as keras
from keras import models, layers
from gradcam import VizGradCAM
import matplotlib.pyplot as plt
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import data_handler
from tensorflow import keras
import tensorflow as tf
import matplotlib.cm as cm


def visualize_images(
        image_paths,
        num,
        path_prefix="data/images/iu_xray/",
        model=None,
        img_width=224,
        method=None,
        lime_samples=1000,
        lime_features=10,
        save_figs=False
):
    def _predict(pasted_image):
        x1_data, x2_data = [], []

        for image in pasted_image:
            img1 = image[:, 0:img_width, :]
            img2 = image[:, img_width:, :]

            x1_data.append(img1)
            x2_data.append(img2)

        encoded_images = [np.array(x1_data), np.array(x2_data)]
        return model.predict(encoded_images)

    # Grad CAM methods
    def get_img_array(img_path, size):
        # `img` is a PIL image of size 224x224
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (224, 224, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch" of size (1, 224, 224, 3)
        array = np.expand_dims(array, axis=0)
        return array

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

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
        return heatmap.numpy()

    def rescale_heatmap(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

    if save_figs:
        if os.path.exists("plots"):
            shutil.rmtree("plots")

        os.mkdir("plots")

    for i in range(num):
        image_path_key = list(image_paths.keys())[i]

        encoded_images, img1, img2 = data_handler.encode_images(
            {image_path_key: image_paths[image_path_key]},
            path_prefix,
            return_images=True
        )
        stitched_images = _stitchImages(img1, img2)

        fig, axs = plt.subplots(2, 2)
        fig.suptitle("File: {}".format(image_path_key))
        top_left = axs[0][0]
        top_right = axs[0][1]
        low_left = axs[1][0]
        low_right = axs[1][1]

        top_left.imshow(stitched_images / 2 + 0.5)
        top_left.set_xlabel("Abnormal image")

        if method == "lime":
            # Lime explainer
            lime_explainer = lime_image.LimeImageExplainer()
            explanation = lime_explainer.explain_instance(
                stitched_images,
                _predict,
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
                fig.savefig("plots/{}_output.png".format(image_path_key))

        elif method == "grad":
            # Grad CAM explainer

            # variable configuration
            img_size = (224, 224)

            inner_layers = model.layers[2]
            inner_model = tf.keras.models.Model(inner_layers.inputs, inner_layers.get_layer("avg_pool").output)
            last_conv_layer = inner_model.get_layer("relu")

            s_i_model = models.Sequential()
            s_i_model.add(inner_model)
            s_i_model.add(layers.RepeatVector(2))
            s_i_model.add(layers.Reshape((1, 2048)))
            s_i_model.add(model.layers[4])

            grad_model = tf.keras.models.Model(
                tf.keras.utils.get_source_inputs(s_i_model.inputs), [last_conv_layer.output, s_i_model.output]
            )

            pred_index = 0

            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(encoded_images)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            # This is the gradient of the output neuron (top predicted or chosen)
            # with regard to the output feature map of the last conv layer
            grads = tape.gradient(class_channel, last_conv_layer_output)

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

            # save rescaled heatmap on xray in plots folder


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
