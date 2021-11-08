import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import data_handler


def show_images(
        image_paths,
        num,
        path_prefix="data/images/iu_xray/",
        explainability=False,
        model=None,
        img_width=224,
        lime_samples=1000):
    nested_list = [elem.split(";") for elem in image_paths]
    flattened_list = [item for sublist in nested_list for item in sublist]

    def _predict(pasted_image):
        x1_data, x2_data = [], []

        for image in pasted_image:
            img1 = image[:, 0:img_width, :]
            img2 = image[:, img_width:, :]

            x1_data.append(img1)
            x2_data.append(img2)

        encoded_images = [np.array(x1_data), np.array(x2_data)]
        return model.predict(encoded_images)

    for i in range(num):
        image_path_key = list(image_paths.keys())[i]

        encoded_images, img1, img2 = data_handler.encode_images(
            {image_path_key: image_paths[image_path_key]},
            path_prefix,
            return_images=True
        )
        stitched_images = _stitchImages(img1, img2)

        cax = plt.imshow(stitched_images / 2 + 0.5)
        plt.title("Abnormal image", fontsize=25)
        plt.xlabel("File: {}".format(image_path_key))
        plt.show()

        if explainability:
            lime_explainer = lime_image.LimeImageExplainer()
            explanation = lime_explainer.explain_instance(
                stitched_images,
                _predict,
                top_labels=1,
                hide_color=0,
                num_samples=lime_samples)

            # overlay
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                        hide_rest=False)
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            plt.title("LIME explanation", fontsize=25)
            plt.xlabel("File: {}".format(image_path_key))
            plt.show()

            # heatmap

            ind = explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
            plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
            plt.colorbar()
            plt.title("Importance heatmap", fontsize=25)
            plt.xlabel("File: {}".format(image_path_key))
            plt.show()



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
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])