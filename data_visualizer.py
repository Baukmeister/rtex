import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_images(image_paths, num, path_prefix = "data/images/iu_xray/"):
    nested_list = [elem.split(";") for elem in image_paths]
    flattened_list = [item for sublist in nested_list for item in sublist]
    for i in range(num):
        image_file_path = flattened_list[i]
        img = mpimg.imread(path_prefix + image_file_path)
        cax = plt.imshow(img)
        plt.title("Abnormal image", fontsize=25)
        plt.xlabel("File: {}".format(image_file_path))
        plt.show()