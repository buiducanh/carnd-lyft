import numpy
import os
import cv2
from matplotlib import pyplot as plt

lyft_data_path = "lyft_training_data/Train"

def preprocess_labels(label_image):
    labels_new = numpy.copy(label_image)
    # Identify lane marking pixels (label is 6)
    lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()
    # Set lane marking pixels to road (label is 7)
    labels_new[lane_marking_pixels] = 7

    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = 0
    # Return the preprocessed label image
    return labels_new

def prep():
    import scipy.misc

    lyft_data_path = "lyft_training_data/Train"
    input_path = os.path.abspath(lyft_data_path + "/CameraSeg")
    output_path = os.path.abspath(lyft_data_path + "/CameraPrep")

    for f in os.listdir(input_path):
        img = scipy.misc.imread(os.path.join(input_path, f), mode = 'RGB')
        prep_img = preprocess_labels(img)
        scipy.misc.imsave(os.path.join(output_path,f), prep_img)


def poke():
    path = lyft_data_path + "/CameraRGB"

    for f in os.listdir(path)[:1]:
        img = cv2.imread(os.path.join(path, f))

        img = img[:, :, ::-1]
        plt.imshow(img)
        plt.show()
