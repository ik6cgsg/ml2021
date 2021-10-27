import argparse
import glob
import random
#from intelligent_placer_lib import intelligent_placer
import numpy as np
import cv2
from typing import List
import matplotlib.pyplot as plt
import skimage.transform
import common.image
import common.common

TAG = '[intelligent_checker]'


def generate_test_image(
    object_images: List[np.ndarray],
    object_grayscale_images: List[np.ndarray],
    background_image: np.ndarray,
    objects_num: int
) -> np.ndarray:
    objects: List[np.ndarray] = []
    indices = random.sample(range(len(object_images)), objects_num)
    # cut out objects
    for i in indices:
        mask = common.common.extract_object_masks(object_grayscale_images[i])[-1]
        cut_out = common.common.cut_out_object(object_images[i], mask)
        a4_mask = common.common.extract_background_mask(object_grayscale_images[i])
        _, k, _, _ = common.common.get_perspective_matrix_and_scale(a4_mask)
        scale = 2 * k
        m = np.float32([
            [scale, 0, cut_out.shape[1] * (1 - scale) / 2],
            [0, scale, cut_out.shape[0] * (1 - scale) / 2]
        ])
        cut_out = cv2.warpAffine(cut_out, m, (cut_out.shape[1], cut_out.shape[0]))
        common.image.show(cut_out, "cut out")
        objects.append(cut_out)

    result = background_image.copy()
    # linear shift
    for i, object_image in enumerate(objects):
        shift = 450
        x = 0 #-shift // 2 + shift * i / (len(objects) - 1)
        y = 0
        if len(objects) > 1:
            y = -shift // 2 + shift * i / (len(objects) - 1)
        # angle = (i + 1) * 2 * 3.14 / 3
        #
        # m = np.float32([[1, 0, -object_image.shape[1] // 2], [0, 1, -object_image.shape[0] // 2]])
        # object_image = cv2.warpAffine(object_image, m, (object_image.shape[1], object_image.shape[0]))
        # m = np.float32([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0]])
        # object_image = cv2.warpAffine(object_image, m, (object_image.shape[1], object_image.shape[0]))
        # m = np.float32([[1, 0, object_image.shape[1] // 2], [0, 1, object_image.shape[0] // 2]])
        # object_image = cv2.warpAffine(object_image, m, (object_image.shape[1], object_image.shape[0]))

        m = np.float32([[1, 0, x], [0, 1, y]])
        object_image = cv2.warpAffine(object_image, m, (object_image.shape[1], object_image.shape[0]))
        result[object_image > 0] = object_image[object_image > 0]

    return result


def test_n_random_objects_larger_rect(objects, objects_gray, background, max):
    for i in range(1, max + 1):
        img = generate_test_image(objects, objects_gray, background, i)
        common.image.show(img, "test_image_" + str(i))
    # todo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--images_folder", default="objects/",
                        type=str, help="local path to 10 images with single objects placed on a4")
    parser.add_argument("-b", "--background", default="back.jpg",
                        type=str, help="local path to image with background")
    parser.add_argument("-r", "--restrictions", default="restrictions.json",
                        type=str, help="local path to restrictions json")
    args = parser.parse_args()

    if args.images_folder == "" or args.background == "" or args.restrictions == "":
        print("Some of params are empty(")
        exit(0)

    background = common.image.open_image_rgb(args.background)
    common.image.show(background, "back")
    object_paths = glob.glob(args.images_folder+"*")
    objects = [common.image.open_image_rgb(name) for name in object_paths]
    objects_gray = [common.image.open_image(name) for name in object_paths]

    test_n_random_objects_larger_rect(objects, objects_gray, background, 4)
