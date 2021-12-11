import cv2
import argparse
import numpy as np

from typing import List, Dict

from intelligent_checker_lib.util import common
from intelligent_checker_lib.util.restrictions import RestrictionHandler

TAG = '[intelligent_checker]'


def generate_test_image(objects: List[np.ndarray], background_image: np.ndarray) -> np.ndarray:
    cur_x = 1
    cur_y = 1
    result_image = background_image.copy()
    W = background_image.shape[1]
    H = background_image.shape[0]

    max_h_in_current_row = 0

    for obj in objects:
        if obj.shape[0] < H - cur_y and obj.shape[1] < W - cur_x:
            common.insert_image_into_another_image(
                img_to_insert=obj,
                img_to_insert_into=result_image,
                x=cur_x,
                y=cur_y
            )
            cur_x = cur_x + obj.shape[1] + 1
            if max_h_in_current_row < obj.shape[0]:
                max_h_in_current_row = obj.shape[0]
        else:
            cur_x = 1
            cur_y = cur_y + max_h_in_current_row + 1
            max_h_in_current_row = 0

    return result_image


def warp_to_scale(image: np.ndarray, a4_image_grayscale: np.ndarray, ppm: float) -> np.ndarray:
    a4_mask = common.extract_background_mask(a4_image_grayscale)
    _, k, _, _ = common.get_perspective_matrix_and_scale(a4_mask)
    scale = ppm * k
    m = np.float32([
        [scale, 0, image.shape[1] * (1 - scale) / 2],
        [0, scale, image.shape[0] * (1 - scale) / 2]
    ])
    return cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))


def cut_out_object(image_grayscale: np.ndarray, image_color: np.ndarray) -> np.ndarray:
    mask = common.extract_object_masks(image_grayscale)[-1]
    return common.apply_mask(image_color, mask)


def generate_contour():
    pass


def test(object_images: np.ndarray, object_grayscale_images: np.ndarray,
         background_image: np.ndarray, restrictions: Dict) -> None:
    # cut out objects
    cut_objects: List[np.ndarray] = []
    for i in range(object_images):
        cut_objects.append(cut_out_object(object_grayscale_images[i], object_images[i]))
    # generate test images for different restrictions

    # generate contour

    # test stuff
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--images_folder", default="objects/",
                        type=str, help="local path to 10 images with single objects placed on a4")
    parser.add_argument("-b", "--background", default="back.jpg",
                        type=str, help="local path to image with background")
    parser.add_argument("-r", "--restrictions", default="default_config.yaml",
                        type=str, help="local path to restrictions json")
    args = parser.parse_args()

    if args.images_folder == "" or args.background == "" or args.restrictions == "":
        print("Some of params are empty(")
        exit(0)

    RestrictionHandler.load(args.restrictions)
    print(RestrictionHandler.current)
    if RestrictionHandler.has("polygon_vertex_num"):
        mn, mx = RestrictionHandler.get("polygon_vertex_num")
        print("min res = ", mn)
        print("max res = ", mx)

    #background = intelligent_checker_lib.util.image.open_image_rgb(args.background)
    #intelligent_checker_lib.util.image.show(background, "back")
    #object_paths = glob.glob(args.images_folder+"*")
    #objects = [intelligent_checker_lib.util.image.open_image_rgb(name) for name in object_paths]
    #objects_gray = [intelligent_checker_lib.util.image.open_image(name) for name in object_paths]

    #test_n_random_objects_larger_rect(objects, objects_gray, background, 1)
