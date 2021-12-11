from builtins import object

import cv2
import glob
import argparse
import numpy as np
from typing import List, Dict
# from intelligent_placer_lib.intelligent_placer import check_image
from intelligent_checker_lib.util import image
from intelligent_checker_lib.util import common
from intelligent_checker_lib.util.restrictions import RestrictionHandler
import itertools


TAG = '[intelligent_checker]'


DEFAULT_PPM = 3.0


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


def scale_transform(image: np.ndarray, scale: float) -> np.ndarray:
    m = np.float32([
        [scale, 0, image.shape[1] * (1 - scale) / 2],
        [0, scale, image.shape[0] * (1 - scale) / 2]
    ])
    return cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))


def warp_to_scale(image: np.ndarray, a4_image_grayscale: np.ndarray, ppm: float) -> np.ndarray:
    a4_mask = common.extract_background_mask(a4_image_grayscale)
    # k = 1 / ppm of a4 image
    _, k, _, _ = common.get_perspective_matrix_and_scale(a4_mask)
    return scale_transform(image, ppm * k)


def cut_out_object(image_grayscale: np.ndarray, image_color: np.ndarray) -> np.ndarray:
    mask = common.extract_object_masks(image_grayscale)[-1]
    cut_out = common.apply_mask(image_color, mask)
    return warp_to_scale(cut_out, image_grayscale, DEFAULT_PPM)


def generate_polygon():
    pass


def add_padding(image: np.ndarray, padding: int) -> np.ndarray:
    pass


class TestCase:
    name: str
    image: np.ndarray
    polygon: List
    target_result: bool

    def __init__(self, name: str, image: np.ndarray, polygon: List, target_result: bool):
        self.name = name
        self.image = image
        self.polygon = polygon
        self.target_result = target_result


def test(object_images: np.ndarray, object_grayscale_images: np.ndarray,
         background_image: np.ndarray) -> None:
    # cut out objects
    cut_objects: List[np.ndarray] = []
    for i in range(len(object_images)):
        cut_objects.append(cut_out_object(object_grayscale_images[i], object_images[i]))
        image.save_image(cut_objects[-1], f"cut_objects/object_{i}.jpg")

    # for i in range(len(object_images)):
    #     cut_objects.append(image.open_image_rgb(f"cut_objects/object_{i}.jpg"))

    test_cases: List[TestCase] = []

    # generate object subsets
    indices = range(len(cut_objects))
    test_case_object_indices: List[List[int]] = []
    for k in range(1, RestrictionHandler.get("obj_num")[1] + 1):
        for subset in itertools.combinations(indices, k):
            test_case_object_indices.append(subset)

    # generate test case images
    for case_index, object_indices in enumerate(test_case_object_indices):
        test_case_objects = [cut_objects[i] for i in object_indices]

        # "aspect_ratio", to background

        # figure out scaling
        # calculate object area
        object_area = 0
        for object_image in test_case_objects:
            object_area += object_image.shape[0] * object_image.shape[1]

        # calculate area percentage
        # add padding as necessary

        # # set scaling as necessary?????
        # scale = 2
        # for i, image in enumerate(test_case_objects):
        #     test_case_objects[i] = scale_transform(image, scale)

        # generate image
        test_image = generate_test_image(test_case_objects, background_image)
        image.show(test_image, "")

        # "resolution", to generated image
        # ...

        # "camera_shift"?????

        # noise
        # ...

        # blur
        # ...

        test_cases.append(TestCase(f"{case_index}", test_image, [], True))

    # "rotation",

    # "min_dist_between_obj",

    # "min_dist_between_obj_polygon",
    # "max_dist_between_obj_center",

    # "polygon_vertex_num",
    # "polygon_angle",

    # ,

    # "area_ratio",

    # "same_obj_num",

    # "back_shadows",

    # "back_diff_obj",

    # generate polygon

    # test
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for test_case in test_cases:
        # result = check_image("", test_case.polygon)
        result = True
        if test_case.target_result:
            if result:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if not result:
                true_negative += 1
            else:
                false_negative += 1

    # show infographic
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

    background = image.open_image_rgb(args.background)
    # intelligent_checker_lib.util.image.show(background, "back")
    object_paths = glob.glob(args.images_folder+"*")
    objects = [image.open_image_rgb(name) for name in object_paths]
    objects_gray = [image.open_image(name) for name in object_paths]

    test(objects, objects_gray, background)
    # test_n_random_objects_larger_rect(objects, objects_gray, background, 1)
