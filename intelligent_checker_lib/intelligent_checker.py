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
#from intelligent_gritsaenko_placer.intelligent_placer_lib.intelligent_placer import check_image


TAG = '[intelligent_checker]'


DEFAULT_PPM = 3.0


def generate_test_image(objects: List[np.ndarray], background_image: np.ndarray) -> np.ndarray:
    cur_x = 1
    cur_y = 1
    result_image = background_image.copy()
    W = background_image.shape[1]
    H = background_image.shape[0]

    max_h_in_current_row = 0
    obj_index = 0
    row_index = 0

    objects_with_coordinates_and_shapes = []
    rows_distance_to_right_side = []

    not_placed = False

    while obj_index < len(objects):
        obj = objects[obj_index]
        if obj.shape[0] < H - cur_y and obj.shape[1] < W - cur_x:
            objects_with_coordinates_and_shapes.append([obj, cur_x, cur_y, row_index])
            cur_x = cur_x + obj.shape[1] + 1
            if max_h_in_current_row < obj.shape[0]:
                max_h_in_current_row = obj.shape[0]
            obj_index += 1
        else:
            if not_placed:
                not_placed = False
                obj_index += 1
                continue
            not_placed = True

            rows_distance_to_right_side.append(W - cur_x)
            row_index += 1
            cur_x = 1
            cur_y = cur_y + max_h_in_current_row + 1
            max_h_in_current_row = 0

    rows_distance_to_right_side.append(W - cur_x)

    dist_to_bottom_side = H - cur_y - max_h_in_current_row

    for elem in objects_with_coordinates_and_shapes:
        result_image = common.insert_image_into_another_image(
            img_to_insert=elem[0],
            img_to_insert_into=result_image,
            x=elem[1] + rows_distance_to_right_side[elem[3]] // 2,
            y=elem[2] + dist_to_bottom_side // 2
        )

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


def generate_polygons(objects: List[np.ndarray], ppm: float) -> np.ndarray:
    sum_width = 0
    sum_height = 0
    max_width = 0
    max_height = 0

    for obj in objects:
        sum_height += obj.shape[0]
        sum_width += obj.shape[1]
        max_height = np.max(max_height, obj.shape[0])
        max_width = np.max(max_width, obj.shape[1])
        sum_height += obj.shape[0] - DEFAULT_PPM
        sum_width += obj.shape[1] - DEFAULT_PPM
        max_height = np.max(max_height, obj.shape[0] - DEFAULT_PPM)
        max_width = np.max(max_width, obj.shape[1] - DEFAULT_PPM)

    # max width, sum height
    polygons = np.ndarray((4, 4, 2))
    polygons[0, 0] = [0, 0]
    polygons[0, 1] = [max_width * ppm, 0]
    polygons[0, 2] = [max_width * ppm, sum_height * ppm]
    polygons[0, 3] = [0, sum_height * ppm]
    # max height, sum width
    polygons[1, 0] = [0, 0]
    polygons[1, 1] = [sum_width * ppm, 0]
    polygons[1, 2] = [sum_width * ppm, max_height * ppm]
    polygons[1, 3] = [0, max_height * ppm]

    return polygons


def add_padding(image: np.ndarray, padding: int) -> np.ndarray:
    output = np.zeros((image.shape[0] + padding, image.shape[1] + padding, image.shape[2]))
    output[padding:padding + image.shape[0], padding:padding + image.shape[1]] = image
    return output


class TestCase:
    name: str
    image: np.ndarray
    polygon: np.ndarray
    target_result: bool

    def __init__(self, name: str, image: np.ndarray, polygon: np.ndarray,
                 target_result: bool):
        self.name = name
        self.image = image
        self.polygon = polygon
        self.target_result = target_result


def test(object_images: np.ndarray, object_grayscale_images: np.ndarray,
         background_image: np.ndarray) -> None:
    # cut out objects
    cut_objects: List[np.ndarray] = []

    ppm = DEFAULT_PPM

    #for i in range(len(object_images)):
    #    cut_objects.append(cut_out_object(object_grayscale_images[i], object_images[i]))
    #    image.save_image(cut_objects[-1], f"placer_cut_objects/object_{i}.png")

    for i in range(len(object_images)):
        # obj = cut_out_object(object_grayscale_images[i], object_images[i]))
        # image.save_image(cut_objects[-1], f"cut_objects/object_{i}.jpg")
        obj = image.open_image_rgb(f"cut_objects/object_{i}.png")
        obj = add_padding(obj, RestrictionHandler.get("min_dist_between_obj")[0] * ppm / 2)
        cut_objects.append(obj)

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

        # generate image
        test_image = generate_test_image(test_case_objects, background_image)
        image.save_image(test_image, f"test_cases/case_{case_index}.jpg")

        # true cases
        polygons = generate_polygons(test_case_objects, ppm)
        for i, polygon in enumerate(polygons):
            test_cases.append(TestCase(f"{case_index}_case_{i}_poly_true",
                                       test_image, polygon, True))
        # false cases
        polygons /= 2
        for i, polygon in enumerate(polygons):
            test_cases.append(TestCase(f"{case_index}_case_{i}_poly_false",
                                       test_image, polygon, False))

    # test
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for test_case in test_cases:
        #result = check_image("", test_case.polygon)
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
    parser.add_argument("-f", "--images_folder", default="../objects/",
                        type=str, help="local path to 10 images with single objects placed on a4")
    parser.add_argument("-b", "--background", default="back.jpg",
                        type=str, help="local path to image with background")
    parser.add_argument("-r", "--restrictions",
                        default="../intelligent_gritsaenko_placer/intelligent_placer_lib/default_config.yaml",
                        type=str, help="local path to restrictions json")
    args = parser.parse_args()

    if args.images_folder == "" or args.background == "" or args.restrictions == "":
        print("Some of params are empty(")
        exit(0)

    RestrictionHandler.load(args.restrictions)
    background = image.open_image_rgb(args.background)
    object_paths = glob.glob(args.images_folder+"*")
    objects = [image.open_image_rgb(name) for name in object_paths]
    objects_gray = [image.open_image(name) for name in object_paths]
    test(objects, objects_gray, background)
