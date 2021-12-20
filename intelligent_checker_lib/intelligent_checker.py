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
from intelligent_gritsaenko_placer.intelligent_placer_lib.intelligent_placer import check_image
from tabulate import tabulate


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
    sum_width = 0.0
    sum_height = 0.0
    max_width = 0.0
    max_height = 0.0

    for obj in objects:
        sum_height += obj.shape[0] - DEFAULT_PPM
        sum_width += obj.shape[1] - DEFAULT_PPM
        max_height = max(max_height, obj.shape[0] - DEFAULT_PPM)
        max_width = max(max_width, obj.shape[1] - DEFAULT_PPM)

    sum_width += 2.0
    sum_height += 2.0
    max_width += 2.0
    max_height += 2.0

    # max width, sum height
    polygons = np.ndarray((2, 4, 2))
    polygons[0, 0] = [0, 0]
    polygons[0, 1] = [max_width / ppm, 0]
    polygons[0, 2] = [max_width / ppm, sum_height / ppm]
    polygons[0, 3] = [0, sum_height / ppm]
    # max height, sum width
    polygons[1, 0] = [0, 0]
    polygons[1, 1] = [sum_width / ppm, 0]
    polygons[1, 2] = [sum_width / ppm, max_height / ppm]
    polygons[1, 3] = [0, max_height / ppm]

    return polygons


def add_padding(image: np.ndarray, padding: int) -> np.ndarray:
    output = np.zeros((image.shape[0] + padding, image.shape[1] + padding, image.shape[2]))
    output[padding:padding + image.shape[0], padding:padding + image.shape[1]] = image
    return output


class TestCase:
    name: str
    image_path: str
    polygon: np.ndarray
    target_result: bool
    objects_indices: List[int]

    def __init__(self, name: str, image_path: str, polygon: np.ndarray,
                 target_result: bool, objects_indices: List[int]):
        self.name = name
        self.image_path = image_path
        self.polygon = polygon
        self.target_result = target_result
        self.objects_indices = objects_indices


def test(object_images: np.ndarray, object_grayscale_images: np.ndarray,
         background_image: np.ndarray) -> None:
    # cut out objects
    cut_objects: List[np.ndarray] = []

    ppm = DEFAULT_PPM

    #for i in range(len(object_images)):
    #    cut_objects.append(cut_out_object(object_grayscale_images[i], object_images[i]))
    #    image.save_image(cut_objects[-1], f"test_cut_objects/object_{i}.png")
    #return
    for i in range(len(object_images)):
        obj = image.open_image_rgb(f"test_cut_objects/object_{i}.png")
        obj = add_padding(obj, int(RestrictionHandler.get("min_dist_between_obj")[0] * ppm / 2))
        cut_objects.append(obj)

    test_cases: List[TestCase] = []

    # generate object subsets
    indices = range(len(cut_objects))
    test_case_object_indices: List[List[int]] = []
    for k in range(RestrictionHandler.get("obj_num")[0], RestrictionHandler.get("obj_num")[1] + 1):
        for subset in itertools.combinations(indices, k):
            test_case_object_indices.append(subset)

    # generate test case images
    for case_index, object_indices in enumerate(test_case_object_indices):
        test_case_objects = [cut_objects[i] for i in object_indices]
        obj_num = len(test_case_objects)

        # generate image
        test_image = generate_test_image(test_case_objects, background_image)
        test_image_path = f"test_cases/case_{case_index}.jpg"
        image.save_image(test_image, test_image_path)

        # true cases
        polygons = generate_polygons(test_case_objects, ppm)
        for i, polygon in enumerate(polygons):
            test_cases.append(TestCase(
                f"{case_index}_case_{obj_num}_objects_{i}_poly_true",
                test_image_path,
                polygon.copy(),
                True,
                object_indices
            ))
        # false cases
        polygons /= 2
        for i, polygon in enumerate(polygons):
            test_cases.append(TestCase(
                f"{case_index}_case_{obj_num}_objects_{i}_poly_false",
                test_image_path,
                polygon.copy(),
                False,
                object_indices
            ))

    # test
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    stats = np.zeros(shape=(len(cut_objects), 4))
    for test_case in test_cases:
        print(f"~~~~~~~~~~~~~~~~~~~~~ {test_case.name} ~~~~~~~~~~~~~~~~~~~~~")
        print(f"image: {test_case.image_path} polygon: {test_case.polygon}")
        result = check_image(test_case.image_path, test_case.polygon)
        for obj_id in test_case.objects_indices:
            if test_case.target_result:
                if result:
                    stats[obj_id, 0] += 1
                else:
                    stats[obj_id, 1] += 1
            else:
                if not result:
                    stats[obj_id, 2] += 1
                else:
                    stats[obj_id, 3] += 1
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
        print(f"Expected: {test_case.target_result} Actual: {result}")

    # show infographic
    print("-------------------- TEST RESULTS --------------------")
    print("~~~~~~~~~~~~ GLOBAL ~~~~~~~~~~~~")
    table = [['', 'Expected positive', 'Expected negative'],
             ['Actual positive', f'{true_positive}', f'{false_positive}'],
             ['Actual negative', f'{false_negative}', f'{true_negative}']]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    for i, stat in enumerate(stats):
        print(f"~~~~~~~~~~~~ OBJECT[{i}] ~~~~~~~~~~~~")
        table = [['', 'Expected positive', 'Expected negative'],
                 ['Actual positive', f'{stat[0]}', f'{stat[1]}'],
                 ['Actual negative', f'{stat[3]}', f'{stat[2]}']]
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--images_folder", default="test_objects/",
                        type=str, help="local path to 10 images with single objects placed on a4")
    parser.add_argument("-b", "--background", default="back_white.jpg",
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
