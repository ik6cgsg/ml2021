from intelligent_placer_lib import intelligent_placer
import numpy as np
import cv2
from typing import List
import matplotlib.pyplot as plt
import skimage.transform


from common import common

TAG = '[intelligent_checker]'


def cut_out_object(image: np.ndarray, mask: np.ndarray):
    return cv2.bitwise_and(image.astype("uint8") + 1, image.astype("uint8") + 1,
                           mask=mask.astype("uint8"))


def make_test_image(object_images: List[np.ndarray],
                    object_grayscale_images: List[np.ndarray],
                    background_image: np.ndarray,
                    objects: List[np.ndarray]) -> np.ndarray:
    # extract objects
    objects = []
    for i, image in enumerate(object_images):
        # cut-out object
        mask = common.extract_object_masks(object_grayscale_images[i])[0]
        cut_out = cut_out_object(image, mask)

        a4_mask = common.extract_background_mask(object_grayscale_images[i])
        _, k = common.get_perspective_matrix_and_scale(a4_mask)
        scale = 2 * k
        m = np.float32([
            [scale, 0, cut_out.shape[1] * (1 - scale) / 2],
            [0, scale, cut_out.shape[0] * (1 - scale) / 2]
        ])
        cut_out = cv2.warpAffine(cut_out, m, (cut_out.shape[1], cut_out.shape[0]))

        objects.append(cut_out)

        common.save_image(cut_out, "object" + str(i + 1) + "_cut.jpg")

    result = background_image.copy()

    for i, object_image in enumerate(objects):
        shift = 450
        x = 0 #-shift // 2 + shift * i / (len(objects) - 1)
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


def test_intelligent_placer():
    print(f'{TAG} test_intelligent_placer()')
    common.do_common_stuff()
    res = intelligent_placer.check_image('30', [(30, 30)])
    print(f'{TAG} res = {res}')
    assert res


if __name__ == '__main__':
    print(f'{TAG} start checking...')
    # test_intelligent_placer()

    background = common.open_image_rgb("background.jpg")

    object_names = ["object1.jpg", "object2.jpg", "object3.jpg"]
    # object_names = ["object1.jpg"]
    objects = [common.open_image_rgb(name) for name in object_names]
    objects_gray = [common.open_image(name) for name in object_names]

    cut_objects_names = ["object1_cut.jpg", "object2_cut.jpg", "object3_cut.jpg"]
    cut_objects = [common.open_image_rgb(name) for name in cut_objects_names]

    test_image = make_test_image(objects, objects_gray, background, cut_objects)

    plt.figure()
    plt.imshow(test_image)
    plt.show()
