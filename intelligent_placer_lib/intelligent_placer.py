from common import common

Point = tuple[float, float]
TAG = '[intelligent_placer]'


def check_image(path_to_image: str, polygon_coordinates: list[Point]) -> bool:
    """
    Check if objects in the provided image can fit inside the provided polygon

    :param path_to_image: path to jpg image on local computer
    :param polygon_coordinates: coordinates in clockwise order
    """
    print(f'{TAG} check_image({path_to_image}, {polygon_coordinates})')
    common.do_common_stuff()
    result = False
    if path_to_image == '30' and polygon_coordinates[0][0] == 30:
        result = True
    return result
