from intelligent_placer_lib import intelligent_placer
from common import common

TAG = '[intelligent_checker]'


def test_intelligent_placer():
    print(f'{TAG} test_intelligent_placer()')
    common.do_common_stuff()
    res = intelligent_placer.check_image('30', [(30, 30)])
    print(f'{TAG} res = {res}')
    assert res


if __name__ == '__main__':
    print(f'{TAG} start checking...')
    test_intelligent_placer()
