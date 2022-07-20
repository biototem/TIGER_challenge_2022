import json
import os.path

from basic import config, join
from utils import Timer
from process import do_convert


def main():
    with Timer() as T:
        with open(join('~/resource/manual/image_uses.json')) as f:
            images = json.load(f)
        # print(len(images))
        # return
        targets = [
            '/YOUR_DIR/wsirois/roi-level-annotations/tissue-cells/images/122S_[27872, 11652, 29086, 12830].png'
        ]
        sources = [v['image'] for k, v in images.items()]
        save_root = '~/resource/data/color_normalized'
        do_convert(targets=targets, sources=sources, save_root=save_root)


if __name__ == '__main__':
    main()
