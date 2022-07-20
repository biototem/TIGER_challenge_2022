import os.path

from basic import config
from utils import Timer
from process import do_valid


def main():
    with Timer() as T:
        do_valid(T=T)


if __name__ == '__main__':
    main()
