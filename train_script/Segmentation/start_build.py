import os.path

from basic import config
from utils import Timer
from process import do_build


def main():
    with Timer() as T:
        
        do_build(T=T)


if __name__ == '__main__':
    main()
