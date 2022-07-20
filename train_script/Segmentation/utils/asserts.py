import os


class Assert:
    @staticmethod
    def file_exist(*paths: str):
        for path in paths:
            assert path is not None, f'File path is None! Please provide a available path!'
            assert os.path.exists(path), f'File path not exist! Check: {path}'
            assert os.path.isfile(path), f'Target path is not a file! Check: {path}'

    @staticmethod
    def dir_exist(*paths: str):
        for path in paths:
            assert path is not None, f'Directory path is None! Please provide a available path!'
            assert os.path.exists(path), f'Directory path not exist! Check: {path}'
            assert os.path.isdir(path), f'Target path is not a Directory! Check: {path}'

    @staticmethod
    def not_none(**kwargs):
        for key, val in kwargs.items():
            assert val is not None, f'{key} could not be none!'
