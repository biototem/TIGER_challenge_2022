import abc


class Dataset(abc.ABC):
    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented
