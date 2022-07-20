from .caculate import do_calculate
from .evaluate import do_evaluate
from .visual_dice import visual as dice
from .visual_score import visual as score
from .visual_val import visual as val
from .visual_low import visual as low
from .visual_matrix import visual as matrix
from .visual_best import visual as best
from .visual_select import visual as select


__all__ = ['do_calculate', 'do_evaluate', 'Visualizer']


class Visualizer(object):
    def __init__(self, root: str):
        self.root = root

    def dice(self, *args, **kwargs):
        dice(self.root, *args, **kwargs)

    def score(self, *args, **kwargs):
        score(self.root, *args, **kwargs)

    def val(self, *args, **kwargs):
        val(self.root, *args, **kwargs)

    def low(self, low_value: float = 0.8, *args, **kwargs):
        low(self.root, low=low_value, *args, **kwargs)

    def matrix(self, *args, **kwargs):
        matrix(self.root, *args, **kwargs)

    def best(self, *args, **kwargs):
        best(self.root, *args, **kwargs)

    def select(self, *args, **kwargs):
        select(self.root, *args, **kwargs)
