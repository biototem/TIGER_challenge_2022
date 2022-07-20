from utils import Timer
from .caculate import do_calculate
from .evaluate import do_evaluate


__all__ = ['do_predict']


def do_predict(model_name: str, root: str, T: Timer = None, visual: bool = False):
    do_calculate(model_name=model_name, root=root, T=T, visual=visual)
    do_evaluate(model_name=model_name, root=root)

