# TODO: 事实表明，流程这东西会越弄越多，所以早晚有一天，要给它改成动态 import 的写法，以免各流程间的预备状态相互干扰或拖累
from .build import do_build
from .train import do_train
from .valid import do_valid
from .evaluate import do_calculate, do_evaluate, Visualizer
from .predict import do_predict
from .full_predict import do_full_predict
# from .submit_predict import do_submit_predict
from .full_predict_speedup import do_full_predict_speedup
# from .submit_predict_speedup import do_submit_predict_speedup
from .convert import do_convert


__all__ = [
    'do_build',
    'do_train',
    'do_valid',
    'do_calculate',
    'do_evaluate',
    'Visualizer',
    'do_predict',
    'do_full_predict',
    # 'do_submit_predict',
    'do_full_predict_speedup',
    # 'do_submit_predict_speedup',
    'do_convert',
]
