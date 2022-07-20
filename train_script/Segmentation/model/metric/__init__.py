from .renal import Renal
from .dice import MDice
from .mfscore import MFscore
from .special import Special


def metric():
    return [
        Special(tag='score'),
        Special(tag='dice_tumor'),
        Special(tag='dice_stroma'),
        # MDice(only=(1, 2, 5, 6, 7), name='mdice(1,2,5,6,7)'),
        # MDice(only=(1,), name='dice-tumor1'),
        # MDice(only=(2,), name='dice-stroma1'),
        # MDice(only=(3,), name='dice-tumor2'),
        # MDice(only=(4,), name='dice-normal'),
        # MDice(only=(5,), name='dice-necrosis'),
        # MDice(only=(6,), name='dice-stroma2'),
        # MDice(only=(7,), name='dice-rest'),
    ]
