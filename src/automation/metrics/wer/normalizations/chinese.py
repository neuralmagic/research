from automation.metrics.wer.normalizations.speechio.cn_tn import TextNorm
from jiwer import (
    AbstractTransform,
    Compose,
    Strip,
    ReduceToListOfListOfWords,
)

class ChineseNormalization(AbstractTransform):
    def __init__(self):
        super().__init__()
        self.normalizer = TextNorm()

    def process_string(self, string):
        return self.normalizer(string)


chinese_normalizer = Compose(
    [
        ChineseNormalization(),
        Strip(),
        ReduceToListOfListOfWords(),
    ]
)