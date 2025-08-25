from automation.metrics.wer.normalizations.whisper.basic import BasicTextNormalizer
from jiwer import (
    AbstractTransform,
    Compose,
    Strip,
    ReduceToListOfListOfWords,
)

class BasicNormalization(AbstractTransform):
    def __init__(self):
        super().__init__()
        self.normalizer = BasicTextNormalizer()

    def process_string(self, string):
        return self.normalizer(string)


basic_normalizer = Compose(
    [
        BasicNormalization(),
        Strip(),
        ReduceToListOfListOfWords(),
    ]
)