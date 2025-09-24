from automation.metrics.wer.normalizations.whisper.english import EnglishTextNormalizer
from jiwer import (
    AbstractTransform,
    Compose,
    Strip,
    ReduceToListOfListOfWords,
)

class EnglishNormalization(AbstractTransform):
    def __init__(self):
        super().__init__()
        self.normalizer = EnglishTextNormalizer()

    def process_string(self, string):
        return self.normalizer(string)


english_normalizer = Compose(
    [
        EnglishNormalization(),
        Strip(),
        ReduceToListOfListOfWords(),
    ]
)