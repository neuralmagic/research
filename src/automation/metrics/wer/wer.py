from tarfile import CompressionError
import numpy
from jiwer import wer
from automation.metrics.wer.normalizations import basic_normalizer, english_normalizer, chinese_normalizer

normalization_map = {
    "english": english_normalizer,
    "en": english_normalizer,
    "chinese": chinese_normalizer,
    "cn": chinese_normalizer,
    "cmn_hans": chinese_normalizer,
    "yue_hant": chinese_normalizer,
}

class WERMetric:
    def __init__(self, language):
        self.language = language
        if language not in normalization_map:
            self.normalizer = basic_normalizer
        else:
            self.normalizer = normalization_map[language]
        self.values = []

    def __call__(self, reference, hypothesis, accumulate=True):
        wer_value = wer(reference, hypothesis, reference_transform=self.normalizer, hypothesis_transform=self.normalizer)
        if accumulate:
            self.values.append(wer_value)
        return wer_value

    def mean(self):
        return numpy.mean(self.values)

    def std(self):
        return numpy.std(self.values)

