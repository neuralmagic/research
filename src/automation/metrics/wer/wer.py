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
        self.metadata = []

    def __call__(self, reference, hypothesis, metadata=None, accumulate=True):
        wer_value = wer(reference, hypothesis, reference_transform=self.normalizer, hypothesis_transform=self.normalizer)
        if accumulate:
            self.values.append(wer_value)
        if metadata is not None:
            self.metadata.append(metadata)
        return wer_value

    def mean(self):
        return numpy.mean(self.values)

    def std(self):
        return numpy.std(self.values)

    def to_dict(self):
        if len(self.metadata) > 0:
            values = []
            for value, metadata in zip(self.values, self.metadata):
                value_dict = metadata.copy()
                value_dict["value"] = value
                values.append(value_dict)
        else:
            values = self.values
        
        return {
            "language": self.language,
            "wer": {
                "mean": self.mean().item(),
                "std": self.std().item(),
                "values": values,
            },
        }

