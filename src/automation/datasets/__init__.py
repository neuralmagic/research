from automation.datasets.calibration import load_calibration_dataset
from automation.datasets.calibration import DATASET_PATH as CALIBRATION_DATASET
from automation.datasets.openthoughts import load_openthoughts_dataset
from automation.datasets.openthoughts import DATASET_PATH as OPENTHOUGHTSDATASET
from automation.datasets.utils import load_llm_messages, load_vlm_messages
from automation.datasets.fleurs import load_fleurs_dataset
from automation.datasets.tulu import make_tulu_prompt
from automation.datasets.openplatypus import make_openplatypus_prompt
from automation.datasets.alpaca import make_alpaca_prompt
from automation.datasets.defaults import make_default_prompt

SUPPORTED_DATASETS = {
    "calibration": load_calibration_dataset,
    CALIBRATION_DATASET: load_calibration_dataset,
    "openthoughts": load_openthoughts_dataset,
    OPENTHOUGHTSDATASET: load_openthoughts_dataset, 
}

__all__ = [
    "load_calibration_dataset",
    "load_openthoughts_dataset",
    "load_llm_messages",
    "load_vlm_messages",
    "make_tulu_prompt",
    "make_openplatypus_prompt",
    "make_alpaca_prompt",
    "make_default_prompt",
    "load_fleurs_dataset",
    "SUPPORTED_DATASETS",
]
