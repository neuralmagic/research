from automation.datasets.calibration import load_calibration_dataset
from automation.datasets.calibration import DATASET_PATH as CALIBRATION_DATASET
from automation.datasets.openthoughts import load_openthoughts_dataset
from automation.datasets.openthoughts import DATASET_PATH as OPENTHOUGHTSDATASET
from automation.datasets.utils import load_dataset_messages

SUPPORTED_DATASETS = {
    "calibration": load_calibration_dataset,
    CALIBRATION_DATASET: load_calibration_dataset,
    "openthoughts": load_openthoughts_dataset,
    OPENTHOUGHTSDATASET: load_openthoughts_dataset, 
}