from dataset.JsonFromFiles import JsonFromFilesDataset
from .CauseActionDataset import CauseActionDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "CauseAction": CauseActionDataset,
}
