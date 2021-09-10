from .nlp.JsonFromFiles import JsonFromFilesDataset
from .LawDataset import LawDataset
dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "law": LawDataset,
}
