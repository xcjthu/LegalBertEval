from .nlp.JsonFromFiles import JsonFromFilesDataset
from .LawDataset import LawDataset
from .CNNLawDataset import CNNLawDataset
dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "law": LawDataset,
    "cnn": CNNLawDataset
}
