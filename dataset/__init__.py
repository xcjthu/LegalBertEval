from .nlp.JsonFromFiles import JsonFromFilesDataset
from .LawDataset import LawDataset
from .CNNLawDataset import CNNLawDataset
from .zjjdDataset import zyjdDataset
from .zyjdTestDataset import zyjdTestDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "law": LawDataset,
    "cnn": CNNLawDataset,
    "zyjd": zyjdDataset,
    "test": zyjdTestDataset
}
