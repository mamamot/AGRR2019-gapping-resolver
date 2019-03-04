from fastai.text import *

ENCODER_OUTPUT_DIM = 400
DECODER_OUTPUT_DIM = 3
ARTIFACTS_PATH = Path("artifacts")
DATABUNCH_PATH = ARTIFACTS_PATH/ "gapping_data_2"
MODELS_PATH = ARTIFACTS_PATH/"models"
CLS_MODEL_NAME = "gapping_classifier_2_3"
TAG_MODEL_NAME = "gapping_resolver_10.pt"
TAG_ITOS_PATH = ARTIFACTS_PATH/"tag_itos.pkl"
DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH/"train.csv"
DEV_PATH = DATA_PATH/"dev.csv"
CLS_ITOS_PATH = DATABUNCH_PATH/"itos.pkl"

SPACE = " "
PREDICATE = "P"
AFTER_ELIDED = "A"

out_stoi = {
    'A': 0,
    'P': 1,
    'O': 2
}

out_itos = list(sorted([v for v in out_stoi.keys()], key=lambda x: out_stoi[x]))
