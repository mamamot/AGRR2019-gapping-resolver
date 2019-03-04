import argparse
import logging

from utils.imports import *
from utils.model import load_tag_model, predict_binary, predict_labels
from utils.preprocess import preprocess_strings, get_dataloader
from utils.postprocess import post_process

def get_device(gpu):
    if gpu:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def resolve(df, gpu=False):
    device = get_device(gpu)
    
    logging.info("Loading classification model...")
    # Load fast.ai databunch
    with open(CLS_ITOS_PATH, "rb") as f:
        cls_itos = pickle.load(f)
    
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    dev_df = pd.read_csv(DEV_PATH, sep="\t")

    train_df['is_valid'] = False
    dev_df['is_valid'] = True

    cls_df = pd.concat([train_df, dev_df])

    cls_data = (TextList.from_df(df=cls_df, path=ARTIFACTS_PATH, vocab=Vocab(cls_itos), cols="text")
            .split_from_df(col="is_valid")
            .label_from_df(cols="class").add_test(df.text))

    classifier = text_classifier_learner(cls_data.databunch(), path=ARTIFACTS_PATH)

    # Load fast.ai classifier
    classifier.load(CLS_MODEL_NAME)

    # Classify sentences
    logging.info("Loaded, starting classification.")
    binary_predictions = predict_binary(classifier)
    # Unload classifier and databunch
    del cls_data
    del classifier
    if gpu:
        torch.cuda.empty_cache()

    logging.info("Done, starting sequence processing.")

    with open(TAG_ITOS_PATH, "rb") as f:
        cls_itos = pickle.load(f)
    
    # Prepare texts and spans for seq2seq
    in_tokens, out_tokens, offsets = preprocess_strings(df)
    
    dl = get_dataloader(in_tokens, out_tokens, cls_itos)
    
    logging.info("Loaded, loading tagging model...")

    # Load seq2seq model
    model = load_tag_model(MODELS_PATH/TAG_MODEL_NAME, device, cls_itos)
    model.to(device)
    
    logging.info("Done, predicting sequences...")
    # Predict tokens
    _, labels_predictions, probas = predict_labels(model, dl, device)

    # Transform tokens back into spans
    
    logging.info("Post-processing...")
    df_filled = post_process(df, binary_predictions, labels_predictions, offsets)
    
    logging.info("Done.")
    return df_filled

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input file in AGRR format")
    parser.add_argument("output_file", help="Output file")
    parser.add_argument("--gpu", action="store_true", help="Whether to use GPU")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    gpu = args.gpu

    in_df = pd.read_csv(input_file, sep="\t").fillna("")
    out_df = resolve(in_df, gpu)
    out_df.to_csv(output_file, sep="\t", index=None)
