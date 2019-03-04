from .imports import *

def post_process(df, binary_predictions, labels_predictions, offsets):
    """
    Given the empty (text-only) AGRR-formatted dataframe, binary predictions and
    label predictions, return an annotated dataframe.
    """
    def get_dict():
        return {
            "text": "",
            "class": "",
            "cV": "",
            "cR1": "",
            "cR2": "",
            "V": "",
            "R1": "",
            "R2": "",
        }

    output_lines = list()
    columns = ['text', 'class', 'cV', 'cR1', 'cR2', 'V', 'R1', 'R2']
    
    for i, row in df.iterrows():
        output_line = get_dict()
        output_line["text"] = row["text"]
        binary_pred = binary_predictions[i]
        output_line["class"] = binary_pred
        if binary_pred:
            V_tags = list()
            cV_tags = list()
            for n, tag in enumerate(labels_predictions[i]):
                tag = out_itos[tag]
                try:
                    if tag == "A":
                        V_tags.append(":".join([
                            str(offsets[i][n][0]), str(offsets[i][n][0])
                        ]))
                    elif tag == "P":
                        cV_tags.append(":".join([
                            str(offsets[i][n][0]), str(offsets[i][n][1])
                        ]))
                except:
                    print(labels_predictions[i])
                    print(offsets[i])
                    print(i)
                    raise
            output_line["cV"] = " ".join(cV_tags)
            output_line["V"] = " ".join(V_tags)
        output_lines.append(output_line)
    
    return pd.DataFrame.from_dict(output_lines)[columns]
