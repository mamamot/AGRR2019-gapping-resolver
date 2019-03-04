from .imports import *
from collections import defaultdict

class LinearDecoder(nn.Module):
    def __init__(self, encoder, output_size, dropout_p=0.1, weights=None):
        super(LinearDecoder, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_to_tag = nn.Linear(ENCODER_OUTPUT_DIM, output_size)
        self.activation = nn.LogSoftmax(dim=2)
        self.loss_f = nn.NLLLoss(reduction="none", weight=weights)
        
    def forward(self, input_seq, output_seq, mask):
        """
        Returns predictions and loss
        """
        encoded_states = self.dropout(self.encoder(input_seq)[0][2])
        predictions = self.activation(self.hidden_to_tag(encoded_states))
        losses = list()
        for i, (pred, active, out) in enumerate(zip(predictions, mask, output_seq)):
            loss = self.loss_f(pred, out)
            masked_loss = loss * active
            losses.append(torch.sum(masked_loss))
        losses = torch.stack(losses)
        total_loss = torch.sum(losses)

        return predictions, total_loss
    
def get_encoder(itos):
    emb_sz=400
    lin_ftrs = [50]
    layers = [emb_sz*3] + lin_ftrs + [2]

    drops = np.array([0.4, 0.1])

    model = get_rnn_classifier(bptt=70, 
                               max_seq=70*20, 
                               vocab_sz=len(itos), 
                               emb_sz=400, 
                               n_hid=1150, 
                               n_layers=3,
                               pad_token=1,
                               layers=layers,
                               drops=drops,
                               qrnn=False
                               )
    enc = model[0]
    enc.eval()
    return enc

def load_tag_model(path, device, itos):
    encoder = get_encoder(itos)
    model = LinearDecoder(encoder, DECODER_OUTPUT_DIM)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_binary(classifier):
    preds = classifier.get_preds(ordered=True, ds_type=DatasetType.Test)
    return [l.argmax().item() for l in preds[0]]

def predict_labels(model, data, device, thresh=0.0):
    flatten = lambda x: [item for sublist in x for item in sublist]
    model.eval()
    total_loss = 0
    
    all_labels = list()
    predicted_labels = list()
    predicted_proba = list()
    
    with torch.no_grad():
        for inputs, outputs, masks in data:
            inputs, outputs, masks = inputs.to(device), outputs.to(device), masks.to(device)
            predictions, loss = model(inputs, outputs, masks)
            total_loss += loss.item()
            
            outputs_ = outputs.cpu().detach().numpy()
            masks_ = masks.cpu().detach().numpy()
            predictions_ = predictions.cpu().detach().numpy()
            
            for output_, mask_, pred_ in zip(outputs_, masks_, predictions_):
                predicted = defaultdict(list)
                true_labels = output_[mask_.astype(bool)]
                preds = np.argmax(pred_, axis=1)[mask_.astype(bool)]
                preds_probas = np.exp(np.max(pred_, axis=1)[mask_.astype(bool)])
                thresholded = list()
                for i, (l, p) in enumerate(zip(preds, preds_probas)):
                    if p > thresh:
                        thresholded.append(l)
                        predicted[l].append((i, p))
                    else:
                        thresholded.append(out_stoi["O"])
                if len(predicted[out_stoi["A"]]) < len(predicted[out_stoi["P"]]):
                    less_probable = list(sorted(predicted[out_stoi["P"]], key=lambda x: x[1]))
                    for i, _ in less_probable[:-len(predicted[out_stoi["A"]])]:
                        thresholded[i] = out_stoi["O"]
                predicted_labels.append(thresholded)
                all_labels.extend(true_labels)
                predicted_proba.append(preds_probas)

    return total_loss, predicted_labels, predicted_proba
