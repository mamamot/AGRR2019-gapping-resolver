from .imports import *

def tokenize(s):
    """
    Space-preserving tokenization
    """
    tokens = list()
    for char in s:
        if char.isalnum():
            if len(tokens) > 0 and tokens[-1][-1].isalnum():
                tokens[-1].append(char)
            else:
                tokens.append([char])
        else:
            tokens.append([char])
    return [''.join(t) for t in tokens]

SPACE = " "

def get_token_offsets(s_tok):
    """
    Get offsets for tokens to match the dataset notation
    """
    start_indices = list()
    end_indices = list()
    pointer = 0
    for t in s_tok:
        if t != SPACE:
            start_indices.append(pointer)
            end_indices.append(pointer + len(t))
        pointer += len(t)
    indices = list(zip(start_indices, end_indices))
    return indices

PREDICATE = "P"
AFTER_ELIDED = "A"

def generate_target_string(row, demo=False):
    """
    Generate a target string for a sequence-to-sequence model
    """
    vs = row.V.strip()
    cvs = row.cV.strip()
    s = row.text.lower()
    s_tok = tokenize(s)
    s_tok_nospace = [t for t in s_tok if t != SPACE]
    target = ["O" for _ in range(len(s_tok_nospace))]
    offsets = get_token_offsets(s_tok)
    if row["class"] == 0:
        return s_tok_nospace, target, offsets
    if cvs:
        cvs = cvs.split(" ")
        for cv in cvs:
            if demo:
                print(cv)
            b, e = (int(c) for c in cv.split(":"))
            for i, index in enumerate(offsets):
                if b >= index[0] and e <= index[1]:
                    target[i] = PREDICATE
                if index[0] > e:
                    break
    if vs:
        vs = vs.split(" ")
        for v in vs:
            if demo:
                print(v)
            v = int(v.split(":")[0])
            for i, index in enumerate(offsets):
                if v == index[0]:
                    target[i] = AFTER_ELIDED
                if index[0] > v:
                    break
    if demo:
        print(s)
        print(list(zip(offsets, s_tok_nospace, target)))
    else:
        return s_tok_nospace, target, offsets
    
def preprocess_strings(df):
    """
    For each row of the dataframe in AGRR format, return lists of:
    1. Tokenized strings
    2. Token-level annotations
    3. Offsets for each token
    """
    res_input, res_output, res_offsets = list(), list(), list()
    row_res = df.apply(generate_target_string, axis=1)
    for _, v in row_res.iteritems():
        i, o, offset = v
        res_input.append(i)
        res_output.append(o)
        res_offsets.append(offset)
    return res_input, res_output, res_offsets

def get_dataloader(tokens, labels, itos):
    stoi = {s: i for i, s in enumerate(itos)}
    
    vocab = Vocab(stoi)
    
    def generate_seqs(tokenized, output_seq, max_len=256):
        tokenized = tokenized[:max_len]
        masked_tokens = ['xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxi']
        numericalized = np.array([stoi['xxpad']] * max_len)
        numericalized[:len(tokenized)] = vocab.numericalize(tokenized)
        mask = np.array([0] * max_len)
        mask[:len(tokenized)] = np.array([int(t not in masked_tokens) for t in tokenized])
        out = np.array([out_stoi['O']] * max_len)
        out_n = 0
        for i, masked in enumerate(mask):
            if masked:
                tok = out_stoi[output_seq[out_n]]
                out[i] = tok
                out_n += 1
        numericalized = torch.tensor(numericalized)
        out = torch.tensor(out)
        mask = torch.tensor(mask, dtype=torch.float)
        return numericalized, out, mask
    
    def data_to_tensors(input_seqs, output_seqs):
        text_tensors, output_tensors, masks = zip(*[generate_seqs(i, o) for i, o in zip(input_seqs, output_seqs)])
        texts = torch.stack(text_tensors)
        output = torch.stack(output_tensors)
        masks = torch.stack(masks)
        return texts, output, masks
    
    ds = TensorDataset(*data_to_tensors(tokens, labels))
    dl = DataLoader(ds, shuffle=False, batch_size=32)
    
    return dl
