"""
Hallucination Detection in Small Language Models

# Files you can edit:
    - aggregation.py — layer selection and token pooling 
    - aggregation.py | extract_geometric_features — optional hand-crafted features 
    - probe.py | HallucinationProbe — probe classifier (nn.Module subclass) 
    - splitting.py | split_data — train / validation / test split strategy 

# Fixed infrastructure (do not edit)
    - model.py | LLM loader (get_model_and_tokenizer) 
    - evaluate.py | Evaluation loop, summary table, JSON output 

# Data Format — ChatML and Special Tokens
    The `prompt` column uses ChatML (Chat Markup Language), the conversation
    template built into Qwen models.  Each message is wrapped in role markers:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    ... question and context ... <|im_end|>
    <|im_start|>assistant

    Special tokens and their roles:

    - `<|im_start|>` — opens a chat turn; the role (`system`, `user`, or `assistant`) immediately follows
    - `<|im_end|>` — closes the current chat turn
    - `<|endoftext|>` — end-of-sequence (EOS) token appended by the model at the end of its response

    The `prompt` ends right after `<|im_start|>assistant\n` — it provides the
    full context up to (but not including) the model's reply.  The `response`
    column holds the actual generated text, ending with `<|endoftext|>`.

    We feed the concatenation of `prompt + response` to the feature extractor
    so the hidden states capture both the question context and the model's
    specific answer — the hallucination signal lives in that joint representation.


"""

import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data

import nltk

# ---------------------------------------------------------------------

DATA_FILE     = "./data/dataset.csv"   # path to the dataset CSV
OUTPUT_FILE   = "results.json"         # where to write the results summary
BATCH_SIZE    = 4
TEST_FILE        = "./data/test.csv"   # competition test set (labels are null)
PREDICTIONS_FILE = "predictions.csv"   # output file with predicted labels

assert OUTPUT_FILE == "results.json"
assert PREDICTIONS_FILE == "predictions.csv"
# ---------------------------------------------------------------------

def define_important_tokens (all_texts: list, tokenizer) -> list:
    question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose']
    start = 'Here is the question: '
    end = 'Your answer'

    def get_question (text: str) -> str:
        start_index = text.index(start) + len(start)
        end_index = text.index(end, start_index)
        match = text[start_index:end_index]
        if match:
            for question in question_words:
                if question in match.lower():
                    return question
            return False
        else:
            return False
    
    def get_pos (question_word: str, text: str) -> list:
        pos_tokens = nltk.word_tokenize (text)
        pos_tags = nltk.pos_tag (pos_tokens)
        if question_word == 'who':
            return ['PRP', 'NNP', 'NNPS'], pos_tokens, pos_tags
        if question_word == 'where':
            return ['NN', 'NNS'], pos_tokens, pos_tags
        if question_word == 'when':
            return ['RB', 'RBR', 'RBS', 'VBN'], pos_tokens, pos_tags
        if question_word == 'why':
            return ['CC', 'VB', 'VBD', 'VBG', 'VBN'], pos_tokens, pos_tags
        if question_word == 'how':
            return ['RB', 'RBR', 'RBS', 'VBN'], pos_tokens, pos_tags
        if question_word == 'which':
            return ['JJ', 'JJR', 'JJS'], pos_tokens, pos_tags
        if question_word == 'whose':
            return ['PRP', 'NNP', 'NNPS'], pos_tokens, pos_tags
        if question_word == 'what': # can be about description or object
            if pos_tags [(next((i for i, token in enumerate(pos_tokens) if token.lower() == 'what'), None)+1)][1] [:2]=='NN':
                return ['JJ', 'JJR', 'JJS'], pos_tokens, pos_tags
            else:
                return ['NN', 'NNS'], pos_tokens, pos_tags

    def tokens_dict (text: str, tokens: list, tokenizer) -> list:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # match tokens and text as str
        tok_text_match = []
        start_search = 0
        for token in tokens:
            start_index = text.find(token, start_search)
            if start_index == -1:
                start_index = start_search
                end_index = start_search
            else:
                end_index = start_index + len (token) -1
            tok_text_match.append ({token: [start_index, end_index]})
            start_search = end_index

        # match encoding and text as str
        text_len = len (text)
        encoding_len = sum(1 for e in encoding ['attention_mask'][0] if e == 1)

        text_inds = list(range (text_len))
        symb_token_groups = [x.tolist() for x in np.array_split(text_inds, encoding_len)]

        # match tokens and encoding based on text
        tok_enc_match = []
        for i in range (0, len (tok_text_match)):
            token = list(tok_text_match [i].keys()) [0]
            start_text = tok_text_match [i] [token] [0]
            end_text = tok_text_match [i] [token] [1]

            start_enc = [i for i, e in enumerate(symb_token_groups) if start_text in e] [0]
            end_enc = [i for i, e in enumerate(symb_token_groups) if end_text in e] [0]
            tok_enc_match.append ({token: [ind for ind in range(start_enc, end_enc + 1)]})

            # save prompt - response tag
            if token == 'assistant':
                prompt_answer_ind = end_enc
        return tok_enc_match, prompt_answer_ind

    def get_important_tokens (token_enc_match: list, pos: list, tags: list) -> list:
        ids = []
        for t in range (0, len (tags)):
            token = tags [t][0]
            token_pos = tags [t][1]
            if token_pos in pos:
                ids.append (token_enc_match[t][token])
        ids = [item for sublist in ids for item in sublist]
        ids = sorted(list(set(ids)))
        return ids
    
    important_tokens_ids = []
    for text in all_texts:
        question_word = get_question (text)
        if question_word:
            pos, tokens, tags = get_pos (question_word, text)
            token_enc_match, prompt_answer_ind = tokens_dict (text, tokens, tokenizer)
            important_tokens_ids.append ([get_important_tokens (token_enc_match, pos, tags), prompt_answer_ind])
        else: # then consider the first and the last token (its ind will be defined later)
            important_tokens_ids.append ([[0], 1])
    return important_tokens_ids


if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device       : {device}")
    print(f"Data         : {DATA_FILE}")
    print(f"Max length   : {MAX_LENGTH} tokens")

    df = pd.read_csv(DATA_FILE)

    # Build the text fed to the LLM: concatenation of prompt and response.
    all_texts  = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])

    n_total = len(all_labels)
    print(f"Loaded {n_total} samples  "
        f"({all_labels.sum()} hallucinated / {(all_labels == 0).sum()} truthful)")
    
    # Preview the raw data
    print(f"Columns : {df.columns.tolist()}")
    print(f"Rows    : {len(df)}")
    print(f"Labels  : {dict(df['label'].value_counts().sort_index())}")
    print()

    # Show the first sample (truncated for readability)
    row0 = df.iloc[0]
    print("── prompt (first 500 chars) " + "─" * 34)
    print(row0["prompt"][:500])
    print()
    print("── response (first 300 chars) " + "─" * 31)
    print(row0["response"][:300])
    print()
    label_str = "hallucinated" if int(row0["label"]) else "truthful"
    print(f"── label : {int(row0['label'])}  ({label_str})")


    # Load the LLM
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # Get important tokens ids
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    print ('Defining important tokens...')
    important_tokens_ids = define_important_tokens (all_texts, tokenizer)

    all_features: list = []
    t0 = time.time()

    for start in tqdm(range(0, len(all_texts), BATCH_SIZE),
                    desc="Extracting & aggregating", unit="batch"):

        # ── 1. Tokenise the current mini-batch ───────────────────────────────
        batch_texts = all_texts[start : start + BATCH_SIZE]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # ── 2. LLM forward pass ──────────────────────────────────────────────
        # outputs.hidden_states: tuple of (n_layers+1) tensors,
        # each with shape (batch, seq_len, hidden_dim).
        # Index 0 → token embeddings; index k → transformer layer k.
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # ── 3. Stack all layers into one tensor, move to CPU ─────────────────
        # Shape: (batch, n_layers, seq_len, hidden_dim)
        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask   = attention_mask.cpu()

        # ── 4. Aggregate each sample and store the compact feature vector ─────
        # The raw `hidden` tensor is released at the end of this loop iteration.
        for i in range(hidden.size(0)):
            # Leave vectors only from important tokens
            hidden_to_consider = hidden[i]
            tokens_ids = important_tokens_ids [start+i] [0]
            token_response_start = next((index for index, element in enumerate(tokens_ids) if element >= important_tokens_ids [start+i] [1]), None)
            if len (tokens_ids) <= 1:
                hidden_to_consider = hidden[i][:, [0, hidden.shape [1]-1], :]
            else:
                hidden_to_consider = hidden[i][:, tokens_ids, :]

            feat = aggregation_and_feature_extraction(
                hidden_to_consider,   # (n_layers, seq_len, hidden_dim)
                mask[i],     # (seq_len,)
                token_response_start,
            )
            all_features.append(feat.cpu())

    extract_time = time.time() - t0
    print(f"Done in {extract_time:.1f} s  —  {len(all_features)} feature vectors extracted")

    # Stack into the (N, feature_dim) matrix used by the probe.
    X = np.vstack([f.numpy() for f in all_features])   # shape: (N, feature_dim)
    y = all_labels                                       # shape: (N,)
    
    print(f"Feature matrix : {X.shape}  (feature_dim = {X.shape[1]})")

    splits = split_data(y, df)

    print(f"Splits : {len(splits)} fold(s)")
    for i, (tr, va, te) in enumerate(splits):
        print(f"  Fold {i + 1}: train={len(tr)}  "
            f"val={len(va) if va is not None else 'N/A'}  test={len(te)}")

    fold_results = run_evaluation(splits, X, y, HallucinationProbe)
    
    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    

    # ── Load test data ────────────────────────────────────────────────────────
    df_test    = pd.read_csv(TEST_FILE)
    test_texts = [f"{row['prompt']}{row['response']}" for _, row in df_test.iterrows()]
    test_ids   = df_test.index
    print(f"Test set loaded: {len(test_texts)} samples")

    # ── Extract features for test set (same loop as Section 4) ───────────────
    test_features: list = []

    # Get important tokens ids
    important_tokens_ids = define_important_tokens (test_texts, tokenizer)

    for start in tqdm(range(0, len(test_texts), BATCH_SIZE),
                    desc="Test extraction & aggregation", unit="batch"):

        batch_texts = test_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask   = attention_mask.cpu()

        for i in range(hidden.size(0)):
            # Leave vectors only from important tokens
            hidden_to_consider = hidden[i]

            tokens_ids = important_tokens_ids [start+i] [0]
            token_response_start = next((index for index, element in enumerate(tokens_ids) if element >= important_tokens_ids [start+i] [1]), None)
            if len (tokens_ids) <= 1:
                hidden_to_consider = hidden[i][:, [0, hidden.shape [1]-1], :]
            else:
                hidden_to_consider = hidden[i][:, tokens_ids, :]

            feat = aggregation_and_feature_extraction(
                hidden_to_consider, mask[i], token_response_start,
            )
            test_features.append(feat.cpu())

    X_test = np.vstack([f.numpy() for f in test_features])  # (n_test, feature_dim)

    print(f"Feature matrix : {X_test.shape}  (feature_dim = {X_test.shape[1]})")

    # ── Fit final probe on training + validation data only ──────────────────
    # Collect the union of all train and validation indices across every split.
    # For a single split this excludes idx_test; for k-fold every sample appears
    # in a training fold, so all samples are used (same as fitting on X, y).
    idx_non_test = np.unique(np.concatenate([
        np.concatenate([idx_tr, idx_va]) if idx_va is not None else idx_tr
        for idx_tr, idx_va, _ in splits
    ]))
    final_probe = HallucinationProbe()
    final_probe.fit(X[idx_non_test], y[idx_non_test])

    # ── Predict and save ────────────────────────────────────────────────────
    save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)
