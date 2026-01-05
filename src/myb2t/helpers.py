import torch
import re
from spellchecker import SpellChecker
from torch.utils.data import Dataset
import re
import unicodedata
import pandas as pd

def compute_class_weights(ds, vocab, device):
    from collections import Counter
    counts = Counter()
    for _, y, _, _ in ds:
        counts.update(y.tolist())

    num_classes = vocab.size
    weights = torch.ones(num_classes, dtype=torch.float32)
    pad_idx = vocab.PAD

    total_non_pad = sum(c for i, c in counts.items() if i != pad_idx)
    for i in range(num_classes):
        if i == pad_idx:
            weights[i] = 0.0
        else:
            c = counts.get(i, 1)
            weights[i] = total_non_pad / c

    weights = weights / weights.mean()
    return weights.to(device)

def make_key_padding_mask_from_lens(seq_lens, T, device=None):
    """
    seq_lens: [B] (int)
    T: int
    returns mask: [B, T] bool (True = pad)
    """
    if seq_lens.dim() != 1:
        seq_lens = seq_lens.view(-1)

    dev = device if device is not None else seq_lens.device
    seq_lens = seq_lens.to(device=dev)

    B = seq_lens.size(0)
    t = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
    return t >= seq_lens.unsqueeze(1)

def make_causal_mask(batch, device=None):
    """
    """
    B, T, _ = batch.size()
    mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask

def corrupt_sequence(in_seq, p=0.5, pad=0, bos=1, eos=2):
    """
    """

    out_seq = in_seq.clone()
    if p == 0:
        out_seq = in_seq.clone()
    elif p == 1:
        mask = (in_seq != pad) & (in_seq != bos) & (in_seq != eos)
        out_seq[mask] = pad
    else:
        mask = (in_seq != pad) & (in_seq != bos) & (in_seq != eos)
        probs = torch.rand(in_seq.shape, device=in_seq.device)
        out_seq[(mask) & (probs < p)] = pad

    return out_seq

def _spell_check_word(word: str) -> str:
    """
    Spell-check a single word and return the corrected word (or original).
    """

    spell = SpellChecker()

    # Try to correct misspelled words
    corrected = word
    if word in spell.unknown([word]):
        corrected = spell.correction(word)

    # Revert the change if the spell checker failed to correct the word
    if corrected is None:
        corrected = word

    return corrected

def spell_check_sentence(sentence: str) -> str:
    """
    Spell-check words in a sentence, preserving spaces and punctuation.
    """
    return re.sub(r'\w+', lambda m: _spell_check_word(m.group(0)), sentence)

def make_default_config():
    """
    """

    config = {

        # Model architecture
        "d_model": 512,
        "d_ff": 2048,
        "d_session": 16,
        "n_encoder_layers": 4,
        "n_decoder_layers": 2,
        "n_attn_heads": 8,

        # Logistics
        "max_tgt_seq_len": 128,

        # Weirdos
        "alpha": 0.8, # Balances objectives (higher values emphasizes characters, lower values emphasize phonemes)

        # Regularization
        "dropout": 0.15,
        "corruption": 0.1,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "patience": 5,
        "batch_size": 32,
        "weight_decay": 0.003,

        # Re-scoring/ranking hyperparameters
        "use_lm": True,
        "lm_name": "gpt2",
        "beam_size": 10,
        "min_dec_seq_len": 3,
        "max_dec_seq_len": 128,
        "am_weight": 0.2,
        "lm_weight": 0.3,
        "length_bonus": 0.0,
        "cache_lm_scores": True,

        # Learning rate scheduling hyperparameters
        "max_iter": 100,
        "lr": 0.00005,
        "max_warmup_steps": 2000, # Maximum number of steps used in warmup
        "warmup_fraction": 0.1, # Target fraction of training for warmup
        "hold_fraction": 0.05 # Fraction of training for holding LR steady (at maximum value)
    }

    return config

class SubsetWithAttrs(Dataset):
    """
    """

    def __init__(self, dataset, indices, attrs=("sentences", "_sentences")):
        self.dataset = dataset
        self.indices = list(indices)

        # Forward/slice specified attributes if present
        for name in attrs:
            if hasattr(dataset, name):
                val = getattr(dataset, name)
                setattr(self, name, [val[i] for i in self.indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

def normalize_sentence(s):
    """
    """

    #
    _re_keep = re.compile(r"[^a-z0-9'\s]+")
    _re_ws = re.compile(r"\s+")

    #
    if s is None:
        return ""

    # Unicode normalize (handles odd apostrophes, etc.)
    s = unicodedata.normalize("NFKC", s)

    # Lowercase
    s = s.lower()

    # Convert “smart quotes” apostrophes to plain apostrophe
    s = s.replace("’", "'").replace("`", "'")

    # Remove disallowed characters (keep letters, digits, whitespace, apostrophe)
    s = _re_keep.sub(" ", s)

    # Collapse whitespace
    s = _re_ws.sub(" ", s).strip()

    return s

def generate_kaggle_submission(est, ds, dst, algo="beam", check_spelling=False):
    """
    """

    tokens, sentences = est.predict(ds, algo=algo, check_spelling=check_spelling)
    df = pd.DataFrame({
        "id": list(range(len(sentences))),
        "text": sentences
    })
    df.to_csv(dst, index=False)

    return

        