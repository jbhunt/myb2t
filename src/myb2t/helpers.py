import torch
import re
from spellchecker import SpellChecker

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

def make_key_padding_mask(batch, seq_lens, device=None):
    """
    """

    B, T, N = batch.size()
    mask = torch.full([B, T], True, dtype=torch.bool)
    for i_x, x in enumerate(batch):
        seq_len = int(seq_lens[i_x])
        mask[i_x, :seq_len] = False

    #
    if device is not None:
        mask = mask.to(device=device)

    return mask

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
        "d_model": 512,
        "d_ff": 2048,
        "d_session": 16,
        "n_encoder_layers": 6,
        "n_decoder_layers": 3,
        "n_attn_heads": 8,
        "lr": 0.00005,
        "max_tgt_seq_len": 128,
        "dropout": 0.15,
        "alpha": 0.7,
        "corruption": 0.15,
        "max_iter": 300,
        "early_stopping": True,
        "tolerance": 10,
        "validation_fraction": 0.1,
        "batch_size": 16,
    }

    return config
