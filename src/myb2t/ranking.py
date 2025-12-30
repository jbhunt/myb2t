from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
from myb2t.helpers import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class BeamSearchMixin:
    """
    Mixin for:
      - initializing an optional HF LM
      - decoding char token sequences
      - scoring text with LM robustly (guards against empty inputs)
    """

    lm = None

    def _init_lm(self):
        """
        Initialize the language model (optional)
        """
        lm_name = self.config.get("lm_name", "gpt2")
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
        self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
        self.lm.to(self.device)
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad = False
        return

    def _decode_char_sequence(self, token_seq_1d):
        """
        Translate character token IDs into a string (drops BOS/EOS/PAD).
        """
        special_tokens = [self.v_chr.BOS, self.v_chr.EOS, self.v_chr.PAD]

        if torch.is_tensor(token_seq_1d):
            token_seq_1d = token_seq_1d.detach().cpu().numpy()

        token_seq = np.asarray(token_seq_1d)
        mask = ~np.isin(token_seq, special_tokens)
        filtered = token_seq[mask]

        chars = self.v_chr.decode(filtered)
        return "".join(chars)

    def _score_with_lm(self, sentence):
        """
        Return a scalar LM score (total log-prob; higher is better).
        Robust to empty inputs.
        """
        if not self.config.get("use_lm", False):
            return 0.0

        # Guard
        if sentence is None:
            return -1e9

        sentence = sentence.strip()
        if len(sentence) == 0:
            return -1e9

        input_ids = self.lm_tokenizer.encode(sentence, return_tensors="pt").to(self.device)

        # Guard against empty tokenization
        if input_ids.numel() == 0 or input_ids.shape[1] == 0:
            return -1e9

        with torch.no_grad():
            outputs = self.lm(input_ids, labels=input_ids)
            nll = float(outputs.loss.item())
            avg_log_prob = -nll
            T = int(input_ids.shape[1])
            total_log_prob = avg_log_prob * T

        return total_log_prob

    def _run_beam_search(
        self,
        X_single,
        z_single,
        seq_len,
        max_tgt_seq_len,
        beam_size,
        min_len=1,
        ):
        """
        Run beam search for a single sample.

        Returns:
          best_tokens_dec: np.ndarray [T]
          all_tokens: list[np.ndarray]
          all_scores: list[float] (decoder log-prob sums)
        """

        X_b = X_single.unsqueeze(0)
        z_b = z_single.unsqueeze(0)
        seq_lens_b = seq_len.unsqueeze(0)

        beams = [{
            "tokens": torch.tensor([self.v_chr.BOS], device=self.device, dtype=torch.long),
            "log_prob": 0.0,
            "finished": False,
        }]

        for t in range(max_tgt_seq_len - 1):
            if all(b["finished"] for b in beams):
                break

            new_beams = []

            for b in beams:
                if b["finished"]:
                    new_beams.append(b)
                    continue

                y = b["tokens"].unsqueeze(0)  # [1, t_len]
                _, logits_chr, _ = self.model(
                    X_b, None, y, z_b, seq_lens_b,
                    chr_pad_token=self.v_chr.PAD
                )

                last_logits = logits_chr[:, -1, :]                 # [1, V]
                log_probs = F.log_softmax(last_logits, dim=-1)[0]  # [V]

                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    tok = topk_tokens[k]
                    tok_id = int(tok.item())

                    # Enforce min_len before EOS (true constraint, not just reranking hack)
                    # Here min_len counts total tokens incl BOS. Adjust if you want "content length".
                    if tok_id == self.v_chr.EOS and len(b["tokens"]) < min_len:
                        continue

                    tok_lp = float(topk_log_probs[k].item())
                    new_tokens = torch.cat([b["tokens"], tok.unsqueeze(0)], dim=0)
                    new_log_prob = b["log_prob"] + tok_lp
                    finished = (tok_id == self.v_chr.EOS)

                    new_beams.append({
                        "tokens": new_tokens,
                        "log_prob": new_log_prob,
                        "finished": finished,
                    })

            new_beams.sort(key=lambda bb: bb["log_prob"], reverse=True)
            beams = new_beams[:beam_size]

        all_tokens = [b["tokens"].detach().cpu().numpy() for b in beams]
        all_scores = [float(b["log_prob"]) for b in beams]

        best_idx = int(np.argmax(all_scores))
        best_tokens_dec = all_tokens[best_idx]

        return best_tokens_dec, all_tokens, all_scores

    def _predict_with_beam_seach(
        self,
        ds,
        max_tgt_seq_len=128,
        batch_size=16,
        check_spelling=True,
        print_progress=True
        ):
        """
        Beam decoding + (optional) LM reranking.
        """

        #
        if self.config.get("use_lm") and self.lm is None:
                self._init_lm()

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_tokens = []
        all_sentences = []

        self.model.eval()
        n_batches = len(loader)

        beam_size = int(self.config.get("beam_size", 10))
        beta = float(self.config.get("beta", 0.0))
        alpha = float(self.config.get("alpha", 1.0))
        beta_len = float(self.config.get("beta_len", 0.0))
        min_len = int(self.config.get("min_len", 3))  # recommend >=2 so BOS->EOS is disallowed
        cache_lm = bool(self.config.get("cache_lm_scores", True))

        lm_cache = {}

        def _effective_dec_len(tok_seq):
            tok_seq = np.asarray(tok_seq)
            eos = np.where(tok_seq == self.v_chr.EOS)[0]
            if len(eos) > 0:
                return int(eos[0] + 1)
            return int(tok_seq.shape[0])

        def _lm_score_avg(sentence_raw):
            """
            Normalize BEFORE LM scoring, and return avg log-prob per LM token.
            """
            s_norm = normalize_sentence(sentence_raw)

            if len(s_norm) == 0:
                return -1e9  # make empty strings extremely unattractive

            if cache_lm and (s_norm in lm_cache):
                return lm_cache[s_norm]

            if not self.config.get("use_lm", False):
                score = 0.0
            else:
                lm_total = self._score_with_lm(s_norm)
                lm_ids = self.lm_tokenizer.encode(s_norm)
                T_lm = max(len(lm_ids), 1)
                score = lm_total / T_lm

            if cache_lm:
                lm_cache[s_norm] = score

            return score
        
        with torch.no_grad():
            for i_batch, (_, X_batch, _, _, z_batch, seq_lens_batch) in enumerate(loader):

                if print_progress:
                    print(f"Working on batch {i_batch + 1} out of {n_batches}")

                X_batch = X_batch.to(self.device)
                z_batch = z_batch.to(self.device)
                seq_lens_batch = seq_lens_batch.to(self.device)

                B = X_batch.shape[0]

                for b in range(B):
                    X_single = X_batch[b]
                    z_single = z_batch[b]
                    seq_lens_single = seq_lens_batch[b]

                    best_tokens_dec, beam_tokens, beam_scores = self._run_beam_search(
                        X_single,
                        z_single,
                        seq_lens_single,
                        max_tgt_seq_len=max_tgt_seq_len,
                        beam_size=beam_size,
                        min_len=min_len
                    )

                    if self.config.get("use_lm", False):
                        best_score = -float("inf")
                        best_tokens = best_tokens_dec
                        best_sentence = self._decode_char_sequence(best_tokens_dec)

                        for tok_seq, dec_score in zip(beam_tokens, beam_scores):
                            raw_sentence = self._decode_char_sequence(tok_seq)

                            T_dec = max(_effective_dec_len(tok_seq), 1)
                            dec_lp = float(dec_score) / T_dec

                            lm_lp = _lm_score_avg(raw_sentence)

                            # Combined score (length-normalized + optional explicit length term)
                            total_score = (
                                alpha * dec_lp +
                                beta * lm_lp +
                                beta_len * T_dec
                            )

                            if total_score > best_score:
                                best_score = total_score
                                best_tokens = tok_seq
                                best_sentence = raw_sentence

                    else:
                        best_tokens = best_tokens_dec
                        best_sentence = self._decode_char_sequence(best_tokens)

                    # Postprocess AFTER selecting best
                    if check_spelling:
                        best_sentence = spell_check_sentence(best_sentence)

                    best_sentence = normalize_sentence(best_sentence)

                    all_tokens.append(best_tokens)
                    all_sentences.append(best_sentence)

        tokens_array = np.array(all_tokens, dtype=object)
        return tokens_array, all_sentences
    
class GreedyDecodingMixin():
    """
    """

    def _predict_with_greedy_decoding(self, ds, max_tgt_seq_len=128, batch_size=16, check_spelling=False, print_progress=True):
        """
        Batched greedy decoding
        """

        # Dataset + DataLoader for batching
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        #
        tokens = []
        sentences = []

        #
        self.model.eval()
        special_tokens = [self.v_chr.BOS, self.v_chr.EOS, self.v_chr.PAD]
        n_batches = len(loader)
        with torch.no_grad():
            for i_batch, (_, X_batch, _, _, z_batch, seq_lens_batch) in enumerate(loader):
               
                #
                if print_progress:
                    print(f"Working on batch {i_batch + 1} out of {n_batches}")

                # Move this batch to the target device
                X_batch = X_batch.to(device=self.device)
                z_batch = z_batch.to(device=self.device)
                seq_lens_batch = seq_lens_batch.to(device=self.device)

                B = X_batch.shape[0]

                # Start-of-sequence tokens
                y = torch.full(
                    (B, 1),
                    self.v_chr.BOS,
                    dtype=torch.long,
                    device=self.device,
                )

                finished = torch.zeros(B, dtype=torch.bool, device=self.device)

                # Greedy decoding for this batch
                for t in range(max_tgt_seq_len - 1):
                    logits_pho, logits_chr, _ = self.model(X_batch, None, y, z_batch, seq_lens_batch)
                    next_token_logits = logits_chr[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=1)

                    # Force EOS for sequences that are already finished
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, self.v_chr.EOS),
                        next_token,
                    )

                    y = torch.cat([y, next_token.unsqueeze(1)], dim=1)

                    finished = finished | (next_token == self.v_chr.EOS)
                    if finished.all():
                        break

                # Ensure fixed length is the same as max_tgt_seq_len
                if y.shape[1] < max_tgt_seq_len:
                    pad = torch.full(
                        (B, max_tgt_seq_len - y.shape[1]),
                        self.v_chr.PAD,
                        dtype=torch.long,
                        device=self.device,
                    )
                    y = torch.cat([y, pad], dim=1)

                # Move predictions back to CPU
                y_detached = y.to(device="cpu")

                # Drop all tokens after the first EOS (set to PAD)
                for i_seq, tgt_seq in enumerate(y_detached):
                    indices = torch.where(tgt_seq == self.v_chr.EOS)[0]
                    if len(indices) == 0:
                        continue
                    index = indices[0]
                    y_detached[i_seq, index + 1 :] = self.v_chr.PAD

                #
                seqs = y_detached.numpy()
                tokens.append(seqs)

                # Decode to sentences and spell-check
                for in_seq in seqs:
                    mask = np.isin(in_seq, special_tokens)
                    filtered = in_seq[~mask]
                    characters = self.v_chr.decode(filtered)
                    sentence = "".join(characters)
                    if check_spelling:
                        sentence = spell_check_sentence(sentence)
                    sentences.append(sentence)

        # Concatenate all batches along sample axis
        tokens = np.concatenate(tokens, axis=0)

        return tokens, sentences
