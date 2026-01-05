from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from myb2t.helpers import normalize_sentence, spell_check_sentence

class BeamSearchMixin:
    """
    Beam decoding + optional external LM reranking (GPT-2).

    Config keys used:
      - use_lm (bool)
      - lm_name (str)
      - beam_size (int)
      - min_dec_seq_len (int)  # minimum decoded token length constraint (includes BOS in current implementation)
      - max_dec_seq_len (int)
      - am_weight (float)      # weight on model/decoder score (length-normalized)
      - lm_weight (float)      # weight on LM score (avg log-prob per LM token)
      - length_bonus (float)   # linear bonus/penalty on decoded length (T_dec)
      - cache_lm_scores (bool)
    """

    lm = None

    def _init_lm(self):
        """Initialize the external language model (optional)."""
        lm_name = self.config.get("lm_name", "gpt2")
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
        self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
        self.lm.to(self.device)
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad = False

    def _decode_char_sequence(self, token_seq_1d) -> str:
        """Translate character token IDs into a string (drops BOS/EOS/PAD)."""
        special_tokens = {self.v_chr.BOS, self.v_chr.EOS, self.v_chr.PAD}

        if torch.is_tensor(token_seq_1d):
            token_seq_1d = token_seq_1d.detach().cpu().numpy()

        token_seq = np.asarray(token_seq_1d)
        filtered = token_seq[~np.isin(token_seq, list(special_tokens))]
        chars = self.v_chr.decode(filtered)
        return "".join(chars)

    def _score_with_lm_total_logprob(self, sentence: str) -> float:
        """
        Return LM total log-prob over its own tokens (higher is better).
        Robust to empty inputs.
        """
        if not self.config.get("use_lm", False):
            return 0.0

        if sentence is None:
            return -1e9

        sentence = sentence.strip()
        if len(sentence) == 0:
            return -1e9

        input_ids = self.lm_tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        if input_ids.numel() == 0 or input_ids.shape[1] == 0:
            return -1e9

        with torch.no_grad():
            outputs = self.lm(input_ids, labels=input_ids)
            nll = float(outputs.loss.item())          # mean NLL per LM token
            avg_log_prob = -nll
            T_lm = int(input_ids.shape[1])
            total_log_prob = avg_log_prob * T_lm

        return total_log_prob

    @staticmethod
    def _first_eos_len(tok_seq: np.ndarray, eos_id: int) -> int:
        """
        Effective decoded length including EOS if present, otherwise full length.
        """
        tok_seq = np.asarray(tok_seq)
        eos = np.where(tok_seq == eos_id)[0]
        if len(eos) > 0:
            return int(eos[0] + 1)
        return int(tok_seq.shape[0])

    def _run_beam_search_single(
        self,
        X_single: torch.Tensor,
        z_single: torch.Tensor,
        seq_len_single: torch.Tensor,
        beam_size: int,
        max_dec_seq_len: int,
        min_dec_seq_len: int,
    ):
        """
        Beam search for a single sample using *decoder-only* scores.

        Returns:
          best_tokens_dec: np.ndarray [T]
          all_tokens: list[np.ndarray]
          all_scores: list[float]  (decoder log-prob sums, unnormalized)
        """
        X_b = X_single.unsqueeze(0)
        z_b = z_single.unsqueeze(0)
        seq_lens_b = seq_len_single.unsqueeze(0)

        beams = [{
            "tokens": torch.tensor([self.v_chr.BOS], device=self.device, dtype=torch.long),
            "log_prob": 0.0,
            "finished": False,
        }]

        # We will generate up to max_dec_seq_len tokens total (including BOS/EOS).
        for _t in range(max_dec_seq_len - 1):
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

                    # Hard constraint: disallow EOS until min_dec_seq_len reached.
                    # NOTE: this counts tokens INCLUDING BOS. If you want "min content length",
                    # use: (len(b["tokens"]) - 1) < min_content_len.
                    if tok_id == self.v_chr.EOS and len(b["tokens"]) < min_dec_seq_len:
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

    def _predict_with_beam_search(
        self,
        ds,
        batch_size: int = 16,
        check_spelling: bool = True,
        print_progress: bool = True,
        ):
        """
        Beam decoding + optional LM reranking.

        Returns:
          tokens_array: np.ndarray(dtype=object), shape [N]
          all_sentences: list[str], length N
        """
        use_lm = bool(self.config.get("use_lm", False))
        if use_lm and self.lm is None:
            self._init_lm()

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        n_batches = len(loader)

        self.model.eval()

        # Config (new nomenclature)
        beam_size = int(self.config.get("beam_size", 10))
        min_dec_seq_len = int(self.config.get("min_dec_seq_len", 3))
        max_dec_seq_len = int(self.config.get("max_dec_seq_len", 128))

        am_weight = float(self.config.get("am_weight", 1.0))
        lm_weight = float(self.config.get("lm_weight", 0.0))
        length_bonus = float(self.config.get("length_bonus", 0.0))

        cache_lm = bool(self.config.get("cache_lm_scores", True))
        lm_cache = {}

        def lm_score_avg(sentence_raw: str) -> float:
            """
            Normalize BEFORE LM scoring, return avg log-prob per LM token.
            """
            s_norm = normalize_sentence(sentence_raw)
            if len(s_norm) == 0:
                return -1e9

            if cache_lm and (s_norm in lm_cache):
                return lm_cache[s_norm]

            if not use_lm:
                score = 0.0
            else:
                lm_total = self._score_with_lm_total_logprob(s_norm)
                lm_ids = self.lm_tokenizer.encode(s_norm)
                T_lm = max(len(lm_ids), 1)
                score = lm_total / T_lm

            if cache_lm:
                lm_cache[s_norm] = score

            return score

        all_tokens = []
        all_sentences = []

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

                    # 1) Generate candidates (decoder-only beam)
                    best_tokens_dec, beam_tokens, beam_scores = self._run_beam_search_single(
                        X_single=X_single,
                        z_single=z_single,
                        seq_len_single=seq_lens_single,
                        beam_size=beam_size,
                        max_dec_seq_len=max_dec_seq_len,
                        min_dec_seq_len=min_dec_seq_len,
                    )

                    # 2) Optional LM reranking
                    if use_lm:
                        best_score = -float("inf")
                        best_tokens = best_tokens_dec
                        best_sentence_raw = self._decode_char_sequence(best_tokens_dec)

                        for tok_seq, dec_score_sum in zip(beam_tokens, beam_scores):
                            raw_sentence = self._decode_char_sequence(tok_seq)

                            T_dec = max(self._first_eos_len(tok_seq, self.v_chr.EOS), 1)

                            # Decoder average log-prob per decoded token (length-normalized)
                            am_score = float(dec_score_sum) / T_dec

                            # LM average log-prob per LM token
                            lm_score = lm_score_avg(raw_sentence)

                            # Combined reranking score
                            total_score = (
                                am_weight * am_score +
                                lm_weight * lm_score +
                                length_bonus * T_dec
                            )

                            if total_score > best_score:
                                best_score = total_score
                                best_tokens = tok_seq
                                best_sentence_raw = raw_sentence
                    else:
                        best_tokens = best_tokens_dec
                        best_sentence_raw = self._decode_char_sequence(best_tokens)

                    # 3) Postprocess AFTER selection
                    if check_spelling:
                        best_sentence_raw = spell_check_sentence(best_sentence_raw)

                    best_sentence = normalize_sentence(best_sentence_raw)

                    all_tokens.append(best_tokens)
                    all_sentences.append(best_sentence)

        tokens_array = np.array(all_tokens, dtype=object)
        return tokens_array, all_sentences


class GreedyDecodingMixin:
    def _predict_with_greedy_decoding(
        self,
        ds,
        batch_size: int = 16,
        check_spelling: bool = False,
        print_progress: bool = True,
        ):
        """
        Batched greedy decoding. Uses max_dec_seq_len from config unless overridden.
        """
        max_dec_seq_len = int(self.config.get("max_dec_seq_len", 128))

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        tokens = []
        sentences = []

        self.model.eval()
        special_tokens = {self.v_chr.BOS, self.v_chr.EOS, self.v_chr.PAD}
        n_batches = len(loader)

        with torch.no_grad():
            for i_batch, (_, X_batch, _, _, z_batch, seq_lens_batch) in enumerate(loader):
                if print_progress:
                    print(f"Working on batch {i_batch + 1} out of {n_batches}")

                X_batch = X_batch.to(self.device)
                z_batch = z_batch.to(self.device)
                seq_lens_batch = seq_lens_batch.to(self.device)

                B = X_batch.shape[0]

                y = torch.full(
                    (B, 1),
                    self.v_chr.BOS,
                    dtype=torch.long,
                    device=self.device,
                )

                finished = torch.zeros(B, dtype=torch.bool, device=self.device)

                for _t in range(max_dec_seq_len - 1):
                    _, logits_chr, _ = self.model(X_batch, None, y, z_batch, seq_lens_batch)
                    next_token_logits = logits_chr[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=1)

                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, self.v_chr.EOS),
                        next_token,
                    )

                    y = torch.cat([y, next_token.unsqueeze(1)], dim=1)
                    finished = finished | (next_token == self.v_chr.EOS)
                    if finished.all():
                        break

                # pad to fixed length max_dec_seq_len
                if y.shape[1] < max_dec_seq_len:
                    pad = torch.full(
                        (B, max_dec_seq_len - y.shape[1]),
                        self.v_chr.PAD,
                        dtype=torch.long,
                        device=self.device,
                    )
                    y = torch.cat([y, pad], dim=1)

                y_cpu = y.to("cpu")

                # Drop all tokens after first EOS
                for i_seq, tgt_seq in enumerate(y_cpu):
                    indices = torch.where(tgt_seq == self.v_chr.EOS)[0]
                    if len(indices) == 0:
                        continue
                    index = indices[0]
                    y_cpu[i_seq, index + 1:] = self.v_chr.PAD

                seqs = y_cpu.numpy()
                tokens.append(seqs)

                for in_seq in seqs:
                    mask = np.isin(in_seq, list(special_tokens))
                    filtered = in_seq[~mask]
                    characters = self.v_chr.decode(filtered)
                    sentence = "".join(characters)
                    if check_spelling:
                        sentence = spell_check_sentence(sentence)
                    sentences.append(sentence)

        tokens = np.concatenate(tokens, axis=0)
        return tokens, sentences
