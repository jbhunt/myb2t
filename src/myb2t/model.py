from torch import nn
from torch.nn import functional as F
import torch
import math
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from myb2t.vocab import PhonemeVocabulary, CharacterVocabulary
from myb2t.helpers import *
from myb2t.pretrain import Pretraining
import pathlib as pl
import copy
from jiwer import wer
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Constants
FLOAT_DTYPE = torch.float32
INTEGER_DTYPE = torch.long

class PositionalEncoding(nn.Module):
    """
    """

    def __init__(self, d_model, max_seq_len=1000, base=10000, dropout=None):
        """
        """

        super().__init__()
        pe = torch.zeros(max_seq_len, d_model, dtype=FLOAT_DTYPE)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

        return
    
    def forward(self, X):
        """
        """

        B, T, N = X.size()
        X_encoded = X + self.pe[:T, :].unsqueeze(0).to(X.device) # B x T x D
        if self.dropout is not None:
            X_encoded = self.dropout(X_encoded)
        return X_encoded
    
class TemporalSubsampler(nn.Module):
    def __init__(self, d_model, stride=2, kernel_size=5, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(d_model)

        # cache these for mask conv
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (kernel_size // 2) * dilation
        self.dilation = dilation

    def forward(self, x, seq_lens):
        """
        """
        B, T, D = x.shape
        device = x.device
        seq_lens = seq_lens.to(device=device, dtype=torch.long).view(-1)

        # --- main conv ---
        x_ds = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T', D]
        x_ds = self.norm(x_ds)

        # --- robust length computation via mask conv ---
        # mask: 1 for valid timesteps, 0 for pad
        t = torch.arange(T, device=device).unsqueeze(0)               # [1, T]
        valid = (t < seq_lens.unsqueeze(1)).float().unsqueeze(1)      # [B, 1, T]

        # Use a conv with all-ones kernel to see whether any valid input contributes
        # to each output position.
        ones_kernel = torch.ones(
            1, 1, self.kernel_size, device=device, dtype=valid.dtype
        )
        valid_ds = torch.nn.functional.conv1d(
            valid,
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )  # [B, 1, T']

        # output timestep is valid if it sees any valid input in its receptive field
        out_valid = (valid_ds.squeeze(1) > 0)             # [B, T']
        new_seq_lens = out_valid.sum(dim=1).to(torch.long)    # [B]

        # Safety: never exceed T'
        new_seq_lens = torch.clamp(new_seq_lens, max=x_ds.size(1))

        return x_ds, new_seq_lens

class Frontend(nn.Module):
    """
    Frontend module
    """

    def __init__(
        self,
        d_session=16,
        d_model=128,
        dropout=0.1,
        gate_hidden=128,
        stride=2,
        kernel_size=5
        ):
        """
        """

        super().__init__()

        # Model width
        self.d_model = d_model

        # Project each modality to model space
        self.proj_spikes = nn.Linear(256, d_model, dtype=FLOAT_DTYPE)
        self.proj_lfp = nn.Linear(256, d_model, dtype=FLOAT_DTYPE)

        # Gate
        self.gate_norm = nn.LayerNorm(2 * d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, gate_hidden, dtype=FLOAT_DTYPE),
            nn.GELU(),
            nn.Linear(gate_hidden, 1, dtype=FLOAT_DTYPE),
        )

        # Session embedding
        self.session_embedding = nn.Linear(512, d_session, dtype=FLOAT_DTYPE)

        # Map session embedding into FiLM parameters for d_model
        self.to_gamma = nn.Linear(d_session, d_model, dtype=FLOAT_DTYPE)
        self.to_beta  = nn.Linear(d_session, d_model, dtype=FLOAT_DTYPE)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

        #
        self.subsampler = TemporalSubsampler(
            d_model=d_model,
            stride=stride,
            kernel_size=kernel_size
        )

        #
        self.positional_encoding = PositionalEncoding(
            d_model,
            max_seq_len=3000
        )

        #
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        return

    def forward(self, X, z, seq_lens):
        """
        """

        # Split modalities
        X_spikes = X[:, :, :, 1]                        # [B, T, 256]
        X_lfp = X[:, :, :, 0]                           # [B, T, 256]

        # Project modalities
        x_spikes = self.proj_spikes(X_spikes)           # [B, T, d_model]
        x_lfp = self.proj_lfp(X_lfp)                    # [B, T, d_model]

        # Gate
        gate_in = torch.cat([x_spikes, x_lfp], dim=-1)  # [B, T, 2 * d_model]
        gate_in_normed = self.gate_norm(gate_in)        # [B, T, 2 * d_model]
        g = torch.sigmoid(self.gate(gate_in_normed))    # [B, T, 1]

        # Fuse
        x = g * x_spikes + (1.0 - g) * x_lfp            # [B, T, d_model]

        # Session encode
        s = self.session_embedding(z)                   # [B, d_session]

        # FiLM
        gamma = self.to_gamma(s).unsqueeze(1)           # [B, 1, d_model]
        beta = self.to_beta(s).unsqueeze(1)             # [B, 1, d_model]
        x = x * (1.0 + gamma) + beta

        # Downsample
        x, new_lens = self.subsampler(x, seq_lens)

        # Positional encoding
        x = self.positional_encoding(x)

        # Normalize and dropout
        x = self.norm(x)
        x = self.dropout(x)

        return x, g, new_lens

class BrainToCharacterTransformer(nn.Module):
    """
    """

    def __init__(
        self,
        phoneme_vocab_size,
        character_vocab_size,
        d_session=16,
        d_model=128,
        dim_ff=1024,
        dropout=0.1,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=3,
        ):
        """
        """


        super().__init__()

        #
        self.frontend = Frontend(
            d_session,
            d_model,
            dropout=dropout
        )
        self.neural_activity_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                nhead=n_heads,
                dim_feedforward=dim_ff,
                batch_first=True,
                dropout=dropout
            ),
            num_layers=n_encoder_layers
        )
        self.character_embedding = nn.Embedding(
            character_vocab_size,
            d_model,
        )
        self.encoder_input_dropout = nn.Dropout(dropout)
        self.decoder_input_dropout = nn.Dropout(dropout)
        self.character_positional_encoding = PositionalEncoding(
            d_model,
            dropout=dropout
        )
        self.character_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_decoder_layers
        )
        self.character_head= nn.Linear(
            in_features=d_model,
            out_features=character_vocab_size,
        )
        self.phoneme_head = nn.Linear(
            in_features=d_model,
            out_features=phoneme_vocab_size
        )

        return

    def forward(self, X, y_pho_seq, y_chr_seq, z, seq_lens, chr_pad_token=0):
        """
        """

        # Check data types
        device = X.device
        X = X.to(dtype=FLOAT_DTYPE)
        if y_pho_seq is not None:
            y_pho_seq = y_pho_seq.to(dtype=INTEGER_DTYPE)
        y_chr_seq = y_chr_seq.to(dtype=INTEGER_DTYPE)
        z = z.to(dtype=FLOAT_DTYPE)
        seq_lens = seq_lens.to(dtype=INTEGER_DTYPE)

        #
        enc_in, g, new_seq_lens = self.frontend(X, z, seq_lens[:, 0])
        new_seq_lens = new_seq_lens.to(device=X.device, dtype=torch.long)
        enc_in = self.encoder_input_dropout(enc_in)

        # Build key padding masks using sequence lengths
        T_enc_new = enc_in.size(1)
        memory_key_padding_mask = make_key_padding_mask_from_lens(
            new_seq_lens,
            T_enc_new,
            device=device
        )
        tgt_key_padding_mask = (y_chr_seq == chr_pad_token)

        # Encoder input
        memory = self.neural_activity_encoder(
            enc_in,
            src_key_padding_mask=memory_key_padding_mask
        )

        #
        logits_phoneme = self.phoneme_head(memory)

        # Decoder input
        dec_in = self.character_embedding(y_chr_seq)
        dec_in = self.decoder_input_dropout(dec_in) 
        dec_in = self.character_positional_encoding(dec_in)

        # Decoder output
        tgt_mask = make_causal_mask(dec_in, device=device)
        dec_out = self.character_decoder(
            tgt=dec_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits_chr = self.character_head(dec_out)
        return logits_phoneme, logits_chr, new_seq_lens
   
class BrainToTextDecoder():
    """
    """

    def __init__(
        self,
        config,
        out_dir=None,
        device=None,
        snapshot=None
        ):
        """
        """

        #
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Vocabs
        self.v_pho = PhonemeVocabulary()
        self.v_chr = CharacterVocabulary()

        # 
        self.config = config
        self.model = BrainToCharacterTransformer(
            d_model=config["d_model"],
            dim_ff=config["d_ff"],
            d_session=config["d_session"],
            n_encoder_layers=config["n_encoder_layers"],
            n_decoder_layers=config["n_decoder_layers"],
            n_heads=config["n_attn_heads"],
            dropout=config["dropout"],
            phoneme_vocab_size=self.v_pho.size,
            character_vocab_size=self.v_chr.size,
        ).to(self.device)

        #
        self.out_dir = out_dir
        if self.out_dir is not None:
            self.out_dir = pl.Path(self.out_dir)
            if self.out_dir.exists() == False:
                self.out_dir.mkdir()
            ckpt_dir = self.out_dir.joinpath("snapshots")
            if ckpt_dir.exists() == False:
                ckpt_dir.mkdir()

        #
        self.best_state_dict = None
        self.best_epoch = None
        self.loss_train = None
        self.loss_valid = None

        #
        if snapshot is not None:
            self.load(snapshot)

        #
        self.lm = None
        if self.config.get("use_lm"):
            self._init_lm()

        return
    
    def _init_lm(self):
        """
        Initialize the lange model (optional)
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
        Translate tokens into characters
        """

        special_tokens = [self.v_chr.BOS, self.v_chr.EOS, self.v_chr.PAD]
        if torch.is_tensor(token_seq_1d):
            token_seq_1d = token_seq_1d.cpu().numpy()
        token_seq = np.array(token_seq_1d)

        mask = ~np.isin(token_seq, special_tokens)
        filtered = token_seq[mask]

        chars = self.v_chr.decode(filtered)
        sentence = "".join(chars)

        return sentence
    
    def _score_with_lm(self, sentence):
        """
        Returns a scalar log-probability score using a language model
        """

        if self.config.get("use_lm") == False:
            return 0.0

        # Encode sentence
        input_ids = self.lm_tokenizer.encode(
            sentence,
            return_tensors="pt"
        ).to(self.device)    # shape: [1, T]

        with torch.no_grad():
            outputs = self.lm(input_ids, labels=input_ids)
            # outputs.loss is average NLL per token
            nll = outputs.loss.item()          # >= 0
            avg_log_prob = -nll                # log-prob per token
            seq_len = input_ids.shape[1]
            total_log_prob = avg_log_prob * seq_len

        return total_log_prob
    
    def _run_beam_search(
        self,
        X_single,        # [T_enc, n_channels] or [1, T_enc, n_channels] depending on your model
        z_single,        # [d_session] or [1, d_session]
        seq_len,         # [2] or [1,2]
        max_tgt_seq_len,
        beam_size,
        ):
        """
        Run beam search for a single sample
        """

        # Ensure shapes are batched (B=1) for your model call
        X_b = X_single.unsqueeze(0)              # [1, T_enc, C]
        z_b = z_single.unsqueeze(0)              # [1, d_session]
        seq_lens_b = seq_len.unsqueeze(0)        # [1,2] or similar

        # Each beam is a dict: {"tokens": 1D tensor, "log_prob": float, "finished": bool}
        beams = [{
            "tokens": torch.tensor([self.v_chr.BOS], device=self.device, dtype=torch.long),
            "log_prob": 0.0,
            "finished": False,
        }]

        for t in range(max_tgt_seq_len - 1):
            # Check if all beams finished
            if all(b["finished"] for b in beams):
                break

            new_beams = []

            # Expand each current beam
            for b in beams:
                if b["finished"]:
                    # Keep finished beams as-is
                    new_beams.append(b)
                    continue

                y = b["tokens"].unsqueeze(0)  # [1, t_len]
                logits_pho, logits_chr, _ = self.model(
                    X_b, None, y, z_b, seq_lens_b,
                    chr_pad_token=self.v_chr.PAD
                )   # logits_chr: [1, t_len, V_chr]

                # Get log-probs over vocab for last step
                last_logits = logits_chr[:, -1, :]          # [1, V_chr]
                log_probs = F.log_softmax(last_logits, dim=-1)[0]  # [V_chr]

                # Take top-k extensions
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    tok = topk_tokens[k]
                    tok_lp = topk_log_probs[k].item()

                    new_tokens = torch.cat([b["tokens"], tok.unsqueeze(0)], dim=0)
                    new_log_prob = b["log_prob"] + tok_lp
                    finished = (tok.item() == self.v_chr.EOS)

                    new_beams.append({
                        "tokens": new_tokens,
                        "log_prob": new_log_prob,
                        "finished": finished,
                    })

            # Prune to top beam_size beams by log_prob
            new_beams.sort(key=lambda bb: bb["log_prob"], reverse=True)
            beams = new_beams[:beam_size]

        # After decoding, collect beams
        all_tokens = [b["tokens"].detach().cpu().numpy() for b in beams]
        all_scores = [b["log_prob"] for b in beams]

        # Let the caller decide which one is "best"
        best_idx = int(np.argmax(all_scores))
        best_tokens = all_tokens[best_idx]

        return best_tokens, all_tokens, all_scores

    def enable_lm(self):
        self.config["use_lm"] = True

    def disable_lm(self):
        self.config["use_lm"] = False
    
    def load(self, filepath):
        """
        Load checkpoint (saved with `save`) onto CPU or CUDA if available.
        """

        checkpoint = torch.load(filepath)
        if checkpoint is None:
            print("Could not load snapshot")
            return

        # Get state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("best_state_dict"))
        if state_dict is None:
            raise KeyError("Checkpoint missing both 'model_state_dict' and 'best_state_dict'.")

        self.model.load_state_dict(state_dict)

        # Restore config if present
        old_config = checkpoint.get("config")
        self.config.update(old_config)

        # Decide device and move model there
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Keep a reference to best_state_dict if you want
        self.best_state_dict = state_dict

        return checkpoint
    
    def save(self, filepath):
        """
        Save a CPU-portable checkpoint (parameters and config dict).
        """

        filepath = pl.Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prefer the tracked best_state_dict, fall back to current model weights
        if self.best_state_dict is not None:
            state_dict = self.best_state_dict
        else:
            state_dict = self.model.state_dict()

        # Force everything to CPU for maximum portability
        cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}

        checkpoint = {
            "best_state_dict": cpu_state_dict,
            "config": getattr(self, "config", None),
        }

        torch.save(checkpoint, filepath)

        return
    
    def _evaluate_loss_function(
        self,
        X, y_pho, y_chr, z, seq_lens, chr_loss_fn, pho_loss_fn,
        corrupt_inputs=False
        ):
        """
        """

        #
        X = X.to(device=self.device)
        y_pho = y_pho.to(device=self.device)
        y_chr = y_chr.to(device=self.device)
        z = z.to(device=self.device)
        seq_lens = seq_lens.to(device=self.device)

        #
        if corrupt_inputs:
            y_inputs = corrupt_sequence(
                y_chr[:, :-1].clone(),
                p=self.config.get("corruption"),
                pad=self.v_chr.PAD,
                bos=self.v_chr.BOS,
                eos=self.v_chr.EOS
            )
        else:
            y_inputs = y_chr[:, :-1].clone()
        y_outputs = y_chr[:, 1:]

        #
        logits_pho, logits_chr , new_seq_lens = self.model(
            X,
            y_pho,
            y_inputs,
            z,
            seq_lens,
            chr_pad_token=self.v_chr.PAD
        )
        new_seq_lens = new_seq_lens.to(device=self.device, dtype=torch.long)

        # Character loss
        B, T_chr, V_chr = logits_chr.size()
        loss_chr = chr_loss_fn(
            logits_chr.view(-1, V_chr),
            y_outputs.reshape(-1).long(),
        )

        # Phoneme loss
        B, T_enc, V_pho = logits_pho.size()
        log_probs = F.log_softmax(logits_pho, dim=-1)  # [B, T_enc, V_pho]
        log_probs = log_probs.transpose(0, 1)          # [T_enc, B, V_pho]

        # encoder lengths (time dimension)
        # input_lengths = seq_lens[:, 0].long()          # [B]

        # phoneme target lengths
        target_lengths = seq_lens[:, 1].long()         # [B]
        mask = (y_pho != self.v_pho.PAD)
        targets_ctc = y_pho[mask].long()               # [sum(target_lengths)]

        #
        loss_pho = pho_loss_fn(
            log_probs,       # [T_enc, B, V_pho]
            targets_ctc,     # [sum(target_lengths)]
            new_seq_lens,
            target_lengths,  # [B]
        )

        #
        alpha = self.config.get("alpha_mtl")
        loss = alpha * loss_chr + (1 - alpha) * loss_pho

        return loss, loss_chr, loss_pho
    
    # TODO: Finish coding this
    def pretrain(self, ds):
        """
        """

        pt = Pretraining()

        return
    
    def fit(self, ds):
        """
        """

        # Dataset(s)
        if self.config.get("early_stopping"):
            n_total = len(ds)
            n_valid = max(1, int(n_total * self.config.get("validation_fraction")))
            n_train = n_total - n_valid
            all_indices = np.arange(n_total)
            train_indices = np.random.choice(all_indices, size=n_train, replace=False)
            valid_indices = np.delete(all_indices, train_indices)
            ds_train = SubsetWithAttrs(ds, train_indices)
            ds_valid = SubsetWithAttrs(ds, valid_indices)
            loader_valid = DataLoader(ds_valid, batch_size=self.config.get("batch_size"), shuffle=True)
        else:
            ds_train = ds
            ds_valid = None
            loader_valid = None
        loader_train = DataLoader(ds_train, batch_size=self.config.get("batch_size"), shuffle=True)

        # Loss functions
        chr_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.v_chr.PAD
        )
        pho_loss_fn = nn.CTCLoss(
            blank=self.v_pho.PAD,
            zero_infinity=True,
        )

        # Initialize optimizer
        optimizer = Adam(self.model.parameters(), lr=self.config.get("lr"))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,   # epochs with no val improvement
        )

        # Loss tracking
        self.loss_train = np.full(self.config.get("max_iter"), np.nan)
        self.loss_valid = np.full(self.config.get("max_iter"), np.nan)

        #
        n_epochs_without_improvement = 0
        self.best_state_dict = None
        best_loss_train = np.inf
        best_loss_valid = np.inf

        #
        snapshot_index = 0

        #
        for i_epoch in range(self.config.get("max_iter")):

            # Train
            self.model.train()
            batch_loss_train = 0.0    
            for i_batch, (i_trial, X_, y_pho, y_chr, z_, seq_lens_) in enumerate(loader_train):

                # Compute loss
                loss, loss_chr, loss_pho = self._evaluate_loss_function(
                    X_, y_pho, y_chr, z_, seq_lens_, chr_loss_fn, pho_loss_fn,
                    corrupt_inputs=True
                )
                batch_loss_train += loss.item()

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #
            batch_loss_train /= len(loader_train)

            # Validation
            batch_loss_valid = np.nan
            if self.config.get("early_stopping"):
                self.model.eval()
                batch_loss_valid = 0.0
                with torch.no_grad():
                    for i_batch, (i_trial, X_, y_pho, y_chr, z_, seq_lens_) in enumerate(loader_valid):
                        loss, _, _ = self._evaluate_loss_function(
                            X_, y_pho, y_chr, z_, seq_lens_, chr_loss_fn, pho_loss_fn,
                            corrupt_inputs=False
                        )
                        batch_loss_valid += loss.item()
                batch_loss_valid /= len(loader_valid)

            # Estimate WER (downsample to 3x the batch size to save time)
            if ds_valid is not None:
                n_samples = int(3 * self.config.get("batch_size"))
                if n_samples > len(ds_valid):
                    n_samples = len(ds_valid)
                idxs = np.random.choice(np.arange(n_samples), size=n_samples, replace=False)
                ds_valid_subset = SubsetWithAttrs(ds_valid, idxs)
                tokens, hypothesis = self.predict(ds_valid_subset, algo="greedy", print_progress=False)
                reference = [ds_valid_subset.dataset.sentences[i] for i in ds_valid_subset.indices]
                wer_valid = wer(reference, hypothesis)
            else:
                wer_valid = np.nan

            # Update learning rate
            if self.config.get("early_stopping") and not np.isnan(batch_loss_valid):
                scheduler.step(batch_loss_valid)
            else:
                scheduler.step(batch_loss_train) 
            
            # Print out loss values
            self.loss_train[i_epoch] = batch_loss_train
            self.loss_valid[i_epoch] = batch_loss_valid
            print(f'Epoch {i_epoch + 1} out of {self.config.get("max_iter")}: Training loss = {batch_loss_train:0.6f}, Validation loss = {batch_loss_valid:0.6f}, Validation WER = {wer_valid:.2f}')

            # Update best state dict
            if batch_loss_train < best_loss_train:
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                best_loss_train = batch_loss_train

            # Save snapshot (optional)
            if self.out_dir is not None:
                if ((i_epoch + 1) % 10) == 0:
                    dst = self.out_dir.joinpath("snapshots", f"snapshot-{snapshot_index + 1}.pkl")
                    self.save(dst)
                    snapshot_index += 1

            # Check for early stopping condition (and save best state and epoch)
            if self.config.get("early_stopping"):
                if batch_loss_valid < best_loss_valid:
                    best_loss_valid = batch_loss_valid
                    self.best_epoch = i_epoch
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    n_epochs_without_improvement = 0
                else:
                    n_epochs_without_improvement += 1
            else:
                if batch_loss_train < best_loss_train:
                    best_loss_train = batch_loss_train
                    self.best_epoch = i_epoch
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
            if n_epochs_without_improvement >= self.config.get("tolerance"):
                print(f"Early stopping condition met: loss has not improved in {self.config.get('tolerance')} epochs")
                break

        # Save the best snapshot
        if self.out_dir is not None:
            dst = self.out_dir.joinpath("snapshots", f"snapshot-final.pkl")
            self.save(dst)

        return
    
    def predict(self, ds, algo="beam", max_tgt_seq_len=128, batch_size=16, check_spelling=True, print_progress=True):
        """
        """

        if algo == "greedy":
            tokens, sentences = self._predict_with_greedy_decoding(
                ds,
                max_tgt_seq_len=max_tgt_seq_len,
                batch_size=batch_size,
                check_spelling=check_spelling,
                print_progress=print_progress
            )
        elif algo == "beam":
            tokens, sentences = self._predict_with_beam_seach(
                ds,
                max_tgt_seq_len=max_tgt_seq_len,
                batch_size=batch_size,
                check_spelling=check_spelling,
                print_progress=print_progress
            )
        else:
            raise Exception("{algo} is not a valid algorithm")
        
        #
        sentences_normalized = list()
        for s in sentences:
            sentences_normalized.append(normalize_sentence(s))

        return tokens, sentences_normalized

    def _predict_with_greedy_decoding(self, ds, max_tgt_seq_len=128, batch_size=16, check_spelling=True, print_progress=True):
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
    
    def _predict_with_beam_seach(self, ds, max_tgt_seq_len=128, batch_size=16, check_spelling=True, print_progress=True):

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_tokens = []
        all_sentences = []

        self.model.eval()
        n_batches = len(loader)

        with torch.no_grad():
            for i_batch, (_, X_batch, _, _, z_batch, seq_lens_batch) in enumerate(loader):

                #
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

                    # Beam search to get K candidates with decoder scores
                    best_tokens_dec, beam_tokens, beam_scores = self._run_beam_search(
                        X_single,
                        z_single,
                        seq_lens_single,
                        max_tgt_seq_len=max_tgt_seq_len,
                        beam_size=self.config.get("beam_size"),
                    )

                    # Optionally rescore with LM
                    if self.config.get("use_lm"):
                        best_score = -float("inf")
                        best_tokens = None
                        best_sentence = None

                        for tok_seq, dec_score in zip(beam_tokens, beam_scores):
                            raw_sentence = self._decode_char_sequence(tok_seq)
                            lm_score = self._score_with_lm(raw_sentence)
                            total_score = (
                                self.config["alpha_model"] * dec_score +
                                self.config["beta_lm"] * lm_score
                            )
                            if total_score > best_score:
                                best_score = total_score
                                best_tokens = tok_seq
                                best_sentence = raw_sentence

                    # Fallback: just use decoder best
                    else:
                        best_tokens = best_tokens_dec
                        best_sentence = self._decode_char_sequence(best_tokens)

                    # Optional spell check after rescoring
                    if check_spelling:
                        best_sentence = spell_check_sentence(best_sentence)

                    #
                    all_tokens.append(best_tokens)
                    all_sentences.append(best_sentence)

        #
        tokens_array = np.array(all_tokens, dtype=object)

        return tokens_array, all_sentences
    
    def score(self, ds, hypothesis=None):
        """
        Score prediction using the word error rate metric
        """

        reference = list()
        for i_sample, (_, _, _, r, _, _) in enumerate(ds):
            r_dec = self.v_chr.decode(r)
            sentence = self.v_chr.translate([r_dec])[0]
            reference.append(sentence)
        if hypothesis is None:
            tokens, hypothesis = self.predict(ds)
        score = wer(reference=reference, hypothesis=hypothesis)

        return score
    
    def generate_submission(self, ds, dst, algo="beam", check_spelling=True):
        """
        """

        tokens, sentences = self.predict(ds, algo=algo, check_spelling=check_spelling)
        df = pd.DataFrame({
            "id": list(range(len(sentences))),
            "text": sentences
        })
        df.to_csv(dst, index=False)

        return



