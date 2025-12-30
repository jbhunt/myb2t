from torch import nn
from torch.nn import functional as F
import torch
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from myb2t.vocab import PhonemeVocabulary, CharacterVocabulary
from myb2t.helpers import *
import pathlib as pl
import copy
from jiwer import wer, cer
import pandas as pd
from myb2t.ranking import BeamSearchMixin, GreedyDecodingMixin

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
   
class BrainToTextDecoder(BeamSearchMixin, GreedyDecodingMixin):
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

        super().__init__()

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
        if self.config is None or len(self.config) == 0:
            self.model = None
        else:
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

        return
    
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

        # Restore config if present
        old_config = checkpoint.get("config")
        if self.config is None:
            self.config = {}
        self.config.update(old_config)

        #
        if self.model is None:
            self.model = BrainToCharacterTransformer(
                d_model=self.config["d_model"],
                dim_ff=self.config["d_ff"],
                d_session=self.config["d_session"],
                n_encoder_layers=self.config["n_encoder_layers"],
                n_decoder_layers=self.config["n_decoder_layers"],
                n_heads=self.config["n_attn_heads"],
                dropout=self.config["dropout"],
                phoneme_vocab_size=self.v_pho.size,
                character_vocab_size=self.v_chr.size,
            ).to(self.device)
        self.model.load_state_dict(state_dict)

        # Decide device and move model there
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Keep a reference to best_state_dict if you want
        self.best_state_dict = state_dict

        return checkpoint
    
    def save(self, filepath, state_dict_gpu=None):
        """
        Save a CPU-portable checkpoint (parameters and config dict).
        """

        filepath = pl.Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prefer the tracked best_state_dict, fall back to current model weights
        if state_dict_gpu is None:
            if self.best_state_dict is not None:
                state_dict_gpu = self.best_state_dict
            else:
                state_dict_gpu = self.model.state_dict()

        # Force everything to CPU for maximum portability
        state_dict_cpu = {k: v.detach().cpu() for k, v in state_dict_gpu.items()}

        checkpoint = {
            "best_state_dict": state_dict_cpu,
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
        lambda_ = self.config.get("lambda")
        loss = lambda_ * loss_chr + (1 - lambda_) * loss_pho

        return loss, loss_chr, loss_pho
    
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
            loader_valid = DataLoader(ds_valid, batch_size=self.config.get("batch_size"), shuffle=False)
        else:
            ds_train = ds
            ds_valid = None
            loader_valid = None
        loader_train = DataLoader(ds_train, batch_size=self.config.get("batch_size"), shuffle=True)

        # Loss functions
        chr_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.v_chr.PAD,
            label_smoothing=0.1
        )
        pho_loss_fn = nn.CTCLoss(
            blank=self.v_pho.PAD,
            zero_infinity=True,
        )

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])

        # Init LR scheduler
        n_steps_total = self.config["max_iter"] * len(loader_train)
        n_steps_warmup = max(1, min(1500, int(0.05 * n_steps_total)))
        n_steps_hold = int(0.15 * n_steps_total)
        def lr_lambda(step):
            # Warmup
            if step < n_steps_warmup:
                return (step + 1) / n_steps_warmup

            # Hold at max LR
            if step < n_steps_warmup + n_steps_hold:
                return 1.0

            # Cosine decay
            decay_steps = n_steps_total - n_steps_warmup - n_steps_hold
            progress = (step - n_steps_warmup - n_steps_hold) / max(1, decay_steps)
            progress = min(progress, 1.0)

            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return 0.1 + 0.9 * cosine   # floor at 10% of base LR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Loss tracking
        self.loss_train = np.full(self.config.get("max_iter"), np.nan)
        self.loss_valid = np.full(self.config.get("max_iter"), np.nan)
        self.wer_valid = np.full(self.config.get("max_iter"), np.nan)

        #
        n_epochs_without_improvement = 0
        self.best_state_dict = None
        best_wer_valid = np.inf

        #
        snapshot_index = 0
        i_step = 0 # Counter for steps (NOT EPOCHS)

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                #
                i_step += 1

            #
            batch_loss_train /= len(loader_train)

            # Evaluate loss on the validation dataset
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

            # Estimate WER on a downsampled subset of the validation dataset
            if ds_valid is not None:
                _, hypothesis = self.predict(ds_valid, algo="greedy", print_progress=False)
                reference = [ds_valid.dataset.sentences[i] for i in ds_valid.indices]
                wer_valid = wer(reference, hypothesis)
                cer_valid = cer(reference, hypothesis)
            else:
                wer_valid = np.nan
                cer_valid = np.nan
            
            # Print out progress report
            self.loss_train[i_epoch] = batch_loss_train
            self.loss_valid[i_epoch] = batch_loss_valid
            self.wer_valid[i_epoch] = wer_valid
            current_lr = optimizer.param_groups[0]["lr"]
            if i_step < n_steps_warmup:
                phase = "Warmup"
            elif i_step < n_steps_warmup + n_steps_hold:
                phase = "Hold"
            else:
                phase = "Cosine decay"
            print(f'Epoch {i_epoch + 1} out of {self.config.get("max_iter")} [{phase}]: Learning rate={current_lr:.9f}, Loss (train)={batch_loss_train:0.3f}, Loss (Validation)={batch_loss_valid:0.3f}, WER={wer_valid:.3f}, CER={cer_valid:.3f}, N epochs without improvement={n_epochs_without_improvement}/{self.config["tolerance"]}')

            # Update best state dict using the WER
            if wer_valid < best_wer_valid:
                self.best_epoch = i_epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                best_wer_valid = wer_valid
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

            # Save snapshot (optional)
            if self.out_dir is not None:
                if ((i_epoch + 1) % 1) == 0:
                    dst = self.out_dir.joinpath("snapshots", f"snapshot-{snapshot_index + 1}.pkl")
                    self.save(dst, state_dict_gpu=self.model.state_dict())
                    snapshot_index += 1

            # Check for early stopping condition
            if n_epochs_without_improvement >= self.config.get("tolerance"):
                print(f"Early stopping condition met: WER has not improved in {self.config.get('tolerance')} epochs")
                break

        # Save the best snapshot
        if self.out_dir is not None:
            dst = self.out_dir.joinpath("snapshots", f"snapshot-best.pkl")
            self.save(dst)

        return
    
    def predict(self, ds, algo="beam", max_tgt_seq_len=128, batch_size=16, check_spelling=False, print_progress=True):
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
    
    def generate_submission(self, ds, dst, algo="beam", check_spelling=False):
        """
        """

        tokens, sentences = self.predict(ds, algo=algo, check_spelling=check_spelling)
        df = pd.DataFrame({
            "id": list(range(len(sentences))),
            "text": sentences
        })
        df.to_csv(dst, index=False)

        return


