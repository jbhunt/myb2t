import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from myb2t.helpers import make_causal_mask
from myb2t.datasets import OpusDataset, CharacterVocabulary
import pathlib as pl
import numpy as np

# Constants
FLOAT_DTYPE = torch.float32
INTEGER_DTYPE = torch.long

class PositionalEncoding(nn.Module):
    """
    """

    def __init__(self, d_model, max_seq_len=1000, base=10000, dropout=0.0):
        """
        """

        super().__init__()
        pe = torch.zeros(max_seq_len, d_model, dtype=FLOAT_DTYPE)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

        return
    
    def forward(self, X):
        """
        """

        B, T, N = X.size()
        X_encoded = X + self.pe[:T, :].unsqueeze(0).to(X.device) # B x T x D
        X_encoded = self.dropout(X_encoded)
        return X_encoded
    
class CharacterLevelLanguageModel(nn.Module):
    """
    """

    def __init__(
        self,
        d_model,
        d_ff,
        vocab_size,
        n_heads,
        n_layers,
        dropout
        ):
        """
        """

        super().__init__()
        self.character_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )
        self.positional_encoding = PositionalEncoding(
            d_model,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.head = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
        )

        return
    
    def forward(self, X, pad_token):
        """
        """

        device = X.device
        enc_in = self.character_embedding(X)
        enc_in = self.positional_encoding(enc_in)
        src_mask = make_causal_mask(enc_in, device=device)
        src_key_padding_mask = (X == pad_token)
        enc_out = self.encoder(
            enc_in,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        logits = self.head(enc_out)

        return logits
    
# TODO: Implement early stopping to prevent overfitting the language model to this dataset
class Pretraining():
    """
    """
    
    def __init__(
        self,
        config,
        batch_size=32,
        early_stopping=True,
        validation_fraction=0.1
        ):
        """
        """

        self.config = config
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.loss_train = None
        self.loss_valid = None
        self.v_chr = CharacterVocabulary()

        #
        self.model = CharacterLevelLanguageModel(
            d_model=self.config["d_model"],
            d_ff=self.config["d_ff"],
            vocab_size=self.v_chr.size, # Check this
            n_heads=self.config["n_attn_heads"],
            n_layers=self.config["n_decoder_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)

        return
    
    def pretrain(self, ds, max_iter=10, lr=0.00005):
        """
        Pre-train with the OPUS dataset
        """

        # Dataset(s)
        if self.early_stopping:
            n_total = len(ds)
            n_valid = max(1, int(n_total * self.validation_fraction))
            n_train = n_total - n_valid
            ds_train, ds_valid = random_split(ds, [n_train, n_valid])
            loader_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=True)
        else:
            ds_train = ds
            ds_valid = None
            loader_valid = None
        loader_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)

        #
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.v_chr.PAD)
        self.loss_train = np.full(int(max_iter), np.nan)
        self.loss_valid = np.full(int(max_iter), np.nan)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        #
        for i_epoch in range(max_iter):

            #
            self.model.train()
            batch_loss_train = 0.0
            for X_batch in loader_train:

                #
                X_batch = X_batch.to(self.device).long()
                X_in = X_batch[:, :-1]
                X_out= X_batch[:, 1:]
                logits = self.model(X_in, pad_token=ds.v_chr.PAD)

                #
                B, T, V = logits.size()
                loss_train = loss_fn(
                    logits.view(B * T, V),
                    X_out.reshape(B * T)
                )
                batch_loss_train += loss_train.item()

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            #
            self.model.eval()
            batch_loss_valid = 0.0
            for X_batch in loader_valid:

                #
                X_batch = X_batch.to(self.device).long()
                X_in = X_batch[:, :-1]
                X_out= X_batch[:, 1:]
                logits = self.model(X_in, pad_token=ds.v_chr.PAD)

                #
                B, T, V = logits.size()
                loss_valid = loss_fn(
                    logits.view(B * T, V),
                    X_out.reshape(B * T)
                )
                batch_loss_valid += loss_valid.item()


            #
            batch_loss_train /= len(loader_train)
            batch_loss_valid /= len(loader_valid)
            print(f"Epoch {i_epoch + 1}: Training loss = {batch_loss_train:.6f}, Validation loss = {batch_loss_valid:.6f}")
            self.loss_train[i_epoch] = batch_loss_train
            self.loss_valid[i_epoch] = batch_loss_valid

        return
    
    def finetune(self, ds, max_iter=10, lr=0.00005):
        """
        Fine tune using the Brain-to-Text dataset
        """

        # Dataset(s)
        if self.early_stopping:
            n_total = len(ds)
            n_valid = max(1, int(n_total * self.validation_fraction))
            n_train = n_total - n_valid
            ds_train, ds_valid = random_split(ds, [n_train, n_valid])
            loader_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=True)
        else:
            ds_train = ds
            ds_valid = None
            loader_valid = None
        loader_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)

        #
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.v_chr.PAD)
        self.loss_train = np.full(int(max_iter), np.nan)
        self.loss_valid = np.full(int(max_iter), np.nan)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        #
        for i_epoch in range(max_iter):

            #
            self.model.train()
            batch_loss_train = 0.0
            for _, _, _, X_batch, _, _ in loader_train:

                #
                X_batch = X_batch.to(self.device).long()
                X_in = X_batch[:, :-1]
                X_out= X_batch[:, 1:]
                logits = self.model(X_in, pad_token=ds.v_chr.PAD)

                #
                B, T, V = logits.size()
                loss_train = loss_fn(
                    logits.view(B * T, V),
                    X_out.reshape(B * T)
                )
                batch_loss_train += loss_train.item()

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            #
            self.model.eval()
            batch_loss_valid = 0.0
            for _, _, _, X_batch, _, _ in loader_valid:

                #
                X_batch = X_batch.to(self.device).long()
                X_in = X_batch[:, :-1]
                X_out= X_batch[:, 1:]
                logits = self.model(X_in, pad_token=ds.v_chr.PAD)

                #
                B, T, V = logits.size()
                loss_valid = loss_fn(
                    logits.view(B * T, V),
                    X_out.reshape(B * T)
                )
                batch_loss_valid += loss_valid.item()


            #
            batch_loss_train /= len(loader_train)
            batch_loss_valid /= len(loader_valid)
            print(f"Epoch {i_epoch + 1}: Training loss = {batch_loss_train:.6f}, Validation loss = {batch_loss_valid:.6f}")
            self.loss_train[i_epoch] = batch_loss_train
            self.loss_valid[i_epoch] = batch_loss_valid

        return
    
    def load(self, src):
        state_dict = torch.load(src)
        self.model.load_state_dict(state_dict)
        return
    
    def save(self, dst):
        torch.save(self.model.state_dict(), dst)
        return
    
    def transfer(self, cls):
        """
        Transfer LM weights into BrainToCharacterTransformer's character decoder stack.

        Copies:
        - character_embedding
        - character_positional_encoding.pe (buffer; truncated to min length)
        - per-layer: self_attn, linear1, linear2, norm1, norm3
        - character_head

        Does NOT copy:
        - cross-attention (multihead_attn) or norm2
        - frontend, neural_activity_encoder, phoneme_head
        """

        with torch.no_grad():

            # -------------------------
            # Embedding
            # -------------------------
            if self.model.character_embedding.weight.shape != cls.model.character_embedding.weight.shape:
                raise ValueError(
                    "Embedding shape mismatch: "
                    f"pretrain {tuple(self.model.character_embedding.weight.shape)} vs "
                    f"target {tuple(cls.model.character_embedding.weight.shape)}"
                )
            cls.model.character_embedding.weight.copy_(self.model.character_embedding.weight)

            # -------------------------
            # Positional encoding buffer: (max_seq_len, d_model)
            # Copy as much as fits to avoid max_seq_len mismatch issues.
            # -------------------------
            if not (hasattr(self.model.positional_encoding, "pe") and hasattr(cls.model.character_positional_encoding, "pe")):
                raise AttributeError("Missing .pe buffer on positional encodings.")

            pe_src = self.model.positional_encoding.pe
            pe_tgt = cls.model.character_positional_encoding.pe

            if pe_src.size(1) != pe_tgt.size(1):
                raise ValueError(
                    "Positional encoding d_model mismatch: "
                    f"pretrain d_model={pe_src.size(1)} vs target d_model={pe_tgt.size(1)}"
                )

            L = min(pe_src.size(0), pe_tgt.size(0))
            pe_tgt[:L, :].copy_(pe_src[:L, :])

            # -------------------------
            # Decoder stack: copy self-attn + FFN (+ correct norms)
            # -------------------------
            pre_layers = self.model.encoder.layers                # TransformerEncoderLayer list
            dec_layers = cls.model.character_decoder.layers        # TransformerDecoderLayer list
            n_copy = min(len(pre_layers), len(dec_layers))

            for i in range(n_copy):
                enc_layer = pre_layers[i]
                dec_layer = dec_layers[i]

                # Self-attention (decoder-side masked self-attn)
                dec_layer.self_attn.load_state_dict(enc_layer.self_attn.state_dict())

                # Feed-forward
                dec_layer.linear1.load_state_dict(enc_layer.linear1.state_dict())
                dec_layer.linear2.load_state_dict(enc_layer.linear2.state_dict())

                # Norms:
                # Encoder: norm1 (self-attn), norm2 (ffn)
                # Decoder: norm1 (self-attn), norm2 (cross-attn), norm3 (ffn)
                dec_layer.norm1.load_state_dict(enc_layer.norm1.state_dict())
                dec_layer.norm3.load_state_dict(enc_layer.norm2.state_dict())

                # Intentionally do NOT copy:
                #   dec_layer.multihead_attn (cross-attn)
                #   dec_layer.norm2 (cross-attn norm)

            # -------------------------
            # Output head
            # -------------------------
            if self.model.head.weight.shape != cls.model.character_head.weight.shape:
                raise ValueError(
                    "Character head weight shape mismatch: "
                    f"pretrain {tuple(self.model.head.weight.shape)} vs "
                    f"target {tuple(cls.model.character_head.weight.shape)}"
                )
            cls.model.character_head.load_state_dict(self.model.head.state_dict())

        return
