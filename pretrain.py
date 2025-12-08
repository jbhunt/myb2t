import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from myb2t.helpers import make_causal_mask
from myb2t.datasets import OpusDataset
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
        lr=0.0001,
        max_iter=300,
        batch_size=32
        ):
        """
        """

        self.config = config
        self.max_iter = max_iter
        self.lr = lr
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.loss_train = None
        self.loss_valid = None

        #
        self.model = CharacterLevelLanguageModel(
            d_model=self.config["d_model"],
            d_ff=self.config["d_ff"],
            vocab_size=31,
            n_heads=self.config["n_attn_heads"],
            n_layers=self.config["n_decoder_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)

        return
    
    def fit(self, ds):
        """
        """

        #
        self.model = CharacterLevelLanguageModel(
            d_model=self.config["d_model"],
            d_ff=self.config["d_ff"],
            vocab_size=ds.v_chr.size,
            n_heads=self.config["n_attn_heads"],
            n_layers=self.config["n_decoder_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)

        #
        loader = DataLoader(ds, batch_size=self.batch_size)
        loss_fn = nn.CrossEntropyLoss(ignore_index=ds.v_chr.PAD)
        self.loss_train = np.full(self.max_iter, np.nan)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        #
        for i_epoch in range(self.max_iter):

            #
            batch_loss = 0.0
            for X_batch in loader:

                #
                X_batch = X_batch.to(self.device).long()
                # B, T = X_batch.size()
                # if T > self.config["max_tgt_seq_len"]:
                #     X_batch = X_batch[:, :self.config["max_tgt_seq_len"]]
                X_in = X_batch[:, :-1]
                X_out= X_batch[:, 1:]
                logits = self.model(X_in, pad_token=ds.v_chr.PAD)

                #
                B, T, V = logits.size()
                loss = loss_fn(
                    logits.view(B * T, V),
                    X_out.reshape(B * T)
                )
                batch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #
            batch_loss /= len(loader)
            print(f"Epoch {i_epoch + 1}: Training loss = {batch_loss:.6f}")
            self.loss_train[i_epoch] = batch_loss

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
        """

        with torch.no_grad():

            # embeddings and positional encoding
            cls.model.character_embedding.weight.copy_(self.model.character_embedding.weight)
            cls.model.character_positional_encoding.pe.copy_(self.model.positional_encoding.pe)

            # Encoder weights
            for i in range(len(self.model.encoder.layers)):
                enc_layer = self.model.encoder.layers[i]
                dec_layer = cls.model.character_decoder.layers[i]

                # Self-attention block
                dec_layer.self_attn.load_state_dict(enc_layer.self_attn.state_dict())

                # Feed-forward block
                dec_layer.linear1.load_state_dict(enc_layer.linear1.state_dict())
                dec_layer.linear2.load_state_dict(enc_layer.linear2.state_dict())

                # Norms that surround self-attn and FFN
                dec_layer.norm1.load_state_dict(enc_layer.norm1.state_dict())
                dec_layer.norm2.load_state_dict(enc_layer.norm2.state_dict())

            #
            cls.model.character_head.load_state_dict(self.model.head.state_dict())

        return