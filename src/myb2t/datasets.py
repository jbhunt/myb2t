import numpy as np
from collections import Counter
import pathlib as pl
import h5py
from torch.utils.data import Dataset
from myb2t.vocab import PhonemeVocabulary, CharacterVocabulary
import os

class BrainToText2025(Dataset):
    """
    """

    def __init__(self, root=None, T=int(0.02 * 1000 * 1), split="train", norm=False, chr_seq_len=128):
        """
        """

        super().__init__()
        self.root = root
        self.T = T
        self.split = split
        self.norm = norm
        self._z = None
        self._seq_lens = None
        self.v_pho = PhonemeVocabulary()
        self.v_chr = CharacterVocabulary()
        self._X = None
        self._y_1 = None
        self._y_2 = None
        self._sentences = None
        self._z = None
        self._trial_indices = None
        self.chr_seq_len = chr_seq_len
    
        return
    
    def load(self, p_skip=0, padding_value=0):
        """
        """

        if self.root is None:
            raise Exception("Root directory not sepecified during instantiation")

        self._X = list()
        self._y_1 = list()
        self._y_2 = list()
        self._sentences = list()
        self._z = list()
        self._seq_lens = list()
        self._trial_indices = list()

        #
        target_files = list()
        for folder in pl.Path(self.root).iterdir():
            for file in folder.iterdir():

                #
                split = file.stem.split("data_")[1]
                if self.split == "train" and split in ["test"]:
                    continue
                if self.split == "test" and split in ["train", "val"]:
                    continue

                #
                target_files.append(file)
        target_files_sorted = sorted(
            target_files,
            key=lambda f: tuple(map(int, f.parent.name.split(".")[1:4]))
        )

        #
        trial_index = 0
        for file in target_files_sorted:

            #
            X, y_1, y_2, sentences, seq_lens = list(), list(), list(), list(), list()
            with h5py.File(file, 'r') as stream:
                for trial_key in stream.keys():

                    #
                    if p_skip == 0 or p_skip is None:
                        skip = False
                    elif p_skip == 1:
                        skip = True
                    else:
                        skip = bool(np.random.choice([0, 1], p=[1 - p_skip, p_skip]).item())
                    if skip:
                        continue

                    #
                    xi_spikes = np.array(
                        stream[trial_key]["input_features"][:self.T, 0: 256],
                        dtype=np.float16
                    )  # T time bins x N channels
                    xi_lfp = np.array(
                        stream[trial_key]["input_features"][:self.T, 256:],
                        dtype=np.float16
                    )
                    xi = np.dstack([xi_spikes, xi_lfp])
                    seq_len_1, n_features, n_modes = xi.shape
                    if seq_len_1 < self.T:
                        n_elements = self.T - seq_len_1
                        padding = np.full([n_elements, n_features, 2], padding_value, dtype=np.float16)
                        xi = np.vstack([xi, padding])
                    X.append(xi)

                    #
                    self._trial_indices.append(trial_index)
                    trial_index += 1

                    # No ground truth for the test split
                    if self.split == "test":
                        y_1.append(np.nan)
                        y_2.append(np.nan)
                        sentences.append("")
                        seq_lens.append([seq_len_1, 0, 0])
                        continue
                    
                    # Phoneme sequence
                    yi_1 = np.array(stream[trial_key]["seq_class_ids"][:], dtype=np.int16)
                    yi_1 = self.v_pho.process_raw_sequence(yi_1, tgt_seq_len=128)
                    seq_len_2 = np.delete(yi_1, yi_1 == self.v_pho.PAD).size
                    y_1.append(yi_1)

                    # Character sequence
                    yi_2 = np.array(stream[trial_key]["transcription"][:], dtype=np.int16)
                    yi_2 = self.v_chr.process_raw_sequence(yi_2, tgt_seq_len=self.chr_seq_len)
                    sentence = "".join(self.v_chr.decode(np.delete(yi_2, np.isin(yi_2, [self.v_chr.PAD, self.v_chr.BOS, self.v_chr.EOS]))))
                    sentences.append(sentence)
                    seq_len_3 = np.delete(yi_2, yi_2 == self.v_chr.PAD).size
                    y_2.append(yi_2)

                    #
                    seq_lens.append(np.array([seq_len_1, seq_len_2, seq_len_3], dtype=np.int16))

            #
            X = np.array(X)
            if X.size == 0:
                continue
            n_trials, T, n_features, n_modes = X.shape
            mean = np.nanmean(X, axis=(0, 1))
            # std = np.nanstd(X, axis=(0, 1))
            # if self.norm:
            #     X = (X - mean) / std
            z = np.tile(mean, n_trials).reshape(n_trials, -1).astype(np.float16)

            #
            for xi, yi_1, yi_2, sentence, zi, seq_len in zip(X, y_1, y_2, sentences, z, seq_lens):
                self._X.append(xi)
                self._y_1.append(yi_1)
                self._y_2.append(yi_2)
                self._sentences.append(sentence)
                self._z.append(zi)
                self._seq_lens.append(seq_len)   


        #
        self._X = np.array(self._X)
        self._z = np.array(self._z)
        self._seq_lens = np.array(self._seq_lens)
        self._trial_indices = np.array(self._trial_indices)

        if self.split == "train":
            self._y_1 = np.array(self._y_1)
            self._y_2 = np.array(self._y_2)
        else:
            self._y_1 = None
            self._y_2 = None

        return

    @property
    def X(self):
        return self._X
    
    @property
    def y_1(self):
        return self._y_1
    
    @property
    def y_2(self):
        return self._y_2
    
    @property
    def sentences(self):
        return self._sentences

    @property
    def z(self):
        return self._z
    
    @property
    def seq_lens(self):
        return self._seq_lens
    
    @property
    def trial_indices(self):
        return self._trial_indices
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        if self.split == "train":
            return self.trial_indices[index], self.X[index], self.y_1[index], self.y_2[index], self.z[index], self.seq_lens[index]
        else:
            return self.trial_indices[index], self.X[index], [], [], self.z[index], self._seq_lens[index]
        
class OpusDataset(Dataset):
    """
    """

    def __init__(self, root=None):
        """
        """

        if root is not None:
            root = pl.Path(root)
        self.root = root
        self.v_chr = CharacterVocabulary()
        self._X = None
        self._sentences = None

        return
    
    def load(self, n_seqs=100, min_seq_len=10, max_seq_len=50, tgt_seq_len=128, save_sentences=False,
             seed=42, max_tries=100000):
        """
        """

        #
        corpus = None
        file_size = None
        for file in self.root.iterdir():
            if file.stem == "corpus":
                file_size = os.path.getsize(file)
                corpus = file
                break
        if corpus is None:
            raise Exception("Could not locate corpus")
        
        #
        i_seq = 0
        rng = np.random.default_rng(seed)
        self._X = np.full([n_seqs, tgt_seq_len], 0, dtype=np.int16)
        self._sentences = list()
        n_tries = 0
        n_selected = 0
        with open(corpus, "rb") as stream:
            while n_selected < n_seqs and n_tries < max_tries:
                n_tries += 1
                byte_index = int(rng.integers(0, file_size))
                stream.seek(byte_index)
                stream.readline()
                ln = stream.readline().decode("utf-8", errors="ignore").strip()
                if not ln:
                    continue
                if len(ln) > max_seq_len or len(ln) < min_seq_len:
                    continue
                seq = self.v_chr.encode(ln, tgt_seq_len=tgt_seq_len).astype(np.uint8)
                tokens = self.v_chr.decode(seq)
                tokens = np.delete(tokens, np.isin(tokens, ["<PAD>", "<BOS>", "<EOS>"]))
                sentence = "".join(tokens)

                # Check for duplicates
                duplicate = np.any(np.all(self._X == seq, axis=1))
                if duplicate:
                    continue

                #
                self._X[i_seq, :] = seq
                if save_sentences:
                    self._sentences.append(sentence)
                i_seq += 1
                n_selected += 1

        return
    
    @property
    def X(self):
        return self._X
    
    @property
    def sentences(self):
        return self._sentences
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index]