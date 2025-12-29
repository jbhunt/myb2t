import numpy as np
import re

class PhonemeVocabulary():
    """
    """

    PAD = 0
    BOS = 1
    EOS = 2
    map = {
        0: "<PAD>",
        1: "<BOS>",
        2: "<EOS>",
        3: "<BLANK>",
        4: "AA",
        5: "AE",
        6: "AH",
        7: "AO",
        8: "AW",
        9: 'AY',
        10: 'B',
        11: 'CH',
        12: 'D',
        13: 'DH',
        14: 'EH',
        15: 'ER',
        16: 'EY',
        17: 'F',
        18: 'G',
        19: 'HH',
        20: 'IH', 
        21: 'IY',
        22: 'JH',
        23: 'K',
        24: 'L',
        25: 'M',
        26: 'N',
        27: 'NG',
        28: 'OW',
        29: 'OY',
        30: 'P',
        31: 'R',
        32: 'S',
        33: 'SH',
        34: 'T',
        35: 'TH',
        36: 'UH',
        37: 'UW',
        38: 'V',
        39: 'W',
        40: 'Y',
        41: 'Z',
        42: 'ZH',
        43: '<SILENCE>'
    }

    def __init__(self):
        """
        """

        return
    
    def process_raw_sequence(self, in_seq, tgt_seq_len=128, padding_token=0):
        """
        """

        in_seq = np.array(in_seq)
        out_seq = np.copy(in_seq)
        mask = np.array(out_seq) != padding_token
        out_seq[mask] += 2
        i = np.where(in_seq ==  0)[0][0]
        out_seq = np.insert(out_seq, i, self.EOS)
        out_seq = np.concatenate([
            np.array([self.BOS]),
            out_seq
        ])

        #
        if tgt_seq_len is not None:
            if len(out_seq) > tgt_seq_len:
                out_seq = out_seq[:tgt_seq_len]
            if len(out_seq) < tgt_seq_len:
                out_seq = np.concatenate([
                    out_seq,
                    np.full(tgt_seq_len - len(out_seq), self.PAD)
                ])

        return out_seq
    
    def decode(self, in_seq):
        """
        """

        out_seq = list()
        for el in in_seq:
            token = self.map[el]
            out_seq.append(token)

        return np.array(out_seq)
    
    @property
    def size(self):
        return max(self.map.keys()) + 1
    
    def __len__(self):
        return len(self.map)
    
import numpy as np

class CharacterVocabulary():
    """
    Vocabulary for character-level decoding.

    Output vocabulary is restricted to:
      - special tokens: <PAD>, <BOS>, <EOS>, <UNKNOWN>
      - space: " "
      - apostrophe: "'"
      - lowercase alphabetic characters: a–z
    """

    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self):
        """
        Build mappings from raw codepoints to a compact, normalized vocabulary.
        """

        # Raw input mapping from integer codepoints to characters/tokens
        # (same as your original, but 8217 is treated as an apostrophe-like char)
        self._in_map = {
            0: "<PAD>",
            32: " ",
            33: "!",
            34: '"',
            35: "#",
            36: "$",
            37: "%",
            38: "&",
            39: "'",
            40: "(",
            41: ")",
            42: "*",
            43: "+",
            44: ",",
            45: "-",
            46: ".",
            47: "/",
            48: "0",
            49: "1",
            50: "2",
            51: "3",
            52: "4",
            53: "5",
            54: "6",
            55: "7",
            56: "8",
            57: "9",
            58: ":",
            59: ";",
            60: "<",
            61: "=",
            62: ">",
            63: "?",
            64: "@",
            65: "A",
            66: "B",
            67: "C",
            68: "D",
            69: "E",
            70: "F",
            71: "G",
            72: "H",
            73: "I",
            74: "J",
            75: "K",
            76: "L",
            77: "M",
            78: "N",
            79: "O",
            80: "P",
            81: "Q",
            82: "R",
            83: "S",
            84: "T",
            85: "U",
            86: "V",
            87: "W",
            88: "X",
            89: "Y",
            90: "Z",
            91: "[",
            92: "\\",
            93: "]",
            94: "^",
            95: "_",
            96: "`",
            97: "a",
            98: "b",
            99: "c",
            100: "d",
            101: "e",
            102: "f",
            103: "g",
            104: "h",
            105: "i",
            106: "j",
            107: "k",
            108: "l",
            109: "m",
            110: "n",
            111: "o",
            112: "p",
            113: "q",
            114: "r",
            115: "s",
            116: "t",
            117: "u",
            118: "v",
            119: "w",
            120: "x",
            121: "y",
            122: "z",
            123: "{",
            124: "|",
            125: "}",
            126: "~",
            127: "DEL",
            8217: "’",   # right single quotation mark / curly apostrophe
        }

        # Target (compact) character set:
        #   indices: 0=<PAD>, 1=<BOS>, 2=<EOS>, then <UNKNOWN>, space, apostrophe, a–z
        all_characters = [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            " ",
            "'",
        ] + [chr(c) for c in range(ord('a'), ord('z') + 1)]

        # Build maps between ids and tokens
        self._out_map = {i: ch for i, ch in enumerate(all_characters)}
        self._out_map_inv = {ch: i for i, ch in self._out_map.items()}

    def _normalize_char(self, ch):
        """
        Normalize a raw character into the restricted vocabulary
        """

        # Leave special token strings as-is
        if ch in ("<PAD>", "<BOS>", "<EOS>"):
            return ch

        # Normalize curly apostrophe
        if ch == "’":
            ch = "'"

        # Normalize uppercase letters
        if len(ch) == 1 and "A" <= ch <= "Z":
            ch = ch.lower()

        # If this normalized char isn't in the vocabulary, mark as unknown
        if ch not in self._out_map_inv:
            return "<PAD>"

        return ch

    def process_raw_sequence(self, in_seq, tgt_seq_len=128):
        """
        Map a raw sequence of integer codepoints to the compact vocabulary,
        apply normalization, and insert BOS/EOS.

        Assumes raw 0 corresponds to a PAD token in the input.
        """

        out_tokens = []

        for in_token in in_seq:
            in_token = int(in_token)

            # Look up raw character, defaulting to <PAD> if unseen
            ch = self._in_map.get(in_token, "<PAD>")

            # Normalize to target character set
            ch = self._normalize_char(ch)

            # Convert to compact vocab id
            out_token = self._out_map_inv[ch]
            out_tokens.append(out_token)

        out_seq = np.array(out_tokens, dtype=np.int64)

        # Insert EOS before the first PAD, or at the end if no PAD
        pad_positions = np.where(out_seq == self.PAD)[0]
        if pad_positions.size > 0:
            first_pad = pad_positions[0]
            out_seq = np.insert(out_seq, first_pad, self.EOS)
        else:
            out_seq = np.concatenate([out_seq, np.array([self.EOS], dtype=np.int64)])

        # Prepend BOS
        out_seq = np.concatenate([np.array([self.BOS], dtype=np.int64), out_seq])

        #
        if tgt_seq_len is not None:
            if len(out_seq) > tgt_seq_len:
                out_seq = out_seq[:tgt_seq_len]
            if len(out_seq) < tgt_seq_len:
                out_seq = np.concatenate([
                    out_seq,
                    np.full(tgt_seq_len - len(out_seq), self.PAD)
                ])

        return out_seq
    
    def encode(self, phrase, tgt_seq_len=None):
        """
        """

        #
        phrase = phrase.lower()
        phrase = phrase.replace("’", "'").replace("‘", "'").replace("´", "'").replace("`", "'")
        phrase = re.sub(r"\s+", " ", phrase)
        phrase = re.sub(r"[^a-z' ]", "", phrase)
        phrase = phrase.strip()

        # Build sequence
        out_seq = [self.BOS]
        for chr in phrase:
            normed = self._normalize_char(chr)
            token = self._out_map_inv[normed]
            out_seq.append(token)
        out_seq.append(self.EOS)
        out_seq = np.array(out_seq)

        #
        if tgt_seq_len is not None:
            if len(out_seq) > tgt_seq_len:
                out_seq = out_seq[:tgt_seq_len]
            if len(out_seq) < tgt_seq_len:
                out_seq = np.concatenate([
                    out_seq,
                    np.full(tgt_seq_len - len(out_seq), self.PAD)
                ])

        return out_seq

    def decode(self, in_seq):
        """
        Map a sequence of compact-vocab ids back to characters/tokens.
        (No stripping of BOS/EOS/PAD—handle that upstream if you want.)
        """

        out_seq = []
        for in_token in in_seq:
            in_token = int(in_token)
            ch = self._out_map[in_token]
            out_seq.append(ch)

        return np.array(out_seq, dtype=object)
    
    def translate(self, in_seqs, special_tokens=["<PAD>", "<BOS>", "<EOS>"]):
        """
        """

        sentences = list()
        for in_seq in in_seqs:
            mask = np.isin(in_seq, special_tokens)
            filtered = in_seq[~mask]
            sentence = "".join(filtered)
            sentences.append(sentence)

        return sentences

    @property
    def size(self):
        return len(self._out_map)

    def __len__(self):
        return len(self._out_map)
