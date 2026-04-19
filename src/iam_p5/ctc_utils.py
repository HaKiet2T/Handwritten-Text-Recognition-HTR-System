"""
CTC Utilities
- CTCLabelConverter: encode text → tensor, decode logits → text
- blank_index = 0 (consistent with nn.CTCLoss(blank=0))
- CER calculation via Levenshtein distance
"""
import string
from typing import List, Tuple

import torch

# Vocabulary: blank(0) + a-z(1-26) + 0-9(27-36)
CHARS = string.ascii_lowercase + string.digits   # 36 chars
BLANK = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR  = {i + 1: c  for i, c in enumerate(CHARS)}
NUM_CLASSES  = len(CHARS) + 1   # 37 total (blank + 36)


class CTCLabelConverter:
    """Encode/decode labels for CTC training."""

    def __init__(self):
        self.char_to_idx = CHAR_TO_IDX
        self.idx_to_char = IDX_TO_CHAR
        self.num_classes = NUM_CLASSES

    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of strings to flat tensor + lengths.
        Required format for nn.CTCLoss.

        Returns:
            targets : (sum_of_lengths,)  LongTensor — flat encoded chars
            lengths : (B,)               LongTensor — length of each text
        """
        encoded = []
        lengths  = []
        for text in texts:
            text = text.lower()
            # Keep only chars in vocab
            valid = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
            encoded.extend(valid)
            lengths.append(len(valid))
        return torch.LongTensor(encoded), torch.LongTensor(lengths)

    def decode_greedy(self, log_probs: torch.Tensor) -> List[str]:
        """
        CTC greedy decode from log_probs.

        Args:
            log_probs: (T, B, num_classes)  log-softmax probabilities
        Returns:
            List[str] of length B
        """
        indices = log_probs.argmax(dim=2)  # (T, B)
        results = []
        for b in range(indices.size(1)):
            seq = indices[:, b].tolist()
            # Step 1: collapse consecutive repeated tokens
            collapsed = [seq[0]] if seq else []
            for tok in seq[1:]:
                if tok != collapsed[-1]:
                    collapsed.append(tok)
            # Step 2: remove blank token (index 0)
            chars = [self.idx_to_char[tok] for tok in collapsed if tok != BLANK]
            results.append(''.join(chars))
        return results


# ---------------------------------------------------------------------------
# CER helpers
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1,
                            curr[j]     + 1,
                            prev[j]     + (c1 != c2)))
        prev = curr
    return prev[len(s2)]


def compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate for a single pair."""
    gt = gt.lower()
    pred = pred.lower()
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _levenshtein(pred, gt) / len(gt)


def batch_cer(preds: List[str], gts: List[str]) -> float:
    """
    Aggregate CER over a batch.
    Formula: sum(edit_dist) / sum(len(gt))
    """
    total_dist = 0
    total_len  = 0
    for p, g in zip(preds, gts):
        g = g.lower()
        p = p.lower()
        if len(g) == 0:
            continue
        total_dist += _levenshtein(p, g)
        total_len  += len(g)
    return total_dist / max(total_len, 1)
