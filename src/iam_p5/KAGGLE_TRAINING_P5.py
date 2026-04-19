# ===========================================================================
# KAGGLE_TRAINING_P5.py — Fixed for nibinv23 Dataset
# ===========================================================================
import os, sys, string, time, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as tvdatasets

# ---------------------------------------------------------------------------
# CONFIG — Đã chỉnh lại iam_root cho khớp với dataset nibinv23
# ---------------------------------------------------------------------------
CFG = {
    'iam_root'   : '/kaggle/input/datasets/nibinv23/iam-handwriting-word-database/iam_words',
    'emnist_root': '/kaggle/working/emnist_data',
    'img_h'      : 32,
    'img_w'      : 128,

    'pretrain_epochs'   : 20,
    'pretrain_batch'    : 128,
    'pretrain_lr'       : 1e-3,
    'pretrain_patience' : 4,

    'finetune_epochs'   : 40,
    'finetune_batch'    : 64,
    'finetune_lr'       : 3e-4,
    'finetune_patience' : 4,

    'weight_decay': 1e-4,
    'lr_factor'   : 0.5,
    'min_lr'      : 1e-5,
    'early_stop'  : 8,
    'save_path'   : '/kaggle/working/best_crnn_ctc.pth',
    'seed'        : 42,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])

# ===========================================================================
# 1. VOCABULARY & CTC UTILS
# ===========================================================================
CHARS       = string.ascii_lowercase + string.digits
BLANK       = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c  for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

class CTCLabelConverter:
    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, lengths = [], []
        for text in texts:
            valid = [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]
            encoded.extend(valid)
            lengths.append(len(valid))
        return torch.LongTensor(encoded), torch.LongTensor(lengths)

    def decode_greedy(self, log_probs: torch.Tensor) -> List[str]:
        indices = log_probs.argmax(dim=2)
        results = []
        for b in range(indices.size(1)):
            seq = indices[:, b].tolist()
            collapsed = [seq[0]] if seq else []
            for tok in seq[1:]:
                if tok != collapsed[-1]:
                    collapsed.append(tok)
            results.append(''.join(IDX_TO_CHAR[t] for t in collapsed if t != BLANK))
        return results

def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2): return _levenshtein(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
        prev = curr
    return prev[len(s2)]

def batch_cer(preds: List[str], gts: List[str]) -> float:
    total_d, total_l = 0, 0
    for p, g in zip(preds, gts):
        if len(g) == 0: continue
        total_d += _levenshtein(p.lower(), g.lower())
        total_l += len(g)
    return total_d / max(total_l, 1)

# ===========================================================================
# 2. MODEL
# ===========================================================================
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn    = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.linear = nn.Linear(nHidden * 2, nOut)
    def forward(self, x):
        rec, _ = self.rnn(x)
        T, B, H = rec.size()
        return self.linear(rec.view(T * B, H)).view(T, B, -1)

class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 512),
            BidirectionalLSTM(512, 256, num_classes),
        )
    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2).permute(2, 0, 1)
        return self.rnn(feat)

# ===========================================================================
# 3. DATASETS
# ===========================================================================
_EMNIST_MAP = list('0123456789') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')

class EMNISTCharDataset(Dataset):
    def __init__(self, root: str, train: bool, img_h: int, img_w: int):
        self.transform = T.Compose([T.Resize((img_h, img_w)), T.ToTensor(), T.Normalize([0.5], [0.5])])
        raw = tvdatasets.EMNIST(root=root, split='balanced', train=train, download=True)
        self.samples = []
        for img, label in raw:
            char = _EMNIST_MAP[label].lower()
            if char in CHAR_TO_IDX:
                self.samples.append((img, char))
        print(f"EMNIST {'train' if train else 'val'}: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, char = self.samples[idx]
        img = T.functional.rotate(img, -90)
        img = T.functional.hflip(img)
        return self.transform(img.convert('L')), char

# --------------------------------------------------------------------------
# 3B. IAM Dataset
# --------------------------------------------------------------------------
def load_iam_samples(data_root: str):
    words_txt = os.path.join(data_root, 'words.txt')
    if not os.path.exists(words_txt): return []
    
    samples = []
    with open(words_txt, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.strip().split()
            if len(parts) < 9 or parts[1] != 'ok': continue
            
            label = parts[-1].lower()
            if not all(c in CHAR_TO_IDX for c in label) or not label: continue
            
            word_id = parts[0]
            pid = word_id.split('-')
            # Cấu trúc: root/words/a01/a01-000/a01-000-00-00.png
            img_path = os.path.join(data_root, 'words', pid[0], f"{pid[0]}-{pid[1]}", f"{word_id}.png")
            
            if os.path.exists(img_path):
                samples.append((img_path, label))
    
    print(f"IAM: loaded {len(samples)} valid samples")
    return samples

class IAMDataset(Dataset):
    def __init__(self, samples, img_h, img_w, augment=False):
        self.samples = samples
        trans = [T.Grayscale(1), T.Resize((img_h, img_w))]
        if augment:
            trans += [T.RandomAffine(5, (0.04, 0.04), (0.92, 1.08), 3), T.ColorJitter(0.2, 0.2)]
        trans += [T.ToTensor(), T.Normalize([0.5], [0.5])]
        self.transform = T.Compose(trans)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('L')
            return self.transform(img), label
        except Exception:
            # Nếu ảnh lỗi, lấy đại 1 ảnh khác
            return self.__getitem__(random.randint(0, len(self.samples)-1))

def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), list(labels)

# ===========================================================================
# 4. TRAINING HELPERS
# ===========================================================================
def train_one_epoch(model, loader, optimizer, criterion, converter, device):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        targets, t_lengths = converter.encode(labels)
        log_probs = torch.log_softmax(model(imgs), dim=2)
        T, B, _ = log_probs.size()
        loss = criterion(log_probs, targets.to(device), torch.full((B,), T, dtype=torch.long, device=device), t_lengths.to(device))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, converter, device):
    model.eval()
    all_p, all_g = [], []
    for imgs, labels in loader:
        log_probs = torch.log_softmax(model(imgs.to(device)), dim=2)
        all_p.extend(converter.decode_greedy(log_probs))
        all_g.extend(labels)
    return batch_cer(all_p, all_g)

def run():
    model = CRNN(NUM_CLASSES).to(device)
    converter = CTCLabelConverter()
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    # --- PHASE A ---
    print("\n--- PHASE A: EMNIST PRETRAIN ---")
    ds_train = EMNISTCharDataset(CFG['emnist_root'], True, CFG['img_h'], CFG['img_w'])
    ds_val = EMNISTCharDataset(CFG['emnist_root'], False, CFG['img_h'], CFG['img_w'])
    train_loader = DataLoader(ds_train, CFG['pretrain_batch'], True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(ds_val, CFG['pretrain_batch'], False, collate_fn=collate_fn, num_workers=2)

    opt = optim.AdamW(model.parameters(), lr=CFG['pretrain_lr'], weight_decay=CFG['weight_decay'])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=CFG['lr_factor'], patience=CFG['pretrain_patience'])

    best_cer = 1.0
    for epoch in range(1, CFG['pretrain_epochs'] + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, criterion, converter, device)
        cer = evaluate(model, val_loader, converter, device)
        sched.step(cer)
        print(f"[EMNIST] Epoch {epoch:2d} loss={loss:.4f} cer={cer*100:.2f}% time={time.time()-t0:.0f}s")
        if cer < best_cer:
            best_cer = cer
            torch.save({'model_state': model.state_dict(), 'val_cer': cer}, CFG['save_path'])

    # --- PHASE B ---
    print("\n--- PHASE B: IAM FINE-TUNE ---")
    iam_samples = load_iam_samples(CFG['iam_root'])
    if not iam_samples: raise RuntimeError(f"IAM Dataset not found at {CFG['iam_root']}")
    
    random.shuffle(iam_samples)
    split = int(len(iam_samples) * 0.7)
    train_ds = IAMDataset(iam_samples[:split], CFG['img_h'], CFG['img_w'], augment=True)
    val_ds = IAMDataset(iam_samples[split:], CFG['img_h'], CFG['img_w'], augment=False)
    
    train_loader = DataLoader(train_ds, CFG['finetune_batch'], True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, CFG['finetune_batch'], False, collate_fn=collate_fn, num_workers=2)

    model.load_state_dict(torch.load(CFG['save_path'])['model_state'])
    opt = optim.AdamW(model.parameters(), lr=CFG['finetune_lr'], weight_decay=CFG['weight_decay'])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=CFG['lr_factor'], patience=CFG['finetune_patience'])

    best_cer = 1.0
    for epoch in range(1, CFG['finetune_epochs'] + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, criterion, converter, device)
        cer = evaluate(model, val_loader, converter, device)
        sched.step(cer)
        print(f"[IAM] Epoch {epoch:2d} loss={loss:.4f} cer={cer*100:.2f}% lr={opt.param_groups[0]['lr']:.2e}")
        if cer < best_cer:
            best_cer = cer
            torch.save({'model_state': model.state_dict(), 'val_cer': cer}, CFG['save_path'])
            print(f" ✅ Saved Best IAM Model (CER: {cer*100:.2f}%)")

if __name__ == '__main__':
    run()