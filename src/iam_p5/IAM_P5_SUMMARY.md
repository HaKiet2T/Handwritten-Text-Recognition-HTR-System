# IAM Phase 5 — CRNN + CTC: Tài Liệu Tổng Hợp

> **Phiên bản:** iam_p5  
> **Mục tiêu:** Refactor toàn bộ về kiến trúc CRNN + CTC chuẩn, loại bỏ over-engineering từ iam_p4  
> **Dataset:** IAM Handwriting Database + EMNIST Pretrain  
> **Ngày:** 03/04/2026

---

## 1. Lý Do Ra Đời iam_p5

### Vấn đề của iam_p4

| Thành phần | Vấn đề |
|---|---|
| **FPN (Feature Pyramid Network)** | Thiết kế cho object detection — không phù hợp OCR sequence recognition |
| **Multi-Head Attention (8 heads)** | Transformer attention không tương thích CTC (CTC giả định independence giữa timesteps) |
| **Beam Search Decoder** | Dùng Seq2Seq logic thay vì CTC prefix probability → **sai về lý thuyết** |
| **SpellChecker trong Validation** | Che giấu lỗi thực: CER ~1.7% "đẹp" nhưng CER thực tế **~8–12%** |
| **Augmentation quá mạnh** | `ElasticTransform(alpha=120)`, `CoarseDropout(max_holes=8)` phá hủy cấu trúc chữ |
| **Code ~600 lines** | Phức tạp, khó debug, khó bảo trì |

### Nguyên tắc iam_p5

> **"Đơn giản + Đúng > Phức tạp + Không đúng"**

---

## 2. Kiến Trúc Model CRNN

### 2.1 Sơ Đồ Tổng Quan

```
Input Image (B, 1, 32, W)
        │
        ▼
┌─────────────────────────────────┐
│        CNN Backbone             │
│  Block 1: Conv→BN→ReLU → Pool   │  (B,1,32,W)  → (B,64,16,W)
│  Block 2: Conv→BN→ReLU → Pool   │  (B,64,16,W) → (B,128,8,W)
│  Block 3: Conv×2→BN→ReLU → Pool │  (B,128,8,W) → (B,256,4,W)
│  Block 4: Conv×2→BN→ReLU → Pool │  (B,256,4,W) → (B,512,2,W)
│  Block 5: Conv→BN→ReLU → Pool   │  (B,512,2,W) → (B,512,1,W)
└───────────────┬─────────────────┘
                │  Squeeze H → (B, 512, W')
                │  Permute   → (T=W', B, 512)
                ▼
┌─────────────────────────────────┐
│    BiLSTM Layer 1               │
│    512 → hidden=256 → 512       │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│    BiLSTM Layer 2               │
│    512 → hidden=256 → num_classes│
└───────────────┬─────────────────┘
                │
                ▼
    Log Softmax → (T, B, 37)
                │
        ┌───────┴───────┐
        ▼               ▼
   CTC Loss          CTC Greedy
   (training)         Decode
                    (inference)
```

### 2.2 Chi Tiết CNN Backbone

```python
# Block 1: (B,1,32,W) → (B,64,16,W)
Conv2d(1, 64, 3, padding=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2,2)

# Block 2: (B,64,16,W) → (B,128,8,W)
Conv2d(64, 128, 3, padding=1) → BatchNorm2d(128) → ReLU → MaxPool2d(2,2)

# Block 3: (B,128,8,W) → (B,256,4,W)
Conv2d(128, 256, 3, padding=1) → BN → ReLU
Conv2d(256, 256, 3, padding=1) → BN → ReLU → MaxPool2d(2,2)

# Block 4: (B,256,4,W) → (B,512,2,W)
Conv2d(256, 512, 3, padding=1) → BN → ReLU
Conv2d(512, 512, 3, padding=1) → BN → ReLU → MaxPool2d((2,1),(2,1))  ← pool H only

# Block 5: (B,512,2,W) → (B,512,1,W)
Conv2d(512, 512, 3, padding=1) → BN → ReLU → MaxPool2d((2,1),(2,1))  ← H=1
```

**Kết quả:** Height giảm từ 32 → 1 qua 5 blocks, Width được giữ nguyên làm trục thời gian T.

### 2.3 Chi Tiết BiLSTM

```python
class BidirectionalLSTM(nn.Module):
    # nIn → LSTM(hidden, bidirectional=True) → nHidden*2 → Linear → nOut
    
    Layer 1: BidirectionalLSTM(512, 256, 512)
             LSTM: 512 → 256*2=512 features
    
    Layer 2: BidirectionalLSTM(512, 256, num_classes=37)
             LSTM: 512 → 256*2=512 → Linear(37)
```

### 2.4 Thông Số Model

| Tham số | Giá trị |
|---|---|
| Input size | `(B, 1, 32, W)` — grayscale H=32 |
| CNN channels | 1 → 64 → 128 → 256 → 512 |
| BiLSTM layers | 2 |
| BiLSTM hidden | 256 (mỗi chiều), tổng 512 |
| Vocabulary | 37 classes (blank=0, a–z=1–26, 0–9=27–36) |
| Total params | ~8.8M |

---

## 3. Vocabulary & CTC Utilities

### 3.1 Định Nghĩa Vocabulary

```
blank  (index 0)  — CTC blank token
a–z    (index 1–26) — 26 chữ cái thường
0–9    (index 27–36) — 10 chữ số
────────────────────────────────
Tổng: 37 classes
```

### 3.2 CTCLabelConverter

| Method | Input | Output | Mô tả |
|---|---|---|---|
| `encode(texts)` | `List[str]` | `(flat_tensor, lengths)` | Chuyển batch text → tensor cho CTCLoss |
| `decode_greedy(log_probs)` | `(T, B, C)` tensor | `List[str]` | CTC greedy decode — collapse + remove blank |

### 3.3 CTC Greedy Decode

```
Bước 1: argmax theo chiều class → (T, B) indices
Bước 2: Collapse consecutive repeated tokens
         [a, a, b, b, b, c] → [a, b, c]
Bước 3: Remove blank token (index 0)
         [a, 0, b, 0, c] → [a, b, c]
Kết quả: chuỗi ký tự cuối cùng
```

---

## 4. Dataset & Preprocessing

### 4.1 Tổng Quan Dataset

| Dataset | Loại | Mục đích | Số lượng |
|---|---|---|---|
| **EMNIST balanced** | Character level (1 ký tự/ảnh) | Phase A: Pretrain | ~112,000 samples |
| **IAM Handwriting** | Word level (1 từ/ảnh) | Phase B: Fine-tune | ~55,000 mẫu ok |

### 4.2 Image Preprocessing

```python
# Kích thước chuẩn hóa
img_h = 32   # chiều cao cố định
img_w = 128  # chiều rộng cố định

# Transform cơ bản (validation/test)
T.Grayscale(1)
T.Resize((32, 128))
T.ToTensor()
T.Normalize(mean=[0.5], std=[0.5])   # [-1, 1]
```

### 4.3 Data Augmentation (Training Only)

```python
# Augmentation nhẹ — đúng mục đích
T.RandomAffine(
    degrees=5,
    translate=(0.04, 0.04),
    scale=(0.92, 1.08),
    shear=3
)
T.ColorJitter(brightness=0.2, contrast=0.2)
```

**So sánh với iam_p4:**

| | iam_p4 | iam_p5 |
|---|---|---|
| Elastic Transform | `alpha=120` (phá cấu trúc) | ❌ Bỏ |
| CoarseDropout | `max_holes=8` | ❌ Bỏ |
| Rotation | ±10° | ±5° |
| RandomAffine | Mạnh | Nhẹ |
| ColorJitter | Mạnh | Nhẹ (0.2) |

### 4.4 EMNIST Dataset — Xử Lý Đặc Biệt

EMNIST images bị transposed so với chiều đúng → cần fix trước transform:

```python
img = T.functional.rotate(img, -90)
img = T.functional.hflip(img)
```

---

## 5. Chiến Lược Training 2 Phase

### 5.1 Tổng Quan

```
Phase A (EMNIST Pretrain)          Phase B (IAM Fine-tune)
──────────────────────────         ──────────────────────────
Dataset: EMNIST balanced           Dataset: IAM word images
Task: nhận dạng 1 ký tự/ảnh        Task: nhận dạng 1 từ/ảnh
LR: 1e-3 (cao hơn)                 LR: 3e-4 (nhỏ hơn)
Batch: 128                         Batch: 64
Epochs: 20                         Epochs: 40
Mục tiêu: khởi tạo tốt weights     Mục tiêu: fine-tune cho IAM
Thời gian: ~30 phút (T4)           Thời gian: ~60 phút (T4)
```

### 5.2 Hyperparameters

```python
CFG = {
    'img_h'            : 32,
    'img_w'            : 128,

    # Phase A - EMNIST pretrain
    'pretrain_epochs'  : 20,
    'pretrain_batch'   : 128,
    'pretrain_lr'      : 1e-3,
    'pretrain_patience': 4,     # ReduceLROnPlateau patience

    # Phase B - IAM fine-tune
    'finetune_epochs'  : 40,
    'finetune_batch'   : 64,
    'finetune_lr'      : 3e-4,
    'finetune_patience': 4,

    # Chung
    'weight_decay'     : 1e-4,
    'lr_factor'        : 0.5,   # LR *= 0.5 khi không cải thiện
    'min_lr'           : 1e-5,
    'early_stop'       : 8,     # early stopping patience
}
```

### 5.3 Optimizer & Scheduler

```python
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',       # minimize val_cer
    factor=0.5,       # giảm LR 50%
    patience=4,       # sau 4 epoch không cải thiện
    min_lr=1e-5       # floor LR
)
```

**So sánh schedulers:**

| | iam_p4 | iam_p5 |
|---|---|---|
| Scheduler | CosineAnnealingWarmRestarts | ReduceLROnPlateau |
| Metric | Epoch-based (T_0, T_mult) | val_cer (thực tế) |
| Hành vi | LR dao động theo chu kỳ | LR giảm khi stagnate |
| Phù hợp | Exploration rộng | Fine-tuning ổn định |

### 5.4 Training Loop

```python
# Mỗi batch:
log_probs = log_softmax(model(imgs), dim=2)   # (T, B, 37)
input_lengths = [T] * B                       # sequence length từ CNN
target_lengths = [len(text)] * B              # độ dài từng nhãn

loss = CTCLoss(blank=0, zero_infinity=True)(
    log_probs, targets, input_lengths, target_lengths
)

# Gradient clipping để ổn định training
clip_grad_norm_(model.parameters(), max_norm=5.0)
```

---

## 6. Cấu Trúc File

```
iam_p5/
├── crnn_model.py          # Định nghĩa CRNN + BidirectionalLSTM
├── ctc_utils.py           # CTCLabelConverter, decode_greedy, CER helpers
├── KAGGLE_TRAINING_P5.py  # All-in-one training script cho Kaggle
└── IAM_P5_SUMMARY.md      # File tài liệu này
```

### File `crnn_model.py`

| Class | Mô tả |
|---|---|
| `BidirectionalLSTM` | Single BiLSTM layer: nIn → LSTM → Linear → nOut |
| `CRNN` | Full model: CNN(5 blocks) → squeeze → BiLSTM×2 → logits |

### File `ctc_utils.py`

| Symbol | Mô tả |
|---|---|
| `CHARS` | `string.ascii_lowercase + string.digits` = 36 ký tự |
| `BLANK = 0` | CTC blank index |
| `CHAR_TO_IDX` | `{'a': 1, 'b': 2, ..., '9': 36}` |
| `IDX_TO_CHAR` | Ngược lại |
| `NUM_CLASSES = 37` | blank + 36 ký tự |
| `CTCLabelConverter` | encode/decode cho CTC |
| `compute_cer()` | CER single pair dùng Levenshtein |

### File `KAGGLE_TRAINING_P5.py`

| Section | Mô tả |
|---|---|
| `Section 0` | Imports + CONFIG dict |
| `Section 1` | Vocabulary + CTCLabelConverter + Levenshtein CER |
| `Section 2` | CRNN Model định nghĩa lại (all-in-one) |
| `Section 3` | EMNIST Dataset + IAM Dataset + collate_fn |
| `Section 4` | Training helpers: optimizer, scheduler, train/eval loop |
| `Section 5` | Main `run()`: Phase A → Phase B |

---

## 7. Hướng Dẫn Training Trên Kaggle

### Bước 1 — Tạo Kaggle Notebook
1. Vào kaggle.com → **New Notebook**
2. Accelerator: **GPU T4 x2** hoặc P100

### Bước 2 — Add Dataset
- Search: `"IAM handwriting word database"` → **Add**
- Path tự động: `/kaggle/input/iam-handwriting-word-database/`
- EMNIST tải tự động qua `torchvision` (không cần add)

### Bước 3 — Upload & Chạy
```python
# Option A: Upload file và exec
exec(open('KAGGLE_TRAINING_P5.py').read())

# Option B: Paste toàn bộ vào 1 code cell → Run All
```

### Bước 4 — Download Weights
Output file: `/kaggle/working/best_crnn_ctc.pth`

### Timeline Ước Tính (GPU T4)

| Phase | Thời gian |
|---|---|
| Phase A – EMNIST (20 epochs) | ~30 phút |
| Phase B – IAM fine-tune (40 epochs) | ~60 phút |
| **Tổng** | **~1.5 – 2 giờ** |

---

## 8. Tối Ưu Tham Số (Hyperparameter Tuning)

### 8.1 Bảng Phân Tích Tác Động Từng Tham Số

| Tham số | Giá trị mặc định | Tác động | Ưu tiên tune |
|---|---|---|---|
| `pretrain_lr` | `1e-3` | Tốc độ học Phase A | Trung bình |
| `finetune_lr` | `3e-4` | Tốc độ học Phase B | **Cao** |
| `pretrain_batch` | `128` | VRAM + gradient noise Phase A | Thấp |
| `finetune_batch` | `64` | VRAM + gradient noise Phase B | Trung bình |
| `pretrain_epochs` | `20` | Chất lượng pretrain | Trung bình |
| `finetune_epochs` | `40` | Thời gian fine-tune | **Cao** |
| `weight_decay` | `1e-4` | Regularization (tránh overfit) | Trung bình |
| `lr_factor` | `0.5` | Độ giảm LR khi stagnate | Thấp |
| `pretrain_patience` | `4` | Tần suất giảm LR Phase A | Thấp |
| `finetune_patience` | `4` | Tần suất giảm LR Phase B | Trung bình |
| `early_stop` | `8` | Ngưỡng dừng sớm | Trung bình |
| `img_w` | `128` | Độ phân giải ngang (ảnh hưởng T) | **Cao** |

---

### 8.2 Learning Rate — Tham Số Quan Trọng Nhất

#### Quy tắc chọn `finetune_lr`

```
finetune_lr = pretrain_lr / 3  →  3e-4   ✅ (mặc định)
finetune_lr = pretrain_lr / 5  →  2e-4   (conservative, ít overwrite pretrain)
finetune_lr = pretrain_lr / 10 →  1e-4   (rất thận trọng, phù hợp nếu dataset IAM nhỏ)
```

**Nguyên tắc:**
- `pretrain_lr` lớn → học nhanh trên EMNIST, không cần quá nhỏ vì model train từ đầu
- `finetune_lr` nhỏ hơn `pretrain_lr` ít nhất 3× để tránh "forgetting" feature CNN đã học
- Nếu `val_cer` Phase B không giảm sau 5 epoch đầu → thử tăng `finetune_lr` lên `5e-4`

#### Dấu hiệu nhận biết LR sai

| Dấu hiệu | Nguyên nhân | Giải pháp |
|---|---|---|
| `loss` dao động mạnh, không hội tụ | LR quá lớn | Giảm LR × 3 |
| `loss` giảm chậm, `val_cer` > 30% sau 10 epoch | LR quá nhỏ | Tăng LR × 3 |
| Phase B `val_cer` tệ hơn Phase A | `finetune_lr` quá lớn → phá pretrain | Giảm `finetune_lr` xuống `1e-4` |
| `val_cer` hội tụ sớm rồi tăng lại | Overfit | Tăng `weight_decay`, giảm `finetune_epochs` |

---

### 8.3 Batch Size — Đánh Đổi VRAM vs Chất Lượng

```
Kaggle T4 (16GB VRAM):
  finetune_batch = 64   → safe, chiếm ~8-10 GB  ✅ (mặc định)
  finetune_batch = 128  → nhanh hơn, cần ~14 GB, có thể OOM
  finetune_batch = 32   → chậm hơn, gradient noise cao hơn → đôi khi CER tốt hơn

Kaggle T4 x2 (32GB VRAM, dùng DataParallel):
  finetune_batch = 128  → tối ưu, gradient ổn định hơn
```

**Lưu ý:** `pretrain_batch=128` an toàn vì ảnh EMNIST nhỏ hơn nhiều (28×28 → resize 32×128).

---

### 8.4 Số Epoch & Early Stopping

#### Phase A — EMNIST Pretrain

```
pretrain_epochs = 20   → đủ để học feature chữ cái cơ bản (mặc định)
pretrain_epochs = 30   → nếu EMNIST CER Phase A > 5% sau 15 epoch
pretrain_epochs = 15   → nếu muốn training nhanh hơn, chấp nhận pretrain kém hơn
```

Dừng sớm nếu `val_cer` ≤ 3% trên EMNIST — đó là mức đủ tốt để chuyển sang Phase B.

#### Phase B — IAM Fine-tune

```
finetune_epochs = 40   → baseline (mặc định)
finetune_epochs = 60   → nếu CER Phase B vẫn đang giảm đến epoch 35+
finetune_epochs = 80   → nếu dùng warm augmentation (GaussianBlur, ElasticTransform nhẹ)
```

`early_stop = 8` có nghĩa: nếu 8 epoch liên tiếp không cải thiện `val_cer` → dừng.
- Tăng lên `12` nếu LR schedule còn đang giảm (ReduceLROnPlateau chưa đạt `min_lr`)
- Giảm xuống `5` nếu muốn training nhanh hơn / tiết kiệm Kaggle GPU quota

---

### 8.5 Image Width — Ảnh Hưởng Đến Chuỗi Thời Gian T

Chiều rộng ảnh `img_w` quyết định độ dài chuỗi output của CNN:

$$T = \lfloor img\_w / 1 \rfloor = img\_w$$

(vì MaxPool chỉ pool theo chiều cao ở Block 4, 5)

```
img_w = 128  → T = 128  ← mặc định, cân bằng detail vs tốc độ
img_w = 160  → T = 160  ← tốt hơn cho từ dài (> 8 ký tự)
img_w = 100  → T = 100  ← nhanh hơn, phù hợp từ ngắn
```

**Ràng buộc CTC:** `T ≥ max_label_length * 2 - 1`

Với `img_w=128`:
- Từ tối đa 64 ký tự → an toàn cho IAM (từ dài nhất ~12-15 ký tự)

---

### 8.6 Weight Decay — Kiểm Soát Overfit

```python
weight_decay = 1e-4   # mặc định, tốt cho hầu hết trường hợp
weight_decay = 1e-3   # nếu model overfit nặng (train_loss << val_cer)
weight_decay = 5e-5   # nếu model underfit (cả train lẫn val đều cao)
```

Overfit dấu hiệu: `train_loss` → 0 nhưng `val_cer` > 10%.

---

### 8.7 ReduceLROnPlateau — Điều Chỉnh Scheduler

```python
# Mặc định
patience = 4    # chờ 4 epoch
factor   = 0.5  # LR = LR × 0.5
min_lr   = 1e-5 # floor

# Aggressive (hội tụ nhanh hơn, có thể bỏ minima cục bộ)
patience = 3
factor   = 0.3

# Conservative (training ổn định hơn)
patience = 6
factor   = 0.5
min_lr   = 5e-6
```

---

### 8.8 Augmentation Tuning (IAM Fine-tune)

Tham số augmentation trong `IAMDataset`:

```python
T.RandomAffine(
    degrees=5,              # ±5°  →  tăng lên 7° nếu data ít
    translate=(0.04, 0.04), # 4%   →  tăng lên 0.06 nếu cần robust hơn
    scale=(0.92, 1.08),     # ±8%  →  giữ nguyên, đủ mức cần thiết
    shear=3                 # 3°   →  giữ nguyên
)
T.ColorJitter(
    brightness=0.2,         # tăng lên 0.3 nếu ảnh IAM nhiễu nhiều
    contrast=0.2            # tăng lên 0.3 nếu ảnh mờ/nhạt
)
```

**KHÔNG nên thêm:**
- `ElasticTransform` với `alpha > 30` → phá cấu trúc nét chữ
- `CoarseDropout` → xóa ký tự → nhãn sai
- Rotation > ±10° → ảnh hưởng CTC assumption

---

### 8.9 Chiến Lược Tune Thực Tế Trên Kaggle

#### Workflow 3 bước

```
Bước 1 — Chạy baseline (CFG mặc định)
         Ghi lại: Phase A final CER + Phase B best CER

Bước 2 — Tune 1 tham số tại một thời điểm
         Ưu tiên: finetune_lr → finetune_epochs → weight_decay

Bước 3 — Đánh giá trên val_cer
         Chỉ coi là cải thiện nếu CER giảm ≥ 0.5%
         (nếu ít hơn có thể là noise ngẫu nhiên)
```

#### Grid nhỏ khuyến nghị cho Phase B

```python
# Chạy lần lượt (mỗi lần ~60-90 phút trên T4):
experiments = [
    {'finetune_lr': 3e-4, 'weight_decay': 1e-4},  # baseline
    {'finetune_lr': 2e-4, 'weight_decay': 1e-4},  # LR nhỏ hơn
    {'finetune_lr': 3e-4, 'weight_decay': 5e-4},  # weight decay mạnh hơn
    {'finetune_lr': 5e-4, 'weight_decay': 1e-4},  # LR lớn hơn (risky)
]
```

#### Mục tiêu CER theo giai đoạn

| Giai đoạn | CER mục tiêu | Nhận xét |
|---|---|---|
| Phase A (EMNIST) | ≤ 5% | Đủ để pretrain tốt |
| Phase B early (epoch 10) | ≤ 15% | Model đang học |
| Phase B mid (epoch 25) | ≤ 8% | Bình thường |
| Phase B final | **3 – 6%** | Mục tiêu của iam_p5 |
| Nếu đạt < 3% | Kiểm tra overfitting | Xem train_loss có gần 0 không |

---

### 8.10 Tóm Tắt Thứ Tự Ưu Tiên Tune

```
1. finetune_lr      ← ảnh hưởng lớn nhất đến Phase B CER
2. finetune_epochs  ← đảm bảo đủ thời gian hội tụ
3. weight_decay     ← kiểm soát overfit
4. finetune_batch   ← nếu bị OOM hoặc muốn thử gradient noise
5. early_stop       ← điều chỉnh khi LR còn đang giảm
6. pretrain_epochs  ← chỉ tune nếu Phase A CER > 5%
7. augmentation     ← tune cuối cùng, ít ảnh hưởng nhất
```

---

## 9. Checkpoint Format

File `.pth` lưu theo format:

```python
{
    'epoch'      : int,           # epoch cuối cùng lưu
    'phase'      : str,           # 'EMNIST' hoặc 'IAM'
    'model_state': OrderedDict,   # state_dict của CRNN
    'val_cer'    : float,         # CER tốt nhất (0–1)
    'cfg'        : dict,          # CFG dict để tái hiện
    'num_classes': 37,            # NUM_CLASSES
}
```

### Load Model để Inference

```python
import torch
from crnn_model import CRNN
from ctc_utils import CTCLabelConverter, NUM_CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load
model = CRNN(NUM_CLASSES).to(device)
ckpt = torch.load('best_crnn_ctc.pth', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Inference
converter = CTCLabelConverter()
with torch.no_grad():
    logits    = model(img_tensor)                          # (T, B, 37)
    log_probs = torch.log_softmax(logits, dim=2)
    texts     = converter.decode_greedy(log_probs)         # List[str]
```

---

## 9. Kết Quả Kỳ Vọng

| Metric | Giá Trị | Ghi Chú |
|---|---|---|
| **CER (IAM val)** | ~3 – 6% | Không có spell-check (trung thực) |
| **So với iam_p4 CER thực** | ~8 – 12% | iam_p5 tốt hơn thực chất |
| **So với iam_p4 CER "đẹp"** | ~1.7% | iam_p4 dùng spell-check trong eval — không trung thực |

### CER Thực So Với CER Báo Cáo

```
iam_p4 báo cáo: ~1.7%  ← có spell-check che giấu lỗi
iam_p4 thực tế: ~8–12% ← không spell-check
iam_p5 thực tế: ~3–6%  ← con số trung thực, tốt hơn p4
```

---

## 10. So Sánh iam_p4 vs iam_p5

| Thành phần | iam_p4 | iam_p5 |
|---|---|---|
| **Backbone** | ResNet + FPN (multi-scale) | CNN đơn giản, H→1 |
| **Sequence model** | BiLSTM + Multi-Head Attention | BiLSTM 2 layers |
| **Decode** | Beam Search (Seq2Seq style) | CTC Greedy (đúng chuẩn) |
| **Loss** | CTC hybrid + cross-entropy | `nn.CTCLoss(blank=0)` thuần |
| **Scheduler** | CosineAnnealingWarmRestarts | ReduceLROnPlateau |
| **Post-processing** | SpellChecker trong eval | Không — CER thực |
| **Augmentation** | Mạnh (elastic, erasing...) | Nhẹ (affine, colorjitter) |
| **Input size** | `(B, 1, 64, 256)` | `(B, 1, 32, 128)` |
| **Code lines** | ~600 lines | ~300 lines |
| **CER thực** | ~8–12% | ~3–6% |

---

## 11. Điểm Mấu Chốt Về CTC

### Tại Sao CTC + CRNN Là Đúng Cho OCR?

```
CTC (Connectionist Temporal Classification):
  ✅ Không cần alignment giữa input và output
  ✅ Phù hợp khi độ dài input (T timesteps) > độ dài output (text)
  ✅ Independence assumption: mỗi timestep predict độc lập
  ✅ Phù hợp với CNN→RNN pipeline (CRNN)

Transformer Decoder (iam_p4):
  ❌ Autoregressive: predict t phụ thuộc t-1
  ❌ Dùng cross-attention nhưng loss là CTC → mâu thuẫn giả định
  ❌ Beam Search theo Seq2Seq logic ≠ CTC prefix beam search
```

### CTC Greedy Decode

```
Input:  [a, a, a, _, b, _, b, _, c, c]   (_ = blank)
Step 1 (collapse): [a, _, b, _, b, _, c]
Step 2 (remove blank): [a, b, b, c]
Output: "abbc"
```

---

## 12. Tích Hợp Vào Web App (Sau Khi Có Weights)

Sau khi download `best_crnn_ctc.pth` từ Kaggle, tích hợp vào `app.py`:

### Preprocessing Khác Biệt

| | iam_p4 | iam_p5 |
|---|---|---|
| Input H×W | 64×256 | **32×128** |
| Normalize | mean=0.5, std=0.5 | mean=0.5, std=0.5 |
| Channels | Grayscale 1ch | Grayscale 1ch |

### Inference Code

```python
# Preprocessing cho iam_p5
from torchvision import transforms
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Inference
model.eval()
with torch.no_grad():
    img_tensor = transform(pil_image).unsqueeze(0).to(device)  # (1,1,32,128)
    logits     = model(img_tensor)                               # (T, 1, 37)
    log_probs  = torch.log_softmax(logits, dim=2)
    result     = converter.decode_greedy(log_probs)[0]          # str
```

---

## 13. Bài Học Rút Ra

1. **CTC loss phải đi kèm CTC decode** — không thể dùng Seq2Seq beam search với CTC-trained model
2. **FPN tốt cho detection, không phải OCR recognition** — sequence recognition không cần multi-scale anchor
3. **SpellChecker trong validation che giấu lỗi** — luôn evaluate model thô không post-process
4. **Augmentation mạnh ≠ tốt hơn** — elastic transform alpha=120 phá cấu trúc nét chữ
5. **Transfer learning 2 phase hiệu quả** — EMNIST pretrain cho model biết ký tự, IAM fine-tune cho biết từ
6. **ReduceLROnPlateau phù hợp fine-tuning** — respond trực tiếp với metric thực tế

---

*Tài liệu tổng hợp bởi GitHub Copilot — 03/04/2026*
