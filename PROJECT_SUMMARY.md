# 📋 PROJECT SUMMARY — Nhận Diện Chữ Viết Tay (HTR)

> **Tên dự án:** Handwritten Text Recognition Web Application  
> **Dataset:** IAM Handwriting Database  
> **Ngày tổng hợp:** 30/03/2026

---

## 1. Tổng Quan Dự Án

Hệ thống nhận diện chữ viết tay (Handwritten Text Recognition — HTR) dưới dạng **Web Application**, cho phép người dùng vẽ/upload ảnh chữ viết tay và nhận kết quả nhận dạng văn bản tức thì. Dự án phát triển qua **5 phiên bản model (iam_p1 → iam_p5)**, tập trung cải thiện độ chính xác, kiến trúc, và triển khai thực tế.

---

## 2. Môi Trường & Ngôn Ngữ

### 🖥️ Môi Trường Phát Triển

| Thành phần | Chi tiết |
|---|---|
| **Hệ điều hành** | Windows (local dev) |
| **Training** | Kaggle Notebooks (GPU T4/P100) |
| **Python** | 3.x (virtual env `.venv`) |
| **CUDA / Device** | CUDA nếu có GPU, fallback CPU tự động |

### 📦 Ngôn Ngữ & Thư Viện Chính

**Backend (Python):**
- `PyTorch` — deep learning framework chính
- `Flask` + `flask-cors` + `flask-limiter` — REST API web server
- `OpenCV` (`cv2`) — xử lý ảnh
- `Pillow` (PIL) — load/transform ảnh
- `NumPy` — tính toán số học
- `pyspellchecker` — spell checking

**Frontend (Web):**
- `HTML5` + `CSS3` + `Vanilla JavaScript`
- `Canvas API` — bảng vẽ chữ tay trực tiếp trên browser
- Không dùng framework JS (React/Vue...)

---

## 3. Dataset

| Thông tin | Chi tiết |
|---|---|
| **Tên dataset** | IAM Handwriting Database |
| **Loại dữ liệu** | Ảnh chữ viết tay từng từ (word-level) |
| **Tập gốc** | ~78,000 samples |
| **Bổ sung (Phase 1)** | +25,000 EMNIST/MNIST digits & letters |
| **Synthetic digits** | +10,000 ảnh chữ số tổng hợp |
| **Tổng sau mở rộng** | ~113,000 samples (+36%) |
| **Vocabulary** | 39 ký tự: `a-z` + `0-9` + `<PAD>` `<SOS>` `<EOS>` |

---

## 4. Kiến Trúc Model Qua Các Phiên Bản

### 4.1 Tiến Trình Phát Triển

```
iam_p1  ──►  iam_p2  ──►  iam_p3  ──►  iam_p4  ──►  iam_p5
(Baseline)  (ResNet)   (Larger)   (Full Phases) (Refactor/CRNN)
CRNN cơ bản  d=384     d=384,6+4   Transformer    CRNN+CTC thuần
```

---

### 4.2 Model Production Hiện Tại — `iam_p4` (Transformer Encoder-Decoder)

**Tên file:** `iam_p4/best_encoder_decoder.pth`  
**Được load bởi:** `app.py` → `src/models/handwriting_model.py`

#### Kiến Trúc `EncoderDecoderHTR`

```
Input (B, 1, 64, 256)
   │
   ▼
┌─────────────────────────────────────────────┐
│  CNN Backbone — ResNetBackbone (d_model=384) │
│  Conv → BN → ReLU → MaxPool                │
│  Layer1 (64)  → Layer2 (128)               │
│  Layer3 (256) → Layer4 (384)               │
│  Output: (B, 384, 4, 16)                   │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
         2D Positional Encoding
                    │
          Flatten → (B, 64, 384)
                    │
                    ▼
┌──────────────────────────────────┐
│  Transformer Encoder (6 layers)  │
│  d_model=384, nhead=8            │
│  FFN dim=1536, GELU, Pre-LN      │
└───────────────┬──────────────────┘
                │  memory
                ▼
┌──────────────────────────────────┐
│  Transformer Decoder (4 layers)  │
│  Autoregressive: <SOS> → tokens  │
│  Cross-attention ↔ encoder memory│
└───────────────┬──────────────────┘
                │
                ▼
       Linear → vocab_size (39)
                │
                ▼
   Greedy / Beam Search Decode
```

**Tham số mô hình:**

| Tham số | Giá trị |
|---|---|
| `d_model` | 384 |
| `enc_layers` | 6 |
| `dec_layers` | 4 |
| `nhead` | 8 |
| `ffn_dim` | 1536 |
| `dropout` | 0.2 |
| `max_seq_len` | 50 |
| `vocab_size` | 39 |

---

### 4.3 Model Refactored — `iam_p5` (CRNN + CTC)

Phiên bản refactor hoàn toàn nhằm **loại bỏ over-engineering** và trở về kiến trúc chuẩn:

```
Input (B, 1, 32, W)
   │
   ▼
CNN (5 blocks, H: 32 → 1)
  Block1: Conv→BN→ReLU→Pool2d    (→ H=16)
  Block2: Conv→BN→ReLU→Pool2d    (→ H=8)
  Block3: Conv×2→BN→ReLU→Pool2d  (→ H=4)
  Block4: Conv×2→BN→ReLU→Pool(H) (→ H=2)
  Block5: Conv→BN→ReLU→Pool(H)   (→ H=1)
   │
   ▼
Squeeze H → (B, 512, W')
Permute  → (T=W', B, 512)
   │
   ▼
BiLSTM Layer 1: 512 → 256 → 512
BiLSTM Layer 2: 512 → 256 → num_classes
   │
   ▼
CTC Loss (train) / CTC Greedy Decode (infer)
```

---

## 5. Thuật Toán & Kỹ Thuật Sử Dụng

### 5.1 Thuật Toán Core

| Thuật toán | Phiên bản | Mục đích |
|---|---|---|
| **Transformer Encoder-Decoder** | iam_p4 | Nhận dạng chuỗi ký tự từ ảnh |
| **ResNet Backbone** (BasicBlock) | iam_p4 | Trích xuất CNN features |
| **2D Positional Encoding** | iam_p4 | Mã hóa vị trí không gian ảnh |
| **Multi-Head Attention** (8 heads) | iam_p4 | Self-attention + Cross-attention |
| **Beam Search Decoding** (width=5) | iam_p4 | Tìm chuỗi tối ưu thay vì greedy |
| **CTC Loss + CTC Greedy Decode** | iam_p5 | Alignment-free sequence training |
| **BiLSTM** (2 layers) | iam_p5 | Mô hình chuỗi thời gian |

### 5.2 Kỹ Thuật Training

| Kỹ thuật | Mô tả |
|---|---|
| **Pre-LN Transformer** (`norm_first=True`) | Ổn định training hơn Post-LN |
| **GELU Activation** | Thay ReLU trong Transformer layers |
| **TrOCR-style Weight Init** | `trunc_normal_(std=0.02)` cho Linear/Embedding, `kaiming_normal_` cho Conv |
| **AdamW Optimizer** | Weight decay để regularize |
| **CosineAnnealingWarmRestarts** (p4) | Giảm LR theo chu kỳ |
| **ReduceLROnPlateau** (p5) | Giảm LR khi val_cer không cải thiện |
| **Dynamic Class Weights** | Cân bằng học theo tần suất ký tự |
| **Confusion Multipliers** | Tăng trọng số cặp dễ nhầm (0/O, 1/l/I, 8/B...) |

### 5.3 Augmentation

**Phase 1-4 (mạnh):**
- Random Erasing / CutOut — giả lập che khuất
- Elastic Distortion — biến dạng nét bút
- Grid Distortion — cong giấy
- Ink Bleeding — loang mực
- Shadow/Lighting — artifact từ scan
- Rotation, Shear, Scale, Blur, Noise

**Phase 5 (nhẹ, đúng mục đích):**
```python
RandomAffine(degrees=5, translate=(0.04,0.04), scale=(0.92,1.08), shear=3)
ColorJitter(brightness=0.2, contrast=0.2)
```

### 5.4 Post-Processing (Phase 3)

| Component | Mô tả |
|---|---|
| **OCR Spell Checker** | Sửa lỗi confusion phổ biến (0↔O, 1↔l/I, 8↔B, 5↔S...) |
| **N-gram Language Model** (trigram) | Rescore dựa trên xác suất chuỗi ký tự |
| **Context-Aware Corrector** | Rule-based: số xung quanh → 0, chữ xung quanh → l |

---

## 6. Thay Đổi & Cải Thiện Qua Các Phases

### Phase 1 — Quick Wins (19/12/2025)
**Mục tiêu:** CER 4.2% → 2.5%

| Thay đổi | Trước | Sau | Cải thiện |
|---|---|---|---|
| Digit samples (EMNIST) | 2,500 | 15,000 | **+6×** |
| Synthetic digits | 0 | 10,000 | **+10k** |
| Tổng dataset | 83,000 | 113,000 | **+36%** |
| Digit accuracy | 75–80% | 88–92% | **+15%** |
| CER | 4.20% | 2.50% | **−40%** |
| WER | 15–20% | 8–12% | **−40%** |

**Kỹ thuật thêm:**
- Dynamic class weights (data-driven thay vì hard-coded)
- Confusion multipliers: `'0': 2.0×`, `'1': 1.5×`, `'5': 1.8×`, `'8': 2.0×`
- Advanced augmentation (elastic, grid distortion, ink bleed...)

---

### Phase 2 — Architecture (19/12/2025)
**Mục tiêu:** CER 2.5% → 2.0%

| Thay đổi | Mô tả | CER giảm |
|---|---|---|
| **FPN (Feature Pyramid Network)** | Multi-scale features (P2, P3, P4 fused) | −0.3% → −0.6% |
| **Refined Attention** | Position-aware bias + Pre-LN residual | −0.1% → −0.3% |
| **Beam Search** (width=5) | Thay greedy decode | −0.2% → −0.5% |
| **Model Ensemble** | Average logits của 3 models | −0.3% → −0.6% |

---

### Phase 3 — Post-Processing (19/12/2025)
**Mục tiêu:** CER 2.0% → 1.7% *(không cần retrain)*

| Component | Cải thiện CER | Cải thiện WER |
|---|---|---|
| OCR Spell Checker | −0.5% → −1.0% | −2% → −5% |
| N-gram Language Model | −0.1% → −0.3% | −0.5% → −1.0% |
| Context-Aware Correction | Rule-based, instant | - |

---

### Phase 4 — Deployment Optimization
**Mục tiêu:** Production-ready

| Kỹ thuật | Trước | Sau | Thay đổi |
|---|---|---|---|
| **INT8 Quantization** | FP32, 140 MB | INT8, 35 MB | **−75% size, +100% speed** |
| **ONNX Export** | PyTorch only | Cross-platform | Windows/Linux/macOS |
| **FastAPI Service** | Flask dev | REST API production | - |
| **Memory** | 600 MB | 200 MB | **−67%** |
| Character Accuracy | 95.8% | 95.5% | −0.3% (chấp nhận) |

---

### Phase 5 — Refactor (iam_p5)
**Quyết định:** Quay về CRNN + CTC thuần, loại bỏ over-engineering

#### ❌ Những gì bị xóa và lý do

| Thành phần | Lý do xóa |
|---|---|
| **FPN (Feature Pyramid Network)** | FPN dùng cho object detection, không phù hợp OCR sequence-to-sequence |
| **Multi-Head Attention (8 heads)** | Transformer attention không match CTC decode (CTC giả định independence giữa timesteps) |
| **Beam Search Decoder** | Implementation cũ dùng Seq2Seq logic thay vì CTC prefix probability → sai về lý thuyết |
| **SpellChecker trong Validation** | Che giấu lỗi thực, CER "đẹp" nhưng không trung thực (~1.7% là ảo) |
| **Augmentation quá mạnh** | `ElasticTransform(alpha=120)`, `CoarseDropout(max_holes=8)` phá hủy cấu trúc chữ |

#### ✅ Những gì cải thiện trong iam_p5

| Thành phần | Cũ (iam_p4) | Mới (iam_p5) |
|---|---|---|
| Backbone | ResNet + FPN | CNN đơn giản (H→1) |
| Sequence | BiLSTM + Multi-Head Attention | BiLSTM 2 layer (chuẩn CRNN) |
| Decode | Beam Search (Seq2Seq style) | CTC Greedy (đúng chuẩn) |
| Loss | CTC hybrid + cross-entropy | `nn.CTCLoss(blank=0, zero_infinity=True)` |
| Scheduler | CosineAnnealingWarmRestarts | ReduceLROnPlateau (theo val_cer) |
| Post-proc | SpellChecker trong eval | Không có — CER thực |
| Code | ~600 lines (phức tạp) | ~300 lines (modular) |

#### CER Thực so với CER Báo Cáo

| Phiên bản | CER báo cáo | CER thực (không spell-check) |
|---|---|---|
| iam_p4 | **~1.7%** (có spell-check) | **~8–12%** (thực tế) |
| iam_p5 | — | **~5–8%** (trung thực) |

> **Kết luận:** iam_p5 trông "CER cao hơn" nhưng là con số **đáng tin cậy** để đánh giá model thực sự.

---

## 7. Cấu Trúc Dự Án

```
WEB_AI/
├── app.py                    # Flask web server chính
├── index.html                # Frontend - Canvas vẽ chữ
├── requirements.txt          # Dependencies
├── static/
│   ├── main.js               # Frontend logic (Canvas API, API calls)
│   └── style.css             # Giao diện
├── src/
│   ├── models/
│   │   └── handwriting_model.py   # EncoderDecoderHTR (iam_p4)
│   ├── data/
│   │   ├── handwriting_preprocessing.py
│   │   └── segmentation.py
│   ├── postprocessing/
│   │   ├── spellcheck.py          # SpellCorrector
│   │   └── enhanced_corrector.py  # OCR-aware corrector (Phase 3)
│   ├── config/
│   │   └── logging_config.py
│   └── utils/
│       └── validators.py
├── iam_p4/
│   ├── best_encoder_decoder.pth   # Model weights (production)
│   ├── phase1_quickwins.py
│   ├── phase2_architecture.py
│   ├── phase3_postprocessing.py
│   ├── phase4_deployment.py
│   └── PHASE[1-4]_REPORT.md
├── iam_p5/
│   ├── crnn_model.py              # CRNN thuần (refactored)
│   ├── ctc_utils.py               # CTC encoder/decoder
│   └── IAM_P5_SUMMARY.md
└── saved_models/
    └── digit_recognition_optimized_colab.h5   # Keras model phụ (chữ số)
```

---

## 8. Luồng Hoạt Động Web App

```
[Browser]
   │  Vẽ chữ trên Canvas (HTML5 Canvas API)
   │  POST /predict  (base64 image + params)
   ▼
[Flask app.py]
   │
   ├─ validate_image() — kiểm tra input an toàn
   ├─ preprocess_handwriting_image() — resize, normalize
   ├─ segment_text_image() — tách từng từ (multi-word mode)
   │
   ├─ EncoderDecoderHTR.forward()
   │     ├─ ResNet CNN → features
   │     ├─ Transformer Encoder → memory
   │     └─ Transformer Decoder (Greedy / Beam Search) → tokens
   │
   ├─ decode_sequence() → raw text
   ├─ enhanced_corrector (Phase 3) → corrected text
   └─ SpellCorrector → final text
   │
   ▼
[Response JSON]  { "result": "hello world", "raw": "hell0 w0rld" }
```

---

## 9. Metrics Tổng Hợp

| Phiên bản | CER | Character Acc | Digit Acc | Ghi chú |
|---|---|---|---|---|
| **Baseline (trước p1)** | 4.20% | 95.80% | 75–80% | Model ban đầu |
| **After Phase 1** | 2.50% | 97.50% | 88–92% | +data, +weights |
| **After Phase 2** | 2.00% | 98.00% | ~92% | +FPN, Beam Search |
| **After Phase 3** | 1.70% | 98.30% | ~93% | +spell check (có bias) |
| **After Phase 4 (INT8)** | 1.80% | 95.50% | - | −75% size, −0.3% acc |
| **iam_p5 (honest)** | ~5–8% | - | - | Không post-process |

---

## 10. Điểm Nổi Bật & Bài Học

### ✅ Điểm Mạnh
- **Full pipeline**: từ vẽ tay trên browser → kết quả văn bản
- **Phase-by-phase improvement**: cải tiến có hệ thống và đo lường rõ ràng
- **Web-based**: không cần cài đặt client
- **Spell-checking tích hợp**: sửa lỗi OCR phổ biến (0↔O, 1↔l...)
- **Deployment-ready**: INT8 quantization + ONNX export

### ⚠️ Bài Học Rút Ra
1. **Spell-checker trong validation = "đẹp số giả"** — CER giảm mạnh nhưng không phản ánh model thực
2. **Transformer ≠ luôn tốt hơn CRNN cho OCR** — Khi loss function (CTC) không khớp với decode (autoregressive), model học sai hướng
3. **FPN không phù hợp OCR** — Multi-scale tốt cho detection, không cần thiết cho sequence recognition
4. **Augmentation mạnh quá = hại** — Elastic transform mạnh phá cấu trúc chữ, khiến model học noise
5. **"Đơn giản + Đúng > Phức tạp + Không đúng"** (nguyên tắc iam_p5)

---

*Tổng hợp bởi GitHub Copilot — 30/03/2026*
