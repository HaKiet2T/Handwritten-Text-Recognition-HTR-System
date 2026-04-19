"""
Handwriting Recognition Model (PyTorch)
Encoder-Decoder Architecture for handwriting text recognition
"""
import torch
import torch.nn as nn
import math
import string


# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

# Build vocabulary - NO SPACE (match iam_p2 model)
chars = list(string.ascii_lowercase + string.digits)  # a-z + 0-9 (NO SPACE)
vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + chars  # 3 special + 36 chars = 39 total

char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

PAD_IDX = char_to_idx[PAD_TOKEN]
SOS_IDX = char_to_idx[SOS_TOKEN]
EOS_IDX = char_to_idx[EOS_TOKEN]


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for image features (H x W)"""
    def __init__(self, d_model, max_h=100, max_w=1000):
        super().__init__()
        self.d_model = d_model
        
        # Create 2D positional encodings [max_h, max_w, d_model]
        pe = torch.zeros(max_h, max_w, d_model)
        
        # Split d_model in half for height and width
        d_h = d_model // 2
        
        # Height encoding (first half) - [max_h, d_h//2]
        pos_h = torch.arange(0, max_h).unsqueeze(1).float()  # [max_h, 1]
        div_term_h = torch.exp(torch.arange(0, d_h, 2).float() * (-math.log(10000.0) / d_h))  # [d_h//2]
        
        sin_h = torch.sin(pos_h * div_term_h)  # [max_h, d_h//2]
        cos_h = torch.cos(pos_h * div_term_h)  # [max_h, d_h//2]
        
        # Broadcast to all width positions
        pe[:, :, 0:d_h:2] = sin_h.unsqueeze(1).expand(-1, max_w, -1)
        pe[:, :, 1:d_h:2] = cos_h.unsqueeze(1).expand(-1, max_w, -1)
        
        # Width encoding (second half) - [max_w, d_h//2]
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()  # [max_w, 1]
        div_term_w = torch.exp(torch.arange(0, d_h, 2).float() * (-math.log(10000.0) / d_h))  # [d_h//2]
        
        sin_w = torch.sin(pos_w * div_term_w)  # [max_w, d_h//2]
        cos_w = torch.cos(pos_w * div_term_w)  # [max_w, d_h//2]
        
        # Broadcast to all height positions
        pe[:, :, d_h::2] = sin_w.unsqueeze(0).expand(max_h, -1, -1)
        pe[:, :, d_h+1::2] = cos_w.unsqueeze(0).expand(max_h, -1, -1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # pe: [H, W, C] → [B, H, W, C] → [B, C, H, W]
        pos_enc = self.pe[:H, :W, :].unsqueeze(0).permute(0, 3, 1, 2)
        return x + pos_enc.to(x.device)


class PositionalEncoding1D(nn.Module):
    """1D positional encoding for decoder (sequence position)"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        # x: [B, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class BasicBlock(nn.Module):
    """ResNet Basic Block"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetBackbone(nn.Module):
    """ResNet-style CNN backbone for HTR (matches iam_p2 model)"""
    def __init__(self, d_model=384):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, d_model, 2, stride=1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, 1, 64, 256]
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # [B, 64, 16, 64]
        x = self.layer1(x)  # [B, 64, 16, 64]
        x = self.layer2(x)  # [B, 128, 8, 32]
        x = self.layer3(x)  # [B, 256, 4, 16]
        x = self.layer4(x)  # [B, d_model, 4, 16]
        return x


class SimplifiedCNN(nn.Module):
    """CNN backbone for feature extraction"""
    def __init__(self, d_model=256):
        super().__init__()
        # 4 conv blocks: 1 → 64 → 128 → 256 → d_model
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2)  # 64x256 → 32x128
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2)  # 32x128 → 16x64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d((2, 1))  # 16x64 → 8x64 (keep width)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
            # NO pooling → output 8x64
        )
    
    def forward(self, x):
        # x: [B, 1, 64, 256]
        x = self.conv1(x)  # [B, 64, 32, 128]
        x = self.conv2(x)  # [B, 128, 16, 64]
        x = self.conv3(x)  # [B, 256, 8, 64]
        x = self.conv4(x)  # [B, d_model, 8, 64]
        return x


class EncoderDecoderHTR(nn.Module):
    """Transformer Encoder-Decoder for Handwriting Recognition"""
    def __init__(self, vocab_size, d_model=384, enc_layers=6, dec_layers=4,
                 nhead=8, ffn_dim=1536, dropout=0.2, max_seq_len=50, use_resnet=True,
                 max_h_2d=100, max_w_2d=500):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # CNN backbone - use ResNet for iam_p2 model
        if use_resnet:
            self.backbone = ResNetBackbone(d_model)
        else:
            self.backbone = SimplifiedCNN(d_model)
        
        # Positional encodings (match iam_p2 naming)
        self.pos_enc_2d = PositionalEncoding2D(d_model, max_h=max_h_2d, max_w=max_w_2d)
        self.tgt_pos_enc = PositionalEncoding1D(d_model, max_len=max_seq_len)
        
        # Token embedding for decoder (match iam_p2 naming: tgt_embed)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        
        # Transformer Encoder (processes image features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        
        # Transformer Decoder (generates text autoregressively)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        
        # Output projection (match iam_p2 naming: out_proj)
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Better initialization (TrOCR-style)"""
        # Linear layers: truncated normal
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def encode(self, images):
        """Encode image to features"""
        # CNN: [B, 1, H, W] → [B, C, h, w]
        features = self.backbone(images)
        
        # Add 2D positional encoding
        features = self.pos_enc_2d(features)
        
        # Flatten: [B, C, h, w] → [B, h*w, C]
        B, C, h, w = features.shape
        features = features.flatten(2).transpose(1, 2)
        
        # Transformer encoder
        memory = self.encoder(features)
        return memory
    
    def decode(self, tgt_tokens, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """Decode text from memory"""
        # Embed tokens: [B, seq_len] → [B, seq_len, d_model] (use tgt_embed)
        tgt_emb = self.tgt_embed(tgt_tokens) * math.sqrt(self.d_model)
        
        # Add 1D positional encoding (use tgt_pos_enc)
        tgt_emb = self.tgt_pos_enc(tgt_emb)
        
        # Transformer decoder (with cross-attention to encoder memory)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary (use out_proj)
        logits = self.out_proj(output)
        return logits
    
    def forward(self, images, tgt_tokens, tgt_mask=None, tgt_key_padding_mask=None):
        """Full forward pass (teacher forcing)"""
        memory = self.encode(images)
        logits = self.decode(tgt_tokens, memory, tgt_mask, tgt_key_padding_mask)
        return logits
    
    def generate(self, images, sos_idx, eos_idx, max_len=50, mode='greedy', beam_width=3, length_penalty=0.6, verbose=False, temperature=1.0, return_confidence=False):
        """Generate text from images
        
        Args:
            images: Input images [B, 1, H, W]
            sos_idx: Start of sequence token
            eos_idx: End of sequence token
            max_len: Maximum sequence length
            mode: 'greedy' or 'beam' decoding
            beam_width: Beam size for beam search (default 10)
            length_penalty: Length normalization factor (0.6-0.8, default 0.6)
            verbose: Print beam candidates (only for beam mode)
            temperature: Softmax temperature (1.0=normal, <1.0=more confident, >1.0=more diverse)
            return_confidence: Return average token confidence scores
        
        Returns:
            tgt_tokens: Generated token sequences [B, seq_len]
            confidences (optional): Average confidence per sequence [B]
        """
        if mode == 'beam':
            return self.beam_search(images, sos_idx, eos_idx, max_len, beam_width, length_penalty, verbose, return_confidence, temperature)
        else:
            return self.greedy_search(images, sos_idx, eos_idx, max_len, temperature, return_confidence)
    
    def greedy_search(self, images, sos_idx, eos_idx, max_len=50, temperature=1.0, return_confidence=False):
        """Greedy decoding with temperature and confidence tracking"""
        self.eval()
        with torch.no_grad():
            memory = self.encode(images)
            B = images.size(0)
            device = images.device
            
            # Start with <SOS>
            tgt_tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            
            # Track confidence scores (log probabilities for geometric mean)
            all_log_probs = []
            
            for _ in range(max_len):
                # Generate causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)
                
                # Create padding mask
                tgt_key_padding_mask = (tgt_tokens == PAD_IDX)
                
                # Decode
                logits = self.decode(tgt_tokens, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                
                # Apply temperature
                logits_temp = logits[:, -1, :] / temperature
                log_probs = torch.log_softmax(logits_temp, dim=-1)
                
                # Get next token (greedy)
                next_token = logits_temp.argmax(dim=-1, keepdim=True)
                
                # Track confidence (log probability of chosen token)
                if return_confidence:
                    token_log_prob = log_probs.gather(1, next_token)  # [B, 1]
                    all_log_probs.append(token_log_prob)
                
                # Mark finished sequences (found EOS)
                finished = finished | (next_token.squeeze(1) == eos_idx)
                
                # Append to sequence (replace with PAD if already finished)
                next_token_masked = next_token.clone()
                next_token_masked[finished] = PAD_IDX  # Use PAD_IDX instead of SOS_IDX
                tgt_tokens = torch.cat([tgt_tokens, next_token_masked], dim=1)
                
                # Stop if all sequences finished
                if finished.all():
                    break
            
            if return_confidence and len(all_log_probs) > 0:
                # Geometric mean: exp(mean(log_probs))
                avg_log_prob = torch.cat(all_log_probs, dim=1).mean(dim=1)  # [B]
                confidences = torch.exp(avg_log_prob)
                return tgt_tokens, confidences
            
            return tgt_tokens
    
    def beam_search(self, images, sos_idx, eos_idx, max_len=50, beam_width=3, length_penalty=0.6, verbose=False, return_confidence=False, temperature=1.0):
        """Optimized beam search with length normalization and confidence tracking"""
        self.eval()
        with torch.no_grad():
            memory = self.encode(images)  # [B, H*W, d_model]
            B = images.size(0)
            device = images.device
            vocab_size = self.vocab_size
            
            # Expand memory for beam search: [B*beam_width, H*W, d_model]
            memory = memory.unsqueeze(1).repeat(1, beam_width, 1, 1)  # [B, beam, seq, d]
            memory = memory.view(B * beam_width, memory.size(2), memory.size(3))
            
            # Initialize sequences: [B*beam_width, 1]
            sequences = torch.full((B * beam_width, 1), sos_idx, dtype=torch.long, device=device)
            
            # Initialize scores: [B, beam_width]
            scores = torch.zeros(B, beam_width, device=device)
            scores[:, 1:] = float('-inf')  # Only first beam is active initially
            
            # Track finished sequences
            finished = torch.zeros(B, beam_width, dtype=torch.bool, device=device)
            
            for step in range(max_len):
                # Current sequence length
                seq_len = sequences.size(1)
                
                # Generate mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                
                # Create padding mask for all beams
                tgt_key_padding_mask = (sequences == PAD_IDX)  # [B*beam_width, seq_len]
                
                # Decode all beams in parallel: [B*beam_width, seq_len, vocab]
                logits = self.decode(sequences, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                
                # Get log probs for next token: [B*beam_width, vocab]
                # Apply temperature for diversity
                logits_temp = logits[:, -1, :] / temperature
                log_probs = torch.log_softmax(logits_temp, dim=-1)
                log_probs = log_probs.view(B, beam_width, vocab_size)  # [B, beam, vocab]
                
                # Add current scores: [B, beam, vocab]
                log_probs = scores.unsqueeze(2) + log_probs
                
                # Mask finished beams (don't expand them, except for padding)
                for b in range(B):
                    for beam in range(beam_width):
                        if finished[b, beam]:
                            log_probs[b, beam, :] = float('-inf')
                            log_probs[b, beam, PAD_IDX] = scores[b, beam]  # Use PAD_IDX for consistency
                
                # Flatten to [B, beam*vocab] and get top beam_width
                log_probs_flat = log_probs.view(B, -1)  # [B, beam*vocab]
                top_scores, top_indices = log_probs_flat.topk(beam_width, dim=1)  # [B, beam]
                
                # Convert flat indices to (beam_idx, token_idx)
                prev_beam_idx = top_indices // vocab_size  # [B, beam]
                next_tokens = top_indices % vocab_size  # [B, beam]
                
                # Verbose output
                if verbose and B == 1:
                    print(f"\n📊 Step {step + 1}: (chọn top-{beam_width} từ {beam_width}×{vocab_size}={beam_width*vocab_size} candidates)")
                    for i in range(min(5, beam_width)):  # Show top 5
                        token_idx = next_tokens[0, i].item()
                        char = idx_to_char.get(token_idx, f'[{token_idx}]')
                        score = top_scores[0, i].item()
                        beam_idx = prev_beam_idx[0, i].item()
                        prev_seq = sequences[beam_idx].tolist()[1:]  # Skip SOS
                        prev_text = ''.join([idx_to_char.get(t, f'[{t}]') for t in prev_seq if t not in [eos_idx, PAD_IDX]])
                        # Show which previous beam this came from
                        print(f"  Beam {i+1}: [từ beam {beam_idx+1} ở step trước] '{prev_text}' + '{char}' = '{prev_text}{char}' (log_prob: {score:.3f})")
                
                # Gather previous sequences
                # [B*beam] indices for gathering
                gather_idx = (torch.arange(B, device=device).unsqueeze(1) * beam_width + prev_beam_idx).view(-1)
                sequences = sequences[gather_idx]  # [B*beam, seq_len]
                
                # Append next tokens
                sequences = torch.cat([sequences, next_tokens.view(-1, 1)], dim=1)  # [B*beam, seq_len+1]
                
                # Update scores with length normalization
                scores = top_scores / (seq_len + 1) ** length_penalty
                
                # Update finished flags
                finished = finished.gather(1, prev_beam_idx) | (next_tokens == eos_idx)
                
                # Early stopping: all beams finished
                if finished.view(B * beam_width).all():
                    break
            
            # Select best sequence for each batch
            # Apply final length normalization
            final_scores = scores / (sequences.size(1) ** length_penalty)
            best_beam_idx = final_scores.argmax(dim=1)  # [B]
            
            # Gather best sequences
            batch_idx = torch.arange(B, device=device) * beam_width + best_beam_idx
            best_sequences = sequences[batch_idx]  # [B, seq_len]
            
            if verbose and B == 1:
                print(f"\n✅ Final best: '{decode_sequence(best_sequences[0], idx_to_char)}'")
            
            if return_confidence:
                # Confidence = geometric mean of token probabilities (same as greedy)
                # Get the raw scores (before final length normalization)
                best_raw_scores = top_scores.gather(1, best_beam_idx.unsqueeze(1)).squeeze(1)  # [B]
                
                # Calculate sequence length (exclude SOS and PAD)
                seq_lengths = ((best_sequences != PAD_IDX) & (best_sequences != sos_idx)).sum(dim=1).float()
                seq_lengths = torch.clamp(seq_lengths, min=1)  # Avoid division by zero
                
                # Geometric mean: exp(avg_log_prob)
                avg_log_prob = best_raw_scores / seq_lengths
                confidences = torch.exp(avg_log_prob)
                
                return best_sequences, confidences
            
            return best_sequences


def decode_sequence(indices, idx_to_char, remove_special=True):
    """Decode sequence of indices to text"""
    chars = []
    for idx in indices:
        idx = idx.item() if torch.is_tensor(idx) else idx
        if idx == EOS_IDX:
            break
        if remove_special and idx in [PAD_IDX, SOS_IDX]:
            continue
        if idx in idx_to_char:
            chars.append(idx_to_char[idx])
    return ''.join(chars)


def load_handwriting_model(model_path, device='cpu'):
    """Load the trained handwriting recognition model - Auto-detect architecture"""
    # Load checkpoint first to check config
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Auto-detect architecture from state dict
    # Check if ResNet backbone (iam_p2) or SimplifiedCNN (iam_p1)
    use_resnet = 'backbone.layer1.0.conv1.weight' in state_dict
    
    # Infer d_model from backbone output
    if use_resnet and 'backbone.layer4.1.conv2.weight' in state_dict:
        d_model = state_dict['backbone.layer4.1.conv2.weight'].shape[0]
    elif 'backbone.conv4.0.weight' in state_dict:
        d_model = state_dict['backbone.conv4.0.weight'].shape[0]
    else:
        d_model = 384 if use_resnet else 256  # Default
    
    # Infer encoder/decoder layers by counting
    enc_layer_keys = [k for k in state_dict.keys() if k.startswith('encoder.layers.')]
    enc_layers = len(set(k.split('.')[2] for k in enc_layer_keys if len(k.split('.')) > 2))
    enc_layers = enc_layers if enc_layers > 0 else (6 if use_resnet else 4)
    
    dec_layer_keys = [k for k in state_dict.keys() if k.startswith('decoder.layers.')]
    dec_layers = len(set(k.split('.')[2] for k in dec_layer_keys if len(k.split('.')) > 2))
    dec_layers = dec_layers if dec_layers > 0 else (4 if use_resnet else 3)
    
    # Infer nhead and ffn_dim
    nhead = 8  # Standard
    if 'encoder.layers.0.linear1.weight' in state_dict:
        ffn_dim = state_dict['encoder.layers.0.linear1.weight'].shape[0]
    else:
        ffn_dim = 1536 if use_resnet else 1024
    
    # Infer max_seq_len from positional encoding
    if 'tgt_pos_enc.pe' in state_dict:
        max_seq_len = state_dict['tgt_pos_enc.pe'].shape[1]
    elif 'pos_enc_1d.pe' in state_dict:
        max_seq_len = state_dict['pos_enc_1d.pe'].shape[1]
    else:
        max_seq_len = 50
    
    # Infer 2D positional encoding size from checkpoint
    if 'pos_enc_2d.pe' in state_dict:
        pe_2d_shape = state_dict['pos_enc_2d.pe'].shape  # [max_h, max_w, d_model]
        max_h_2d = pe_2d_shape[0]
        max_w_2d = pe_2d_shape[1]
    else:
        max_h_2d = 100
        max_w_2d = 1000
    
    dropout = 0.2 if use_resnet else 0.1
    
    print(f"🔍 Auto-detected architecture:")
    print(f"   Backbone: {'ResNet (iam_p2)' if use_resnet else 'SimplifiedCNN (iam_p1)'}")
    print(f"   d_model={d_model}, enc_layers={enc_layers}, dec_layers={dec_layers}")
    print(f"   nhead={nhead}, ffn_dim={ffn_dim}, dropout={dropout}")
    print(f"   max_seq_len={max_seq_len}, vocab_size={len(vocab)}")
    print(f"   pos_enc_2d: max_h={max_h_2d}, max_w={max_w_2d}")
    
    # Create model with detected config
    model = EncoderDecoderHTR(
        vocab_size=len(vocab),
        d_model=d_model,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        nhead=nhead,
        ffn_dim=ffn_dim,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_resnet=use_resnet,
        max_h_2d=max_h_2d,
        max_w_2d=max_w_2d
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    
    return model
