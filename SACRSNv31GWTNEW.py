import os
import re
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# ==========================================
# 0. Configuration & Determinism
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "seq_len": 64,
    "embed_dim": 256,       # Complex dim (Real=128, Imag=128)
    "n_modules": 4,         # Processor, Memory, Semantic, Sensory
    "n_sensory_anchors": 64,# Size of the sensory palette (Hebbian)
    "memory_slots": 128,    # Size of Time Line Memory
    "vocab_size": 2000,     # BPE Vocab Size
    
    # NLP / GWT Settings
    "competition_temp": 0.8,
    "commitment_cost": 0.25,
    "halt_threshold": 0.95, # T.O.T.E. Exit threshold
    "max_recursion": 12,    # Max thinking steps
    
    # Training
    "epochs": 5000,
    "lr": 1e-3,
    "batch_size": 16
}

print(f"--- SACRSN v38: NLP Edition ---")
print(f"Device: {DEVICE}")

# ==========================================
# 1. BPE Tokenizer & Data Loading
# ==========================================
class SimpleBPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.reverse_vocab = {}

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, text):
        print("Training BPE Tokenizer...")
        # Pre-tokenize roughly
        words = re.findall(r"[\w']+|[^\s\w]", text)
        vocab = Counter([" ".join(list(w)) for w in words])
        
        for i in range(self.vocab_size - len(self.special_tokens)):
            pairs = self.get_stats(vocab)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges[best] = i + len(self.special_tokens)
            vocab = self.merge_vocab(best, vocab)
            
        # Build final vocab
        self.vocab = self.special_tokens.copy()
        # Add basic chars
        unique_chars = set(list(text))
        idx = len(self.vocab)
        for c in unique_chars:
            if c not in self.vocab:
                self.vocab[c] = idx
                idx += 1
        
        # Add merges
        for p, _ in self.merges.items():
            token = "".join(p)
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1
                
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"BPE Trained. Final Vocab Size: {len(self.vocab)}")

    def encode(self, text):
        # Very simplified inference for demo
        words = re.findall(r"[\w']+|[^\s\w]", text)
        ids = []
        for w in words:
            # Try full word
            if w in self.vocab:
                ids.append(self.vocab[w])
            else:
                # Fallback to chars (inefficient but safe)
                for c in w:
                    ids.append(self.vocab.get(c, self.vocab["<UNK>"]))
        return ids

    def decode(self, ids):
        tokens = [self.reverse_vocab.get(i, "<UNK>") for i in ids]
        return "".join(tokens).replace("</w>", " ")

def load_training_data(filepath="data.txt"):
    # Fallback text if file doesn't exist
    FALLBACK_TEXT = """
    The neural architecture of the mind is a mirror of the cosmos itself. 
    As above, so below; the filamentary structures of the intergalactic web 
    find their precise echo in the dense, white matter connecting the hemispheres. 
    To understand the nature of thought, one must first understand the nature of the void. 
    It is the negative space that defines the positive, the silence that gives shape to sound.
    We are stardust contemplating its own arrangement.
    Logic is the foundation, but chaos is the architect.
    """
    
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
    else:
        print("File not found. Using internal philosophical fallback data.")
        data = FALLBACK_TEXT
        # Write it for next time
        try:
            with open(filepath, "w") as f: f.write(FALLBACK_TEXT)
        except: pass

    return data

# ==========================================
# 2. Complex & Math Primitives
# ==========================================
class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
    def forward(self, z):
        # (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        real = self.fc_real(z.real) - self.fc_imag(z.imag)
        imag = self.fc_real(z.imag) + self.fc_imag(z.real)
        return torch.complex(real, imag)

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        mag = torch.abs(z) + 1e-6
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag - mean) / torch.sqrt(var + 1e-6)
        norm_mag = norm_mag * self.scale + self.shift
        return torch.complex(norm_mag * torch.cos(z.angle()), norm_mag * torch.sin(z.angle()))

class ComplexAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.scale = dim ** -0.5
    
    def forward(self, query, key=None, value=None):
        if key is None: key = query
        if value is None: value = query
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Complex Dot Product Attention
        # Flatten for matmul: treat Complex as 2x Real size for interaction
        q_flat = torch.cat([q.real, q.imag], dim=-1)
        k_flat = torch.cat([k.real, k.imag], dim=-1)
        
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        v_real = torch.matmul(attn, v.real)
        v_imag = torch.matmul(attn, v.imag)
        return torch.complex(v_real, v_imag), attn

# ==========================================
# 3. Improved Modules (NLP Upgrades)
# ==========================================

# [NEW] Hebbian Sensory Module (Anchoring)
class HebbianSensoryModule(nn.Module):
    def __init__(self, dim, n_anchors):
        super().__init__()
        # Learnable Palettes (Submodalities)
        # Visual: Bright/Dim, Color, Distance
        self.visual_palette = nn.Parameter(torch.randn(n_anchors, dim)) 
        # Auditory: Tone, Pitch, Rhythm
        self.audio_palette = nn.Parameter(torch.randn(n_anchors, dim))
        
        self.salience_head = nn.Linear(dim*2, 1)

    def forward(self, gw_state):
        # Query the palette based on the thought "vibe"
        # Visual Query (Real Part)
        vis_scores = torch.matmul(gw_state.real, self.visual_palette.t())
        vis_attn = F.softmax(vis_scores, dim=-1)
        vis_response = torch.matmul(vis_attn, self.visual_palette)
        
        # Auditory Query (Imag Part)
        aud_scores = torch.matmul(gw_state.imag, self.audio_palette.t())
        aud_attn = F.softmax(aud_scores, dim=-1)
        aud_response = torch.matmul(aud_attn, self.audio_palette)
        
        proposal = torch.complex(vis_response, aud_response)
        
        flat = torch.cat([proposal.real, proposal.imag], dim=-1)
        salience = self.salience_head(flat)
        
        return proposal, salience, vis_attn

# [NEW] Time Line Memory (Differentiable Neural Dictionary)
class TimeLineMemory(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.dim = dim
        self.slots = slots
        # Key: The "Vibe" (Context/Phase), Value: The "Content" (Magnitude)
        self.keys = nn.Parameter(torch.randn(slots, dim))
        self.values = nn.Parameter(torch.randn(slots, dim))
        
        self.salience_head = nn.Linear(dim*2, 1)
        self.write_gate = nn.Linear(dim*2, 1)

    def forward(self, gw_state):
        # 1. Retrieval (Recall)
        # Cosine similarity matching
        q_norm = F.normalize(gw_state.real, dim=-1)
        k_norm = F.normalize(self.keys, dim=-1)
        scores = torch.matmul(q_norm, k_norm.t()) # (Batch, Slots)
        attn = F.softmax(scores * 5.0, dim=-1) # Sharpen
        
        recalled_content = torch.matmul(attn, self.values)
        proposal = torch.complex(recalled_content, torch.zeros_like(recalled_content))
        
        # 2. Storage (Write)
        # Simple FIFO/Update for demo (Cyclic)
        # In a real NLP model, this would overwrite the "Least Used" slot
        with torch.no_grad():
             gate = torch.sigmoid(self.write_gate(torch.cat([gw_state.real, gw_state.imag], dim=-1)))
             if gate.mean() > 0.5:
                 # Shift keys/values and write to index 0
                 self.keys.data = torch.cat([gw_state.real, self.keys.data[:-1]])
                 self.values.data = torch.cat([gw_state.real, self.values.data[:-1]])
        
        flat = torch.cat([proposal.real, proposal.imag], dim=-1)
        salience = self.salience_head(flat)
        return proposal, salience

class SemanticModule(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        # VQ Codebook
        self.codebook = nn.Embedding(vocab_size, dim * 2)
        self.salience_head = nn.Linear(dim * 2, 1)

    def forward(self, gw_state):
        z_flat = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        
        # Find nearest symbol (Vector Quantization)
        # d = ||z - e||^2 = ||z||^2 + ||e||^2 - 2ze
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.weight.t())
            
        min_encoding_indices = torch.argmin(d, dim=-1)
        z_q = self.codebook(min_encoding_indices)
        
        # Straight-Through Estimator
        z_q = z_flat + (z_q - z_flat).detach()
        
        proposal = torch.complex(z_q[..., :z_flat.shape[-1]//2], z_q[..., z_flat.shape[-1]//2:])
        salience = self.salience_head(z_q)
        
        # VQ Loss (Commitment)
        loss = F.mse_loss(z_q.detach(), z_flat) + 0.25 * F.mse_loss(z_q, z_flat.detach())
        
        return proposal, salience, loss, min_encoding_indices

class ProcessorModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = ComplexAttention(dim)
        self.norm = ComplexLayerNorm(dim)
        self.salience_head = nn.Linear(dim*2, 1)
        self.confidence_head = nn.Linear(dim*2, 1)

    def forward(self, input_emb, gw_state):
        # Processor compares Input to Workspace
        combined = input_emb + gw_state
        z, _ = self.attn(combined)
        z = self.norm(z)
        
        flat = torch.cat([z.real, z.imag], dim=-1)
        salience = self.salience_head(flat)
        conf = torch.sigmoid(self.confidence_head(flat))
        
        return z, salience, conf

# ==========================================
# 4. The UberCRSN v38 Model
# ==========================================
class UberCRSN_v38(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim * 2)
        
        # The T.O.T.E. Components
        self.processor = ProcessorModule(dim)
        self.sensory = HebbianSensoryModule(dim, CONFIG["n_sensory_anchors"])
        self.memory = TimeLineMemory(dim, CONFIG["memory_slots"])
        self.semantic = SemanticModule(dim, vocab_size)
        
        # Global Workspace Gate
        self.gate_bias = nn.Parameter(torch.zeros(CONFIG["n_modules"]))
        
        # Decoder
        self.decoder = nn.Linear(dim * 2, vocab_size)

    def forward(self, x):
        # x: (1, 1) token index
        emb_flat = self.embedding(x).squeeze(1)
        gw_state = torch.complex(emb_flat[..., :self.dim], emb_flat[..., self.dim:])
        
        total_vq_loss = 0
        loop_count = 0
        final_symbol = None
        
        # T.O.T.E. Loop (Adaptive Computation)
        for t in range(CONFIG["max_recursion"]):
            loop_count += 1
            
            # 1. Modules Generate Proposals
            p_prop, p_sal, p_conf = self.processor(gw_state, gw_state)
            s_prop, s_sal, _      = self.sensory(gw_state)
            m_prop, m_sal         = self.memory(gw_state)
            sem_prop, sem_sal, vq_loss, sym_idx = self.semantic(gw_state)
            
            total_vq_loss += vq_loss
            
            # 2. Competition
            proposals = torch.stack([p_prop, s_prop, m_prop, sem_prop], dim=1)
            saliences = torch.cat([p_sal, s_sal, m_sal, sem_sal], dim=1)
            
            # Softmax with temp
            weights = F.softmax((saliences + self.gate_bias) / CONFIG["competition_temp"], dim=-1)
            
            # 3. Update Workspace (Complex Weighted Sum)
            weights_c = torch.complex(weights.unsqueeze(-1), torch.zeros_like(weights.unsqueeze(-1)))
            gw_update = torch.sum(proposals * weights_c, dim=1)
            
            # Residual connection with decay
            gw_state = 0.7 * gw_state + 0.3 * gw_update
            
            # 4. T.O.T.E. Exit Check (Dynamic Halting)
            # If Processor is > 95% confident, we stop thinking.
            if p_conf.mean() > CONFIG["halt_threshold"]:
                break
                
        # Decode
        flat_out = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        return logits, total_vq_loss, loop_count, weights

# ==========================================
# 5. Training & Visualization
# ==========================================
def train_and_audit():
    # 1. Load Data
    text_data = load_training_data("data.txt")
    
    # 2. Train Tokenizer
    tokenizer = SimpleBPE(vocab_size=CONFIG["vocab_size"])
    tokenizer.train(text_data)
    
    # 3. Prepare Model
    model = UberCRSN_v38(len(tokenizer.vocab), CONFIG["embed_dim"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- Starting Training (NLP T.O.T.E Loop) ---")
    data_ids = tokenizer.encode(text_data)
    data_tensor = torch.tensor(data_ids, dtype=torch.long).to(DEVICE)
    
    history = {"loss": [], "loops": []}
    
    # Training Loop
    model.train()
    for epoch in range(CONFIG["epochs"]):
        # Simple sliding window for demo
        idx = random.randint(0, len(data_tensor) - 2)
        x = data_tensor[idx].view(1, 1)
        y = data_tensor[idx+1].view(1)
        
        optimizer.zero_grad()
        logits, vq_loss, loops, weights = model(x)
        
        loss_main = criterion(logits, y)
        total_loss = loss_main + 0.1 * vq_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history["loss"].append(total_loss.item())
        history["loops"].append(loops)
        
        if epoch % 50 == 0:
            print(f"Ep {epoch} | Loss: {total_loss.item():.4f} | Avg Think Steps: {np.mean(history['loops'][-50:]):.1f}")

    # ==========================================
    # 6. Audit & Visualization (Visualizing the "Mind")
    # ==========================================
    print("\n--- Auditing Internal States ---")
    model.eval()
    
    # Visual 1: The Hebbian Palette (Anchoring)
    # What does the model "see" inside?
    palette = model.sensory.visual_palette.detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(palette[:20], aspect='auto', cmap='inferno')
    plt.title("Learned Sensory Anchors (Visual Submodalities)")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Anchor Index")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    # Visual 2: T.O.T.E. Efficiency
    plt.figure(figsize=(10, 4))
    plt.plot(history['loops'])
    plt.title("Adaptive Computation (T.O.T.E Cycles)")
    plt.ylabel("Thinking Steps needed")
    plt.xlabel("Training Iteration")
    plt.show()
    
    # Inference Test
    start_word = "The"
    curr_ids = tokenizer.encode(start_word)
    input_t = torch.tensor([curr_ids[-1]], dtype=torch.long, device=DEVICE).view(1,1)
    
    print(f"\nGenerative Dream (Seed: '{start_word}'):")
    out_text = start_word
    
    for _ in range(20):
        with torch.no_grad():
            logits, _, steps, _ = model(input_t)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            out_text += tokenizer.decode([next_id])
            input_t = torch.tensor([next_id], dtype=torch.long, device=DEVICE).view(1,1)
            
    print(f"--> {out_text}")

if __name__ == "__main__":
    train_and_audit()