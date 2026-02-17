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
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # Architecture Dimensions
    "seq_len": 64,
    "embed_dim": 128,       # Complex dim (Real=128, Imag=128)
    "n_sensory_anchors": 64,
    "memory_slots": 128,
    "n_modules": 4,         # Processor, Memory, Semantic, Sensory
    
    # Dynamic settings (Will be set automatically based on data)
    "vocab_size": None,     
    
    # NLP / GWT Settings
    "competition_temp": 0.8,
    "halt_threshold": 0.92, # Confidence required to exit T.O.T.E. loop
    "max_recursion": 12,    # Max "thinking steps" per word
    
    # Training & Plasticity
    "epochs": 5000,
    "base_lr": 2e-3,
    "sleep_interval": 20,   # Consolidate memory every X epochs
    "buffer_capacity": 1000  # Size of Hippocampus
}

print(f"--- SACRSN v38.5: NLP Active Learning Edition ---")
print(f"Running on: {DEVICE}")

# ==========================================
# 1. Data & Automatic BPE Tokenizer
# ==========================================
class AutoBPE:
    def __init__(self, target_vocab_size=2000):
        self.target_size = target_vocab_size
        self.vocab = {}
        self.merges = {}
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
        print("Scanning text to build vocabulary...")
        # Basic pre-tokenization
        words = re.findall(r"[\w']+|[^\s\w]", text)
        vocab = Counter([" ".join(list(w)) for w in words])
        
        # Initialize with special tokens + unique characters
        self.vocab = self.special_tokens.copy()
        unique_chars = set(text)
        idx = len(self.vocab)
        for c in sorted(list(unique_chars)):
            if c not in self.vocab:
                self.vocab[c] = idx
                idx += 1
        
        print(f"Initial char-level vocab size: {len(self.vocab)}")
        
        # BPE Merge Loop
        num_merges = self.target_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges[best] = idx
            self.vocab["".join(best)] = idx
            idx += 1
            vocab = self.merge_vocab(best, vocab)
            
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Automatic Vocab Set. Final Size: {len(self.vocab)}")

    def encode(self, text):
        words = re.findall(r"[\w']+|[^\s\w]", text)
        ids = []
        for w in words:
            # 1. Try full word match
            if w in self.vocab:
                ids.append(self.vocab[w])
            else:
                # 2. Fallback to characters (robustness)
                for c in w:
                    ids.append(self.vocab.get(c, self.vocab["<UNK>"]))
        return ids

    def decode(self, ids):
        tokens = [self.reverse_vocab.get(i, "") for i in ids]
        return "".join(tokens)

def load_data_and_setup(filepath="data.txt"):
    # Fallback philosophical text if file missing
    DEFAULT_TEXT = """The neural architecture of the mind is a mirror of the cosmos itself. As above, so below; the filamentary structures of the intergalactic web find their precise echo in the dense, white matter connecting the hemispheres of the brain. Galaxies cluster like neurons in a cosmic synapse, and the voids between them echo the silence between thought. We are stardust contemplating its own arrangement, a fleeting arrangement of atoms attempting to comprehend the laws that bound them together. To understand the nature of thought, one must first understand the nature of the void. It is the negative space that defines the positive, the silence that gives shape to the sound. Logic is the foundation, but chaos is the architect. Without the rigid framework of logic, the structure collapses; without the unpredictability of chaos, the structure creates nothing new."""
    
    if not os.path.exists(filepath):
        print("File not found. Creating 'data.txt' with default corpus.")
        with open(filepath, 'w') as f: f.write(DEFAULT_TEXT)
        data = DEFAULT_TEXT
    else:
        with open(filepath, 'r', encoding='utf-8') as f: data = f.read()

    # Init and Train Tokenizer
    tokenizer = AutoBPE(target_vocab_size=1000) # Cap at 1000 for small data
    tokenizer.train(data)
    
    # Update Config Automatically
    CONFIG["vocab_size"] = len(tokenizer.vocab)
    
    return data, tokenizer

# ==========================================
# 2. Hippocampal Buffer (Memory Replay)
# ==========================================
class HippocampalBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        # Stores tuples: (input_tensor, target_tensor, surprise_score)
        self.buffer = []
    
    def add(self, x, y, surprise):
        # Prioritize keeping high-surprise events
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y, surprise))
        else:
            # Replace the least surprising memory
            self.buffer.sort(key=lambda k: k[2])
            if surprise > self.buffer[0][2]:
                self.buffer[0] = (x, y, surprise)
                
    def sample(self, batch_size=4):
        if not self.buffer: return None
        k = min(len(self.buffer), batch_size)
        return random.sample(self.buffer, k)

# ==========================================
# 3. Complex Math Primitives
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

class ComplexAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.scale = dim ** -0.5
    
    def forward(self, query):
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        
        # Flatten for matmul
        q_flat = torch.cat([q.real, q.imag], dim=-1)
        k_flat = torch.cat([k.real, k.imag], dim=-1)
        
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        v_real = torch.matmul(attn, v.real)
        v_imag = torch.matmul(attn, v.imag)
        return torch.complex(v_real, v_imag), attn

# ==========================================
# 4. Modules: Sensory, Memory, Processor
# ==========================================

# [Hebbian Anchoring]
class HebbianSensoryModule(nn.Module):
    def __init__(self, dim, n_anchors):
        super().__init__()
        # Learnable palettes for "Submodalities"
        self.visual_palette = nn.Parameter(torch.randn(n_anchors, dim)) 
        self.audio_palette = nn.Parameter(torch.randn(n_anchors, dim))
        self.salience_head = nn.Linear(dim*2, 1)

    def forward(self, gw_state):
        # Dot Product Query: "What does this thought look like?"
        vis_attn = F.softmax(torch.matmul(gw_state.real, self.visual_palette.t()), dim=-1)
        vis_resp = torch.matmul(vis_attn, self.visual_palette)
        
        aud_attn = F.softmax(torch.matmul(gw_state.imag, self.audio_palette.t()), dim=-1)
        aud_resp = torch.matmul(aud_attn, self.audio_palette)
        
        proposal = torch.complex(vis_resp, aud_resp)
        flat = torch.cat([proposal.real, proposal.imag], dim=-1)
        salience = self.salience_head(flat)
        return proposal, salience

# [Time Line Memory]
class TimeLineMemory(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.dim = dim
        self.keys = nn.Parameter(torch.randn(slots, dim))   # Context/Vibe
        self.values = nn.Parameter(torch.randn(slots, dim)) # Content
        self.salience_head = nn.Linear(dim*2, 1)

    def forward(self, gw_state):
        # Retrieval by Vibe (Cosine Sim)
        q_norm = F.normalize(gw_state.real, dim=-1)
        k_norm = F.normalize(self.keys, dim=-1)
        attn = F.softmax(torch.matmul(q_norm, k_norm.t()) * 5.0, dim=-1)
        
        recalled = torch.matmul(attn, self.values)
        proposal = torch.complex(recalled, torch.zeros_like(recalled))
        
        # Simple cyclic write (for demo stability)
        with torch.no_grad():
             self.keys.data = torch.cat([gw_state.real, self.keys.data[:-1]])
             self.values.data = torch.cat([gw_state.real, self.values.data[:-1]])

        flat = torch.cat([proposal.real, proposal.imag], dim=-1)
        salience = self.salience_head(flat)
        return proposal, salience

# [Symbolic Grounding]
class SemanticModule(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.codebook = nn.Embedding(vocab_size, dim * 2)
        self.salience_head = nn.Linear(dim * 2, 1)

    def forward(self, gw_state):
        z_flat = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        
        # Vector Quantization (Euclidean distance)
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.weight.t())
            
        min_indices = torch.argmin(d, dim=-1)
        z_q = self.codebook(min_indices)
        
        # Straight-Through Estimator
        z_q = z_flat + (z_q - z_flat).detach()
        
        proposal = torch.complex(z_q[..., :z_flat.shape[-1]//2], z_q[..., z_flat.shape[-1]//2:])
        salience = self.salience_head(z_q)
        vq_loss = F.mse_loss(z_q.detach(), z_flat) + 0.25 * F.mse_loss(z_q, z_flat.detach())
        
        return proposal, salience, vq_loss

class ProcessorModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = ComplexAttention(dim)
        self.norm = nn.LayerNorm(dim*2) # Simplified norm for stability
        self.salience_head = nn.Linear(dim*2, 1)
        self.conf_head = nn.Linear(dim*2, 1)

    def forward(self, gw_state):
        z, _ = self.attn(gw_state)
        # Norm logic
        flat = torch.cat([z.real, z.imag], dim=-1)
        flat = self.norm(flat)
        z = torch.complex(flat[..., :flat.shape[-1]//2], flat[..., flat.shape[-1]//2:])
        
        salience = self.salience_head(flat)
        conf = torch.sigmoid(self.conf_head(flat))
        return z, salience, conf

# ==========================================
# 5. UberCRSN Model (The Brain)
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim * 2)
        
        self.processor = ProcessorModule(dim)
        self.sensory = HebbianSensoryModule(dim, CONFIG["n_sensory_anchors"])
        self.memory = TimeLineMemory(dim, CONFIG["memory_slots"])
        self.semantic = SemanticModule(dim, vocab_size)
        
        self.decoder = nn.Linear(dim * 2, vocab_size)

    def forward(self, x):
        # x: (1, 1) index
        emb_flat = self.embedding(x).squeeze(1)
        gw_state = torch.complex(emb_flat[..., :self.dim], emb_flat[..., self.dim:])
        
        total_vq = 0
        loop_count = 0
        
        # T.O.T.E. Loop (Test-Operate-Test-Exit)
        for t in range(CONFIG["max_recursion"]):
            loop_count += 1
            
            # 1. Proposals
            p_prop, p_sal, p_conf = self.processor(gw_state)
            s_prop, s_sal         = self.sensory(gw_state)
            m_prop, m_sal         = self.memory(gw_state)
            sem_prop, sem_sal, vq = self.semantic(gw_state)
            
            total_vq += vq
            
            # 2. Competition
            props = torch.stack([p_prop, s_prop, m_prop, sem_prop], dim=1)
            sals = torch.cat([p_sal, s_sal, m_sal, sem_sal], dim=1)
            weights = F.softmax(sals / CONFIG["competition_temp"], dim=-1)
            
            # 3. Update Global Workspace
            weights_c = torch.complex(weights.unsqueeze(-1), torch.zeros_like(weights.unsqueeze(-1)))
            gw_update = torch.sum(props * weights_c, dim=1)
            gw_state = 0.6 * gw_state + 0.4 * gw_update # Residual update
            
            # 4. Exit Check (Metacognitive Confidence)
            if p_conf.mean() > CONFIG["halt_threshold"]:
                break
                
        flat_out = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        return logits, total_vq, loop_count

# ==========================================
# 6. Main Learning Loop (Waking & Sleeping)
# ==========================================
def main():
    # 1. Setup Data & Auto-Vocab
    data_text, tokenizer = load_data_and_setup()
    data_ids = tokenizer.encode(data_text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long).to(DEVICE)
    
    print(f"Data Loaded. Tokens: {len(data_ids)}. Vocab: {CONFIG['vocab_size']}")
    
    # 2. Setup Model
    model = UberCRSN(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["base_lr"])
    criterion = nn.CrossEntropyLoss(reduction='none') # Per-sample loss for surprise
    
    replay_buffer = HippocampalBuffer(CONFIG["buffer_capacity"])
    
    history = {"loss": [], "plasticity": [], "loops": []}
    
    model.train()
    
    print("\n--- Starting Active Learning Process ---")
    
    try:
        for epoch in range(CONFIG["epochs"]):
            # --- A. WAKING PHASE ---
            idx = random.randint(0, len(data_tensor) - 2)
            x = data_tensor[idx].view(1, 1)
            y = data_tensor[idx+1].view(1)
            
            optimizer.zero_grad()
            logits, vq_loss, loops = model(x)
            
            # Calculate Surprise
            raw_loss = criterion(logits, y)
            surprise = raw_loss.item()
            
            # Mechanism 1: Dopamine Plasticity
            # High surprise = High learning rate for this step
            plasticity = min(4.0, 1.0 + (surprise * 0.8))
            
            # Mechanism 2: Metacognitive Penalty
            # Punish high confidence on wrong answers
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max()
            ego_penalty = (confidence * raw_loss) * 0.5
            
            total_loss = (raw_loss * plasticity) + ego_penalty + (0.1 * vq_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Store Memory
            replay_buffer.add(x, y, surprise)
            
            history["loss"].append(surprise)
            history["plasticity"].append(plasticity)
            history["loops"].append(loops)
            
            # --- B. SLEEP PHASE (Consolidation) ---
            if epoch > 0 and epoch % CONFIG["sleep_interval"] == 0:
                dream_batch = replay_buffer.sample(batch_size=5)
                if dream_batch:
                    # Replay high-surprise memories
                    for dx, dy, _ in dream_batch:
                        optimizer.zero_grad()
                        d_logits, d_vq, _ = model(dx)
                        d_loss = criterion(d_logits, dy) + 0.1 * d_vq
                        d_loss.backward()
                        optimizer.step()

            # Logging
            if epoch % 50 == 0:
                avg_loop = np.mean(history["loops"][-50:])
                avg_plast = np.mean(history["plasticity"][-50:])
                print(f"Ep {epoch:03d} | Loss: {surprise:.4f} | Plast: {avg_plast:.2f} | Thoughts: {avg_loop:.1f}")

    except KeyboardInterrupt:
        print("\nStopping early...")

    # ==========================================
    # 7. Visualization & Inference
    # ==========================================
    print("\n--- Visualizing Internal State ---")
    model.eval()
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Learning Dynamics
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Surprise", alpha=0.5, color="red")
    plt.plot(history["plasticity"], label="Plasticity", alpha=0.5, color="blue")
    plt.title("Dopaminergic Response Curve")
    plt.xlabel("Experience Steps")
    plt.legend()
    
    # Plot 2: Sensory Anchors
    plt.subplot(1, 2, 2)
    palette = model.sensory.visual_palette.detach().cpu().numpy()
    plt.imshow(palette[:20], aspect='auto', cmap='plasma')
    plt.title("Learned Visual Anchors")
    plt.xlabel("Embedding Dim")
    plt.tight_layout()
    plt.show()
    
    # Text Generation
    start_word = "The"
    print(f"\nGenerative Output (Seed: '{start_word}'):")
    
    input_ids = tokenizer.encode(start_word)
    curr_t = torch.tensor([[input_ids[-1]]], dtype=torch.long, device=DEVICE)
    
    out_text = start_word + " "
    for _ in range(30):
        with torch.no_grad():
            logits, _, _ = model(curr_t)
            probs = F.softmax(logits, dim=-1)
            # Sample
            next_id = torch.multinomial(probs, 1).item()
            decoded = tokenizer.decode([next_id])
            out_text += decoded
            curr_t = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
            
    print(f"--> {out_text}\n")

if __name__ == "__main__":
    main()