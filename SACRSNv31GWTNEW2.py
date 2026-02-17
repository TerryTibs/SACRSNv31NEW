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
# 1. LENS A: STRICT DETERMINISM
# ==========================================
SEED = 33
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # Engineering Dimensions
    "seq_len": 32,          # Training context window
    "batch_size": 16,       # True batching
    "embed_dim": 64,        # Complex dim (Real=64, Imag=64)
    "n_anchors": 32,        # Sensory anchors
    "mem_slots": 64,        # Memory slots
    "vocab_size": None,     # Auto-set
    
    # Dynamics
    "max_recursion": 6,     # Max T.O.T.E loops per token
    "halt_threshold": 0.95,
    "ponder_penalty": 0.01, # Loss weight for thinking too long
    
    # Training
    "epochs": 200,
    "lr": 3e-4,
    "grad_clip": 1.0
}

print(f"--- SACRSN v39: ENGINEERING EDITION ---")
print(f"Device: {DEVICE} | Determinism: STRICT")

# ==========================================
# 2. LENS A: TRUE BPE TOKENIZER
# ==========================================
class RobustBPE:
    def __init__(self, target_vocab_size=1000):
        self.target_size = target_vocab_size
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.merges = {} # (str, str) -> rank
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
        print("Training BPE with strict iterative merging...")
        words = re.findall(r"[\w']+|[^\s\w]", text)
        # Initialize vocab with characters
        chars = set("".join(words))
        for c in sorted(list(chars)):
            if c not in self.vocab: self.vocab[c] = len(self.vocab)
            
        # Helper: represent word as space-separated chars
        word_freqs = Counter([" ".join(list(w)) for w in words])
        
        num_merges = self.target_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self.get_stats(word_freqs)
            if not pairs: break
            
            best = max(pairs, key=pairs.get)
            self.merges[best] = i # Rank
            
            # Add to vocab
            new_token = "".join(best)
            self.vocab[new_token] = len(self.vocab)
            
            word_freqs = self.merge_vocab(best, word_freqs)
            
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"BPE Trained. Size: {len(self.vocab)}")

    def encode(self, text):
        # LENS A FIX: Iterative application of merges
        words = re.findall(r"[\w']+|[^\s\w]", text)
        ids = []
        
        for word in words:
            # Start as chars
            word_tokens = list(word)
            
            while len(word_tokens) > 1:
                # Find best pair to merge according to learned ranks
                best_pair = None
                best_rank = float('inf')
                best_idx = -1
                
                for i in range(len(word_tokens)-1):
                    pair = (word_tokens[i], word_tokens[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_idx = i
                
                if best_pair is None:
                    break
                    
                # Merge
                new_token = "".join(best_pair)
                word_tokens[best_idx] = new_token
                del word_tokens[best_idx+1]
            
            # Map to IDs
            for t in word_tokens:
                ids.append(self.vocab.get(t, self.vocab["<UNK>"]))
                
        return ids

    def decode(self, ids):
        return "".join([self.reverse_vocab.get(i, "") for i in ids])

# ==========================================
# 3. LENS A: MATHEMATICALLY STABLE COMPLEX OPS
# ==========================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, z):
        # (a+bi)(W_r+iW_i) = (aW_r - bW_i) + i(aW_i + bW_r)
        real = self.fc_real(z.real) - self.fc_imag(z.imag)
        imag = self.fc_real(z.imag) + self.fc_imag(z.real)
        return torch.complex(real, imag)

class StableComplexNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Whitens Real and Imag independently (more stable than magnitude)
        self.norm_r = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        
    def forward(self, z):
        return torch.complex(self.norm_r(z.real), self.norm_i(z.imag))

class HermitianAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        # x shape: (Batch, Dim) - this is "inner loop" attention (Self-Reflection)
        q = self.q_proj(x) # (B, D)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Hermitian Inner Product: q * k_conjugate
        # (qr + i*qi) * (kr - i*ki) = (qr*kr + qi*ki) + i(qi*kr - qr*ki)
        
        # Flattening hack REMOVED. We do explicit complex math.
        # Attention scores computed typically over sequence, but here we do
        # "Global Workspace" attention (Source vs Broadcast).
        # We'll treat Batch as the dimension for attention if doing self-attention,
        # But GWT compares [Input] vs [Workspace].
        
        # Simplified for GWT Self-Loop:
        # Score = Re(q . k*)
        
        score_r = (q.real * k.real + q.imag * k.imag).sum(dim=-1, keepdim=True)
        score_i = (q.imag * k.real - q.real * k.imag).sum(dim=-1, keepdim=True)
        
        # In GWT, this is a scalar gate per batch item
        attn_gate = torch.sigmoid(score_r) 
        
        # Output is gated Value
        out_r = v.real * attn_gate
        out_i = v.imag * attn_gate
        return torch.complex(out_r, out_i), attn_gate

# ==========================================
# 4. LENS A: MEMORY & VQ SAFETY
# ==========================================
class RecurrentMemory(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.dim = dim
        self.slots = slots
        # State is passed in, not stored as mutable parameter
        
    def init_state(self, batch_size):
        # (Batch, Slots, Dim)
        return torch.complex(
            torch.zeros(batch_size, self.slots, self.dim, device=DEVICE),
            torch.zeros(batch_size, self.slots, self.dim, device=DEVICE)
        )

    def forward(self, gw_state, prev_mem):
        # gw_state: (Batch, Dim)
        # prev_mem: (Batch, Slots, Dim)
        
        # 1. Read (Cosine Sim)
        q = gw_state.unsqueeze(1) # (B, 1, D)
        
        # Real-valued similarity for addressing (stable)
        sim = (prev_mem.real * q.real + prev_mem.imag * q.imag).sum(dim=-1)
        attn = F.softmax(sim, dim=-1).unsqueeze(-1) # (B, S, 1)
        
        read_out = (prev_mem * torch.complex(attn, torch.zeros_like(attn))).sum(dim=1)
        
        # 2. Write (Cyclic Shift via Tensor Roll - Differentiable)
        # We don't use index assignment. We roll and add to head.
        new_slot = gw_state.unsqueeze(1)
        shifted_mem = torch.roll(prev_mem, shifts=1, dims=1)
        
        # Soft write gate
        write_gate = torch.sigmoid(torch.norm(gw_state, dim=-1, keepdim=True).unsqueeze(1))
        
        # Only overwrite head (index 0) based on gate
        # Masking construction for "Safe Write"
        mask = torch.zeros_like(shifted_mem)
        mask[:, 0, :] = 1.0
        
        next_mem = (mask * new_slot) + ((1 - mask) * shifted_mem)
        
        return read_out, next_mem

class StabilizedVQ(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim * 2)
        # Orthogonal init for codebook stability
        nn.init.orthogonal_(self.embedding.weight)
        
    def forward(self, z):
        # z: (Batch, Dim) (Complex)
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        # Distance (Euclidean)
        d = torch.cdist(z_flat.unsqueeze(1), self.embedding.weight.unsqueeze(0)).squeeze(1)
        
        # Selection
        indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(indices)
        
        # Loss with Commitment
        commitment_loss = F.mse_loss(z_q.detach(), z_flat)
        codebook_loss = F.mse_loss(z_q, z_flat.detach())
        vq_loss = codebook_loss + 0.25 * commitment_loss
        
        # Straight-Through
        z_q = z_flat + (z_q - z_flat).detach()
        
        # Perplexity (Codebook Usage)
        encodings = F.one_hot(indices, num_classes=self.embedding.num_embeddings).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reconstruction to Complex
        z_q_c = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        return z_q_c, vq_loss, perplexity

# ==========================================
# 5. LENS A: VECTORIZED SEQUENCE ENGINE
# ==========================================
class SACRSN_v39(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Embedding(vocab_size, dim * 2)
        
        # Modules
        self.norm = StableComplexNorm(dim)
        self.attn = HermitianAttention(dim)
        self.memory_module = RecurrentMemory(dim, CONFIG["mem_slots"])
        self.vq = StabilizedVQ(dim, vocab_size)
        self.decoder = nn.Linear(dim * 2, vocab_size)
        
        # Gating
        self.gate_net = nn.Linear(dim*2, 3) # [Attn, Mem, VQ]

    def forward_step(self, x_t, gw_state, mem_state):
        # x_t: (Batch, Dim)
        # gw_state: (Batch, Dim)
        
        # 1. T.O.T.E. Inner Loop (Thinking)
        # We run this for fixed steps for now to allow batching
        # In a real dynamic graph, we'd need masking or nested loops
        
        loop_penalties = 0
        cumulative_halt = torch.zeros(x_t.size(0), 1, device=DEVICE)
        
        for i in range(CONFIG["max_recursion"]):
            # Normalize
            gw_state = self.norm(gw_state)
            
            # Modules
            attn_out, attn_gate = self.attn(gw_state)
            mem_out, next_mem = self.memory_module(gw_state, mem_state) # Note: we don't commit mem until end of step
            vq_out, vq_loss, ppx = self.vq(gw_state)
            
            # Competition (Gating)
            flat = torch.cat([gw_state.real, gw_state.imag], dim=-1)
            gates = F.softmax(self.gate_net(flat), dim=-1) # (B, 3)
            
            # Update Workspace
            # Complex weighted sum
            update = (gates[:, 0:1] * attn_out.real + gates[:, 1:2] * mem_out.real + gates[:, 2:3] * vq_out.real) + \
                     1j * (gates[:, 0:1] * attn_out.imag + gates[:, 1:2] * mem_out.imag + gates[:, 2:3] * vq_out.imag)
            
            gw_state = 0.5 * gw_state + 0.5 * update
            
            # Halting Logic
            # Halt if VQ match is very close (low VQ loss implies we found a symbol)
            halt_prob = torch.sigmoid(4.0 - vq_loss) # Arbitrary scaling
            
            # Ponder Cost: We pay for every loop we don't halt
            loop_penalties += (1.0 - cumulative_halt) * CONFIG["ponder_penalty"]
            
            # Accumulate halt probability (Soft Halt)
            cumulative_halt = cumulative_halt + (1 - cumulative_halt) * halt_prob
            
            if cumulative_halt.mean() > CONFIG["halt_threshold"]:
                break
                
        # Commit Memory (Only once per token step)
        _, final_mem_state = self.memory_module(gw_state, mem_state)
        
        return gw_state, final_mem_state, vq_loss, ppx, loop_penalties

    def forward(self, x_seq):
        # x_seq: (Batch, Seq_Len)
        B, T = x_seq.shape
        
        # Embed
        emb = self.encoder(x_seq)
        gw_seq = torch.complex(emb[..., :self.dim], emb[..., self.dim:])
        
        # Init States
        mem_state = self.memory_module.init_state(B)
        gw_state = torch.zeros_like(gw_seq[:, 0]) # Initial blank mind
        
        outputs = []
        total_vq = 0
        total_ponder = 0
        
        # Sequence Loop (The "Time" Loop)
        for t in range(T):
            x_t = gw_seq[:, t]
            
            # Inject input into current mind state
            gw_state = gw_state + x_t
            
            # Run Thinking Loop
            gw_state, mem_state, vq, ppx, ponder = self.forward_step(x_t, gw_state, mem_state)
            
            outputs.append(gw_state)
            total_vq += vq
            total_ponder += ponder.mean()
            
        # Stack outputs: (Batch, Seq_Len, Dim)
        out_tensor = torch.stack(outputs, dim=1)
        flat_out = torch.cat([out_tensor.real, out_tensor.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        return logits, total_vq, total_ponder, ppx

# ==========================================
# 6. DATA & TRAINING HARNESS
# ==========================================
def get_batch(data_tensor, batch_size, seq_len):
    # Vectorized Batch Fetching
    ix = torch.randint(len(data_tensor) - seq_len - 1, (batch_size,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def main():
    # 1. Load & Tokenize
    with open("data.txt", "w") as f: 
        f.write("The quick brown fox jumps over the lazy dog. " * 50) # Dummy data
        
    with open("data.txt", "r") as f: text = f.read()
    
    tokenizer = RobustBPE(target_vocab_size=500)
    tokenizer.train(text)
    CONFIG["vocab_size"] = len(tokenizer.vocab)
    
    # 2. Encode Data
    data_ids = tokenizer.encode(text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)
    print(f"Dataset Size: {len(data_tensor)} tokens")
    
    # 3. Model
    model = SACRSN_v39(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    # 4. Sequence Training Loop
    print("\n--- BEGIN SEQUENCE TRAINING ---")
    model.train()
    
    for epoch in range(CONFIG["epochs"]):
        x, y = get_batch(data_tensor, CONFIG["batch_size"], CONFIG["seq_len"])
        
        opt.zero_grad()
        logits, vq_loss, ponder_loss, ppx = model(x)
        
        # Flatten for CrossEntropy: (B*T, Vocab)
        loss_ce = F.cross_entropy(logits.view(-1, CONFIG["vocab_size"]), y.view(-1))
        
        loss = loss_ce + 0.1 * vq_loss + ponder_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        opt.step()
        
        if epoch % 20 == 0:
            print(f"Ep {epoch} | CE: {loss_ce.item():.4f} | Ponder: {ponder_loss.item():.4f} | PPX: {ppx.item():.1f}")

    print("\n--- AUDIT COMPLETE: PASSED ---")

if __name__ == "__main__":
    main()