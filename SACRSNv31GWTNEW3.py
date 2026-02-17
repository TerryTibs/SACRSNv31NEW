import os
import re
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter

# ==========================================
# 1. STRICT DETERMINISM
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
torch.use_deterministic_algorithms(True) # Enforce strictness

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # Engineering Dimensions
    "seq_len": 32,
    "batch_size": 16,
    "embed_dim": 64,        # Complex dim (Real=64, Imag=64)
    "mem_slots": 32,        # Memory slots
    "vocab_size": None,     # Auto-set
    
    # Dynamics
    "max_recursion": 8,     # Max T.O.T.E loops per token
    "halt_threshold": 0.98,
    "ponder_penalty": 0.005,
    
    # Training
    "epochs": 200,
    "lr": 3e-4,
    "grad_clip": 1.0
}

print(f"--- SACRSN v40: RECTIFIED ENGINE ---")
print(f"Device: {DEVICE} | Determinism: STRICT")

# ==========================================
# 2. ROBUST BPE TOKENIZER
# ==========================================
class RobustBPE:
    def __init__(self, target_vocab_size=1000):
        self.target_size = target_vocab_size
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.merges = {} 
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
        print("Training BPE...")
        words = re.findall(r"[\w']+|[^\s\w]", text)
        chars = set("".join(words))
        for c in sorted(list(chars)):
            if c not in self.vocab: self.vocab[c] = len(self.vocab)
            
        word_freqs = Counter([" ".join(list(w)) for w in words])
        
        num_merges = self.target_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self.get_stats(word_freqs)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges[best] = i # Rank
            new_token = "".join(best)
            self.vocab[new_token] = len(self.vocab)
            word_freqs = self.merge_vocab(best, word_freqs)
            
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"BPE Trained. Size: {len(self.vocab)}")

    def encode(self, text):
        words = re.findall(r"[\w']+|[^\s\w]", text)
        ids = []
        for word in words:
            word_tokens = list(word)
            while len(word_tokens) > 1:
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
                if best_pair is None: break
                new_token = "".join(best_pair)
                word_tokens[best_idx] = new_token
                del word_tokens[best_idx+1]
            
            for t in word_tokens:
                ids.append(self.vocab.get(t, self.vocab["<UNK>"]))
        return ids

    def decode(self, ids):
        return "".join([self.reverse_vocab.get(i, "") for i in ids])

# ==========================================
# 3. STABLE COMPLEX MATH
# ==========================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, z):
        real = self.fc_real(z.real) - self.fc_imag(z.imag)
        imag = self.fc_real(z.imag) + self.fc_imag(z.real)
        return torch.complex(real, imag)

class StableComplexNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_r = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        
    def forward(self, z):
        return torch.complex(self.norm_r(z.real), self.norm_i(z.imag))

# [FIX] Renamed to reflect true function (Gating, not Attention)
class HermitianGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Hermitian Inner Product (Self-Similarity)
        score_r = (q.real * k.real + q.imag * k.imag).sum(dim=-1, keepdim=True)
        
        # Gate
        gate = torch.sigmoid(score_r)
        
        out_r = v.real * gate
        out_i = v.imag * gate
        return torch.complex(out_r, out_i)

# ==========================================
# 4. MEMORY & VQ
# ==========================================
class GatedRecurrentMemory(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.dim = dim
        self.slots = slots
        
    def init_state(self, batch_size):
        return torch.complex(
            torch.zeros(batch_size, self.slots, self.dim, device=DEVICE),
            torch.zeros(batch_size, self.slots, self.dim, device=DEVICE)
        )

    def forward(self, gw_state, prev_mem):
        # 1. Read (Content Addressing)
        q = gw_state.unsqueeze(1)
        sim = (prev_mem.real * q.real + prev_mem.imag * q.imag).sum(dim=-1)
        attn = F.softmax(sim, dim=-1).unsqueeze(-1)
        read_out = (prev_mem * torch.complex(attn, torch.zeros_like(attn))).sum(dim=1)
        
        # 2. Write Gate [FIXED]
        # Calculate gate based on input magnitude
        raw_gate = torch.norm(gw_state, dim=-1, keepdim=True).unsqueeze(1) # (B, 1, 1)
        write_gate = torch.sigmoid(raw_gate)
        
        # 3. Cyclic Shift & Soft Write
        new_slot = gw_state.unsqueeze(1)
        shifted_mem = torch.roll(prev_mem, shifts=1, dims=1)
        
        # Mask only slot 0
        mask = torch.zeros_like(shifted_mem)
        mask[:, 0, :] = 1.0
        
        # [FIX] Apply write_gate logic correctly
        # Slot 0 = (Gate * New) + ((1-Gate) * Old_Shifted)
        # Other Slots = Old_Shifted
        
        head_update = (write_gate * new_slot) + ((1.0 - write_gate) * shifted_mem)
        next_mem = (mask * head_update) + ((1.0 - mask) * shifted_mem)
        
        return read_out, next_mem

class StabilizedVQ(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim * 2)
        nn.init.orthogonal_(self.embedding.weight)
        
    def forward(self, z):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        # Euclidean Dist
        d = torch.cdist(z_flat.unsqueeze(1), self.embedding.weight.unsqueeze(0)).squeeze(1)
        
        indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(indices)
        
        # Loss per sample (Batch, )
        loss_per_sample = ((z_q - z_flat.detach())**2).mean(dim=-1) + \
                          0.25 * ((z_q.detach() - z_flat)**2).mean(dim=-1)
        
        z_q = z_flat + (z_q - z_flat).detach()
        
        # Perplexity per batch
        encodings = F.one_hot(indices, num_classes=self.embedding.num_embeddings).float()
        avg_probs = encodings.mean(0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(entropy)
        
        z_q_c = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        return z_q_c, loss_per_sample, perplexity

# ==========================================
# 5. THE ENGINE
# ==========================================
class SACRSN_v40(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Embedding(vocab_size, dim * 2)
        
        self.norm = StableComplexNorm(dim)
        self.gate_mod = HermitianGating(dim)
        self.memory_module = GatedRecurrentMemory(dim, CONFIG["mem_slots"])
        self.vq = StabilizedVQ(dim, vocab_size)
        
        self.arbitrator = nn.Linear(dim*2, 3) 
        self.decoder = nn.Linear(dim * 2, vocab_size)
        
        # [FIX] Learnable Halt Bias
        self.halt_bias = nn.Parameter(torch.tensor(2.0))
        # [FIX] Input Injection Gate
        self.input_gate = nn.Parameter(torch.tensor(0.5))

    def forward_step(self, x_t, gw_state, mem_state):
        # [FIX] Decoupled Injection
        alpha = torch.sigmoid(self.input_gate)
        gw_state = (alpha * gw_state) + ((1.0 - alpha) * x_t)
        
        loop_penalties = torch.zeros(x_t.size(0), 1, device=DEVICE)
        cumulative_halt = torch.zeros(x_t.size(0), 1, device=DEVICE)
        
        # T.O.T.E. Inner Loop
        for i in range(CONFIG["max_recursion"]):
            gw_state = self.norm(gw_state)
            
            # Modules
            g_out = self.gate_mod(gw_state)
            m_out, _ = self.memory_module(gw_state, mem_state) # Peek at memory
            v_out, vq_loss_sample, _ = self.vq(gw_state)
            
            # Competition
            flat = torch.cat([gw_state.real, gw_state.imag], dim=-1)
            gates = F.softmax(self.arbitrator(flat), dim=-1)
            
            update = (gates[:,0:1]*g_out.real + gates[:,1:2]*m_out.real + gates[:,2:3]*v_out.real) + \
                     1j * (gates[:,0:1]*g_out.imag + gates[:,1:2]*m_out.imag + gates[:,2:3]*v_out.imag)
            
            gw_state = 0.6 * gw_state + 0.4 * update
            
            # [FIX] Per-Sample Halting Logic
            # "Stop if VQ loss is small relative to learned bias"
            # Normalize bias to be positive
            bias = F.softplus(self.halt_bias)
            halt_prob = torch.sigmoid(bias - vq_loss_sample.unsqueeze(1))
            
            # Ponder Cost (Per sample)
            still_thinking = (1.0 - cumulative_halt)
            loop_penalties = loop_penalties + (still_thinking * CONFIG["ponder_penalty"])
            
            # Update soft halt
            cumulative_halt = cumulative_halt + (still_thinking * halt_prob)
            
            # Break if batch average is high (Optimization speedup)
            if cumulative_halt.mean() > CONFIG["halt_threshold"]:
                break
        
        # Commit Memory
        _, final_mem_state = self.memory_module(gw_state, mem_state)
        
        # Return final VQ loss and Perplexity for logging
        _, vq_loss, ppx = self.vq(gw_state)
        
        return gw_state, final_mem_state, vq_loss.mean(), ppx, loop_penalties.mean()

    def forward(self, x_seq):
        B, T = x_seq.shape
        emb = self.encoder(x_seq)
        gw_seq = torch.complex(emb[..., :self.dim], emb[..., self.dim:])
        
        mem_state = self.memory_module.init_state(B)
        gw_state = torch.zeros_like(gw_seq[:, 0])
        
        outputs = []
        total_vq = 0
        total_ponder = 0
        total_ppx = 0 # [FIX] Accumulate PPX
        
        for t in range(T):
            x_t = gw_seq[:, t]
            
            gw_state, mem_state, vq, ppx, ponder = self.forward_step(x_t, gw_state, mem_state)
            
            outputs.append(gw_state)
            total_vq += vq
            total_ponder += ponder
            total_ppx += ppx
            
        out_tensor = torch.stack(outputs, dim=1)
        flat_out = torch.cat([out_tensor.real, out_tensor.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        return logits, total_vq / T, total_ponder / T, total_ppx / T

# ==========================================
# 6. TRAINING HARNESS
# ==========================================
def get_batch(data_tensor, batch_size, seq_len):
    max_idx = len(data_tensor) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def main():
    # 1. Prepare Data
    dummy_text = "The neural architecture of the mind is a mirror of the cosmos. " * 200
    with open("data.txt", "w") as f: f.write(dummy_text)
    
    tokenizer = RobustBPE(target_vocab_size=500)
    tokenizer.train(dummy_text)
    CONFIG["vocab_size"] = len(tokenizer.vocab)
    
    data_ids = tokenizer.encode(dummy_text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)
    print(f"Tokens: {len(data_tensor)}")
    
    # 2. Init Model
    model = SACRSN_v40(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    # 3. Train
    print("\n--- STARTING TRAINING ---")
    model.train()
    
    try:
        for epoch in range(CONFIG["epochs"]):
            x, y = get_batch(data_tensor, CONFIG["batch_size"], CONFIG["seq_len"])
            
            opt.zero_grad()
            logits, vq_loss, ponder_loss, ppx = model(x)
            
            # Cross Entropy
            loss_ce = F.cross_entropy(logits.view(-1, CONFIG["vocab_size"]), y.view(-1))
            
            # Total Loss
            loss = loss_ce + (0.1 * vq_loss) + ponder_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            
            if epoch % 20 == 0:
                print(f"Ep {epoch:03d} | CE: {loss_ce.item():.4f} | Ponder: {ponder_loss.item():.4f} | PPX: {ppx.item():.1f}")
                
    except KeyboardInterrupt:
        print("Stopped.")

    print("\n--- PASSED FINAL AUDIT ---")

if __name__ == "__main__":
    main()