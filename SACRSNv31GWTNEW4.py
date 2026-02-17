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
# 1. STRICT DETERMINISM & CONFIG
# ==========================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True) # Uncomment if hardware supports it strictly

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "seq_len": 32,
    "batch_size": 16,
    "embed_dim": 64,        # Complex dim (Real=64, Imag=64)
    "mem_slots": 32,
    "vocab_size": None,     # Auto-set
    
    # Adaptive Dynamics
    "max_recursion": 8,
    "halt_bias_init": 3.0,
    "ponder_cost": 0.01,
    
    # Regularization
    "entropy_weight": 0.05, 
    "phase_reg": 0.01,
    
    # Training & Memory
    "epochs": 300,
    "lr": 3e-4,
    "grad_clip": 1.0,
    "replay_size": 200      # Hippocampal Buffer Size
}

print(f"--- SACRSN v42: THE COMPLETE EDITION ---")
print(f"Device: {DEVICE}")

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
        
        for i in range(self.target_size - len(self.vocab)):
            pairs = self.get_stats(word_freqs)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges[best] = i 
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
# 3. CORE ENGINE (v41 Asynchronous)
# ==========================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features)
        self.fc_imag = nn.Linear(in_features, out_features)
    def forward(self, z):
        return torch.complex(self.fc_real(z.real) - self.fc_imag(z.imag),
                             self.fc_real(z.imag) + self.fc_imag(z.real))

class StableComplexNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_r = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
    def forward(self, z):
        return torch.complex(self.norm_r(z.real), self.norm_i(z.imag))

class HermitianGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        score_r = (q.real * k.real + q.imag * k.imag).sum(dim=-1, keepdim=True)
        gate = torch.sigmoid(score_r)
        return torch.complex(v.real * gate, v.imag * gate)

class LearnableGateMemory(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.dim, self.slots = dim, slots
        self.gate_net = nn.Linear(dim * 2, 1)
        
    def init_state(self, batch_size):
        return torch.complex(torch.zeros(batch_size, self.slots, self.dim, device=DEVICE),
                             torch.zeros(batch_size, self.slots, self.dim, device=DEVICE))

    def forward(self, gw_state, prev_mem):
        q = gw_state.unsqueeze(1)
        sim = (prev_mem.real * q.real + prev_mem.imag * q.imag).sum(dim=-1)
        attn = F.softmax(sim, dim=-1).unsqueeze(-1)
        read_out = (prev_mem * torch.complex(attn, torch.zeros_like(attn))).sum(dim=1)
        
        flat_input = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        write_gate = torch.sigmoid(self.gate_net(flat_input)).unsqueeze(1)
        
        new_slot = gw_state.unsqueeze(1)
        shifted_mem = torch.roll(prev_mem, shifts=1, dims=1)
        mask = torch.zeros_like(shifted_mem); mask[:, 0, :] = 1.0
        
        head_update = (write_gate * new_slot) + ((1.0 - write_gate) * shifted_mem)
        next_mem = (mask * head_update) + ((1.0 - mask) * shifted_mem)
        return read_out, next_mem

class EntropyRegularizedVQ(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim * 2)
        nn.init.orthogonal_(self.embedding.weight)
        
    def forward(self, z):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        d = torch.cdist(z_flat.unsqueeze(1), self.embedding.weight.unsqueeze(0)).squeeze(1)
        indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(indices)
        
        loss_sample = ((z_q - z_flat.detach())**2).mean(dim=-1) + 0.25 * ((z_q.detach() - z_flat)**2).mean(dim=-1)
        z_q = z_flat + (z_q - z_flat).detach()
        
        encodings = F.one_hot(indices, num_classes=self.embedding.num_embeddings).float()
        avg_probs = encodings.mean(0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        
        z_q_c = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        return z_q_c, loss_sample, torch.exp(entropy), entropy

# ==========================================
# 4. SACRSN v42 MODEL
# ==========================================
class SACRSN_v42(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Embedding(vocab_size, dim * 2)
        self.norm = StableComplexNorm(dim)
        self.gate_mod = HermitianGating(dim)
        self.memory_module = LearnableGateMemory(dim, CONFIG["mem_slots"])
        self.vq = EntropyRegularizedVQ(dim, vocab_size)
        self.arbitrator = nn.Linear(dim*2, 3) 
        self.decoder = nn.Linear(dim * 2, vocab_size)
        self.halt_bias = nn.Parameter(torch.tensor(CONFIG["halt_bias_init"]))
        self.input_gate = nn.Parameter(torch.tensor(0.0))

    def forward_step(self, x_t, gw_state, mem_state):
        alpha = torch.sigmoid(self.input_gate)
        gw_state = (alpha * gw_state) + ((1.0 - alpha) * x_t)
        
        B = x_t.size(0)
        active_mask = torch.ones(B, 1, device=DEVICE)
        
        accum = {k: torch.zeros(B, device=DEVICE) for k in ['vq', 'ppx', 'ent', 'ponder', 'phase']}
        prev_angle = gw_state.angle()
        
        for i in range(CONFIG["max_recursion"]):
            curr_state = self.norm(gw_state)
            v_out, vq_loss_sample, ppx, entropy = self.vq(curr_state)
            g_out = self.gate_mod(curr_state)
            m_out, _ = self.memory_module(curr_state, mem_state)
            
            flat = torch.cat([curr_state.real, curr_state.imag], dim=-1)
            gates = F.softmax(self.arbitrator(flat), dim=-1)
            
            update = (gates[:,0:1]*g_out.real + gates[:,1:2]*m_out.real + gates[:,2:3]*v_out.real) + \
                     1j * (gates[:,0:1]*g_out.imag + gates[:,1:2]*m_out.imag + gates[:,2:3]*v_out.imag)
            
            cand_state = 0.6 * curr_state + 0.4 * update
            
            # Phase Loss
            diff = torch.abs(cand_state.angle() - prev_angle)
            diff = torch.min(diff, 2*math.pi - diff)
            accum['phase'] += active_mask.squeeze() * diff.mean(dim=-1)
            prev_angle = cand_state.angle()
            
            # Halting
            bias = F.softplus(self.halt_bias)
            halt_prob = torch.sigmoid(bias - vq_loss_sample)
            should_stop = (halt_prob > 0.5).float().unsqueeze(1)
            
            accum['ponder'] += active_mask.squeeze() * CONFIG["ponder_cost"]
            
            # Update final stats if active
            mask_flat = active_mask.squeeze() > 0.5
            accum['vq'] = torch.where(mask_flat, vq_loss_sample, accum['vq'])
            accum['ppx'] = torch.where(mask_flat, ppx.expand(B), accum['ppx'])
            accum['ent'] = torch.where(mask_flat, entropy.expand(B), accum['ent'])
            
            # Update State
            gw_real = torch.where(active_mask > 0.5, cand_state.real, gw_state.real)
            gw_imag = torch.where(active_mask > 0.5, cand_state.imag, gw_state.imag)
            gw_state = torch.complex(gw_real, gw_imag)
            
            active_mask = active_mask * (1.0 - should_stop)
            if active_mask.sum() == 0: break
                
        _, final_mem = self.memory_module(gw_state, mem_state)
        return gw_state, final_mem, accum

    def forward(self, x_seq, mem_state=None):
        B, T = x_seq.shape
        emb = self.encoder(x_seq)
        gw_seq = torch.complex(emb[..., :self.dim], emb[..., self.dim:])
        
        if mem_state is None:
            mem_state = self.memory_module.init_state(B)
        gw_state = torch.zeros_like(gw_seq[:, 0])
        
        outputs = []
        stats = {k: 0 for k in ['vq', 'ppx', 'ent', 'ponder', 'phase']}
        
        for t in range(T):
            x_t = gw_seq[:, t]
            gw_state, mem_state, step_stats = self.forward_step(x_t, gw_state, mem_state)
            outputs.append(gw_state)
            for k in stats: stats[k] += step_stats[k].mean()
            
        out_tensor = torch.stack(outputs, dim=1)
        flat_out = torch.cat([out_tensor.real, out_tensor.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        for k in stats: stats[k] /= T
        return logits, stats, mem_state

# ==========================================
# 5. RESTORED: HIPPOCAMPAL BUFFER
# ==========================================
class HippocampalBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [] # (x, y, loss)
        
    def add(self, x, y, loss):
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y, loss))
        else:
            # Replace easiest memory (lowest loss) with this new hard one
            self.buffer.sort(key=lambda k: k[2])
            if loss > self.buffer[0][2]:
                self.buffer[0] = (x, y, loss)
    
    def sample(self, batch_size):
        if not self.buffer: return None
        k = min(len(self.buffer), batch_size)
        return random.sample(self.buffer, k)

# ==========================================
# 6. RESTORED: GENERATION & VISUALIZATION
# ==========================================
def generate_text(model, tokenizer, seed_text="The", length=50, temp=0.8):
    model.eval()
    ids = tokenizer.encode(seed_text)
    curr_seq = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    # Init memory
    mem_state = model.memory_module.init_state(1)
    
    print(f"\n--- GENERATING (Seed: '{seed_text}') ---")
    generated = seed_text
    
    with torch.no_grad():
        # Prime the memory with the seed
        _, _, mem_state = model(curr_seq, mem_state)
        
        # Last token to start generation
        curr_token = curr_seq[:, -1:] 
        
        for _ in range(length):
            logits, _, mem_state = model(curr_token, mem_state)
            
            # Temperature Sampling
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            next_id = torch.multinomial(probs, 1)
            
            token_str = tokenizer.decode([next_id.item()])
            generated += token_str
            curr_token = next_id
            
            # Print stream
            print(token_str, end="", flush=True)
            
    print("\n\n")
    return generated

def visualize_brain(history):
    print("--- VISUALIZING TRAINING DYNAMICS ---")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Total Loss', alpha=0.7)
    plt.plot(history['ponder'], label='Ponder Cost', alpha=0.7)
    plt.title("Cognitive Effort over Time")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['ppx'], color='green')
    plt.title("Codebook Perplexity (Symbol Usage)")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 7. MAIN LOOP
# ==========================================
def get_batch(data_tensor, batch_size, seq_len):
    max_idx = len(data_tensor) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def main():
    # 1. Prepare Data
    dummy_text = "The neural architecture of the mind is a mirror of the cosmos. As above, so below. " * 200
    with open("data.txt", "w") as f: f.write(dummy_text)
    
    tokenizer = RobustBPE(target_vocab_size=500)
    tokenizer.train(dummy_text)
    CONFIG["vocab_size"] = len(tokenizer.vocab)
    
    data_ids = tokenizer.encode(dummy_text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)
    
    # 2. Init
    model = SACRSN_v42(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    hippocampus = HippocampalBuffer(CONFIG["replay_size"])
    
    history = defaultdict(list)
    
    print("\n--- STARTING TRAINING ---")
    model.train()
    
    try:
        for epoch in range(CONFIG["epochs"]):
            # A. Waking Phase
            x, y = get_batch(data_tensor, CONFIG["batch_size"], CONFIG["seq_len"])
            
            opt.zero_grad()
            logits, stats, _ = model(x)
            
            loss_ce = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
            loss = loss_ce + 0.1*stats['vq'] + stats['ponder'] + CONFIG['phase_reg']*stats['phase'] - CONFIG['entropy_weight']*stats['ent']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            
            # Store in Hippocampus
            hippocampus.add(x.detach().cpu(), y.detach().cpu(), loss.item())
            
            # B. Sleep Phase (Replay)
            if epoch > 0 and epoch % 50 == 0:
                dream_data = hippocampus.sample(4)
                if dream_data:
                    for dx, dy, _ in dream_data:
                        dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                        opt.zero_grad()
                        d_logits, d_stats, _ = model(dx)
                        d_loss = F.cross_entropy(d_logits.reshape(-1, CONFIG["vocab_size"]), dy.reshape(-1))
                        d_loss.backward()
                        opt.step()
            
            # Log
            history['loss'].append(loss.item())
            history['ponder'].append(stats['ponder'].item())
            history['ppx'].append(stats['ppx'].item())
            
            if epoch % 20 == 0:
                print(f"Ep {epoch:03d} | CE: {loss_ce.item():.4f} | Ponder: {stats['ponder'].item():.4f} | PPX: {stats['ppx'].item():.1f}")
                
    except KeyboardInterrupt:
        print("Stopped.")

    # 3. Final Outputs
    visualize_brain(history)
    generate_text(model, tokenizer, seed_text="The", length=100)

if __name__ == "__main__":
    main()