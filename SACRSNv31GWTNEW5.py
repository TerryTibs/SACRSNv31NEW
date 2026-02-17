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
import networkx as nx
from collections import defaultdict, Counter

# ==========================================
# 0. STRICT DETERMINISM
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # Architecture
    "seq_len": 32,
    "batch_size": 16,
    "embed_dim": 64,        # Complex dim (Real=64, Imag=64)
    "mem_slots": 32,
    "vocab_size": None,     # Auto-set
    
    # Cognitive Dynamics
    "max_recursion": 8,
    "halt_bias_init": 3.0,
    "ponder_cost": 0.01,
    
    # Regularization & Learning
    "entropy_weight": 0.05, 
    "phase_reg": 0.01,
    "epochs": 300,
    "lr": 3e-4,
    "grad_clip": 1.0,
    "replay_size": 200,     # Hippocampal Buffer
    "active_plasticity": True # V38.5 Dopamine Mechanism
}

print(f"--- SACRSN v43: THE GRAND SYNTHESIS ---")
print(f"Device: {DEVICE}")

# ==========================================
# 1. DATA LOADING & BPE (Restored)
# ==========================================
def load_training_data(filepath="data.txt"):
    # The V38.5 Philosophical Corpus
    FALLBACK_TEXT = """The neural architecture of the mind is a mirror of the cosmos itself. As above, so below; the filamentary structures of the intergalactic web find their precise echo in the dense, white matter connecting the hemispheres of the brain. Galaxies cluster like neurons in a cosmic synapse, and the voids between them echo the silence between thought. We are stardust contemplating its own arrangement, a fleeting arrangement of atoms attempting to comprehend the laws that bound them together. To understand the nature of thought, one must first understand the nature of the void. It is the negative space that defines the positive, the silence that gives shape to the sound. In the absolute zero of the vacuum, potential energy waits, just as a thought waits on the precipice of expression. Logic is the foundation, but chaos is the architect. Without the rigid framework of logic, the structure collapses; without the unpredictability of chaos, the structure creates nothing new. Entropy is not the enemy of intelligence, but its fuel—the friction that generates the heat of creativity."""
    
    if os.path.exists(filepath):
        print(f"Loading data from '{filepath}'...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
        if len(data) < 10: # Safety check for empty files
            print("File too short. Using fallback data.")
            data = FALLBACK_TEXT
    else:
        print(f"'{filepath}' not found. Creating with Default Philosophical Corpus.")
        with open(filepath, "w") as f: f.write(FALLBACK_TEXT)
        data = FALLBACK_TEXT

    return data

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
        print("Training BPE Tokenizer...")
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
# 2. CORE MATH & MODULES (V42 Engine)
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
        return z_q_c, loss_sample, indices, entropy

# ==========================================
# 3. SACRSN MODEL DEFINITION
# ==========================================
class SACRSN_v43(nn.Module):
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
        
        accum = {k: torch.zeros(B, device=DEVICE) for k in ['vq', 'ent', 'ponder', 'phase']}
        final_indices = torch.zeros(B, dtype=torch.long, device=DEVICE)
        prev_angle = gw_state.angle()
        
        for i in range(CONFIG["max_recursion"]):
            curr_state = self.norm(gw_state)
            v_out, vq_loss_sample, indices, entropy = self.vq(curr_state)
            g_out = self.gate_mod(curr_state)
            m_out, _ = self.memory_module(curr_state, mem_state)
            
            flat = torch.cat([curr_state.real, curr_state.imag], dim=-1)
            gates = F.softmax(self.arbitrator(flat), dim=-1)
            
            update = (gates[:,0:1]*g_out.real + gates[:,1:2]*m_out.real + gates[:,2:3]*v_out.real) + \
                     1j * (gates[:,0:1]*g_out.imag + gates[:,1:2]*m_out.imag + gates[:,2:3]*v_out.imag)
            
            cand_state = 0.6 * curr_state + 0.4 * update
            
            diff = torch.abs(cand_state.angle() - prev_angle)
            diff = torch.min(diff, 2*math.pi - diff)
            accum['phase'] += active_mask.squeeze() * diff.mean(dim=-1)
            prev_angle = cand_state.angle()
            
            bias = F.softplus(self.halt_bias)
            halt_prob = torch.sigmoid(bias - vq_loss_sample)
            should_stop = (halt_prob > 0.5).float().unsqueeze(1)
            
            accum['ponder'] += active_mask.squeeze() * CONFIG["ponder_cost"]
            
            mask_flat = active_mask.squeeze() > 0.5
            accum['vq'] = torch.where(mask_flat, vq_loss_sample, accum['vq'])
            accum['ent'] = torch.where(mask_flat, entropy.expand(B), accum['ent'])
            final_indices = torch.where(mask_flat, indices, final_indices)
            
            gw_real = torch.where(active_mask > 0.5, cand_state.real, gw_state.real)
            gw_imag = torch.where(active_mask > 0.5, cand_state.imag, gw_state.imag)
            gw_state = torch.complex(gw_real, gw_imag)
            
            active_mask = active_mask * (1.0 - should_stop)
            if active_mask.sum() == 0: break
                
        _, final_mem = self.memory_module(gw_state, mem_state)
        return gw_state, final_mem, accum, final_indices

    def forward(self, x_seq, mem_state=None):
        B, T = x_seq.shape
        emb = self.encoder(x_seq)
        gw_seq = torch.complex(emb[..., :self.dim], emb[..., self.dim:])
        
        if mem_state is None:
            mem_state = self.memory_module.init_state(B)
        gw_state = torch.zeros_like(gw_seq[:, 0])
        
        outputs = []
        all_indices = []
        stats = {k: 0 for k in ['vq', 'ent', 'ponder', 'phase']}
        
        for t in range(T):
            x_t = gw_seq[:, t]
            gw_state, mem_state, step_stats, indices = self.forward_step(x_t, gw_state, mem_state)
            outputs.append(gw_state)
            all_indices.append(indices)
            for k in stats: stats[k] += step_stats[k].mean()
            
        out_tensor = torch.stack(outputs, dim=1)
        flat_out = torch.cat([out_tensor.real, out_tensor.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        for k in stats: stats[k] /= T
        return logits, stats, torch.stack(all_indices, dim=1)

# ==========================================
# 4. HIPPOCAMPUS & DIAGNOSTICS
# ==========================================
class HippocampalBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [] 
    def add(self, x, y, loss):
        if len(self.buffer) < self.capacity: self.buffer.append((x, y, loss))
        else:
            self.buffer.sort(key=lambda k: k[2])
            if loss > self.buffer[0][2]: self.buffer[0] = (x, y, loss)
    def sample(self, batch_size):
        if not self.buffer: return None
        k = min(len(self.buffer), batch_size)
        return random.sample(self.buffer, k)

def get_batch(data_tensor, batch_size, seq_len):
    max_idx = len(data_tensor) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- VISUALIZATION SUITE (Restored from v37.1) ---
def visualize_suite(model, history, transitions):
    print("\n--- RUNNING DIAGNOSTICS SUITE ---")
    
    # 1. Training Dynamics
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Loss (Surprise)')
    plt.plot(history['ponder'], label='Ponder Cost')
    plt.title("Cognitive Effort")
    plt.legend()
    
    # 2. Phase Space (Real vs Imag)
    # We grab the last state from the model
    dummy_x = torch.tensor([[0]*10], device=DEVICE)
    with torch.no_grad():
        logits, _, _ = model(dummy_x)
    # Extract weights from first layer of decoder to visualize state space
    w = model.decoder.weight.detach().cpu().numpy()
    plt.subplot(1, 3, 2)
    plt.scatter(w[:, :64].mean(0), w[:, 64:].mean(0), alpha=0.6, c='purple')
    plt.title("Phase Space (Memory Topography)")
    plt.xlabel("Real"); plt.ylabel("Imag")
    
    # 3. Semantic Topology (NetworkX)
    plt.subplot(1, 3, 3)
    G = nx.DiGraph()
    # transitions is a list of (from, to) indices
    counts = Counter(transitions)
    most_common = counts.most_common(50)
    for (u, v), w in most_common:
        G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, node_size=20, width=0.5, alpha=0.7, edge_color='teal')
    plt.title("Semantic Associations (Top 50)")
    
    plt.tight_layout()
    plt.show()

def extract_logic_rules(transitions, tokenizer):
    print("\n--- EXTRACTED LOGIC RULES (Codebook Transitions) ---")
    counts = Counter(transitions)
    most_common = counts.most_common(10)
    
    # Since VQ indices are abstract, we just show the abstract rule strength
    print(f"{'FROM':<6} -> {'TO':<6} | {'STRENGTH':<10}")
    print("-" * 30)
    for (u, v), w in most_common:
        print(f"Sym_{u:<4} -> Sym_{v:<4} | {w:<10}")

def anomaly_detector(model, tokenizer):
    print("\n--- ANOMALY DETECTOR ---")
    normal = "The structure of the mind."
    weird = "The banana of the galaxy eats time."
    
    def get_surprise(text):
        ids = tokenizer.encode(text)
        x = torch.tensor([ids[:-1]], device=DEVICE)
        y = torch.tensor([ids[1:]], device=DEVICE)
        with torch.no_grad():
            logits, stats, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, len(tokenizer.vocab)), y.reshape(-1))
        return loss.item()
    
    s1 = get_surprise(normal)
    s2 = get_surprise(weird)
    print(f"Normal Phrase: '{normal}' -> Surprise: {s1:.4f}")
    print(f"Weird Phrase:  '{weird}' -> Surprise: {s2:.4f}")
    if s2 > s1: print(">> Anomaly Correctly Detected.")
    else: print(">> Anomaly Not Detected (Model is confused).")

def generate_text(model, tokenizer, seed="The", length=50):
    print(f"\n--- GENERATING (Seed: '{seed}') ---")
    model.eval()
    ids = tokenizer.encode(seed)
    curr = torch.tensor([ids], device=DEVICE)
    mem = None
    text = seed
    
    with torch.no_grad():
        _, _, mem = model(curr, mem)
        curr = curr[:, -1:]
        for _ in range(length):
            logits, _, mem = model(curr, mem)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            word = tokenizer.decode([next_id.item()])
            text += word
            curr = next_id
            print(word, end="", flush=True)
    print("\n")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    # 1. Load Data
    data_text = load_training_data("data.txt")
    tokenizer = RobustBPE(target_vocab_size=1000)
    tokenizer.train(data_text)
    CONFIG["vocab_size"] = len(tokenizer.vocab)
    
    data_ids = tokenizer.encode(data_text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)
    print(f"Tokens: {len(data_tensor)}")

    # 2. Setup
    model = SACRSN_v43(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    hippocampus = HippocampalBuffer(CONFIG["replay_size"])
    
    history = defaultdict(list)
    transitions = [] # Track VQ jumps
    
    print("\n--- STARTING TRAINING ---")
    model.train()
    
    try:
        for epoch in range(CONFIG["epochs"]):
            # A. Waking Phase
            x, y = get_batch(data_tensor, CONFIG["batch_size"], CONFIG["seq_len"])
            
            opt.zero_grad()
            logits, stats, indices = model(x)
            
            # Record transitions for graph
            idx_np = indices.detach().cpu().numpy()
            for b in range(idx_np.shape[0]):
                for t in range(idx_np.shape[1]-1):
                    transitions.append((idx_np[b,t], idx_np[b,t+1]))
            
            loss_ce = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
            
            # V38.5 Dopamine Plasticity
            plasticity = 1.0
            if CONFIG["active_plasticity"]:
                surprise = loss_ce.item()
                plasticity = min(3.0, 1.0 + (surprise * 0.5))
            
            loss = (loss_ce * plasticity) + \
                   0.1*stats['vq'] + \
                   stats['ponder'] + \
                   CONFIG['phase_reg']*stats['phase'] - \
                   CONFIG['entropy_weight']*stats['ent']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            
            hippocampus.add(x.detach().cpu(), y.detach().cpu(), loss.item())
            
            # B. Sleep Phase
            if epoch > 0 and epoch % 50 == 0:
                dream = hippocampus.sample(4)
                if dream:
                    for dx, dy, _ in dream:
                        dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                        opt.zero_grad()
                        d_logits, _, _ = model(dx)
                        d_loss = F.cross_entropy(d_logits.reshape(-1, CONFIG["vocab_size"]), dy.reshape(-1))
                        d_loss.backward()
                        opt.step()
            
            history['loss'].append(loss.item())
            history['ponder'].append(stats['ponder'].item())
            
            if epoch % 50 == 0:
                print(f"Ep {epoch:03d} | Loss: {loss.item():.4f} | Ponder: {stats['ponder'].item():.4f}")

    except KeyboardInterrupt:
        print("Stopped.")
        
    # 3. The Full Diagnostic Suite
    generate_text(model, tokenizer, length=80)
    visualize_suite(model, history, transitions)
    extract_logic_rules(transitions, tokenizer)
    anomaly_detector(model, tokenizer)

if __name__ == "__main__":
    main()