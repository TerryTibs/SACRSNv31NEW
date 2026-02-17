import os
import re
import math
import time
import random
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter

# ==========================================
# 0. STRICT DETERMINISM & PRECISION
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

try:
    torch.set_float32_matmul_precision('high')
except:
    pass

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # Architecture
    "seq_len": 32,
    "batch_size": 16,
    "embed_dim": 64,        # Complex dim (Real=64, Imag=64)
    "mem_slots": 32,
    "vocab_size": None,     # Auto-set
    
    # [UPGRADE] Semantic Compression
    "codebook_size": 128,   # Forces abstraction (Vocab ~1000 -> 128 Concepts)
    
    # Cognitive Dynamics
    "max_recursion": 8,     
    "halt_bias_init": 3.0,
    "halt_scale_init": 10.0,
    "ponder_cost": 0.01,
    
    # Regularization
    "entropy_weight": 0.05, 
    "phase_reg": 0.01,
    
    # Training
    "epochs": 300,
    "lr": 5e-4,
    "grad_clip": 1.0,
    "replay_size": 200,
    "active_plasticity": True,
    "surprise_decay": 0.9
}

print(f"--- SACRSN v50: ABSTRACTION EDITION ---")
print(f"Device: {DEVICE}")

# ==========================================
# 1. DATA & BPE
# ==========================================
def load_training_data(filepath="data.txt"):
    FALLBACK_TEXT = """The neural architecture of the mind is a mirror of the cosmos itself. As above, so below; the filamentary structures of the intergalactic web find their precise echo in the dense, white matter connecting the hemispheres of the brain. Galaxies cluster like neurons in a cosmic synapse, and the voids between them echo the silence between thought. We are stardust contemplating its own arrangement, a fleeting arrangement of atoms attempting to comprehend the laws that bound them together. To understand the nature of thought, one must first understand the nature of the void. It is the negative space that defines the positive, the silence that gives shape to the sound. In the absolute zero of the vacuum, potential energy waits, just as a thought waits on the precipice of expression. Logic is the foundation, but chaos is the architect. Without the rigid framework of logic, the structure collapses; without the unpredictability of chaos, the structure creates nothing new. Entropy is not the enemy of intelligence, but its fuel—the friction that generates the heat of creativity."""
    
    if os.path.exists(filepath):
        print(f"Loading data from '{filepath}'...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
        if len(data) < 50: 
            print(">> File too short. Using internal fallback corpus.")
            data = FALLBACK_TEXT
    else:
        print(f">> '{filepath}' not found. Creating with Default Philosophical Corpus.")
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
        print(f"BPE Trained. Final Size: {len(self.vocab)}")

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
# 2. MATH & MODULES
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

class AssociativeMemory(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.dim, self.slots = dim, slots
        self.gate_net = nn.Linear(dim * 2, 1)
        self.address_net = nn.Linear(dim * 2, slots)
        self.mem_norm = StableComplexNorm(dim)
        
    def init_state(self, batch_size):
        return torch.complex(torch.zeros(batch_size, self.slots, self.dim, device=DEVICE),
                             torch.zeros(batch_size, self.slots, self.dim, device=DEVICE))

    def forward(self, gw_state, prev_mem):
        # Read
        q = gw_state.unsqueeze(1)
        sim = (prev_mem.real * q.real + prev_mem.imag * q.imag).sum(dim=-1)
        attn = F.softmax(sim, dim=-1).unsqueeze(-1)
        read_out = (prev_mem * torch.complex(attn, torch.zeros_like(attn))).sum(dim=1)
        
        # Write
        flat_input = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        write_gate = torch.sigmoid(self.gate_net(flat_input)).unsqueeze(1) 
        write_weights = F.softmax(self.address_net(flat_input), dim=-1).unsqueeze(-1)
        
        new_slot = gw_state.unsqueeze(1)
        effective_update = write_gate * write_weights
        
        # Update & Normalize
        next_mem = (1.0 - effective_update) * prev_mem + (effective_update * new_slot)
        next_mem = self.mem_norm(next_mem)
        
        return read_out, next_mem

class EntropyRegularizedVQ(nn.Module):
    def __init__(self, dim, num_concepts):
        # [UPGRADE] Decoupled VQ Size (num_concepts)
        super().__init__()
        self.embedding = nn.Embedding(num_concepts, dim * 2)
        nn.init.orthogonal_(self.embedding.weight)
        
    def forward(self, z):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        # Optimized Distance
        x_sq = torch.sum(z_flat**2, dim=1, keepdim=True)
        y_sq = torch.sum(self.embedding.weight**2, dim=1)
        d = x_sq + y_sq - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        
        indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(indices)
        
        loss_sample = ((z_q - z_flat.detach())**2).mean(dim=-1) + 0.25 * ((z_q.detach() - z_flat)**2).mean(dim=-1)
        z_q = z_flat + (z_q - z_flat).detach()
        
        # Semantic Diversity (Abstract Concept Entropy)
        encodings = F.one_hot(indices, num_classes=self.embedding.num_embeddings).float()
        avg_probs = encodings.mean(0)
        batch_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        
        z_q_c = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        return z_q_c, loss_sample, indices, batch_entropy

# ==========================================
# 3. SACRSN MODEL
# ==========================================
class SACRSN_v50(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Embedding(vocab_size, dim * 2)
        self.norm = StableComplexNorm(dim)
        self.gate_mod = HermitianGating(dim)
        self.memory_module = AssociativeMemory(dim, CONFIG["mem_slots"])
        
        # [UPGRADE] Pass explicit Codebook Size
        self.vq = EntropyRegularizedVQ(dim, CONFIG["codebook_size"])
        
        self.arbitrator = nn.Linear(dim*2, 3) 
        self.decoder = nn.Linear(dim * 2, vocab_size)
        
        self.halt_bias = nn.Parameter(torch.tensor(CONFIG["halt_bias_init"]))
        self.halt_scale = nn.Parameter(torch.tensor(CONFIG["halt_scale_init"]))
        self.input_gate = nn.Parameter(torch.tensor(0.0))

    def forward_step(self, x_t, gw_state, mem_state):
        alpha = torch.sigmoid(self.input_gate)
        gw_state = (alpha * gw_state) + ((1.0 - alpha) * x_t)
        
        B = x_t.size(0)
        active_mask = torch.ones(B, 1, device=DEVICE)
        
        accum = {k: torch.zeros(B, device=DEVICE) for k in ['vq', 'ponder', 'phase']}
        total_batch_entropy = 0 
        steps_taken = 0
        total_steps_batch = torch.zeros(B, device=DEVICE)
        
        final_indices = torch.zeros(B, dtype=torch.long, device=DEVICE)
        prev_angle = gw_state.angle()
        
        for i in range(CONFIG["max_recursion"]):
            steps_taken += 1
            curr_state = self.norm(gw_state)
            
            v_out, vq_loss_sample, indices, batch_ent = self.vq(curr_state)
            
            if active_mask.sum() > 0:
                total_batch_entropy += batch_ent
            
            g_out = self.gate_mod(curr_state)
            m_out, cand_mem_state = self.memory_module(curr_state, mem_state)
            
            flat = torch.cat([curr_state.real, curr_state.imag], dim=-1)
            gates = F.softmax(self.arbitrator(flat), dim=-1)
            
            update_real = (gates[:,0:1]*g_out.real + gates[:,1:2]*m_out.real + gates[:,2:3]*v_out.real)
            update_imag = (gates[:,0:1]*g_out.imag + gates[:,1:2]*m_out.imag + gates[:,2:3]*v_out.imag)
            update = torch.complex(update_real, update_imag)
            
            cand_state = 0.6 * curr_state + 0.4 * update
            
            diff = torch.abs(cand_state.angle() - prev_angle)
            diff = torch.min(diff, 2*math.pi - diff)
            accum['phase'] += active_mask.squeeze() * diff.mean(dim=-1)
            prev_angle = cand_state.angle()
            
            bias = F.softplus(self.halt_bias)
            scale = F.softplus(self.halt_scale)
            halt_prob = torch.sigmoid(bias - (scale * vq_loss_sample))
            
            # STE Halting
            hard_stop = (halt_prob > 0.5).float()
            should_stop = (hard_stop - halt_prob.detach() + halt_prob).unsqueeze(1)
            
            accum['ponder'] += active_mask.squeeze() * CONFIG["ponder_cost"]
            total_steps_batch += active_mask.squeeze()
            accum['vq'] += (active_mask.squeeze() * vq_loss_sample)
            
            mask_flat = active_mask.squeeze() > 0.5
            final_indices = torch.where(mask_flat, indices, final_indices)
            
            gw_real = torch.where(active_mask > 0.5, cand_state.real, gw_state.real)
            gw_imag = torch.where(active_mask > 0.5, cand_state.imag, gw_state.imag)
            gw_state = torch.complex(gw_real, gw_imag)
            
            mem_mask = active_mask.unsqueeze(-1)
            mem_real = torch.where(mem_mask > 0.5, cand_mem_state.real, mem_state.real)
            mem_imag = torch.where(mem_mask > 0.5, cand_mem_state.imag, mem_state.imag)
            mem_state = torch.complex(mem_real, mem_imag)
            
            active_mask = active_mask * (1.0 - should_stop)
            if active_mask.sum() == 0: break
        
        accum['vq'] = accum['vq'] / torch.max(total_steps_batch, torch.ones_like(total_steps_batch))
        avg_batch_ent = total_batch_entropy / max(1, steps_taken)
        avg_depth = total_steps_batch.mean()
        
        return gw_state, mem_state, accum, final_indices, avg_batch_ent, avg_depth

    def forward(self, x_seq, mem_state=None):
        B, T = x_seq.shape
        emb = self.encoder(x_seq)
        gw_seq = torch.complex(emb[..., :self.dim], emb[..., self.dim:])
        
        if mem_state is None: mem_state = self.memory_module.init_state(B)
        gw_state = torch.zeros_like(gw_seq[:, 0])
        
        outputs, all_indices = [], []
        stats = {k: 0 for k in ['vq', 'ponder', 'phase', 'ent', 'depth']}
        
        for t in range(T):
            x_t = gw_seq[:, t]
            gw_state, mem_state, step_stats, indices, batch_ent, depth = self.forward_step(x_t, gw_state, mem_state)
            outputs.append(gw_state)
            all_indices.append(indices)
            for k in ['vq', 'ponder', 'phase']: stats[k] += step_stats[k].mean()
            stats['ent'] += batch_ent
            stats['depth'] += depth
            
        out_tensor = torch.stack(outputs, dim=1)
        flat_out = torch.cat([out_tensor.real, out_tensor.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        for k in stats: stats[k] /= T
        
        return logits, stats, mem_state, torch.stack(all_indices, dim=1)

# ==========================================
# 4. HIPPOCAMPUS & UTILS
# ==========================================
class HippocampalBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [] 
    def add(self, x, y, loss):
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, (loss, x, y))
        else:
            if loss > self.buffer[0][0]:
                heapq.heapreplace(self.buffer, (loss, x, y))
    def sample(self, batch_size):
        if not self.buffer: return None
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

def get_batch(data_tensor, batch_size, seq_len):
    max_idx = len(data_tensor) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ==========================================
# 5. DIAGNOSTICS & VISUALIZATION
# ==========================================
def visualize_suite(model, history, transitions):
    print("\n--- RUNNING DIAGNOSTICS SUITE ---")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Total Loss', linewidth=1.5)
    plt.plot(history['ponder'], label='Ponder Cost', alpha=0.7)
    plt.title("Cognitive Effort"); plt.legend(); plt.grid(True, alpha=0.3)
    
    # [UPGRADE] Correct Dummy Input Logic
    plt.subplot(1, 3, 2)
    with torch.no_grad():
        w = model.decoder.weight.detach().cpu().numpy()
        plt.scatter(w[:, :64].mean(0), w[:, 64:].mean(0), alpha=0.2, c='gray', s=5)
        
        # Valid thought simulation: Empty Mind + Zero Input
        gw_dummy = torch.zeros(1, 64, dtype=torch.complex64, device=DEVICE)
        x_dummy = torch.zeros(1, 64, dtype=torch.complex64, device=DEVICE)
        mem_trace = model.memory_module.init_state(1)
        
        traj_real, traj_imag = [], []
        for _ in range(20):
            gw_dummy, _, _, _, _, _ = model.forward_step(x_dummy, gw_dummy, mem_trace)
            traj_real.append(gw_dummy.real.mean().item())
            traj_imag.append(gw_dummy.imag.mean().item())
            
        plt.plot(traj_real, traj_imag, 'r-', linewidth=2, label='Null Thought Trace')
    plt.title("Dynamic Phase Space"); plt.xlabel("Real"); plt.ylabel("Imag"); plt.legend()

    plt.subplot(1, 3, 3)
    G = nx.DiGraph()
    for (u, v), w in Counter(transitions).most_common(50): G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, node_size=20, width=0.5, alpha=0.7, edge_color='teal')
    plt.title(f"Abstract Associations (Vocab:{CONFIG['vocab_size']}->{CONFIG['codebook_size']})")
    plt.tight_layout(); plt.savefig('sacrsn_diagnostics.png'); plt.show()

def extract_logic_rules(transitions, tokenizer):
    print("\n--- EXTRACTED CONCEPT RULES ---")
    print(f"{'FROM':<8} -> {'TO':<8} | {'STRENGTH':<10}")
    print("-" * 35)
    # [UPGRADE] Labels now reflect abstract concepts, not symbols
    for (u, v), w in Counter(transitions).most_common(10):
        print(f"Concept_{u:<3} -> Concept_{v:<3} | {w:<10}")

def anomaly_detector(model, tokenizer):
    print("\n--- ANOMALY DETECTOR ---")
    normal = "The structure of the mind."
    weird = "The banana of the galaxy eats time."
    def score(t):
        ids = tokenizer.encode(t)
        x = torch.tensor([ids[:-1]], device=DEVICE)
        with torch.no_grad():
            logits, _, _, _ = model(x)
            y = torch.tensor([ids[1:]], device=DEVICE)
            return F.cross_entropy(logits.reshape(-1, len(tokenizer.vocab)), y.reshape(-1)).item()
    s1, s2 = score(normal), score(weird)
    print(f"Normal: {s1:.4f} | Weird: {s2:.4f}")
    if s2 > s1: print(">> Anomaly Detected.")

def generate_text(model, tokenizer, seed="The", length=50):
    print(f"\n--- GENERATING (Seed: '{seed}') ---")
    model.eval()
    ids = tokenizer.encode(seed)
    curr = torch.tensor([ids], device=DEVICE)
    mem = None; text = seed
    
    with torch.no_grad():
        _, _, mem, _ = model(curr, mem)
        curr = curr[:, -1:]
        for _ in range(length):
            logits, _, mem, _ = model(curr, mem)
            next_id = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            word = tokenizer.decode([next_id.item()])
            text += word; curr = next_id
            print(word, end="", flush=True)
    print("\n")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    data_text = load_training_data("data.txt")
    tokenizer = RobustBPE(target_vocab_size=1000)
    tokenizer.train(data_text)
    CONFIG["vocab_size"] = len(tokenizer.vocab)
    
    data_ids = tokenizer.encode(data_text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)
    print(f"Tokens: {len(data_tensor)}")

    # [UPGRADE] Pass vocab_size AND codebook_size implicitly via Config access inside classes
    model = SACRSN_v50(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])
    hippocampus = HippocampalBuffer(CONFIG["replay_size"])
    
    history, transitions = defaultdict(list), []
    avg_surprise = 2.0 
    
    print("\n--- STARTING TRAINING ---")
    model.train()
    try:
        for epoch in range(CONFIG["epochs"]):
            x, y = get_batch(data_tensor, CONFIG["batch_size"], CONFIG["seq_len"])
            opt.zero_grad()
            
            logits, stats, mem, indices = model(x)
            
            idx_np = indices.detach().cpu().numpy()
            for b in range(idx_np.shape[0]):
                for t in range(idx_np.shape[1]-1):
                    transitions.append((idx_np[b,t], idx_np[b,t+1]))
            
            loss_ce = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
            
            plast = 1.0
            if CONFIG["active_plasticity"]:
                avg_surprise = CONFIG["surprise_decay"]*avg_surprise + (1-CONFIG["surprise_decay"])*loss_ce.item()
                plast = min(3.0, 1.0 + avg_surprise*0.5)
            
            loss = (loss_ce * plast) + 0.1*stats['vq'] + stats['ponder'] + \
                   CONFIG['phase_reg']*stats['phase'] - CONFIG['entropy_weight']*stats['ent']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            
            hippocampus.add(x.detach().cpu(), y.detach().cpu(), loss.item())
            
            if epoch > 0 and epoch % 50 == 0:
                dream = hippocampus.sample(4)
                if dream:
                    for _, dx, dy in dream:
                        dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                        opt.zero_grad()
                        d_logits, _, _, _ = model(dx)
                        F.cross_entropy(d_logits.reshape(-1, CONFIG["vocab_size"]), dy.reshape(-1)).backward()
                        opt.step()
            
            scheduler.step()
            
            history['loss'].append(loss.item())
            history['ponder'].append(stats['ponder'].item())
            
            if epoch % 20 == 0:
                lr = opt.param_groups[0]['lr']
                text_ppx = math.exp(loss_ce.item())
                code_ppx = math.exp(stats['ent'].item())
                print(f"Ep {epoch:03d} | Loss: {loss.item():.3f} | TextPPX: {text_ppx:.1f} | CodePPX: {code_ppx:.1f} | Depth: {stats['depth'].item():.2f} | Plast: {plast:.2f} | LR: {lr:.1e}")

    except KeyboardInterrupt: print("Stopped.")
    
    generate_text(model, tokenizer, length=80)
    visualize_suite(model, history, transitions)
    extract_logic_rules(transitions, tokenizer)
    anomaly_detector(model, tokenizer)

if __name__ == "__main__":
    main()