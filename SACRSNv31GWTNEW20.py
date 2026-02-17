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
from collections import defaultdict, Counter, deque

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
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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
    "embed_dim": 64,
    "mem_slots": 32,
    "vocab_size": None,
    "codebook_size": 128,
    
    # [RESTORED] Sensory Dimensions (VAKOG)
    "n_sensory_anchors": 32, 
    
    # Cognitive Dynamics
    "max_recursion": 8,     
    "halt_bias_init": 3.0,
    "halt_scale_init": 10.0,
    "ponder_cost": 0.01,
    
    # VQ Stability
    "vq_commitment_beta": 0.25,
    "vq_decay": 0.99,
    
    # Regularization
    "entropy_weight": 0.05, 
    "gate_sparsity_weight": 0.01,
    "slot_balance_weight": 0.01,
    "phase_reg": 0.01,
    "sensory_reg": 0.01, # [NEW] Force sensory consistency
    
    # Training
    "epochs": 300,
    "lr": 5e-4,
    "grad_clip": 1.0,
    "replay_size": 200,
    "active_plasticity": True,
    "surprise_decay": 0.9
}

print(f"--- SACRSN v54: EMBODIED EDITION ---")
print(f"Device: {DEVICE}")

# ==========================================
# 1. DATA & BPE
# ==========================================
def load_training_data(filepath="data.txt"):
    FALLBACK_TEXT = """The neural architecture of the mind is a mirror of the cosmos itself. As above, so below; the filamentary structures of the intergalactic web find their precise echo in the dense, white matter connecting the hemispheres of the brain. Galaxies cluster like neurons in a cosmic synapse, and the voids between them echo the silence between thought. We are stardust contemplating its own arrangement, a fleeting arrangement of atoms attempting to comprehend the laws that bound them together. To understand the nature of thought, one must first understand the nature of the void. It is the negative space that defines the positive, the silence that gives shape to the sound. In the absolute zero of the vacuum, potential energy waits, just as a thought waits on the precipice of expression. Logic is the foundation, but chaos is the architect. Without the rigid framework of logic, the structure collapses; without the unpredictability of chaos, the structure creates nothing new. Entropy is not the enemy of intelligence, but its fuel—the friction that generates the heat of creativity."""
    if os.path.exists(filepath):
        print(f"Loading data from '{filepath}'...")
        with open(filepath, 'r', encoding='utf-8') as f: data = f.read()
        if len(data) < 50: data = FALLBACK_TEXT
    else:
        print(f">> Creating Default Corpus.")
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
        return [self.vocab["<BOS>"]] + ids + [self.vocab["<EOS>"]]

    def decode(self, ids):
        valid_ids = [i for i in ids if i > 3]
        return "".join([self.reverse_vocab.get(i, "") for i in valid_ids])

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
        score_r = (q * k.conj()).sum(dim=-1, keepdim=True).real
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
        q = gw_state.unsqueeze(1)
        sim = (prev_mem.conj() * q).sum(dim=-1).real
        attn = F.softmax(sim, dim=-1).unsqueeze(-1)
        read_out = (prev_mem * torch.complex(attn, torch.zeros_like(attn))).sum(dim=1)
        
        flat_input = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        write_gate = torch.sigmoid(self.gate_net(flat_input)).unsqueeze(1) 
        
        logits = self.address_net(flat_input)
        write_weights = F.softmax(logits, dim=-1)
        slot_entropy = -torch.sum(write_weights * torch.log(write_weights + 1e-10), dim=-1).mean()
        
        write_weights = write_weights.unsqueeze(-1)
        new_slot = gw_state.unsqueeze(1)
        effective_update = write_gate * write_weights
        
        next_mem = (1.0 - effective_update) * prev_mem + (effective_update * new_slot)
        next_mem = self.mem_norm(next_mem)
        
        return read_out, next_mem, slot_entropy

class EMA_VQ(nn.Module):
    def __init__(self, dim, num_concepts):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim = dim
        self.embedding = nn.Embedding(num_concepts, dim * 2)
        self.embedding.weight.data.uniform_(-1/num_concepts, 1/num_concepts)
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_concepts))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.decay = CONFIG["vq_decay"]
        self.epsilon = 1e-5

    def forward(self, z):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        x_sq = torch.sum(z_flat**2, dim=1, keepdim=True)
        y_sq = torch.sum(self.embedding.weight**2, dim=1)
        d = x_sq + y_sq - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        
        indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(indices)
        
        if self.training:
            encodings = F.one_hot(indices, self.num_concepts).float()
            n_total = encodings.sum(0)
            self.ema_cluster_size.data.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_concepts * self.epsilon) * n
            dw = torch.matmul(encodings.t(), z_flat)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
            
        loss_sample = CONFIG["vq_commitment_beta"] * ((z_q.detach() - z_flat)**2).mean(dim=-1)
        z_q = z_flat + (z_q - z_flat).detach()
        
        probs = F.one_hot(indices, self.num_concepts).float().mean(0)
        norm_entropy = -torch.sum(probs * torch.log(probs + 1e-10)) / math.log(self.num_concepts)
        
        z_q_c = torch.complex(z_q[..., :self.dim], z_q[..., self.dim:])
        return z_q_c, loss_sample, indices, norm_entropy

# ==========================================
# 3. [RESTORED] SENSORY CORTEX
# ==========================================
class HebbianSensory(nn.Module):
    def __init__(self, dim, n_anchors):
        super().__init__()
        self.dim = dim
        self.visual_palette = nn.Parameter(torch.randn(n_anchors, dim * 2))
        self.audio_palette = nn.Parameter(torch.randn(n_anchors, dim * 2))
        nn.init.orthogonal_(self.visual_palette)
        nn.init.orthogonal_(self.audio_palette)
        self.proj = ComplexLinear(dim, dim) # Project thought to sensation

    def forward(self, gw_state):
        # 1. Project thought
        z_sense = self.proj(gw_state)
        z_flat = torch.cat([z_sense.real, z_sense.imag], dim=-1)
        
        # 2. Visual Resonance (Real part)
        # Dot product with visual palette
        vis_scores = torch.matmul(z_flat, self.visual_palette.t())
        vis_attn = F.softmax(vis_scores, dim=-1)
        vis_out_flat = torch.matmul(vis_attn, self.visual_palette)
        
        # 3. Audio Resonance (Imaginary part logic for variation)
        aud_scores = torch.matmul(z_flat, self.audio_palette.t())
        aud_attn = F.softmax(aud_scores, dim=-1)
        aud_out_flat = torch.matmul(aud_attn, self.audio_palette)
        
        # 4. Reconstruct Complex Signal
        # Mixing Vis/Aud back into complex thought
        vis_c = torch.complex(vis_out_flat[..., :self.dim], vis_out_flat[..., self.dim:])
        aud_c = torch.complex(aud_out_flat[..., :self.dim], aud_out_flat[..., self.dim:])
        
        # Sensory Experience = Visual (Real Dominant) + Audio (Imag Dominant)
        experience = vis_c + torch.complex(-aud_c.imag, aud_c.real) # Rotate audio 90 deg
        
        return experience, vis_attn

# ==========================================
# 4. SACRSN MODEL (EMBODIED)
# ==========================================
class SACRSN_v54(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Embedding(vocab_size, dim * 2)
        self.norm = StableComplexNorm(dim)
        self.gate_mod = HermitianGating(dim)
        self.memory_module = AssociativeMemory(dim, CONFIG["mem_slots"])
        self.vq = EMA_VQ(dim, CONFIG["codebook_size"])
        # [RESTORED] Sensory Module
        self.sensory_cortex = HebbianSensory(dim, CONFIG["n_sensory_anchors"])
        
        # Arbitrator now manages 4 streams: Gate, Memory, VQ, Sensory
        self.arbitrator = nn.Linear(dim*2, 4) 
        self.decoder = nn.Linear(dim * 2, vocab_size)
        
        self.halt_bias = nn.Parameter(torch.tensor(CONFIG["halt_bias_init"]))
        self.halt_scale = nn.Parameter(torch.tensor(CONFIG["halt_scale_init"]))
        self.input_gate = nn.Parameter(torch.tensor(0.0))
        
        self.register_buffer('vq_loss_ema', torch.tensor(1.0))

    def forward_step(self, x_t, gw_state, mem_state):
        alpha = torch.sigmoid(self.input_gate)
        gw_state = (alpha * gw_state) + ((1.0 - alpha) * x_t)
        
        B = x_t.size(0)
        active_mask = torch.ones(B, 1, device=DEVICE)
        
        accum = {k: torch.zeros(B, device=DEVICE) for k in ['vq', 'ponder', 'phase']}
        total_batch_entropy = 0 
        total_gate_entropy = 0 
        total_slot_entropy = 0 
        
        steps_taken = 0
        total_steps_batch = torch.zeros(B, device=DEVICE)
        final_indices = torch.zeros(B, dtype=torch.long, device=DEVICE)
        prev_angle = gw_state.angle()
        
        for i in range(CONFIG["max_recursion"]):
            steps_taken += 1
            curr_state = self.norm(gw_state)
            
            # Modules
            v_out, vq_loss_sample, indices, batch_ent = self.vq(curr_state)
            g_out = self.gate_mod(curr_state)
            m_out, cand_mem_state, slot_ent = self.memory_module(curr_state, mem_state)
            # [RESTORED] Sensory Hallucination
            s_out, _ = self.sensory_cortex(curr_state)
            
            if self.training:
                with torch.no_grad():
                    mean_loss = vq_loss_sample.mean()
                    self.vq_loss_ema = 0.95 * self.vq_loss_ema + 0.05 * mean_loss
            
            if active_mask.sum() > 0:
                total_batch_entropy += batch_ent
                total_slot_entropy += slot_ent
            
            flat = torch.cat([curr_state.real, curr_state.imag], dim=-1)
            gates = F.softmax(self.arbitrator(flat), dim=-1)
            
            if active_mask.sum() > 0:
                gate_ent = -torch.sum(gates * torch.log(gates + 1e-10), dim=-1).mean()
                total_gate_entropy += gate_ent
            
            # Complex Mixing of 4 Streams
            # 0:Gate, 1:Mem, 2:VQ, 3:Sensory
            update_real = (gates[:,0:1]*g_out.real + gates[:,1:2]*m_out.real + 
                           gates[:,2:3]*v_out.real + gates[:,3:4]*s_out.real)
            update_imag = (gates[:,0:1]*g_out.imag + gates[:,1:2]*m_out.imag + 
                           gates[:,2:3]*v_out.imag + gates[:,3:4]*s_out.imag)
            update = torch.complex(update_real, update_imag)
            
            cand_state = 0.6 * curr_state + 0.4 * update
            
            diff = torch.abs(cand_state.angle() - prev_angle)
            diff = torch.min(diff, 2*math.pi - diff)
            accum['phase'] += active_mask.squeeze() * diff.mean(dim=-1)
            prev_angle = cand_state.angle()
            
            bias = F.softplus(self.halt_bias)
            scale = F.softplus(self.halt_scale)
            
            norm_loss = vq_loss_sample / (self.vq_loss_ema + 1e-6)
            halt_prob = torch.sigmoid(bias - (scale * norm_loss))
            
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
            mem_state = mem_state + mem_mask * (cand_mem_state - mem_state)
            
            active_mask = active_mask * (1.0 - should_stop)
            if active_mask.sum() == 0: break
        
        accum['vq'] = accum['vq'] / torch.max(total_steps_batch, torch.ones_like(total_steps_batch))
        avg_batch_ent = total_batch_entropy / max(1, steps_taken)
        avg_gate_ent = total_gate_entropy / max(1, steps_taken)
        avg_slot_ent = total_slot_entropy / max(1, steps_taken)
        avg_depth = total_steps_batch.mean()
        
        return gw_state, mem_state, accum, final_indices, avg_batch_ent, avg_gate_ent, avg_slot_ent, avg_depth

    def forward(self, x_seq, mem_state=None):
        B, T = x_seq.shape
        emb = self.encoder(x_seq)
        gw_seq = torch.complex(emb[..., :self.dim], emb[..., self.dim:])
        
        if mem_state is None: mem_state = self.memory_module.init_state(B)
        gw_state = torch.zeros_like(gw_seq[:, 0])
        
        outputs, all_indices = [], []
        stats = {k: 0 for k in ['vq', 'ponder', 'phase', 'ent', 'gate_ent', 'slot_ent', 'depth']}
        
        for t in range(T):
            x_t = gw_seq[:, t]
            gw_state, mem_state, step_stats, indices, batch_ent, gate_ent, slot_ent, depth = self.forward_step(x_t, gw_state, mem_state)
            outputs.append(gw_state)
            all_indices.append(indices)
            for k in ['vq', 'ponder', 'phase']: stats[k] += step_stats[k].mean()
            stats['ent'] += batch_ent
            stats['gate_ent'] += gate_ent
            stats['slot_ent'] += slot_ent
            stats['depth'] += depth
            
        out_tensor = torch.stack(outputs, dim=1)
        flat_out = torch.cat([out_tensor.real, out_tensor.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        for k in stats: stats[k] /= T
        
        return logits, stats, mem_state, torch.stack(all_indices, dim=1)

# ==========================================
# 5. HIPPOCAMPUS & UTILS
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
# 6. DIAGNOSTICS & VISUALIZATION
# ==========================================
def visualize_suite(model, history, transitions):
    print("\n--- RUNNING DIAGNOSTICS SUITE ---")
    plt.figure(figsize=(15, 10))
    
    # 1. Cognitive Effort
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['ponder'], label='Ponder Cost')
    plt.title("Cognitive Effort"); plt.legend(); plt.grid(True, alpha=0.3)
    
    # 2. Dynamic Phase Space
    plt.subplot(2, 2, 2)
    with torch.no_grad():
        w = model.decoder.weight.detach().cpu().numpy()
        plt.scatter(w[:, :64].mean(0), w[:, 64:].mean(0), alpha=0.1, c='gray')
        gw_dummy = torch.zeros(1, 64, dtype=torch.complex64, device=DEVICE)
        x_dummy = torch.zeros(1, 64, dtype=torch.complex64, device=DEVICE)
        mem_trace = model.memory_module.init_state(1)
        traj_real, traj_imag = [], []
        for _ in range(20):
            gw_dummy, mem_trace, _, _, _, _, _, _ = model.forward_step(x_dummy, gw_dummy, mem_trace)
            traj_real.append(gw_dummy.real.mean().item())
            traj_imag.append(gw_dummy.imag.mean().item())
        plt.plot(traj_real, traj_imag, 'r-', linewidth=2, label='Thought Trace')
    plt.title("Phase Space"); plt.xlabel("Real"); plt.ylabel("Imag"); plt.legend()

    # 3. Topology
    plt.subplot(2, 2, 3)
    G = nx.DiGraph()
    for (u, v), w in Counter(transitions).most_common(50): G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, node_size=20, width=0.5, alpha=0.7, edge_color='teal')
    plt.title(f"Abstract Topology ({CONFIG['codebook_size']} Concepts)")

    # 4. [RESTORED] Sensory Palette
    plt.subplot(2, 2, 4)
    # Visualize the Visual Palette (first 10 anchors)
    vis = model.sensory_cortex.visual_palette.detach().cpu().numpy()[:10]
    plt.imshow(vis, aspect='auto', cmap='plasma')
    plt.title("Sensory Anchors (Visual Modality)")
    plt.colorbar()

    plt.tight_layout(); plt.savefig('sacrsn_diagnostics.png'); plt.show()

def extract_logic_rules(transitions, tokenizer):
    print("\n--- EXTRACTED CONCEPT RULES ---")
    print(f"{'FROM':<8} -> {'TO':<8} | {'STRENGTH':<10}")
    print("-" * 35)
    for (u, v), w in Counter(transitions).most_common(10):
        print(f"Cpt_{u:<4} -> Cpt_{v:<4} | {w:<10}")

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

# [RESTORED] Dream Mode (Graph Walk)
def dream_graph_walk(model, transitions, tokenizer, steps=20):
    print("\n\n--- 🌙 DREAM MODE (ASSOCIATIVE WALK) ---")
    if not transitions:
        print("No memories to dream yet.")
        return
    
    # Build transition map
    graph = defaultdict(list)
    for u, v in transitions:
        graph[u].append(v)
    
    # Pick random start
    curr = random.choice(list(graph.keys()))
    print(f"Start Concept: [{curr}]", end=" ")
    
    for _ in range(steps):
        if curr in graph and graph[curr]:
            curr = random.choice(graph[curr])
            # Decode a representative word for this concept (Approximate)
            # In a real VQ, we'd need a decoder map. Here we just show the path.
            print(f"-> {curr}", end=" ")
        else:
            break
    print("\n")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def calculate_loss(logits, y, stats, plasticity=1.0):
    loss_ce = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
    
    struct_loss = 0.1 * stats['vq'] + \
                  stats['ponder'] + \
                  CONFIG['phase_reg'] * stats['phase'] - \
                  CONFIG['entropy_weight'] * stats['ent'] + \
                  CONFIG['gate_sparsity_weight'] * stats['gate_ent'] - \
                  CONFIG['slot_balance_weight'] * stats['slot_ent']
                  
    total_loss = (loss_ce * plasticity) + struct_loss
    return loss_ce, total_loss

def main():
    data_text = load_training_data("data.txt")
    tokenizer = RobustBPE(target_vocab_size=1000)
    tokenizer.train(data_text)
    CONFIG["vocab_size"] = len(tokenizer.vocab)
    
    data_ids = tokenizer.encode(data_text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)
    print(f"Tokens: {len(data_tensor)}")

    model = SACRSN_v54(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])
    hippocampus = HippocampalBuffer(CONFIG["replay_size"])
    
    history = defaultdict(list)
    transitions = deque(maxlen=10000) 
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
                with torch.no_grad():
                    # Calculate raw surprise before scaling
                    raw_ce = loss_ce.item()
                avg_surprise = CONFIG["surprise_decay"]*avg_surprise + (1-CONFIG["surprise_decay"])*raw_ce
                plast = min(3.0, 1.0 + avg_surprise*0.5)
            
            loss_ce_scaled, final_loss = calculate_loss(logits, y, stats, plasticity=plast)
            
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            
            hippocampus.add(x.detach().cpu(), y.detach().cpu(), final_loss.item())
            
            if epoch > 0 and epoch % 50 == 0:
                dream = hippocampus.sample(4)
                if dream:
                    for _, dx, dy in dream:
                        dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                        opt.zero_grad()
                        d_logits, d_stats, _, _ = model(dx)
                        _, d_loss = calculate_loss(d_logits, dy, d_stats, plasticity=1.0)
                        d_loss.backward()
                        opt.step()
            
            scheduler.step()
            
            history['loss'].append(final_loss.item())
            history['ponder'].append(stats['ponder'].item())
            
            if epoch % 20 == 0:
                lr = opt.param_groups[0]['lr']
                text_ppx = math.exp(loss_ce.item())
                raw_ent = stats['ent'].item() * math.log(CONFIG['codebook_size'])
                code_ppx = math.exp(raw_ent)
                print(f"Ep {epoch:03d} | Loss: {final_loss.item():.3f} | TextPPX: {text_ppx:.1f} | CodePPX: {code_ppx:.1f} | Depth: {stats['depth'].item():.2f} | Plast: {plast:.2f}")

    except KeyboardInterrupt: print("Stopped.")
    
    generate_text(model, tokenizer, length=80)
    dream_graph_walk(model, transitions, tokenizer) # [RESTORED]
    visualize_suite(model, history, transitions)
    extract_logic_rules(transitions, tokenizer)
    anomaly_detector(model, tokenizer)

if __name__ == "__main__":
    main()