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
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    torch.set_float32_matmul_precision('high')
except:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # Architecture
    "seq_len": 32,
    "batch_size": 16,
    "embed_dim": 64,
    "mem_slots": 32,
    "mem_topk": 3,          # [NEW] Prevent memory blurring
    "vocab_size": None,
    "codebook_size": 128,
    "n_sensory_anchors": 32,
    
    # Cognitive Dynamics
    "max_recursion": 8,     
    "halt_bias_init": 3.0,
    "halt_scale_init": 10.0,
    "ponder_cost": 0.01,
    
    # The Shadow
    "shadow_weight": 0.3,
    "critic_weight": 0.1,
    
    # VQ Stability
    "vq_commitment_beta": 0.25,
    "vq_decay": 0.99,
    
    # Regularization
    "entropy_weight": 0.05, 
    "gate_sparsity_weight": 0.01,
    "slot_balance_weight": 0.01,
    "phase_reg": 0.01,
    "sensory_reg": 0.01,
    
    # [FIXED] Decay applies to all cognitive heads
    "synaptic_decay": 1e-5, 
    
    # Training
    "epochs": 300,
    "lr": 5e-4,
    "grad_clip": 1.0,
    "replay_size": 200,
    "active_plasticity": True,
    "surprise_decay": 0.9,
    "target_depth": 4.0,
    "depth_target_weight": 0.005,
}

print(f"--- SACRSN v61: STABILIZED DYNAMICS EDITION ---")
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
        self._merge_set = set(self.merges.keys())
        print(f"BPE Trained. Final Size: {len(self.vocab)}")

    def encode(self, text):
        words = re.findall(r"[\w']+|[^\s\w]", text)
        ids = []
        merge_lookup = getattr(self, '_merge_set', set(self.merges.keys()))
        for word in words:
            word_tokens = list(word)
            while len(word_tokens) > 1:
                best_pair = None
                best_rank = float('inf')
                best_idx = -1
                for i in range(len(word_tokens)-1):
                    pair = (word_tokens[i], word_tokens[i+1])
                    if pair in merge_lookup:
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
        # [FIX] Human-readable decoding (Audit Fix 2)
        raw = " ".join([self.reverse_vocab.get(i, "") for i in valid_ids])
        # Clean up punctuation gaps
        raw = raw.replace(" .", ".").replace(" ,", ",").replace(" ;", ";")
        return raw

# ==========================================
# 2. MATH & MODULES
# ==========================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.Wr = nn.Linear(in_features, out_features)
        self.Wi = nn.Linear(in_features, out_features)

    def forward(self, z):
        real = self.Wr(z.real) - self.Wi(z.imag)
        imag = self.Wi(z.real) + self.Wr(z.imag)
        return torch.complex(real, imag)

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

class CriticModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = ComplexLinear(dim, dim)
    def forward(self, gw_state):
        z = self.net(gw_state)
        return torch.complex(-z.imag, z.real)

class AssociativeMemory(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.dim, self.slots = dim, slots
        self.gate_net = nn.Linear(dim * 2, 1)
        self.address_net = nn.Linear(dim * 2, slots)
        self.mem_norm = StableComplexNorm(dim)
        self.topk = CONFIG["mem_topk"] # [FIX] Load topk config
        
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
        
        # [FIX] Top-K Sparsity to prevent memory homogenization (Audit Fix 3)
        # Only keep the top 3 write locations, zero the rest
        top_vals, top_idx = torch.topk(write_weights, k=self.topk, dim=-1)
        sparse_weights = torch.zeros_like(write_weights).scatter_(-1, top_idx, top_vals)
        # Re-normalize just the top-k
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        sparse_weights = sparse_weights.unsqueeze(-1)
        new_slot = gw_state.unsqueeze(1)
        
        # Update using sparse weights
        effective_update = write_gate * sparse_weights
        next_mem = (1.0 - effective_update) * prev_mem + (effective_update * new_slot)
        next_mem = self.mem_norm(next_mem)
        
        return read_out, next_mem, slot_entropy

class EMA_VQ(nn.Module):
    def __init__(self, dim, num_concepts):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim = dim
        self.embedding = nn.Embedding(num_concepts, dim * 2)
        init_range = 1.0 / math.sqrt(dim)
        self.embedding.weight.data.uniform_(-init_range, init_range)
        
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
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_concepts).float()
                n_total = encodings.sum(0)
                self.ema_cluster_size.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_concepts * self.epsilon) * n
                dw = torch.matmul(encodings.t(), z_flat.detach())
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                self.embedding.weight.copy_(self.ema_w / cluster_size.unsqueeze(1))
            
        loss_sample = CONFIG["vq_commitment_beta"] * ((z_q.detach() - z_flat)**2).mean(dim=-1)
        z_q = z_flat + (z_q - z_flat).detach()
        
        probs = F.one_hot(indices, self.num_concepts).float().mean(0)
        norm_entropy = -torch.sum(probs * torch.log(probs + 1e-10)) / math.log(self.num_concepts)
        
        z_q_c = torch.complex(z_q[..., :self.dim], z_q[..., self.dim:])
        return z_q_c, loss_sample, indices, norm_entropy

class HebbianSensory(nn.Module):
    def __init__(self, dim, n_anchors):
        super().__init__()
        self.dim = dim
        self.visual_palette = nn.Parameter(torch.randn(n_anchors, dim * 2))
        self.audio_palette = nn.Parameter(torch.randn(n_anchors, dim * 2))
        nn.init.orthogonal_(self.visual_palette)
        nn.init.orthogonal_(self.audio_palette)
        self.proj = ComplexLinear(dim, dim) 

    def forward(self, gw_state):
        z_sense = self.proj(gw_state)
        z_flat = torch.cat([z_sense.real, z_sense.imag], dim=-1)
        
        vis_scores = torch.matmul(z_flat, self.visual_palette.t())
        vis_attn = F.softmax(vis_scores, dim=-1)
        vis_out_flat = torch.matmul(vis_attn, self.visual_palette)
        
        aud_scores = torch.matmul(z_flat, self.audio_palette.t())
        aud_attn = F.softmax(aud_scores, dim=-1)
        aud_out_flat = torch.matmul(aud_attn, self.audio_palette)
        
        vis_c = torch.complex(vis_out_flat[..., :self.dim], vis_out_flat[..., self.dim:])
        aud_c = torch.complex(aud_out_flat[..., :self.dim], aud_out_flat[..., self.dim:])
        
        experience = vis_c + torch.complex(-aud_c.imag, aud_c.real) 
        return experience, vis_attn, aud_attn

# ==========================================
# 4. SACRSN MODEL
# ==========================================
class SACRSN_v61(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Embedding(vocab_size, dim * 2)
        self.norm = StableComplexNorm(dim)
        
        self.gate_mod = HermitianGating(dim)
        self.memory_module = AssociativeMemory(dim, CONFIG["mem_slots"])
        self.vq = EMA_VQ(dim, CONFIG["codebook_size"])
        self.sensory_cortex = HebbianSensory(dim, CONFIG["n_sensory_anchors"])
        self.critic = CriticModule(dim)
        
        self.arbitrator = nn.Linear(dim*2, 5) 
        self.meta_head = nn.Linear(dim*2, 4)
        self.decoder = nn.Linear(dim * 2, vocab_size)
        
        self.halt_bias = nn.Parameter(torch.tensor(CONFIG["halt_bias_init"]))
        self.halt_scale = nn.Parameter(torch.tensor(CONFIG["halt_scale_init"]))
        self.input_gate = nn.Parameter(torch.tensor(0.0))
        
        self.register_buffer('vq_loss_ema', torch.tensor(1.0))
        self.register_buffer('train_step', torch.tensor(0))

    def _step_modules(self, curr_state, mem_state):
        v_out, vq_loss_sample, indices, batch_ent = self.vq(curr_state)
        g_out = self.gate_mod(curr_state)
        m_out, cand_mem_state, slot_ent = self.memory_module(curr_state, mem_state)
        s_out, vis_attn, aud_attn = self.sensory_cortex(curr_state)
        c_out = self.critic(curr_state)
        return v_out, vq_loss_sample, indices, batch_ent, g_out, m_out, cand_mem_state, slot_ent, s_out, vis_attn, aud_attn, c_out

    def _step_arbitration(self, curr_state, meta_values):
        flat = torch.cat([curr_state.real, curr_state.imag], dim=-1)
        raw_logits = self.arbitrator(flat)
        w_ethics = meta_values[:, 0]
        ethics_bias = (w_ethics - 0.5) * CONFIG["critic_weight"] * 4.0
        raw_logits = raw_logits.clone()
        raw_logits[:, 4] = raw_logits[:, 4] + ethics_bias
        raw_gates = F.softmax(raw_logits, dim=-1)
        
        w_unc = meta_values[:, 3:4]
        uniform = torch.ones_like(raw_gates) / raw_gates.size(-1)
        min_entropy_mix = 0.05 + 0.10 * w_unc
        gates = (1 - min_entropy_mix) * raw_gates + min_entropy_mix * uniform
        gate_ent = -torch.sum(gates * torch.log(gates + 1e-10), dim=-1).mean()
        return gates, gate_ent

    def _step_halting(self, vq_loss_sample, meta_values):
        bias = F.softplus(self.halt_bias)
        scale = F.softplus(self.halt_scale)
        norm_loss = vq_loss_sample / (self.vq_loss_ema + 1e-6)
        halt_prob = torch.sigmoid(bias - (scale * norm_loss))
        
        w_unc = meta_values[:, 3]
        halt_prob = halt_prob * (1.0 - 0.3 * w_unc)
        halt_prob = halt_prob.clamp(0.0, 1.0)
        
        if self.training and self.train_step < 20:
            halt_prob = halt_prob * 0.1
        
        if self.training:
            noise = torch.rand_like(halt_prob)
            hard_stop = (halt_prob > noise).float()
        else:
            hard_stop = (halt_prob > 0.5).float()
        should_stop = (hard_stop - halt_prob.detach() + halt_prob).unsqueeze(1)
        return should_stop

    def forward_step(self, x_t, gw_state, mem_state):
        alpha = torch.sigmoid(self.input_gate)
        gw_state = (alpha * gw_state) + ((1.0 - alpha) * x_t)
        
        B = x_t.size(0)
        active_mask = torch.ones(B, 1, device=DEVICE)
        accum = {k: torch.zeros(B, device=DEVICE) for k in ['vq', 'ponder', 'phase']}
        total_batch_entropy, total_gate_entropy, total_slot_entropy, total_sensory_entropy = 0, 0, 0, 0
        module_wins = torch.zeros(B, 5, device=DEVICE)
        
        steps_taken = 0
        total_steps_batch = torch.zeros(B, device=DEVICE)
        final_indices = torch.zeros(B, dtype=torch.long, device=DEVICE)
        
        init_mag = torch.abs(gw_state)
        prev_angle = torch.where(init_mag > 1e-6, gw_state.angle(), torch.zeros_like(gw_state.real))
        
        meta_accum_step = torch.zeros(B, 4, device=DEVICE)
        meta_values = None 
        
        for i in range(CONFIG["max_recursion"]):
            steps_taken += 1
            curr_state = self.norm(gw_state)
            
            flat_curr = torch.cat([curr_state.real, curr_state.imag], dim=-1)
            meta_values = torch.sigmoid(self.meta_head(flat_curr))
            meta_accum_step += meta_values
            
            v_out, vq_loss_sample, indices, batch_ent, g_out, m_out, cand_mem_state, slot_ent, s_out, vis_attn, aud_attn, c_out = \
                self._step_modules(curr_state, mem_state)
            
            if self.training:
                with torch.no_grad():
                    mean_loss = vq_loss_sample.mean()
                    self.vq_loss_ema = 0.95 * self.vq_loss_ema + 0.05 * mean_loss
            
            if active_mask.sum() > 0:
                total_batch_entropy += batch_ent
                total_slot_entropy += slot_ent
                vis_ent = -torch.sum(vis_attn * torch.log(vis_attn + 1e-10), dim=-1).mean()
                aud_ent = -torch.sum(aud_attn * torch.log(aud_attn + 1e-10), dim=-1).mean()
                total_sensory_entropy += (vis_ent + aud_ent) * 0.5
            
            gates, gate_ent = self._step_arbitration(curr_state, meta_values)
            module_wins += (gates * active_mask)
            if active_mask.sum() > 0:
                total_gate_entropy += gate_ent
            
            proposals = torch.stack([g_out, m_out, v_out, s_out, c_out], dim=1)
            gates_c = torch.complex(gates.unsqueeze(-1), torch.zeros_like(gates.unsqueeze(-1)))
            winner_vec = torch.sum(proposals * gates_c, dim=1)
            
            shadow_probs = 1.0 - gates
            shadow_norm = shadow_probs / (shadow_probs.sum(dim=1, keepdim=True) + 1e-6)
            shadow_c = torch.complex(shadow_norm.unsqueeze(-1), torch.zeros_like(shadow_norm.unsqueeze(-1)))
            shadow_vec = torch.sum(proposals * shadow_c, dim=1)
            
            update = winner_vec + (CONFIG["shadow_weight"] * shadow_vec)
            cand_state = 0.6 * curr_state + 0.4 * update
            
            mag = torch.abs(cand_state)
            valid_phase = mag > 1e-6
            safe_angle = torch.where(valid_phase, cand_state.angle(), prev_angle)
            
            diff = torch.abs(safe_angle - prev_angle)
            diff = torch.min(diff, 2*math.pi - diff)
            valid_frac = valid_phase.float().mean(dim=-1)
            accum['phase'] += active_mask.squeeze() * diff.mean(dim=-1) * valid_frac
            prev_angle = safe_angle
            
            should_stop = self._step_halting(vq_loss_sample, meta_values)
            
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
        
        if self.training:
            self.train_step += 1
        
        accum['vq'] = accum['vq'] / torch.max(total_steps_batch, torch.ones_like(total_steps_batch))
        avg_batch_ent = total_batch_entropy / max(1, steps_taken)
        avg_gate_ent = total_gate_entropy / max(1, steps_taken)
        avg_slot_ent = total_slot_entropy / max(1, steps_taken)
        avg_sensory_ent = total_sensory_entropy / max(1, steps_taken)
        avg_depth = total_steps_batch.mean()
        accum['depth_per_sample'] = total_steps_batch
        meta_values_avg = meta_accum_step / max(1, steps_taken)
        
        return gw_state, mem_state, accum, final_indices, avg_batch_ent, avg_gate_ent, avg_slot_ent, avg_depth, meta_values_avg, module_wins, avg_sensory_ent

    def forward(self, x_seq, mem_state=None):
        B, T = x_seq.shape
        emb = self.encoder(x_seq)
        gw_seq = torch.complex(emb[..., :self.dim], emb[..., self.dim:])
        
        if mem_state is None: mem_state = self.memory_module.init_state(B)
        gw_state = torch.zeros_like(gw_seq[:, 0])
        
        outputs, all_indices = [], []
        stats = {k: 0 for k in ['vq', 'ponder', 'phase', 'ent', 'gate_ent', 'slot_ent', 'depth', 'sensory_ent']}
        depth_per_sample_accum = torch.zeros(B, device=DEVICE)
        meta_accum = torch.zeros(B, 4, device=DEVICE) 
        final_wins = None
        
        for t in range(T):
            x_t = gw_seq[:, t]
            gw_state, mem_state, step_stats, indices, batch_ent, gate_ent, slot_ent, depth, meta_vals, wins, sensory_ent = \
                self.forward_step(x_t, gw_state, mem_state)
            
            outputs.append(gw_state)
            all_indices.append(indices)
            for k in ['vq', 'ponder', 'phase']: stats[k] += step_stats[k].mean()
            stats['ent'] += batch_ent
            stats['gate_ent'] += gate_ent
            stats['slot_ent'] += slot_ent
            stats['depth'] += depth
            stats['sensory_ent'] += sensory_ent
            depth_per_sample_accum += step_stats['depth_per_sample']
            meta_accum += meta_vals
            final_wins = wins
            
        out_tensor = torch.stack(outputs, dim=1)
        flat_out = torch.cat([out_tensor.real, out_tensor.imag], dim=-1)
        logits = self.decoder(flat_out)
        
        for k in stats: stats[k] /= T
        meta_accum /= T
        stats['depth_per_sample'] = depth_per_sample_accum / T
        
        return logits, stats, mem_state, torch.stack(all_indices, dim=1), meta_accum, final_wins

# ==========================================
# 5. UTILS & VISUALIZATION
# ==========================================
class HippocampalBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self._n, self._mean, self._M2 = 0, 0.0, 0.0

    def _update_stats(self, x):
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    def _z_score(self, x):
        if self._n < 2: return x
        variance = self._M2 / (self._n - 1)
        std = max(math.sqrt(variance), 1e-6)
        return (x - self._mean) / std

    def add(self, x, y, raw_ce_loss):
        self._update_stats(raw_ce_loss)
        priority = self._z_score(raw_ce_loss)
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, (priority, x, y))
        else:
            if priority > self.buffer[0][0]:
                heapq.heapreplace(self.buffer, (priority, x, y))

    def sample(self, batch_size):
        if not self.buffer: return None
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

def get_batch(data_tensor, batch_size, seq_len):
    max_idx = len(data_tensor) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def visualize_suite(model, history, transitions, meta_history):
    print("\n--- RUNNING DIAGNOSTICS SUITE ---")
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['ponder'], label='Ponder Cost')
    plt.title("Cognitive Effort"); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    with torch.no_grad():
        w = model.decoder.weight.detach().cpu().numpy()
        plt.scatter(w[:, :64].mean(0), w[:, 64:].mean(0), alpha=0.1, c='gray')
        gw_dummy = torch.zeros(1, 64, dtype=torch.complex64, device=DEVICE)
        x_dummy = torch.zeros(1, 64, dtype=torch.complex64, device=DEVICE)
        mem_trace = model.memory_module.init_state(1)
        traj_real, traj_imag = [], []
        for _ in range(20):
            gw_dummy, mem_trace, _, _, _, _, _, _, _, _, _ = model.forward_step(x_dummy, gw_dummy, mem_trace)
            traj_real.append(gw_dummy.real.mean().item())
            traj_imag.append(gw_dummy.imag.mean().item())
        plt.plot(traj_real, traj_imag, 'r-', linewidth=2, label='Thought Trace')
    plt.title("Phase Space"); plt.xlabel("Real"); plt.ylabel("Imag"); plt.legend()

    plt.subplot(2, 3, 3)
    G = nx.DiGraph()
    for (u, v), w in Counter(transitions).most_common(50): G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, node_size=20, width=0.5, alpha=0.7, edge_color='teal')
    plt.title(f"Abstract Topology ({CONFIG['codebook_size']} Concepts)")

    plt.subplot(2, 3, 4)
    vis = model.sensory_cortex.visual_palette.detach().cpu().numpy()[:10]
    plt.imshow(vis, aspect='auto', cmap='plasma')
    plt.title("Sensory Anchors (Visual)")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    if history['depth']:
        plt.plot(history['depth'], label='Actual Depth', color='purple')
        plt.axhline(y=CONFIG['target_depth'], color='red', linestyle='--', label='Target')
        plt.title("Recursion Depth"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    if meta_history:
        meta_hist = np.array(meta_history)
        labels = ["Ethics", "Diversity", "Ponder", "Uncertainty"]
        for i in range(4):
            plt.plot(meta_hist[:, i], label=labels[i])
        plt.title("Meta-Value Drift"); plt.legend()

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
            logits, _, _, _, _, _ = model(x)
            y = torch.tensor([ids[1:]], device=DEVICE)
            return F.cross_entropy(logits.reshape(-1, len(tokenizer.vocab)), y.reshape(-1)).item()
    s1, s2 = score(normal), score(weird)
    print(f"Normal: {s1:.4f} | Weird: {s2:.4f}")
    if s2 > s1: print(">> Anomaly Detected.")

def generate_text(model, tokenizer, seed="The", length=50, temp=0.8, verbose_modules=False):
    print(f"\n--- GENERATING (Seed: '{seed}', Temp: {temp}) ---")
    model.eval()
    ids = tokenizer.encode(seed)
    curr = torch.tensor([ids], device=DEVICE)
    mem = None; text = seed
    modules = ["Gate", "Memory", "VQ", "Sensory", "Critic"]
    if not verbose_modules: print("Stream of Consciousness (first 5 tokens):")
    
    with torch.no_grad():
        _, _, mem, _, _, _ = model(curr, mem)
        curr = curr[:, -1:]
        for i in range(length):
            logits, _, mem, _, _, wins = model(curr, mem)
            if verbose_modules or i < 5:
                winner_idx = torch.argmax(wins[0]).item()
                print(f"Token {i}: [{modules[winner_idx]}] dominant")
            next_id = torch.multinomial(F.softmax(logits[:, -1, :] / temp, dim=-1), 1)
            word = tokenizer.decode([next_id.item()])
            text += word; curr = next_id
            print(word, end="", flush=True)
    print("\n")

def dream_graph_walk(model, transitions, tokenizer, steps=20, temp=1.0):
    print("\n\n--- 🌙 DREAM MODE ---")
    if not transitions: return
    transition_counts = Counter(transitions)
    graph = defaultdict(dict)
    for (u, v), count in transition_counts.items(): graph[u][v] = count
    start_weights = [len(graph[k]) for k in graph]
    curr = random.choices(list(graph.keys()), weights=start_weights, k=1)[0]
    print(f"Start: [{curr}]", end=" ")
    for step in range(steps):
        if curr not in graph or not graph[curr]: break
        targets = list(graph[curr].keys())
        raw_counts = [graph[curr][t] for t in targets]
        scaled = [c ** (1.0 / max(temp, 1e-6)) for c in raw_counts]
        total = sum(scaled)
        probs = [s / total for s in scaled]
        curr = random.choices(targets, weights=probs, k=1)[0]
        print(f"-> {curr}", end=" ")
    print("\n")

def log_grad_norms(model):
    module_groups = {
        'encoder': model.encoder, 'gate_mod': model.gate_mod,
        'memory': model.memory_module, 'vq_embed': model.vq.embedding,
        'sensory': model.sensory_cortex, 'critic': model.critic,
        'arbitrator': model.arbitrator, 'meta_head': model.meta_head,
        'decoder': model.decoder,
    }
    norms = {}
    for name, module in module_groups.items():
        total_norm = 0.0
        n_with_grad = 0
        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                n_with_grad += 1
        norms[name] = total_norm ** 0.5 if n_with_grad > 0 else 0.0
    return norms

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def calculate_loss(logits, y, stats, meta_vals, plasticity=1.0):
    loss_ce = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
    w_ethics, w_div, w_ponder, w_unc = meta_vals.mean(0)
    
    if isinstance(stats.get('depth_per_sample'), torch.Tensor):
        depth_samples = stats['depth_per_sample']
        depth_penalty = CONFIG['depth_target_weight'] * ((depth_samples - CONFIG['target_depth'])**2).mean()
    else:
        depth_penalty = CONFIG['depth_target_weight'] * (stats['depth'] - CONFIG['target_depth'])**2
    
    struct_scale = 1.0 + 0.3 * (plasticity - 1.0)
    
    struct_loss = struct_scale * (
        0.1 * stats['vq'] +
        (stats['ponder'] * (1.0 + w_ponder)) +
        CONFIG['phase_reg'] * stats['phase'] -
        (CONFIG['entropy_weight'] * (1.0 + w_div)) * stats['ent'] +
        CONFIG['gate_sparsity_weight'] * stats['gate_ent'] -
        CONFIG['slot_balance_weight'] * stats['slot_ent'] -
        CONFIG['sensory_reg'] * stats['sensory_ent'] +
        depth_penalty
    )
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

    model = SACRSN_v61(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])
    hippocampus = HippocampalBuffer(CONFIG["replay_size"])
    
    history, transitions = defaultdict(list), deque(maxlen=10000)
    meta_history = []
    avg_surprise = 2.0 
    
    print("\n--- STARTING TRAINING ---")
    model.train()
    try:
        for epoch in range(CONFIG["epochs"]):
            x, y = get_batch(data_tensor, CONFIG["batch_size"], CONFIG["seq_len"])
            opt.zero_grad()
            
            # [FIX] Symmetric Decay on all high-level modules
            with torch.no_grad():
                for module in [model.arbitrator, model.meta_head, model.decoder]:
                    for p in module.parameters():
                        if p.requires_grad:
                            p.mul_(1.0 - CONFIG["synaptic_decay"])
            
            logits, stats, mem, indices, meta_vals, _ = model(x)
            
            idx_np = indices.detach().cpu().numpy()
            for b in range(idx_np.shape[0]):
                for t in range(idx_np.shape[1]-1):
                    transitions.append((idx_np[b,t], idx_np[b,t+1]))
            
            loss_ce_raw = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
            plast = 1.0
            if CONFIG["active_plasticity"]:
                with torch.no_grad(): raw_ce = loss_ce_raw.item()
                avg_surprise = CONFIG["surprise_decay"]*avg_surprise + (1-CONFIG["surprise_decay"])*raw_ce
                plast = min(3.0, 1.0 + avg_surprise*0.5)
            
            _, final_loss = calculate_loss(logits, y, stats, meta_vals, plasticity=plast)
            
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            
            if epoch % 20 == 0:
                grad_norms = log_grad_norms(model)
                history['grad_norms'].append(grad_norms)
            
            opt.step()
            
            hippocampus.add(x.detach().cpu(), y.detach().cpu(), loss_ce_raw.item())
            
            if epoch > 0 and epoch % 50 == 0:
                dream = hippocampus.sample(4)
                if dream:
                    for _, dx, dy in dream:
                        dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                        opt.zero_grad()
                        d_logits, d_stats, _, _, d_meta, _ = model(dx)
                        _, d_loss = calculate_loss(d_logits, dy, d_stats, d_meta, plasticity=1.0)
                        d_loss.backward()
                        opt.step()
            
            scheduler.step()
            history['loss'].append(final_loss.item())
            history['ponder'].append(stats['ponder'].item())
            history['depth'].append(stats['depth'].item())
            history['sensory_ent'].append(stats['sensory_ent'].item() if hasattr(stats['sensory_ent'], 'item') else float(stats['sensory_ent']))
            meta_history.append(meta_vals.mean(0).detach().cpu().numpy())
            
            if epoch % 20 == 0:
                lr_val = opt.param_groups[0]['lr']
                text_ppx = math.exp(loss_ce_raw.item())
                raw_ent = stats['ent'].item() * math.log(CONFIG['codebook_size'])
                code_ppx = math.exp(raw_ent)
                # [FIX] Add VQ_EMA logging to monitor halting baseline drift
                vq_ema = model.vq_loss_ema.item()
                print(f"Ep {epoch:03d} | Loss: {final_loss.item():.3f} | TextPPX: {text_ppx:.1f} | CodePPX: {code_ppx:.1f} | Depth: {stats['depth'].item():.2f} | VQ_EMA: {vq_ema:.4f}")
                if history['grad_norms']:
                    gn = history['grad_norms'][-1]
                    dominant = max(gn, key=gn.get)
                    print(f"         Dominant Module: [{dominant}]")

    except KeyboardInterrupt: print("Stopped.")
    
    generate_text(model, tokenizer, length=80)
    dream_graph_walk(model, transitions, tokenizer)
    visualize_suite(model, history, transitions, meta_history)
    extract_logic_rules(transitions, tokenizer)
    anomaly_detector(model, tokenizer)

if __name__ == "__main__":
    main()