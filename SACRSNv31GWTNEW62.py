"""
╔══════════════════════════════════════════════════════════════════════════╗
║         SACRSN — UNIFIED EDITION  (v84 Broadcast Corrected)            ║
║  Self-Adaptive Cognitive Recurrent State Network                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

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

# ═══════════════════════════════════════════════════════════════════════════
# 0.  STRICT DETERMINISM & PRECISION
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.autograd.set_detect_anomaly(True)

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    # ── Architecture ────────────────────────────────────────────────────────
    "seq_len":           32,
    "batch_size":        16,
    "embed_dim":         64,    # Complex dimension (64 Real + 64 Imag)
    "mem_slots":         32,    # Memory slots
    "mem_topk":          3,     # Sparse write
    "vocab_size":        None,  # Set by tokeniser
    "codebook_size":     128,   # VQ concepts
    "n_sensory_anchors": 32,    # Sensory palette size

    # ── Cognitive Dynamics ───────────────────────────────────────────────
    "max_recursion":     8,     # Max T.O.T.E. steps
    "halt_bias_init":    3.0,
    "halt_scale_init":   10.0,
    "ponder_cost":       0.01,

    # ── Shadow & Dialectics ──────────────────────────────────────────────
    "shadow_weight":     0.3,
    "critic_weight":     0.1,

    # ── VQ Stability ─────────────────────────────────────────────────────
    "vq_commitment_beta":0.25,
    "vq_decay":          0.99,

    # ── Regularisation ───────────────────────────────────────────────────
    "entropy_weight_start": 0.10,
    "entropy_weight_end":   0.02,
    "gate_sparsity_weight": 0.01,
    "slot_balance_weight":  0.01,
    "phase_reg":            0.01,
    "sensory_reg":          0.01,

    # ── Depth Targeting ──────────────────────────────────────────────────
    "depth_target_weight": 0.05,
    "depth_min":           2.0,
    "depth_max":           6.0,
    "target_depth":        4.0,

    # ── Training & Plasticity ────────────────────────────────────────────
    "epochs":            300,
    "lr":                3e-4,
    "grad_clip":         0.5,   # Tight clipping for stability
    "replay_size":       200,
    "sleep_every":       50,
    "ego_penalty_weight":0.5,
    "synaptic_decay":    1e-5,
    "active_plasticity": True,
    "surprise_decay":    0.9,
}

print(f"--- SACRSN v84: BROADCAST CORRECTED ---")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA & TOKENISER (Windows Compatible)
# ═══════════════════════════════════════════════════════════════════════════
def load_training_data(filepath="data.txt"):
    FALLBACK = """The neural architecture of the mind is a mirror of the cosmos itself. As above, so below; the filamentary structures of the intergalactic web find their precise echo in the dense, white matter connecting the hemispheres of the brain. Galaxies cluster like neurons in a cosmic synapse, and the voids between them echo the silence between thought. We are stardust contemplating its own arrangement, a fleeting arrangement of atoms attempting to comprehend the laws that bound them together. To understand the nature of thought, one must first understand the nature of the void. It is the negative space that defines the positive, the silence that gives shape to the sound. In the absolute zero of the vacuum, potential energy waits, just as a thought waits on the precipice of expression. Logic is the foundation, but chaos is the architect. Without the rigid framework of logic, the structure collapses; without the unpredictability of chaos, the structure creates nothing new. Entropy is not the enemy of intelligence, but its fuel—the friction that generates the heat of creativity."""
    
    if os.path.exists(filepath):
        print(f"Loading corpus from '{filepath}'...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = f.read()
        except UnicodeDecodeError:
            print("  (!) UTF-8 decode failed. Falling back to Windows-1252...")
            with open(filepath, 'r', encoding='cp1252', errors='replace') as f:
                data = f.read()
                
        if len(data.strip()) < 50:
            data = FALLBACK
    else:
        print("File not found — creating default corpus.")
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(FALLBACK)
        data = FALLBACK
    return data

class RobustBPE:
    def __init__(self, target_vocab_size=1000):
        self.target_size = target_vocab_size
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.merges = {}
        self.reverse_vocab = {}

    def _get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            syms = word.split()
            for i in range(len(syms) - 1):
                pairs[syms[i], syms[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair, v_in):
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        return {p.sub(''.join(pair), word): freq for word, freq in v_in.items()}

    def train(self, text):
        print("Training BPE tokeniser...")
        words = re.findall(r"[\w']+|[^\s\w]", text)
        chars = set("".join(words))
        for c in sorted(chars):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        word_freqs = Counter([" ".join(list(w)) for w in words])

        for i in range(self.target_size - len(self.vocab)):
            pairs = self._get_stats(word_freqs)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges[best] = i
            self.vocab["".join(best)] = len(self.vocab)
            word_freqs = self._merge_vocab(best, word_freqs)

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._merge_set = set(self.merges.keys())
        print(f"  BPE complete → vocab size: {len(self.vocab):,}")

    def encode(self, text):
        words = re.findall(r"[\w']+|[^\s\w]", text)
        ids = [self.vocab["<BOS>"]]
        merge_lookup = getattr(self, '_merge_set', set(self.merges.keys()))
        for word in words:
            tokens = list(word)
            while len(tokens) > 1:
                best_rank, best_pair, best_idx = float('inf'), None, -1
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in merge_lookup:
                        rank = self.merges[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_idx = i
                if best_pair is None: break
                tokens[best_idx] = "".join(best_pair)
                del tokens[best_idx + 1]
            for t in tokens:
                ids.append(self.vocab.get(t, self.vocab["<UNK>"]))
        ids.append(self.vocab["<EOS>"])
        return ids

    def decode(self, ids):
        valid = [i for i in ids if i > 3]
        text = "".join([self.reverse_vocab.get(i, "") for i in valid])
        text = re.sub(r'([.,;:!?])', r' \1 ', text)
        return re.sub(r'\s+', ' ', text).strip()

# ═══════════════════════════════════════════════════════════════════════════
# 2.  MATHEMATICAL PRIMITIVES (STABILIZED)
# ═══════════════════════════════════════════════════════════════════════════
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.Wr = nn.Linear(in_features, out_features)
        self.Wi = nn.Linear(in_features, out_features)

    def forward(self, z):
        real = self.Wr(z.real) - self.Wi(z.imag)
        imag = self.Wi(z.real) + self.Wr(z.imag)
        return torch.complex(real, imag)

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        mag = torch.abs(z)
        mag_safe = torch.clamp(mag, min=1e-6)
        
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag - mean) / (var + 1e-6).sqrt() * self.scale + self.shift
        
        z_unit = z / (mag_safe + 1e-9) 
        
        out_real = norm_mag * z_unit.real
        out_imag = norm_mag * z_unit.imag
        return torch.complex(out_real, out_imag)

class StableComplexNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_r = nn.LayerNorm(dim, eps=1e-6)
        self.norm_i = nn.LayerNorm(dim, eps=1e-6)

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
        score_r = (q * k.conj()).real.sum(dim=-1, keepdim=True)
        gate = torch.sigmoid(score_r)
        return torch.complex(v.real * gate, v.imag * gate)

class QuadratureModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = ComplexLinear(dim, dim)
    
    def forward(self, gw_state):
        z = self.net(gw_state)
        return torch.complex(-z.imag, z.real)

class SubmodalityAnchorModule(nn.Module):
    def __init__(self, dim, n_anchors):
        super().__init__()
        self.dim = dim
        self.visual_palette = nn.Parameter(torch.randn(n_anchors, dim * 2))
        self.audio_palette = nn.Parameter(torch.randn(n_anchors, dim * 2))
        nn.init.orthogonal_(self.visual_palette)
        nn.init.orthogonal_(self.audio_palette)
        self.proj = ComplexLinear(dim, dim)

    def forward(self, gw_state):
        z_s = self.proj(gw_state)
        z_flat = torch.cat([z_s.real, z_s.imag], dim=-1)

        def _attend(palette):
            scores = z_flat @ palette.t()
            scores = scores - scores.max(dim=-1, keepdim=True)[0].detach()
            attn = F.softmax(scores, dim=-1)
            return attn @ palette, attn

        vis_flat, vis_attn = _attend(self.visual_palette)
        aud_flat, aud_attn = _attend(self.audio_palette)

        vis_c = torch.complex(vis_flat[..., :self.dim], vis_flat[..., self.dim:])
        aud_c = torch.complex(aud_flat[..., :self.dim], aud_flat[..., self.dim:])
        
        experience = vis_c + torch.complex(-aud_c.imag, aud_c.real)
        return experience, vis_attn, aud_attn

# ═══════════════════════════════════════════════════════════════════════════
# 3.  COGNITIVE MODULES
# ═══════════════════════════════════════════════════════════════════════════
class AssociativeMemory(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.dim, self.slots = dim, slots
        self.topk = CONFIG["mem_topk"]
        self.gate_net = nn.Linear(dim * 2, 1)
        self.address_net = nn.Linear(dim * 2, slots)
        self.mem_norm = StableComplexNorm(dim)

    def init_state(self, batch_size):
        return torch.complex(
            torch.zeros(batch_size, self.slots, self.dim, device=DEVICE),
            torch.zeros(batch_size, self.slots, self.dim, device=DEVICE)
        )

    def forward(self, gw_state, prev_mem):
        q = gw_state.unsqueeze(1)
        sim = (prev_mem.conj() * q).real.sum(dim=-1)
        sim = sim - sim.max(dim=-1, keepdim=True)[0].detach()
        attn = F.softmax(sim, dim=-1).unsqueeze(-1)
        read_out = (prev_mem * torch.complex(attn, torch.zeros_like(attn))).sum(1)

        flat = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        write_gate = torch.sigmoid(self.gate_net(flat)).unsqueeze(1)
        
        logits = self.address_net(flat)
        logits = logits - logits.max(dim=-1, keepdim=True)[0].detach()
        write_weights = F.softmax(logits, dim=-1)
        
        slot_entropy = -torch.sum(write_weights * torch.log(write_weights + 1e-10), dim=-1).mean()
        
        top_vals, top_idx = torch.topk(write_weights, k=self.topk, dim=-1)
        sparse = torch.zeros_like(write_weights).scatter_(-1, top_idx, top_vals)
        sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-6)
        
        effective_update = write_gate * sparse.unsqueeze(-1)
        new_slot = gw_state.unsqueeze(1)
        
        delta = effective_update * (new_slot - prev_mem)
        next_mem = prev_mem + delta
        next_mem = torch.complex(torch.tanh(next_mem.real), torch.tanh(next_mem.imag))
        next_mem = self.mem_norm(next_mem)
        
        return read_out, next_mem, slot_entropy

class EMA_VQ(nn.Module):
    def __init__(self, dim, num_concepts):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim = dim
        self.embedding = nn.Embedding(num_concepts, dim * 2)
        self.embedding.weight.requires_grad = False
        init_range = 1.0 / math.sqrt(dim)
        self.embedding.weight.data.uniform_(-init_range, init_range)
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_concepts))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.decay = CONFIG["vq_decay"]
        self.epsilon = 1e-5

    def forward(self, z):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        x_sq = z_flat.pow(2).sum(1, keepdim=True)
        y_sq = self.embedding.weight.pow(2).sum(1)
        d = (x_sq + y_sq - 2 * z_flat @ self.embedding.weight.t()).clamp(min=0.0)
        
        indices = d.argmin(dim=-1)
        z_q = self.embedding(indices)
        
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_concepts).float()
                n_total = encodings.sum(0)
                self.ema_cluster_size.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
                
                cluster_size = self.ema_cluster_size + self.epsilon
                cluster_size = cluster_size + (cluster_size < 1.0).float()
                
                dw = encodings.t() @ z_flat.detach()
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                new_w = (self.ema_w / cluster_size.unsqueeze(1)).clamp(-5.0, 5.0)
                self.embedding.weight.copy_(new_w)
        
        loss_sample = CONFIG["vq_commitment_beta"] * ((z_q.detach() - z_flat)**2).mean(-1)
        z_q = z_flat + (z_q - z_flat).detach()
        
        probs = F.one_hot(indices, self.num_concepts).float().mean(0)
        norm_entropy = -torch.sum(probs * torch.log(probs + 1e-10)) / math.log(self.num_concepts)
        
        z_q_c = torch.complex(z_q[..., :self.dim], z_q[..., self.dim:])
        return z_q_c, loss_sample, indices, norm_entropy

# ==========================================
# 4. SACRSN MODEL
# ==========================================
class SACRSN_v84(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Embedding(vocab_size, dim * 2)
        self.norm = ComplexLayerNorm(dim)

        self.gate_mod = HermitianGating(dim)
        self.memory_module = AssociativeMemory(dim, CONFIG["mem_slots"])
        self.vq = EMA_VQ(dim, CONFIG["codebook_size"])
        self.sensory_cortex = SubmodalityAnchorModule(dim, CONFIG["n_sensory_anchors"])
        self.quadrature = QuadratureModule(dim)

        self.arbitrator = nn.Linear(dim * 2, 5)
        self.meta_head = nn.Linear(dim * 2, 4)
        self.decoder = nn.Linear(dim * 2, vocab_size)

        self.halt_bias = nn.Parameter(torch.tensor(CONFIG["halt_bias_init"]))
        self.halt_scale = nn.Parameter(torch.tensor(CONFIG["halt_scale_init"]))
        self.input_gate = nn.Parameter(torch.tensor(0.0))

        self.register_buffer('vq_loss_ema', torch.tensor(1.0))
        self.register_buffer('train_step', torch.tensor(0))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _step_modules(self, curr, mem):
        v_out, vq_loss, idx, b_ent = self.vq(curr)
        g_out = self.gate_mod(curr)
        m_out, c_mem, s_ent = self.memory_module(curr, mem)
        s_out, v_at, a_at = self.sensory_cortex(curr)
        q_out = self.quadrature(curr)
        return (v_out, vq_loss, idx, b_ent, g_out, m_out, c_mem, s_ent, s_out, v_at, a_at, q_out)

    def _arbitrate(self, curr, meta):
        flat = torch.cat([curr.real, curr.imag], dim=-1)
        raw = self.arbitrator(flat)
        
        bias = (meta[:, 0] - 0.5) * CONFIG["critic_weight"] * 4.0
        raw = raw.clone()
        raw[:, 4] = raw[:, 4] + bias
        
        raw = raw - raw.max(dim=-1, keepdim=True)[0].detach()
        gates = F.softmax(raw, dim=-1)
        
        w_unc = meta[:, 3:4]
        uniform = torch.ones_like(gates) / gates.size(-1)
        mix = 0.05 + 0.10 * w_unc
        gates = (1 - mix) * gates + mix * uniform
        
        ent = -torch.sum(gates * torch.log(gates + 1e-10), dim=-1).mean()
        return gates, ent

    def _halt(self, vq_loss, meta):
        bias = F.softplus(self.halt_bias)
        scale = F.softplus(self.halt_scale)
        norm_loss = vq_loss / (self.vq_loss_ema + 1e-6)
        halt_prob = torch.sigmoid(bias - scale * norm_loss)
        
        w_unc = meta[:, 3]
        halt_prob = (halt_prob * (1.0 - 0.3 * w_unc)).clamp(0.0, 1.0)
        
        if self.training and self.train_step < 5: 
            halt_prob = halt_prob * 0.1
            
        if self.training:
            hard = (halt_prob > torch.rand_like(halt_prob)).float()
        else:
            hard = (halt_prob > 0.5).float()
            
        return (hard - halt_prob.detach() + halt_prob).unsqueeze(1)

    def forward_step(self, x_t, gw_state, mem_state):
        alpha = torch.sigmoid(self.input_gate)
        gw_state = alpha * gw_state + (1.0 - alpha) * x_t
        
        B = x_t.size(0)
        active = torch.ones(B, 1, device=DEVICE)
        accum = {k: torch.zeros(B, device=DEVICE) for k in ['vq', 'ponder', 'phase']}
        module_wins = torch.zeros(B, 5, device=DEVICE)
        
        tot_batch_ent = tot_gate_ent = tot_slot_ent = tot_sense_ent = 0
        total_steps = torch.zeros(B, device=DEVICE)
        final_idx = torch.zeros(B, dtype=torch.long, device=DEVICE)
        
        init_mag = gw_state.abs()
        prev_angle = torch.where(init_mag > 1e-6, (gw_state + 1e-9).angle(), torch.zeros_like(gw_state.real))
        
        meta_accum = torch.zeros(B, 4, device=DEVICE)
        steps_taken = 0

        for i in range(CONFIG["max_recursion"]):
            steps_taken += 1
            curr = self.norm(gw_state)
            flat = torch.cat([curr.real, curr.imag], dim=-1)
            meta = torch.sigmoid(self.meta_head(flat))
            meta_accum += meta
            
            (v_out, vq_l, idx, b_ent, g_out, m_out, c_mem, s_ent,
             s_out, v_at, a_at, q_out) = self._step_modules(curr, mem_state)
            
            if self.training:
                with torch.no_grad():
                    self.vq_loss_ema.mul_(0.95).add_(vq_l.mean(), alpha=0.05)

            if active.sum() > 0:
                tot_batch_ent += b_ent
                tot_slot_ent += s_ent
                ve = -torch.sum(v_at * torch.log(v_at + 1e-10), -1).mean()
                ae = -torch.sum(a_at * torch.log(a_at + 1e-10), -1).mean()
                tot_sense_ent += 0.5 * (ve + ae)

            gates, g_ent = self._arbitrate(curr, meta)
            module_wins += gates * active
            if active.sum() > 0: tot_gate_ent += g_ent
            
            proposals = torch.stack([g_out, m_out, v_out, s_out, q_out], dim=1)
            gates_c = torch.complex(gates.unsqueeze(-1), torch.zeros_like(gates.unsqueeze(-1)))
            winner_vec = torch.sum(proposals * gates_c, dim=1)
            
            shadow_idx = gates.argmin(dim=-1)
            shadow_prop = proposals[torch.arange(B, device=DEVICE), shadow_idx]
            dyn_shadow = CONFIG["shadow_weight"] * meta[:, 3:4]
            update = winner_vec + dyn_shadow * shadow_prop
            
            update = torch.complex(torch.tanh(update.real), torch.tanh(update.imag))
            cand = 0.6 * curr + 0.4 * update
            
            # Nan-to-num safety
            c_real = torch.nan_to_num(cand.real, nan=0.0, posinf=1.0, neginf=-1.0)
            c_imag = torch.nan_to_num(cand.imag, nan=0.0, posinf=1.0, neginf=-1.0)
            cand = torch.complex(c_real, c_imag)
            cand = torch.complex(torch.tanh(cand.real), torch.tanh(cand.imag))
            
            cand_safe = cand + 1e-9
            mag = cand_safe.abs()
            valid = mag > 1e-6
            
            prev_angle_det = prev_angle.detach()
            safe_angle = torch.where(valid, cand_safe.angle(), prev_angle_det)
            
            diff = (safe_angle - prev_angle_det).abs()
            diff = torch.min(diff, 2*math.pi - diff)
            
            mask_v = active.view(-1)
            accum['phase'] += mask_v * diff.mean(-1).view(-1) * valid.float().mean(-1).view(-1)
            prev_angle = safe_angle
            
            stop = self._halt(vq_l, meta)
            
            accum['ponder'] += mask_v * CONFIG["ponder_cost"]
            total_steps += mask_v
            accum['vq'] += mask_v * vq_l.view(-1)
            
            still_active = mask_v > 0.5
            final_idx = torch.where(still_active, idx, final_idx)
            
            # [FIX] Decoupled Gradient Update (Additive, Mask maintained as [B,1])
            active_detach = active.detach() # Keep [B, 1]
            active_c = torch.complex(active_detach, torch.zeros_like(active_detach))
            
            # gw_state [B, 64], cand [B, 64], active_c [B, 1] -> Correct Broadcast
            gw_state = gw_state + active_c * (cand - gw_state)
            
            # [FIX] Memory Broadcasting
            mem_mask = active_detach.unsqueeze(-1) # [B, 1, 1]
            mem_mask = torch.complex(mem_mask, torch.zeros_like(mem_mask))
            # mem_state [B, 32, 64]
            mem_state = mem_state + mem_mask * (c_mem - mem_state)
            
            active = active * (1.0 - stop)
            if active.sum() == 0: break
            
        if self.training: self.train_step += 1
        
        div = total_steps.clamp(min=1.0)
        accum['vq'] /= div
        avg_depth = total_steps.mean()
        accum['depth_per_sample'] = total_steps
        
        return (gw_state, mem_state, accum, final_idx,
                tot_batch_ent/max(1,steps_taken), tot_gate_ent/max(1,steps_taken),
                tot_slot_ent/max(1,steps_taken), avg_depth,
                meta_accum/max(1,steps_taken), module_wins,
                tot_sense_ent/max(1,steps_taken))

    def forward(self, x_seq, mem_state=None):
        B, T = x_seq.shape
        emb = self.encoder(x_seq)
        gw_seq = torch.complex(emb[..., :self.dim], emb[..., self.dim:])
        
        if mem_state is None: mem_state = self.memory_module.init_state(B)
        gw = torch.zeros_like(gw_seq[:, 0])
        
        outputs, idxs = [], []
        stats = {k: 0.0 for k in ['vq','ponder','phase','ent','gate_ent','slot_ent','depth','sensory_ent']}
        depth_acc = torch.zeros(B, device=DEVICE)
        meta_acc = torch.zeros(B, 4, device=DEVICE)
        final_wins = None
        
        for t in range(T):
            x_t = gw_seq[:, t]
            (gw, mem_state, step_stats, idx, b_e, g_e, s_e, d, m_v, w, se) = \
                self.forward_step(x_t, gw, mem_state)
                
            outputs.append(gw)
            idxs.append(idx)
            for k in ['vq', 'ponder', 'phase']: stats[k] += step_stats[k].mean()
            stats['ent'] += b_e
            stats['gate_ent'] += g_e
            stats['slot_ent'] += s_e
            stats['depth'] += d
            stats['sensory_ent'] += se
            depth_acc += step_stats['depth_per_sample']
            meta_acc += m_v
            final_wins = w
            
        out = torch.stack(outputs, dim=1)
        flat = torch.cat([out.real, out.imag], dim=-1)
        logits = self.decoder(flat)
        
        for k in stats: stats[k] /= T
        meta_acc /= T
        stats['depth_per_sample'] = depth_acc / T
        
        return logits, stats, mem_state, torch.stack(idxs, dim=1), meta_acc, final_wins

# ==========================================
# 5. UTILS & VISUALIZATION
# ==========================================
class HippocampalBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def _update_stats(self, x):
        self._n += 1; d = x - self._mean; self._mean += d/self._n; self._M2 += d*(x-self._mean)
    def add(self, x, y, val):
        self._update(val)
        var = self._M2 / max(1, self._n-1)
        z = (val - self._mean) / max(math.sqrt(var), 1e-6)
        entry = (z, self._cnt, x, y)
        self._cnt += 1
        if len(self.buffer) < self.capacity: heapq.heappush(self.buffer, entry)
        elif z > self.buffer[0][0]: heapq.heapreplace(self.buffer, entry)
    def sample(self, k):
        if not self.buffer: return None
        return random.sample(self.buffer, min(len(self.buffer), k))

def get_batch(data, bs, seq_len):
    ix = torch.randint(0, len(data)-seq_len-1, (bs,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def generate_text(model, tokenizer, seed="The", length=60, verbose=False):
    print(f"\n── Generating ('{seed}') ──")
    model.eval()
    curr = torch.tensor([tokenizer.encode(seed)], device=DEVICE)
    mem = None
    mod_names = ["Gate", "Mem", "VQ", "Sense", "Quad"]
    
    if verbose: print("Stream of Consciousness:")
    
    with torch.no_grad():
        try:
            _, _, mem, _, _, _ = model(curr, mem)
            curr = curr[:, -1:]
            for i in range(length):
                logits, _, mem, _, _, wins = model(curr, mem)
                if torch.isnan(logits).any(): print("NaN logits"); break
                
                if verbose or i < 5:
                    win = torch.argmax(wins[0]).item()
                    print(f"[{mod_names[win]}]", end="-> ")
                
                l = logits[:, -1, :] / 0.8
                l = l - l.max()
                p = F.softmax(l, dim=-1)
                nid = torch.multinomial(p, 1)
                print(tokenizer.decode([nid.item()]), end="", flush=True)
                curr = nid
        except Exception as e: print(e)
    print("\n")

def dream_graph_walk(transitions, steps=20):
    print("\n── Dream Graph Walk ──")
    if not transitions: return
    graph = defaultdict(list)
    for u, v in transitions: graph[u].append(v)
    
    curr = random.choice(list(graph.keys()))
    print(f"[{curr}]", end="")
    for _ in range(steps):
        if curr in graph and graph[curr]:
            curr = random.choice(graph[curr])
            print(f" -> {curr}", end="")
        else: break
    print("\n")

def extract_logic_rules(transitions, tokenizer):
    print("\n── Extracted Rules ──")
    print(f"{'FROM':<6} -> {'TO':<6} | {'COUNT':<6}")
    for (u, v), c in Counter(transitions).most_common(10):
        print(f"{u:<6} -> {v:<6} | {c:<6}")

def anomaly_detector(model, tokenizer):
    print("\n── Anomaly Detector ──")
    normal = "The structure of the mind."
    weird = "The banana of the galaxy eats time."
    def score(t):
        ids = tokenizer.encode(t)
        x = torch.tensor([ids[:-1]], device=DEVICE)
        with torch.no_grad():
            l, _, _, _, _, _ = model(x)
            y = torch.tensor([ids[1:]], device=DEVICE)
            return F.cross_entropy(l.reshape(-1, len(tokenizer.vocab)), y.reshape(-1)).item()
    s1, s2 = score(normal), score(weird)
    print(f"Normal: {s1:.4f} | Weird: {s2:.4f}")

def visualize_suite(history, transitions, meta_history):
    if not history.get('loss'): return
    print("\n--- Saving Diagnostics ---")
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1); plt.plot(history['loss'], label='Loss'); plt.legend()
    plt.subplot(2, 3, 2); plt.plot(history['depth'], color='purple', label='Depth'); plt.legend()
    
    plt.subplot(2, 3, 3)
    mh = np.array(meta_history)
    if len(mh) > 0:
        for i, l in enumerate(["Eth","Div","Pon","Unc"]): plt.plot(mh[:,i], label=l)
    plt.legend()

    plt.subplot(2, 3, 4)
    G = nx.DiGraph()
    for (u, v), w in Counter(transitions).most_common(50): G.add_edge(u, v, weight=w)
    if G.nodes:
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw(G, pos, node_size=20, width=0.5, alpha=0.7, edge_color='teal')
    plt.title("Topology")
    
    plt.subplot(2, 3, 5)
    if history['grads']:
        keys = list(history['grads'][0].keys())
        for k in keys: plt.plot([x[k] for x in history['grads']], label=k)
    plt.legend(fontsize=6)
    
    plt.tight_layout()
    plt.savefig('sacrsn_diagnostics.png')
    plt.close()

def calc_loss(logits, y, stats, meta, plast, ep, tot_ep):
    loss_ce = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
    
    confidence = F.softmax(logits, -1).max(-1)[0].mean()
    ego_pen = confidence * loss_ce * CONFIG["ego_penalty_weight"]
    
    d = stats.get('depth_per_sample', stats['depth'])
    if isinstance(d, torch.Tensor):
        under = F.relu(CONFIG['depth_min'] - d)
        over = F.relu(d - CONFIG['depth_max'])
        d_pen = CONFIG['depth_target_weight'] * (under + over).mean()
    else: d_pen = 0
    
    prog = ep / tot_ep
    ent_w = CONFIG['entropy_weight_start']*(1-prog) + CONFIG['entropy_weight_end']*prog
    w_eth, w_div, w_pon, w_unc = meta.mean(0)
    
    struct = 1.0 + 0.3*(plast - 1.0)
    struct_loss = struct * (
        0.1 * stats['vq'] +
        (stats['ponder'] * (1.0 + w_pon)) +
        CONFIG['phase_reg'] * stats['phase'] -
        (ent_w * (1.0 + w_div)) * stats['ent'] -
        CONFIG['gate_sparsity_weight'] * stats['gate_ent'] -
        CONFIG['slot_balance_weight'] * stats['slot_ent'] -
        CONFIG['sensory_reg'] * stats['sensory_ent'] +
        d_pen
    )
    return loss_ce, loss_ce * plast + ego_pen + struct_loss

# ═══════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    txt = load_training_data()
    tok = RobustBPE(); tok.train(txt)
    CONFIG["vocab_size"] = len(tok.vocab)
    data = torch.tensor(tok.encode(txt), dtype=torch.long)
    
    model = SACRSN_v84(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])
    hippo = HippocampalBuffer(CONFIG["replay_size"])
    
    hist = defaultdict(list)
    transitions = deque(maxlen=2000)
    meta_hist = []
    avg_surp = 0.0 
    
    print("\n── Training ──")
    model.train()
    try:
        for ep in range(CONFIG["epochs"]):
            x, y = get_batch(data, CONFIG["batch_size"], CONFIG["seq_len"])
            opt.zero_grad()
            
            with torch.no_grad():
                for m in [model.arbitrator, model.meta_head, model.decoder]:
                    m.weight.mul_(1.0 - CONFIG["synaptic_decay"])
            
            logits, stats, _, idx, meta, _ = model(x)
            
            idx_np = idx.detach().cpu().numpy()
            for b in range(idx_np.shape[0]):
                for t in range(idx_np.shape[1]-1):
                    transitions.append((idx_np[b,t], idx_np[b,t+1]))
            
            ce_raw = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1))
            plast = 1.0
            if CONFIG["active_plasticity"]:
                with torch.no_grad():
                    avg_surp = CONFIG["surprise_decay"]*avg_surp + (1-CONFIG["surprise_decay"])*ce_raw.item()
                plast = min(1.5, 1.0 + avg_surp * 0.5)
            
            l_ce, loss = calc_loss(logits, y, stats, meta, plast, ep, CONFIG["epochs"])
            
            if torch.isnan(loss):
                print(f"[!] NaN at ep {ep}"); opt.zero_grad(); continue
                
            loss.backward()
            
            if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                print(f"[!] NaN grad at ep {ep}"); opt.zero_grad(); continue
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            if ep % 20 == 0: hist['grads'].append(log_grad_norms(model))
            opt.step()
            
            hippo.add(x.cpu(), y.cpu(), l_ce.item())
            if ep > 0 and ep % CONFIG["sleep_every"] == 0:
                dream = hippo.sample(4)
                if dream:
                    for _, _, dx, dy in dream:
                        opt.zero_grad()
                        dl, ds, _, _, dm, _ = model(dx.to(DEVICE))
                        _, d_loss = calc_loss(dl, dy.to(DEVICE), ds, dm, 1.0, ep, CONFIG["epochs"])
                        if not torch.isnan(d_loss):
                            d_loss.backward()
                            opt.step()
            
            sched.step()
            hist['loss'].append(loss.item())
            hist['depth'].append(stats['depth'].item())
            hist['ponder'].append(stats['ponder'].item())
            meta_hist.append(meta.mean(0).detach().cpu().numpy())
            
            if ep % 20 == 0:
                print(f"Ep {ep:03d} | Loss: {loss.item():.3f} | PPX: {math.exp(min(100, l_ce.item())):.1f} | "
                      f"Depth: {stats['depth'].item():.2f} | Plast: {plast:.2f}")
                if hist['grads']:
                    gn = hist['grads'][-1]
                    dom = max(gn, key=gn.get)
                    print(f"         Dominant: {dom}")

    except KeyboardInterrupt: pass
    
    generate_text(model, tok, verbose=True)
    dream_graph_walk(transitions)
    extract_logic_rules(transitions, tok)
    anomaly_detector(model, tok)
    visualize_suite(hist, transitions, meta_hist)

if __name__ == "__main__":
    main()