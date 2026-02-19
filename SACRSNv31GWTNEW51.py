"""
╔══════════════════════════════════════════════════════════════════════════╗
║         SACRSN — UNIFIED EDITION  (Merge of v38 → v74)                 ║
║  Self-Adaptive Cognitive Recurrent State Network                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  MATHEMATICAL FRAMEWORK                                                  ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  • Complex-valued state space  z = Re(z) + i·Im(z)                     ║
║  • Hermitian gating: score = Re(q * conj(k))  →  gate = σ(score)       ║
║  • EMA Vector-Quantisation with cluster-normalised codebook update       ║
║  • Phase-coherence regularisation: Δφ = |∠z_t – ∠z_{t-1}|             ║
║  • Topk sparse memory write with associative cosine-similarity read     ║
║                                                                          ║
║  ENGINEERING ARCHITECTURE                                                ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  • 5-module Global Workspace: Gate · Memory · VQ · Sensory · Critic     ║
║  • Adaptive T.O.T.E. halting (learnable bias + scale on VQ loss EMA)   ║
║  • Hippocampal replay with online z-score priority                      ║
║  • Cosine-annealed AdamW, gradient clipping, NaN-guard                  ║
║  • Synaptic decay (L2 weight noise) for continual learning              ║
║                                                                          ║
║  NEUROLINGUISTIC GROUNDING                                               ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  • RobustBPE tokeniser (proper greedy merge ordering)                   ║
║  • Hebbian sensory palettes: visual (Real) + auditory (Imaginary)       ║
║  • T.O.T.E. loop ≡ cortical recurrence / predictive coding cycle       ║
║  • Dopamine plasticity: surprise → learning rate modulation             ║
║  • Ego/metacognitive penalty: confidence on wrong answer is penalised   ║
║  • Shadow (unconscious) processing: complement of winning module        ║
║  • Meta-values: Ethics · Diversity · Ponder · Uncertainty               ║
║  • Sleep consolidation: hippocampal replay of high-surprise events      ║
║  • Dream graph walk: free-associative concept traversal                 ║
║  • Anomaly detection: perplexity divergence from norm                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import math
import heapq
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter, deque

# ═══════════════════════════════════════════════════════════════════════════
# 0.  STRICT DETERMINISM & DEVICE
#     Engineering: ensure bit-for-bit reproducibility across runs.
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    print("Warning: strict determinism unavailable on this hardware.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════
# 0a. UNIFIED CONFIGURATION
#     All hyper-parameters in one place for easy experiment management.
# ═══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Architecture ────────────────────────────────────────────────────────
    "seq_len":           32,
    "batch_size":        16,
    "embed_dim":         64,    # Complex dimension d; each module uses d real + d imag
    "mem_slots":         32,    # Associative memory slot count
    "mem_topk":          3,     # Sparse write: only top-k slots receive update
    "vocab_size":        None,  # Set automatically after tokeniser trains
    "codebook_size":     128,   # VQ codebook concepts
    "n_sensory_anchors": 32,    # Hebbian submodality palette size

    # ── Cognitive Dynamics (T.O.T.E. loop) ───────────────────────────────
    "max_recursion":     5,     # Max cortical cycles per token.
                                 # Reduced from 8: deeper recursion = more
                                 # backward-through-time passes through complex
                                 # ops → gradient explosion. 5 is still rich.
    "halt_bias_init":    3.0,   # Learnable halt sigmoid bias
    "halt_scale_init":   10.0,  # Learnable halt scale
    "ponder_cost":       0.01,  # Penalty per extra iteration (effort = time)

    # ── Shadow (Unconscious) Processing ──────────────────────────────────
    "shadow_weight":     0.15,  # Weakest-module (shadow) influence on workspace update.
                                 # Reduced from 0.3: shadow is scaled by uncertainty too
                                 # (dynamic_shadow = shadow_weight * w_unc), so the
                                 # effective maximum is already 0.15 * 1.0 = 0.15 of the
                                 # winner vec. 0.3 was too strong during early training
                                 # when the weakest module may be incoherent noise.
    "critic_weight":     0.1,   # Strength of the critic's ethical modulation

    # ── EMA-VQ Stability ─────────────────────────────────────────────────
    "vq_commitment_beta":0.25,  # Commitment loss weight
    "vq_decay":          0.99,  # EMA decay for codebook cluster tracking

    # ── Regularisation Schedule ───────────────────────────────────────────
    # CORE terms (keep these; they directly shape learning):
    "entropy_weight_start": 0.10,  # VQ batch entropy weight at epoch 0
    "entropy_weight_end":   0.02,  # VQ batch entropy weight at final epoch
    "gate_sparsity_weight": 0.01,  # Penalise diffuse gate distributions (keep sharp competition)
    "phase_reg":            0.005, # Penalise large inter-step phase jumps
                                   # Halved from 0.01 — phase_reg is second-order;
                                   # too strong and it fights the TOTE recurrence.

    # SECOND-ORDER terms (reduce/zero first if training stagnates):
    "slot_balance_weight":  0.005, # Reward uniform memory-slot usage (halved — advisory)
    "sensory_reg":          0.005, # Reward diverse sensory attention (halved — advisory)

    # ── Depth (T.O.T.E. step count) Targets ──────────────────────────────
    "depth_target_weight": 0.05,
    "depth_min":           2.0,
    "depth_max":           6.0,
    "target_depth":        4.0,

    # ── Metacognition / Neurolinguistics ──────────────────────────────────
    "ego_penalty_weight":  0.5,  # Metacognitive: penalise confident wrong guesses
    "synaptic_decay":      1e-5, # L2 weight noise for continual plasticity
    "surprise_decay":      0.9,  # EMA decay for dopamine signal
    "active_plasticity":   True, # Enable dopamine-modulated learning rate

    # ── Training ──────────────────────────────────────────────────────────
    "epochs":       300,
    "lr":           1e-4,   # Reduced from 3e-4: high LR × plasticity 1.5 amplified
                             # gradient instability during the critical first epochs.
    "grad_clip":    0.5,
    "replay_size":  200,
    "sleep_every":  50,   # Hippocampal replay consolidation interval
}

print("╔══════════════════════════════════╗")
print("║  SACRSN — UNIFIED EDITION        ║")
print(f"║  Device: {DEVICE:<24}║")
print("╚══════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & ROBUST BPE TOKENISER
#     Neurolinguistics: greedy BPE captures morphological structure;
#     proper merge ordering ensures consistency between train and encode.
# ═══════════════════════════════════════════════════════════════════════════
def load_training_data(filepath: str = "data.txt") -> str:
    """Load corpus or fall back to embedded philosophical text."""
    FALLBACK = (
        "The neural architecture of the mind is a mirror of the cosmos itself. "
        "As above, so below; the filamentary structures of the intergalactic web "
        "find their precise echo in the dense white matter connecting the hemispheres. "
        "Galaxies cluster like neurons in a cosmic synapse, and the voids between them "
        "echo the silence between thought. We are stardust contemplating its own "
        "arrangement — a fleeting configuration of atoms attempting to comprehend the "
        "laws that bound them together. To understand the nature of thought, one must "
        "first understand the nature of the void. It is the negative space that defines "
        "the positive, the silence that gives shape to the sound. In the absolute zero "
        "of the vacuum, potential energy waits, just as a thought waits on the precipice "
        "of expression. Logic is the foundation, but chaos is the architect. Without the "
        "rigid framework of logic, the structure collapses; without the unpredictability "
        "of chaos, the structure creates nothing new. Entropy is not the enemy of "
        "intelligence, but its fuel — the friction that generates the heat of creativity."
    )
    if os.path.exists(filepath):
        print(f"Loading corpus from '{filepath}'...")
        # Try common encodings in order; Windows text editors often save
        # in cp1252 (byte 0x97 = em-dash), which is not valid UTF-8.
        for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    data = f.read()
                print(f"  Encoding detected: {enc}")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            # latin-1 accepts every byte 0-255, so this is the guaranteed fallback
            with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
                data = f.read()
            print("  Warning: file encoding unknown — some characters may be replaced.")
        if len(data.strip()) < 50:
            print("File too small — using internal fallback corpus.")
            data = FALLBACK
    else:
        print("File not found — creating default corpus.")
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(FALLBACK)
        data = FALLBACK
    return data


class RobustBPE:
    """
    Byte-Pair Encoding tokeniser with correct greedy merge ordering.

    Neurolinguistic note:
        BPE mirrors the brain's tendency to chunk repeated co-occurrences
        into higher-level morphological units — sub-word tokens act as
        the lexicon's 'phoneme clusters' rather than arbitrary byte slices.

    Mathematical note:
        Each merge iteration selects argmax over bigram frequency;
        the greedy ordering is tracked by rank (self.merges[pair] = i)
        and re-applied in rank order during encode to guarantee
        consistency with training.
    """
    def __init__(self, target_vocab_size: int = 1000):
        self.target_size = target_vocab_size
        self.vocab       = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.merges: dict = {}
        self.reverse_vocab: dict = {}

    # ── Internal helpers ──────────────────────────────────────────────────
    def _get_stats(self, vocab: dict) -> dict:
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            syms = word.split()
            for i in range(len(syms) - 1):
                pairs[syms[i], syms[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair: tuple, v_in: dict) -> dict:
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        return {p.sub(''.join(pair), word): freq for word, freq in v_in.items()}

    # ── Public interface ──────────────────────────────────────────────────
    def train(self, text: str) -> None:
        print("Training BPE tokeniser...")
        words = re.findall(r"[\w']+|[^\s\w]", text)
        chars = set("".join(words))
        for c in sorted(chars):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        word_freqs = Counter([" ".join(list(w)) for w in words])

        for i in range(self.target_size - len(self.vocab)):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges[best]          = i          # rank for encode ordering
            self.vocab["".join(best)]  = len(self.vocab)
            word_freqs                 = self._merge_vocab(best, word_freqs)

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._merge_set    = set(self.merges.keys())
        print(f"  BPE complete → vocab size: {len(self.vocab):,}")

    def encode(self, text: str) -> list:
        """Greedy tokenisation: apply merges in training rank order."""
        words = re.findall(r"[\w']+|[^\s\w]", text)
        ids   = [self.vocab["<BOS>"]]
        for word in words:
            tokens = list(word)
            while len(tokens) > 1:
                best_rank, best_pair, best_idx = float('inf'), None, -1
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self._merge_set and self.merges[pair] < best_rank:
                        best_rank = self.merges[pair]
                        best_pair = pair
                        best_idx  = i
                if best_pair is None:
                    break
                tokens[best_idx] = "".join(best_pair)
                del tokens[best_idx + 1]
            for t in tokens:
                ids.append(self.vocab.get(t, self.vocab["<UNK>"]))
        ids.append(self.vocab["<EOS>"])
        return ids

    def decode(self, ids: list) -> str:
        """Convert token IDs back to a human-readable string.

        Engineering fix (E-02): tokens are concatenated without spaces.
        BPE sub-word tokens already encode word boundaries via the merge
        vocabulary; joining with spaces would produce 'T h e' for any word
        that fell back to character-level tokenisation.
        """
        valid  = [i for i in ids if i > 3]       # skip special tokens
        tokens = [self.reverse_vocab.get(i, "") for i in valid]
        text   = "".join(tokens)                  # [E-02 FIX] no spaces between tokens
        # Minimal detokenisation: restore spacing around punctuation
        text = re.sub(r'([.,;:!?])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ═══════════════════════════════════════════════════════════════════════════
# 2.  MATHEMATICAL PRIMITIVES  (Complex-Valued Neural Algebra)
#
#     The model operates over ℂ^d  where the Real part encodes
#     'what' (magnitude / semantic content) and the Imaginary part
#     encodes 'how / when' (phase / contextual relationship).
# ═══════════════════════════════════════════════════════════════════════════

class ComplexLinear(nn.Module):
    """
    Complex-valued linear map: W·z for z ∈ ℂ^d.

    Math:
        W = Wr + i·Wi  (split real + imaginary weight matrices)
        (Wr + i·Wi)(a + ib) = (Wr·a − Wi·b) + i(Wi·a + Wr·b)

    This is a strict implementation of complex matrix multiplication,
    not a naive concatenation — it preserves the algebraic structure of ℂ.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.Wr = nn.Linear(in_features, out_features)
        self.Wi = nn.Linear(in_features, out_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        real = self.Wr(z.real) - self.Wi(z.imag)
        imag = self.Wi(z.real) + self.Wr(z.imag)
        return torch.complex(real, imag)


class StableComplexNorm(nn.Module):
    """
    Normalise both real and imaginary components independently.

    Used internally in AssociativeMemory to keep slot magnitudes stable.
    NOTE: this distorts the phase angle — do NOT use in the main TOTE loop.
    For the TOTE loop, ComplexLayerNorm (phase-preserving) is used instead.
    See audit finding M-04.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm_r = nn.LayerNorm(dim, eps=1e-6)
        self.norm_i = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.norm_r(z.real), self.norm_i(z.imag))


class ComplexLayerNorm(nn.Module):
    """
    Phase-preserving normalisation — PRIMARY NORM for the TOTE loop.

    Math:
        mag  = |z| = sqrt(Re² + Im²)
        z_norm = ((mag − μ) / σ) * scale + shift
        output = z_norm * e^{i·∠z}   (preserves phase direction)

    Neurolinguistic: analogous to gain control in sensory cortex — the
    'volume' of activation is normalised while its 'direction' (phase)
    is preserved, allowing phase coding to carry contextual information.

    This is the correct choice for the TOTE loop because the phase
    regularisation term measures ∠z changes between steps. Using a
    norm that distorts ∠z (StableComplexNorm) would make that
    regularisation incoherent. See audit finding M-04.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # SAFE MAGNITUDE — epsilon must be INSIDE the sqrt.
        # torch.abs(z) backward at zero magnitude: grad = Re/sqrt(Re²+Im²) = 0/0 = NaN.
        # gw_state is zeros at the start of every token, so this NaN fires on the
        # very first TOTE step of every forward pass and poisons the whole graph.
        # Fix: (Re²+Im²+eps).sqrt() → grad = Re/sqrt(Re²+Im²+eps) = 0/sqrt(eps) = 0.
        mag  = (z.real.pow(2) + z.imag.pow(2) + 1e-8).sqrt()

        mean = mag.mean(dim=-1, keepdim=True)

        # DETACH VAR — grad of 1/sqrt(var+eps) w.r.t. var = -1/(2*(var+eps)^1.5).
        # At var≈0 (uniform near-zero embedding init): -1/(2*(1e-6)^1.5) = -5×10^8.
        # Detach makes the denominator a fixed scaling constant per step, which is
        # equivalent to RMS-Norm behaviour and removes the gradient explosion.
        var  = mag.detach().var(dim=-1, keepdim=True)

        norm_mag = (mag - mean) / (var + 1e-4).sqrt() * self.scale + self.shift

        # Angle detached: atan2(0,0) gradient = 0/0 = NaN at zero magnitude.
        angle = z.angle().detach()
        return torch.complex(norm_mag * torch.cos(angle),
                             norm_mag * torch.sin(angle))


class HermitianGating(nn.Module):
    """
    Self-gating attention via the Hermitian inner product: ⟨q, k⟩ = q·k*.

    Math:
        q·k* = (q_r + i·q_i)(k_r − i·k_i)
             = (q_r·k_r + q_i·k_i) + i(q_i·k_r − q_r·k_i)
        score = Re(q·k*) / sqrt(d)   ← scaled before summing
        gate  = σ(score.sum(-1))     ∈ (0,1)
        output = gate ⊙ v

    Engineering note: the 1/√d scaling is mandatory. Without it, summing
    d independent N(0,1) terms gives variance = d, so raw scores are O(√d)
    before any training. At d=64 that is σ(8) ≈ 1.0 everywhere → zero
    gradient → NaN in backward from epoch 0. Standard scaled dot-product
    attention (Vaswani et al. 2017) uses this scaling for exactly this reason.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        self.scale  = dim ** -0.5     # 1/√d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v  = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        score_r  = (q * k.conj()).real * self.scale   # scale per-dim before sum
        score    = score_r.sum(dim=-1, keepdim=True)
        gate     = torch.sigmoid(score)
        return torch.complex(v.real * gate, v.imag * gate)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  COGNITIVE MODULES  (Global Workspace Theory)
#
#     Each module independently generates a 'proposal' and a 'salience'
#     score.  The arbitrator (meta-gating network) then broadcasts the
#     winning combination back into the shared Global Workspace state z.
#
#     Neurolinguistics:
#         GWT (Baars 1988): consciousness = broadcast of one winner across
#         specialised, otherwise independent, cortical processors.
# ═══════════════════════════════════════════════════════════════════════════

class AssociativeMemory(nn.Module):
    """
    Differentiable associative memory with sparse topk write gate.

    Architecture:
        Read  : cosine-similarity softmax over slots → weighted sum
        Write : address network selects top-k slots; write gate controls
                how much of the new state overwrites existing content.
        Norm  : StableComplexNorm after each write prevents slot decay.

    Neurolinguistic:
        Models episodic memory (hippocampus): context similarity retrieves
        past experience; write gate mirrors consolidation — only salient
        (high gate) events are strongly stored.
    """
    def __init__(self, dim: int, slots: int):
        super().__init__()
        self.dim   = dim
        self.slots = slots
        self.topk  = CONFIG["mem_topk"]

        self.gate_net    = nn.Linear(dim * 2, 1)
        self.address_net = nn.Linear(dim * 2, slots)
        self.mem_norm    = StableComplexNorm(dim)

    def init_state(self, batch_size: int) -> torch.Tensor:
        """Blank memory at session start."""
        return torch.complex(
            torch.zeros(batch_size, self.slots, self.dim, device=DEVICE),
            torch.zeros(batch_size, self.slots, self.dim, device=DEVICE)
        )

    def forward(self, gw_state: torch.Tensor, prev_mem: torch.Tensor):
        # ── READ: Hermitian cosine similarity ────────────────────────────
        q   = gw_state.unsqueeze(1)                  # (B, 1, d)
        sim = (prev_mem.conj() * q).real.sum(dim=-1) # (B, slots)
        sim = sim - sim.max(dim=-1, keepdim=True)[0].detach()
        attn    = F.softmax(sim, dim=-1).unsqueeze(-1)
        read_out = (prev_mem * torch.complex(attn, torch.zeros_like(attn))).sum(1)

        # ── WRITE: learned addressing + topk sparsity ────────────────────
        flat  = torch.cat([gw_state.real, gw_state.imag], dim=-1)
        write_gate = torch.sigmoid(self.gate_net(flat)).unsqueeze(1)

        logits = self.address_net(flat)
        logits = logits - logits.max(dim=-1, keepdim=True)[0].detach()
        write_weights = F.softmax(logits, dim=-1)

        # Slot-usage entropy (reward uniform distribution over slots)
        slot_entropy = -torch.sum(
            write_weights * torch.log(write_weights + 1e-10), dim=-1
        ).mean()

        # Sparsify: keep only top-k address weights
        top_vals, top_idx = torch.topk(write_weights, k=self.topk, dim=-1)
        sparse = torch.zeros_like(write_weights).scatter_(-1, top_idx, top_vals)
        sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-6)

        effective_update = write_gate * sparse.unsqueeze(-1)
        new_slot         = gw_state.unsqueeze(1)
        next_mem         = (1.0 - effective_update) * prev_mem + \
                           effective_update * new_slot
        next_mem         = self.mem_norm(next_mem)

        return read_out, next_mem, slot_entropy


class EMA_VQ(nn.Module):
    """
    Exponential Moving Average Vector Quantisation.

    Math:
        Forward : find nearest codebook vector e_k for input z
                  k* = argmin_k ||z − e_k||²
        Update  : EMA cluster sizes n_k ← α·n_k + (1−α)·count_k
                  EMA weights       w_k ← α·w_k + (1−α)·sum(z_i∈k)
                  new_weight        e_k = w_k / n_k
        Loss    : β · ||sg(e_{k*}) − z||²   (commitment, straight-through)

    Engineering:
        Codebook weights are frozen from the optimiser (pure EMA update).
        Cluster normaliser is clamped to ≥1 to prevent dead-code explosion.
        Codebook weights are clamped to [-5, 5] for training stability.

    Neurolinguistic:
        VQ ≡ categorical perception — continuous sensory input is snapped
        to a discrete 'phoneme-like' category, enabling compositional
        symbol-level reasoning in downstream modules.
    """
    def __init__(self, dim: int, num_concepts: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim          = dim

        self.embedding = nn.Embedding(num_concepts, dim * 2)
        self.embedding.weight.requires_grad = False   # pure EMA update

        init_range = 1.0 / math.sqrt(dim)
        self.embedding.weight.data.uniform_(-init_range, init_range)

        self.register_buffer('ema_cluster_size', torch.zeros(num_concepts))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

        self.decay   = CONFIG["vq_decay"]
        self.epsilon = 1e-5

    def forward(self, z: torch.Tensor):
        z_flat = torch.cat([z.real, z.imag], dim=-1)   # (B, 2d)

        # ── Distance ──────────────────────────────────────────────────────
        x_sq = z_flat.pow(2).sum(1, keepdim=True)
        y_sq = self.embedding.weight.pow(2).sum(1)
        d    = (x_sq + y_sq - 2 * z_flat @ self.embedding.weight.t()).clamp(min=0.0)
        indices = d.argmin(dim=-1)
        z_q     = self.embedding(indices)

        # ── EMA Codebook Update ───────────────────────────────────────────
        if self.training:
            with torch.no_grad():
                encodings   = F.one_hot(indices, self.num_concepts).float()
                n_total     = encodings.sum(0)
                self.ema_cluster_size.mul_(self.decay).add_(n_total, alpha=1 - self.decay)

                # clamp(min=1.0) is cleaner than the additive boolean trick —
                # no fractional jumps, and the EMA math is not distorted.
                cluster_size = (self.ema_cluster_size + self.epsilon).clamp(min=1.0)

                dw = encodings.t() @ z_flat.detach()
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                new_w = (self.ema_w / cluster_size.unsqueeze(1)).clamp(-5.0, 5.0)
                self.embedding.weight.copy_(new_w)

        # ── Commitment Loss + Straight-Through ───────────────────────────
        loss_sample = CONFIG["vq_commitment_beta"] * ((z_q.detach() - z_flat)**2).mean(-1)
        z_q         = z_flat + (z_q - z_flat).detach()

        # Normalised codebook usage entropy  ∈ [0, 1]
        probs      = F.one_hot(indices, self.num_concepts).float().mean(0)
        norm_entropy = -torch.sum(
            probs * torch.log(probs + 1e-10)
        ) / math.log(self.num_concepts)

        z_q_c = torch.complex(z_q[..., :self.dim], z_q[..., self.dim:])
        return z_q_c, loss_sample, indices, norm_entropy


class SubmodalityAnchorModule(nn.Module):
    """
    Sensory submodality retrieval via soft attention over learned anchor palettes.

    NLP Grounding (Bandler & Grinder submodalities):
        Internal representations possess sensory qualities — brightness, tone,
        spatial distance, texture. This module maintains two 'palettes' of
        prototype anchors, one per primary channel, and retrieves the nearest
        anchor to the current workspace state. The retrieved anchor enriches
        the state with sensory texture, mirroring how NLP practitioners
        access and modify submodalities to shift the 'feel' of an internal
        representation.

    Naming correction (audit fix N-02):
        Prior versions labelled this 'HebbianSensory', implying Hebb's
        biological co-occurrence rule (Δw ∝ x·y). The actual update is
        standard gradient descent through cross-entropy — there is no local
        Hebbian correlation signal. The label has been corrected.
        The *spirit* of Hebbian organisation is approximated in that anchors
        which co-occur with frequent workspace states will be pulled toward
        those states during training — but this is not strict Hebbian learning.

    Architecture:
        Visual palette  ← queried by the full complex state (Real = 'what')
        Auditory palette← queried by the full complex state (Imag = 'when/how')
        Palettes are orthogonally initialised to maximally span the embedding
        space at the start of training.

    Math:
        The two palette outputs are combined as:
            experience = vis_c + i · aud_c
        The auditory component is phase-rotated 90° to maintain orthogonality
        between the two sensory channels in ℂ^d.
    """
    def __init__(self, dim: int, n_anchors: int):
        super().__init__()
        self.dim = dim
        self.visual_palette = nn.Parameter(torch.randn(n_anchors, dim * 2))
        self.audio_palette  = nn.Parameter(torch.randn(n_anchors, dim * 2))
        nn.init.orthogonal_(self.visual_palette)
        nn.init.orthogonal_(self.audio_palette)
        self.proj = ComplexLinear(dim, dim)

    def forward(self, gw_state: torch.Tensor):
        z_s    = self.proj(gw_state)
        z_flat = torch.cat([z_s.real, z_s.imag], dim=-1)

        def _attend(palette):
            scores = z_flat @ palette.t()
            scores = scores - scores.max(dim=-1, keepdim=True)[0].detach()
            attn   = F.softmax(scores, dim=-1)
            return attn @ palette, attn

        vis_flat, vis_attn = _attend(self.visual_palette)
        aud_flat, aud_attn = _attend(self.audio_palette)

        vis_c = torch.complex(vis_flat[..., :self.dim], vis_flat[..., self.dim:])
        aud_c = torch.complex(aud_flat[..., :self.dim], aud_flat[..., self.dim:])

        # Auditory channel is 90° phase-rotated (i·aud) to stay orthogonal to visual
        experience = vis_c + torch.complex(-aud_c.imag, aud_c.real)
        return experience, vis_attn, aud_attn


class QuadratureModule(nn.Module):
    """
    Phase-diversity module: rotates the workspace 90° in ℂ (quadrature).

    Math:
        quadrature(z) = i · W(z)   where  i · (a+ib) = −b + ia
        This is a pure phase rotation — the module redirects, never amplifies.

    What this actually does (audit fix N-01):
        The rotation produces a vector orthogonal to the current workspace
        state, providing phase diversity in the arbitration pool. This
        encourages the Global Workspace to explore directions it has not
        yet considered — the NLP equivalent of de Bono's lateral thinking.

    What it does NOT do:
        Despite prior labelling as a 'CriticModule' or 'ethical filter',
        this module has no access to any ethical ground-truth signal and
        cannot learn ethical preferences. Ethical modulation is correctly
        located in the meta-head bias applied to the arbitrator —
        see `_arbitrate()`. If genuine ethical filtering is needed,
        a separate value-head trained on human preference data is required.

    NLP framing (corrected):
        In NLP terms this is the 'contrarian perceptual position' — the
        internal voice that systematically proposes the orthogonal view,
        keeping the workspace from collapsing into a single attractor.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = ComplexLinear(dim, dim)

    def forward(self, gw_state: torch.Tensor) -> torch.Tensor:
        z = self.net(gw_state)
        return torch.complex(-z.imag, z.real)   # multiply by i → 90° rotation


# ═══════════════════════════════════════════════════════════════════════════
# 4.  SACRSN — UNIFIED MODEL
#
#     Global Workspace Theory implementation with:
#       • 5 specialised modules competing for 'broadcast'
#       • Adaptive T.O.T.E. halting (cortical recurrence)
#       • Meta-cognition head (ethics / diversity / ponder / uncertainty)
#       • Shadow (unconscious) integration
#       • Learnable input injection gate
# ═══════════════════════════════════════════════════════════════════════════

class SACRSN_Unified(nn.Module):
    """
    SACRSN Unified — synthesis of v38 → v74, all audit fixes applied.

    Five modules compete each T.O.T.E. cycle:
        0. Gate       (HermitianGating)        — self-attention / working memory gate
        1. Memory     (AssociativeMemory)      — episodic retrieval
        2. VQ         (EMA_VQ)                 — categorical / symbolic grounding
        3. Sensory    (SubmodalityAnchorModule)— submodality palette enrichment
        4. Quadrature (QuadratureModule)       — phase-diversity / lateral thinking
                                                 [renamed from CriticModule, N-01]

    Meta-head produces 4 values at each step:
        w_ethics      — scale quadrature influence (via arbitrator bias)
        w_diversity   — scale entropy reward    (lexical diversity)
        w_ponder      — scale ponder cost       (effort awareness)
        w_uncertainty — dilute halt probability (epistemic humility)

    Audit fixes applied: M-01 M-02 M-03 M-04 M-05 E-01 E-02 E-03 E-04 E-05
                         N-01 N-02 N-03
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim

        # ── Input Encoding ─────────────────────────────────────────────────
        self.encoder = nn.Embedding(vocab_size, dim * 2)
        # [M-04 FIX] Use ComplexLayerNorm (phase-preserving) in the TOTE loop.
        # StableComplexNorm distorts ∠z, making phase regularisation incoherent.
        self.norm    = ComplexLayerNorm(dim)

        # ── The Five Workspace Modules ─────────────────────────────────────
        self.gate_mod        = HermitianGating(dim)
        self.memory_module   = AssociativeMemory(dim, CONFIG["mem_slots"])
        self.vq              = EMA_VQ(dim, CONFIG["codebook_size"])
        # [N-02 FIX] Renamed from HebbianSensory — gradient descent, not Hebb rule
        self.sensory_cortex  = SubmodalityAnchorModule(dim, CONFIG["n_sensory_anchors"])
        # [N-01 FIX] Renamed from CriticModule — quadrature rotation, not ethics
        self.quadrature      = QuadratureModule(dim)

        # ── Arbitration & Meta-Cognition ──────────────────────────────────
        self.arbitrator = nn.Linear(dim * 2, 5)    # 5 module weights
        self.meta_head  = nn.Linear(dim * 2, 4)    # ethics/diversity/ponder/uncertainty

        # ── Output ────────────────────────────────────────────────────────
        self.decoder = nn.Linear(dim * 2, vocab_size)

        # ── Learnable Halting Parameters ──────────────────────────────────
        self.halt_bias  = nn.Parameter(torch.tensor(CONFIG["halt_bias_init"]))
        self.halt_scale = nn.Parameter(torch.tensor(CONFIG["halt_scale_init"]))
        self.input_gate = nn.Parameter(torch.tensor(0.0))   # blending α

        # ── Running State Buffers ─────────────────────────────────────────
        self.register_buffer('vq_loss_ema', torch.tensor(1.0))
        self.register_buffer('train_step',  torch.tensor(0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            # Small std prevents large initial magnitudes that flow into
            # ComplexLayerNorm and HermitianGating, causing NaN at epoch 0.
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ── Module step ───────────────────────────────────────────────────────
    def _step_modules(self, curr_state: torch.Tensor, mem_state: torch.Tensor):
        v_out, vq_loss_s, indices, batch_ent = self.vq(curr_state)
        g_out                                = self.gate_mod(curr_state)
        m_out, cand_mem, slot_ent            = self.memory_module(curr_state, mem_state)
        s_out, vis_attn, aud_attn            = self.sensory_cortex(curr_state)
        q_out                                = self.quadrature(curr_state)  # [N-01 FIX]
        return (v_out, vq_loss_s, indices, batch_ent,
                g_out, m_out, cand_mem, slot_ent,
                s_out, vis_attn, aud_attn, q_out)

    # ── Arbitration ───────────────────────────────────────────────────────
    def _arbitrate(self, curr_state: torch.Tensor,
                   meta_values: torch.Tensor):
        """
        Soft-max arbitration with:
            • Ethical bias on critic channel
            • Uncertainty-driven entropy floor (prevents over-commitment)
        """
        flat       = torch.cat([curr_state.real, curr_state.imag], dim=-1)
        raw_logits = self.arbitrator(flat)

        # Ethical modulation: push critic (channel 4) up when ethics is high
        w_ethics   = meta_values[:, 0]
        ethics_bias = (w_ethics - 0.5) * CONFIG["critic_weight"] * 4.0
        raw_logits  = raw_logits.clone()
        raw_logits[:, 4] = raw_logits[:, 4] + ethics_bias

        raw_logits = raw_logits - raw_logits.max(dim=-1, keepdim=True)[0].detach()
        raw_gates  = F.softmax(raw_logits, dim=-1)

        # Uncertainty-driven entropy floor (prevents winner-take-all collapse)
        w_unc     = meta_values[:, 3:4]
        uniform   = torch.ones_like(raw_gates) / raw_gates.size(-1)
        mix       = 0.05 + 0.10 * w_unc
        gates     = (1 - mix) * raw_gates + mix * uniform

        gate_ent  = -torch.sum(gates * torch.log(gates + 1e-10), dim=-1).mean()
        return gates, gate_ent

    # ── T.O.T.E. Halting ─────────────────────────────────────────────────
    def _halt(self, vq_loss_sample: torch.Tensor,
              meta_values: torch.Tensor) -> torch.Tensor:
        """
        Adaptive halting: P(stop) = σ(bias − scale * norm_vq_loss)

        Engineering:
            • softplus ensures bias/scale remain positive
            • norm_loss is loss relative to running EMA baseline
            • uncertainty meta-value reduces halt probability (more pondering)
            • Hard stop uses straight-through estimator for gradient flow
            • Warmup: first 20 steps use 0.1× halt probability
        """
        bias      = F.softplus(self.halt_bias)
        scale     = F.softplus(self.halt_scale)
        norm_loss = vq_loss_sample / (self.vq_loss_ema + 1e-6)
        halt_prob = torch.sigmoid(bias - scale * norm_loss)

        w_unc     = meta_values[:, 3]
        halt_prob = (halt_prob * (1.0 - 0.3 * w_unc)).clamp(0.0, 1.0)

        if self.training and self.train_step < 100:
            # Warmup: suppress halting for first 100 steps so the model
            # explores recursion depth while gradients are still stabilising.
            # 20 was too short — with halt_prob * 0.1 ≈ 0.09, expected depth
            # is still ~5/5 (max) for the entire warmup period.
            halt_prob = halt_prob * 0.1

        if self.training:
            hard_stop = (halt_prob > torch.rand_like(halt_prob)).float()
        else:
            hard_stop = (halt_prob > 0.5).float()

        # Straight-through: gradient flows through halt_prob
        should_stop = (hard_stop - halt_prob.detach() + halt_prob).unsqueeze(1)
        return should_stop

    # ── Single T.O.T.E. Step ─────────────────────────────────────────────
    def forward_step(self, x_t: torch.Tensor, gw_state: torch.Tensor,
                     mem_state: torch.Tensor):
        """
        Execute one token's T.O.T.E. (Test-Operate-Test-Exit) cycle.

        Neurolinguistic:
            T.O.T.E. (Miller, Galanter & Pribram 1960) is the basic unit of
            goal-directed behaviour: the model sets a 'test' (halt condition),
            operates (updates workspace), tests again, and exits on success.

            Here it models predictive coding: the model iteratively refines
            its internal prediction until confidence threshold is met.
        """
        alpha     = torch.sigmoid(self.input_gate)
        gw_state  = alpha * gw_state + (1.0 - alpha) * x_t    # Input injection

        B         = x_t.size(0)
        active    = torch.ones(B, 1, device=DEVICE)

        accum = {k: torch.zeros(B, device=DEVICE)
                 for k in ['vq', 'ponder', 'phase']}
        total_batch_ent = total_gate_ent = total_slot_ent = total_sensory_ent = 0
        module_wins      = torch.zeros(B, 5, device=DEVICE)
        total_steps      = torch.zeros(B, device=DEVICE)
        final_indices    = torch.zeros(B, dtype=torch.long, device=DEVICE)

        # Phase tracking for regularisation.
        # ALL angle computations here are detached:
        #   - torch.where(cond, x.angle(), zeros): even when cond=False, autograd
        #     computes grad_x = grad_out * cond = NaN * 0 = NaN (IEEE 754).
        #   - gw_state starts as zeros → atan2(0,0) gradient = 0/0 = NaN.
        # Phase reg is a second-order monitoring term (weight 0.005); detaching
        # its angle computations removes the NaN source without loss of function.
        with torch.no_grad():
            init_mag   = gw_state.abs()
            prev_angle = torch.where(init_mag > 1e-6,
                                     gw_state.angle(),
                                     torch.zeros_like(gw_state.real))
        meta_accum = torch.zeros(B, 4, device=DEVICE)
        meta_values = None

        steps_taken = 0
        for i in range(CONFIG["max_recursion"]):
            steps_taken += 1
            curr_state  = self.norm(gw_state)

            # Meta-cognition
            flat        = torch.cat([curr_state.real, curr_state.imag], dim=-1)
            meta_values = torch.sigmoid(self.meta_head(flat))
            meta_accum += meta_values

            # Run all five modules
            (v_out, vq_loss_s, indices, batch_ent,
             g_out, m_out, cand_mem, slot_ent,
             s_out, vis_attn, aud_attn, q_out) = self._step_modules(curr_state, mem_state)

            # Update VQ EMA baseline
            if self.training:
                with torch.no_grad():
                    self.vq_loss_ema.mul_(0.95).add_(
                        vq_loss_s.mean(), alpha=0.05)

            # Accumulate entropy metrics
            if active.sum() > 0:
                total_batch_ent   += batch_ent
                total_slot_ent    += slot_ent
                vis_ent = -torch.sum(vis_attn * torch.log(vis_attn + 1e-10), -1).mean()
                aud_ent = -torch.sum(aud_attn * torch.log(aud_attn + 1e-10), -1).mean()
                total_sensory_ent += 0.5 * (vis_ent + aud_ent)

            # ── Arbitration ───────────────────────────────────────────────
            gates, gate_ent = self._arbitrate(curr_state, meta_values)
            module_wins    += gates * active
            if active.sum() > 0:
                total_gate_ent += gate_ent

            # Broadcast: weighted sum of proposals
            proposals  = torch.stack([g_out, m_out, v_out, s_out, q_out], dim=1)
            gates_c    = torch.complex(gates.unsqueeze(-1),
                                       torch.zeros_like(gates.unsqueeze(-1)))
            winner_vec = (proposals * gates_c).sum(dim=1)

            # ── Shadow Processing — True NLP Shadow (audit fix N-03) ──────
            # The NLP/Jungian shadow is the single most-suppressed voice,
            # NOT the inverse-weighted average of all modules.
            # Prior implementation: shadow = (1-gates) / sum(1-gates) ≈ uniform
            # That approximates peripheral awareness, not a silenced voice.
            # Fixed: retrieve only the single most-suppressed module's proposal.
            w_unc       = meta_values[:, 3:4]                             # (B, 1)
            shadow_idx  = gates.argmin(dim=-1)                            # (B,)
            shadow_prop = proposals[torch.arange(proposals.size(0),
                                                 device=DEVICE), shadow_idx]
            # Integrate shadow only as strongly as epistemic uncertainty warrants
            dynamic_shadow = CONFIG["shadow_weight"] * w_unc              # (B, 1)
            update         = winner_vec + dynamic_shadow * shadow_prop

            # Clamp to unit hypercube in each component
            update     = torch.complex(torch.tanh(update.real),
                                       torch.tanh(update.imag))
            cand_state = 0.6 * curr_state + 0.4 * update
            cand_state = torch.complex(torch.tanh(cand_state.real),
                                       torch.tanh(cand_state.imag))

            # ── Phase Regularisation ──────────────────────────────────────
            # Detached: angle() backward at near-zero magnitude = NaN.
            # Phase reg is second-order (weight 0.005) — monitoring, not driving.
            mask_vec = active.view(-1)   # needed below for ponder/vq too
            with torch.no_grad():
                mag        = cand_state.abs()
                valid      = mag > 1e-6
                safe_angle = torch.where(valid, cand_state.angle(), prev_angle)
                diff       = (safe_angle - prev_angle).abs()
                diff       = torch.min(diff, 2 * math.pi - diff)
                accum['phase'] += (mask_vec * diff.mean(-1).view(-1)
                                   * valid.float().mean(-1).view(-1)).detach()
                prev_angle = safe_angle

            # ── Halting ───────────────────────────────────────────────────
            should_stop = self._halt(vq_loss_s, meta_values)

            accum['ponder'] += mask_vec * CONFIG["ponder_cost"]
            total_steps     += mask_vec
            accum['vq']     += mask_vec * vq_loss_s.view(-1)

            still_active  = mask_vec > 0.5
            final_indices = torch.where(still_active, indices, final_indices)

            # Update GW state only for active samples
            gw_state = torch.complex(
                torch.where(active > 0.5, cand_state.real, gw_state.real),
                torch.where(active > 0.5, cand_state.imag, gw_state.imag)
            )

            # [E-03 FIX] Cast mask to complex dtype so BOTH the real and
            # imaginary parts of memory slots are correctly gated per sample.
            # A real float mask has version-dependent semantics against complex
            # tensors and may silently update only Re(mem) on some CUDA builds.
            mem_mask_c = torch.complex(active.unsqueeze(-1),
                                       torch.zeros_like(active.unsqueeze(-1)))
            mem_state  = mem_state + mem_mask_c * (cand_mem - mem_state)

            active = active * (1.0 - should_stop)
            if active.sum() == 0:
                break

        if self.training:
            self.train_step += 1

        div          = total_steps.clamp(min=1.0)
        accum['vq'] /= div
        avg_depth    = total_steps.mean()
        accum['depth_per_sample'] = total_steps

        return (gw_state, mem_state, accum, final_indices,
                total_batch_ent / max(1, steps_taken),
                total_gate_ent  / max(1, steps_taken),
                total_slot_ent  / max(1, steps_taken),
                avg_depth,
                meta_accum      / max(1, steps_taken),
                module_wins,
                total_sensory_ent / max(1, steps_taken))

    # ── Full Sequence Forward Pass ────────────────────────────────────────
    def forward(self, x_seq: torch.Tensor, mem_state=None):
        B, T    = x_seq.shape
        emb     = self.encoder(x_seq)
        gw_seq  = torch.complex(emb[..., :self.dim], emb[..., self.dim:])

        if mem_state is None:
            mem_state = self.memory_module.init_state(B)

        gw_state  = torch.zeros_like(gw_seq[:, 0])
        outputs, all_indices = [], []
        stats     = {k: 0.0 for k in ['vq', 'ponder', 'phase',
                                       'ent', 'gate_ent', 'slot_ent',
                                       'depth', 'sensory_ent']}
        depth_acc = torch.zeros(B, device=DEVICE)
        meta_acc  = torch.zeros(B, 4, device=DEVICE)
        final_wins = None

        for t in range(T):
            x_t = gw_seq[:, t]
            (gw_state, mem_state, step_stats, indices,
             batch_ent, gate_ent, slot_ent, depth,
             meta_vals, wins, sensory_ent) = self.forward_step(x_t, gw_state, mem_state)

            outputs.append(gw_state)
            all_indices.append(indices)
            for k in ['vq', 'ponder', 'phase']:
                stats[k] += step_stats[k].mean()
            stats['ent']         += batch_ent
            stats['gate_ent']    += gate_ent
            stats['slot_ent']    += slot_ent
            stats['depth']       += depth
            stats['sensory_ent'] += sensory_ent
            depth_acc            += step_stats['depth_per_sample']
            meta_acc             += meta_vals
            final_wins            = wins

        out_tensor = torch.stack(outputs, dim=1)
        flat_out   = torch.cat([out_tensor.real, out_tensor.imag], dim=-1)
        logits     = self.decoder(flat_out)

        for k in stats:
            stats[k] /= T
        meta_acc /= T
        stats['depth_per_sample'] = depth_acc / T

        return logits, stats, mem_state, torch.stack(all_indices, dim=1), meta_acc, final_wins


# ═══════════════════════════════════════════════════════════════════════════
# 5.  HIPPOCAMPAL BUFFER  (Memory Replay with Online Z-Score Priority)
#
#     Engineering: min-heap on z-score of cross-entropy loss ensures we
#     replay the most surprising / anomalous experiences first.
#
#     Neurolinguistic: models memory consolidation during sleep (SWS) —
#     the hippocampus reactivates high-novelty traces for cortical transfer.
# ═══════════════════════════════════════════════════════════════════════════

class HippocampalBuffer:
    """
    Priority replay buffer using online z-score to rank experiences by surprise.

    Engineering fix (E-01):
        heapq resolves ties by comparing successive tuple elements. When two
        experiences have equal float priority, Python would attempt tensor < tensor,
        raising RuntimeError. A monotonically increasing integer counter is inserted
        as the second tuple element so ties are always broken without touching tensors.
        Tuple layout: (priority: float, counter: int, x: Tensor, y: Tensor)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer   = []
        self._counter = 0          # [E-01 FIX] monotonic tie-breaker
        self._n = 0; self._mean = 0.0; self._M2 = 0.0

    def _update_stats(self, x: float):
        self._n  += 1
        delta     = x - self._mean
        self._mean += delta / self._n
        self._M2  += delta * (x - self._mean)

    def _z_score(self, x: float) -> float:
        if self._n < 2:
            return x
        std = max(math.sqrt(self._M2 / (self._n - 1)), 1e-6)
        return (x - self._mean) / std

    def add(self, x: torch.Tensor, y: torch.Tensor, raw_ce: float):
        self._update_stats(raw_ce)
        priority          = self._z_score(raw_ce)
        entry             = (priority, self._counter, x, y)   # [E-01 FIX]
        self._counter    += 1
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, entry)
        elif priority > self.buffer[0][0]:
            heapq.heapreplace(self.buffer, entry)

    def sample(self, batch_size: int):
        if not self.buffer:
            return None
        # Return (priority, counter, x, y) tuples; callers unpack accordingly
        return random.sample(self.buffer, min(len(self.buffer), batch_size))


# ═══════════════════════════════════════════════════════════════════════════
# 6.  LOSS FUNCTION  (Multi-Objective with Neurolinguistic Terms)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_loss(logits: torch.Tensor, y: torch.Tensor,
                   stats: dict, meta_vals: torch.Tensor,
                   plasticity: float = 1.0,
                   epoch: int = 0, total_epochs: int = 1,
                   confidence: torch.Tensor = None):
    """
    Unified loss — all audit fixes applied.

    Components:
        loss_ce        — cross-entropy (primary language modelling objective)
        ego_penalty    — metacognitive: high confidence on wrong answer → penalty.
                         [M-02 FIX] loss_ce must NOT be detached — both operands
                         must carry gradients or the penalty is a dead constant.
                         [M-03 FIX] confidence passed in is computed over ALL
                         token positions (not just last), matching CE loss scope.
        depth_penalty  — [M-05 FIX] quadratic pull toward target_depth provides a
                         continuous gradient everywhere. Old dead-band relu had zero
                         gradient inside [depth_min, depth_max].
        vq_loss        — codebook commitment (symbolic grounding)
        ponder_cost    — penalise unnecessary extra thinking steps
        phase_reg      — penalise large inter-step phase jumps
        gate_ent       — [M-01 FIX] SUBTRACTED to penalise diffuse gate distributions.
                         Old code added it, silently rewarding uniform/diffuse gates
                         and destroying GWT competitive selection.
        slot_ent       — reward uniform memory-slot usage (subtract from loss)
        sensory_ent    — reward diverse sensory attention (subtract from loss)
    """
    loss_ce = F.cross_entropy(
        logits.reshape(-1, CONFIG["vocab_size"]),
        y.reshape(-1)
    )

    # ── Metacognitive Ego Penalty (M-02 + M-03 FIX) ──────────────────────
    # Neurolinguistic: the brain's ACC detects conflict between confidence
    # and outcome; high confidence + high error = dopamine dip → strong update.
    #
    # M-02: loss_ce is NOT detached — ego_penalty now has real gradient flow.
    # M-03: confidence must cover all positions (computed in training loop).
    ego_penalty = torch.tensor(0.0, device=DEVICE)
    if confidence is not None:
        ego_penalty = confidence * loss_ce * CONFIG["ego_penalty_weight"]

    # ── Depth Regularisation (M-05 FIX) ──────────────────────────────────
    # Quadratic soft pull toward target_depth; provides gradient everywhere.
    # Old dead-band: zero gradient inside [depth_min, depth_max].
    depth_penalty = torch.tensor(0.0, device=DEVICE)
    if isinstance(stats.get('depth_per_sample'), torch.Tensor):
        d             = stats['depth_per_sample']
        depth_penalty = CONFIG['depth_target_weight'] * \
                        (d - CONFIG['target_depth']).pow(2).mean()

    # ── Entropy Weight Schedule (anneal from aggressive to gentle) ────────
    progress  = epoch / max(1, total_epochs)
    ent_w     = (CONFIG['entropy_weight_start'] * (1 - progress) +
                 CONFIG['entropy_weight_end']   * progress)

    w_ethics, w_div, w_ponder, w_unc = meta_vals.mean(0)
    struct_scale = 1.0 + 0.3 * (plasticity - 1.0)

    struct_loss = struct_scale * (
        # ── CORE terms — always keep ─────────────────────────────────────
        0.1  * stats['vq']                                    # codebook commitment
        + (stats['ponder'] * (1.0 + w_ponder))               # effort cost
        - ent_w * (1.0 + w_div) * stats['ent']              # VQ diversity reward
        - CONFIG['gate_sparsity_weight'] * stats['gate_ent']  # penalise diffusion

        # ── SECOND-ORDER terms — reduce/zero first if training stagnates ─
        + CONFIG['phase_reg']            * stats['phase']     # phase coherence
        - CONFIG['slot_balance_weight']  * stats['slot_ent']  # reward slot balance
        - CONFIG['sensory_reg']          * stats['sensory_ent']

        # ── Depth target (smooth pull, not dead-band) ─────────────────────
        + depth_penalty
    )

    total_loss = loss_ce * plasticity + ego_penalty + struct_loss
    return loss_ce, total_loss


# ═══════════════════════════════════════════════════════════════════════════
# 7.  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def get_batch(data_tensor: torch.Tensor, batch_size: int, seq_len: int):
    max_idx = len(data_tensor) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x  = torch.stack([data_tensor[i:i + seq_len]     for i in ix])
    y  = torch.stack([data_tensor[i + 1:i + seq_len + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def log_grad_norms(model: SACRSN_Unified) -> dict:
    """Per-module gradient norm for training diagnostics."""
    groups = {
        'encoder':    model.encoder,
        'gate_mod':   model.gate_mod,
        'memory':     model.memory_module,
        'vq':         model.vq,
        'sensory':    model.sensory_cortex,
        'quadrature': model.quadrature,          # [N-01 FIX] renamed from critic
        'arbitrator': model.arbitrator,
        'meta_head':  model.meta_head,
        'decoder':    model.decoder,
    }
    norms = {}
    for name, module in groups.items():
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        )
        norms[name] = total ** 0.5
    return norms


# ═══════════════════════════════════════════════════════════════════════════
# 8.  GENERATION, DREAMING & ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def generate_text(model: SACRSN_Unified, tokenizer: RobustBPE,
                  seed: str = "The", length: int = 60,
                  temp: float = 0.8, verbose_modules: bool = False) -> str:
    """
    Auto-regressive text generation with optional module-level tracing.

    Neurolinguistic: with verbose_modules=True you can see which cognitive
    module dominated each prediction — revealing whether the model was
    drawing on memory, sensory grounding, VQ (symbolic), or the critic.
    """
    model.eval()
    module_names = ["Gate", "Memory", "VQ", "Sensory", "Quadrature"]  # [N-01 FIX]
    ids  = tokenizer.encode(seed)
    curr = torch.tensor([ids], device=DEVICE)
    mem  = None
    text = seed

    print(f"\n── Generating (seed='{seed}', T={temp}) ──")
    with torch.no_grad():
        try:
            _, _, mem, _, _, _ = model(curr, mem)
            curr = curr[:, -1:]
            for i in range(length):
                logits, _, mem, _, _, wins = model(curr, mem)

                if torch.isnan(logits).any():
                    print("[!] NaN in logits — stopping.")
                    break

                if verbose_modules or i < 5:
                    dom = torch.argmax(wins[0]).item()
                    print(f"  t={i}: [{module_names[dom]}]", end="  ")

                logits_s = logits[:, -1, :] / temp
                logits_s = logits_s - logits_s.max(dim=-1, keepdim=True)[0]
                probs    = F.softmax(logits_s, dim=-1)
                next_id  = torch.multinomial(probs, 1)
                word     = tokenizer.decode([next_id.item()])
                text    += word
                curr     = next_id
                print(word, end="", flush=True)
        except Exception as e:
            print(f"\n[!] Generation error: {e}")
    print("\n")
    return text


def dream_graph_walk(transitions: deque, steps: int = 20, temp: float = 1.0):
    """
    Free-associative walk over the learned concept transition graph.

    Neurolinguistic:
        During REM sleep, the hippocampus replays concept transitions in a
        temperature-scaled random walk — 'dreaming' as creative recombination.
        High temperature → surreal/associative; low → focused/logical.
    """
    print("\n── Dream Mode (Concept Graph Walk) ──")
    if not transitions:
        print("  (No transitions recorded yet.)")
        return
    graph = defaultdict(dict)
    for (u, v), count in Counter(transitions).items():
        graph[u][v] = count

    if not graph:
        return
    start_w = [len(graph[k]) for k in graph]
    curr    = random.choices(list(graph.keys()), weights=start_w, k=1)[0]
    print(f"  [{curr}]", end="")
    for _ in range(steps):
        if curr not in graph or not graph[curr]:
            break
        targets = list(graph[curr].keys())
        scaled  = [c ** (1.0 / max(temp, 1e-6)) for c in graph[curr].values()]
        total   = sum(scaled)
        probs   = [s / total for s in scaled]
        curr    = random.choices(targets, weights=probs, k=1)[0]
        print(f" → {curr}", end="")
    print("\n")


def anomaly_detector(model: SACRSN_Unified, tokenizer: RobustBPE):
    """
    Compare perplexity of a normal vs. semantically anomalous sentence.

    A well-trained model should assign dramatically higher cross-entropy
    to the anomalous sentence — demonstrating acquired world-model.
    """
    print("\n── Anomaly Detector ──")
    normal = "The structure of the mind."
    weird  = "The banana of the galaxy eats time."

    def score(text):
        ids = tokenizer.encode(text)
        x   = torch.tensor([ids[:-1]], device=DEVICE)
        y   = torch.tensor([ids[1:]],  device=DEVICE)
        with torch.no_grad():
            logits, _, _, _, _, _ = model(x)
            return F.cross_entropy(
                logits.reshape(-1, len(tokenizer.vocab)), y.reshape(-1)
            ).item()

    s_n, s_w = score(normal), score(weird)
    print(f"  Normal:    CE = {s_n:.4f}")
    print(f"  Anomalous: CE = {s_w:.4f}")
    if s_w > s_n:
        print("  ✓ Anomaly detected (higher perplexity on weird sentence)")
    else:
        print("  ✗ Model not yet discriminating (continue training)")


def extract_concept_rules(transitions: deque, top_k: int = 10):
    """Print the strongest learned concept-to-concept associations."""
    print(f"\n── Top-{top_k} Concept Transitions ──")
    print(f"{'FROM':>8}  →  {'TO':<8}  {'STRENGTH':>10}")
    print("─" * 34)
    for (u, v), w in Counter(transitions).most_common(top_k):
        print(f"  Cpt_{u:<4}  →  Cpt_{v:<4}  {w:>10}")


# ═══════════════════════════════════════════════════════════════════════════
# 9.  FULL DIAGNOSTIC VISUALISATION SUITE
# ═══════════════════════════════════════════════════════════════════════════

def visualise_suite(model: SACRSN_Unified, history: dict,
                    transitions: deque, meta_history: list):
    """
    6-panel diagnostic dashboard:
        1. Loss / ponder cost curves
        2. Phase-space thought trajectory
        3. Abstract concept topology (NetworkX graph)
        4. Learned visual sensory anchors
        5. Meta-value drift over training
        6. Per-module gradient norms
    """
    print("\n── Visualisation Suite ──")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ① Cognitive Effort ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(history.get('loss', []),   label='Total Loss',  alpha=0.8)
    ax.plot(history.get('ponder', []), label='Ponder Cost', alpha=0.8)
    ax.set_title("Cognitive Effort");  ax.legend(); ax.grid(alpha=0.3)

    # ② Phase-Space Thought Trajectory ────────────────────────────────────
    ax = axes[0, 1]
    with torch.no_grad():
        gw = torch.zeros(1, CONFIG["embed_dim"],
                         dtype=torch.complex64, device=DEVICE)
        xt = torch.zeros_like(gw)
        mem = model.memory_module.init_state(1)
        rs, ims = [], []
        for _ in range(30):
            gw, mem, _, _, _, _, _, _, _, _, _ = model.forward_step(xt, gw, mem)
            rs.append(gw.real.mean().item())
            ims.append(gw.imag.mean().item())
    ax.plot(rs, ims, 'r-', lw=2, label='Thought Trace')
    ax.set_title("Phase Space (Re vs Im)")
    ax.set_xlabel("Real"); ax.set_ylabel("Imag"); ax.legend()

    # ③ Concept Topology ──────────────────────────────────────────────────
    ax = axes[0, 2]
    G  = nx.DiGraph()
    for (u, v), w in Counter(transitions).most_common(50):
        G.add_edge(u, v, weight=w)
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw(G, pos, ax=ax, node_size=20, width=0.5,
                alpha=0.7, edge_color='teal')
    ax.set_title(f"Concept Topology ({CONFIG['codebook_size']} codes)")

    # ④ Sensory Anchors ───────────────────────────────────────────────────
    ax   = axes[1, 0]
    vis  = model.sensory_cortex.visual_palette.detach().cpu().numpy()[:10]
    im   = ax.imshow(vis, aspect='auto', cmap='plasma')
    ax.set_title("Hebbian Visual Anchors (Submodalities)")
    plt.colorbar(im, ax=ax)

    # ⑤ Meta-Value Drift ──────────────────────────────────────────────────
    ax = axes[1, 1]
    if meta_history:
        mh     = np.array(meta_history)
        labels = ["Ethics", "Diversity", "Ponder", "Uncertainty"]
        for i, lbl in enumerate(labels):
            ax.plot(mh[:, i], label=lbl, alpha=0.85)
    ax.set_title("Meta-Value Drift"); ax.legend(); ax.grid(alpha=0.3)

    # ⑥ Gradient Norms ────────────────────────────────────────────────────
    ax = axes[1, 2]
    if history.get('grad_norms'):
        modules = list(history['grad_norms'][0].keys())
        colors  = plt.cm.tab10(range(len(modules)))
        for i, mod in enumerate(modules):
            vals = [gn.get(mod, 0) for gn in history['grad_norms']]
            ax.plot(vals, label=mod, color=colors[i], alpha=0.85, lw=1.2)
        ax.set_title("Gradient Norms by Module")
        ax.legend(fontsize=6); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('sacrsn_unified_diagnostics.png', dpi=150)
    print("  Saved: sacrsn_unified_diagnostics.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 10. MAIN TRAINING LOOP
#     Waking phase  → active learning with dopamine plasticity
#     Sleeping phase → hippocampal replay / memory consolidation
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Data & Tokeniser ──────────────────────────────────────────────────
    data_text = load_training_data("data.txt")
    tokenizer = RobustBPE(target_vocab_size=1000)
    tokenizer.train(data_text)
    CONFIG["vocab_size"] = len(tokenizer.vocab)

    data_ids    = tokenizer.encode(data_text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)
    print(f"\n  Corpus: {len(data_ids):,} tokens   Vocab: {CONFIG['vocab_size']:,}")

    # ── Model & Optimiser ─────────────────────────────────────────────────
    model     = SACRSN_Unified(CONFIG["vocab_size"], CONFIG["embed_dim"]).to(DEVICE)
    opt       = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CONFIG["epochs"])
    hippocampus = HippocampalBuffer(CONFIG["replay_size"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── History ───────────────────────────────────────────────────────────
    history = defaultdict(list)
    transitions: deque = deque(maxlen=10_000)
    meta_history = []
    avg_surprise = 2.0   # initial EMA value for dopamine

    # ── Training ──────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════")
    print("  WAKING PHASE — Active Learning")
    print("══════════════════════════════════════")
    model.train()

    try:
        for epoch in range(CONFIG["epochs"]):
            x, y = get_batch(data_tensor, CONFIG["batch_size"], CONFIG["seq_len"])
            opt.zero_grad()

            # Synaptic decay: mild L2 on select projections (continual learning)
            with torch.no_grad():
                for layer in [model.arbitrator, model.meta_head, model.decoder]:
                    layer.weight.mul_(1.0 - CONFIG["synaptic_decay"])

            logits, stats, mem, indices, meta_vals, _ = model(x)

            # Record concept transitions
            idx_np = indices.detach().cpu().numpy()
            for b in range(idx_np.shape[0]):
                for t in range(idx_np.shape[1] - 1):
                    transitions.append((idx_np[b, t], idx_np[b, t + 1]))

            # ── Dopamine Plasticity (Surprise-Modulated LR) ───────────────
            ce_raw = F.cross_entropy(
                logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1)
            )
            plast = 1.0
            if CONFIG["active_plasticity"]:
                with torch.no_grad():
                    avg_surprise = (CONFIG["surprise_decay"] * avg_surprise
                                    + (1 - CONFIG["surprise_decay"]) * ce_raw.item())
                plast = min(1.5, 1.0 + avg_surprise * 0.5)   # capped for safety

            # ── Ego / Metacognitive Penalty (M-02 + M-03 FIX) ────────────
            # [M-02 FIX] compute OUTSIDE no_grad so confidence carries gradient
            # [M-03 FIX] use ALL token positions, not just the last one
            probs      = F.softmax(logits, dim=-1)                 # (B, T, V)
            confidence = probs.max(dim=-1)[0].mean()              # scalar, in graph

            _, final_loss = calculate_loss(
                logits, y, stats, meta_vals,
                plasticity=plast,
                epoch=epoch, total_epochs=CONFIG["epochs"],
                confidence=confidence
            )

            # NaN guard — loss level
            if torch.isnan(final_loss):
                print(f"[!] NaN loss at epoch {epoch} — skipping step.")
                opt.zero_grad()
                continue

            final_loss.backward()

            # Clip FIRST — may rescue large-but-finite gradients before they
            # become NaN. (Previous order: check NaN then clip was backwards.)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

            # After clipping, check for genuine NaN (true numerical instability).
            # Zero out bad gradients rather than skipping the entire step —
            # this lets the healthy parameters still update.
            nan_in_grads = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if nan_in_grads:
                print(f"[!] NaN gradients at epoch {epoch} — zeroing bad grads.")
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            if epoch % 20 == 0:
                history['grad_norms'].append(log_grad_norms(model))

            opt.step()          # always step after clipping; scheduler follows

            # ── SLEEP PHASE: Hippocampal Consolidation ────────────────────
            hippocampus.add(x.detach().cpu(), y.detach().cpu(), ce_raw.item())
            if epoch > 0 and epoch % CONFIG["sleep_every"] == 0:
                print(f"\n  ── Sleep Phase (Ep {epoch}) ──")
                dream = hippocampus.sample(4)
                if dream:
                    # [E-01 FIX] Buffer now stores 4-tuples: (priority, counter, x, y)
                    for _, _cnt, dx, dy in dream:
                        dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                        opt.zero_grad()
                        d_logits, d_stats, _, _, d_meta, _ = model(dx)
                        _, d_loss = calculate_loss(
                            d_logits, dy, d_stats, d_meta,
                            plasticity=1.0,
                            epoch=epoch, total_epochs=CONFIG["epochs"]
                        )
                        if not torch.isnan(d_loss):
                            d_loss.backward()
                            nan_dream = any(
                                p.grad is not None and torch.isnan(p.grad).any()
                                for p in model.parameters()
                            )
                            if not nan_dream:
                                opt.step()
                            else:
                                opt.zero_grad()
                print(f"  Resuming waking phase...")

            scheduler.step()

            # Record
            history['loss'].append(final_loss.item())
            history['ponder'].append(
                stats['ponder'].item() if hasattr(stats['ponder'], 'item')
                else float(stats['ponder']))
            history['depth'].append(
                stats['depth'].item() if hasattr(stats['depth'], 'item')
                else float(stats['depth']))
            meta_history.append(meta_vals.mean(0).detach().cpu().numpy())

            # Logging
            if epoch % 20 == 0:
                lr_val   = opt.param_groups[0]['lr']
                text_ppx = math.exp(min(100, ce_raw.item()))
                raw_ent  = (stats['ent'].item() if hasattr(stats['ent'], 'item')
                            else float(stats['ent']))
                code_ppx = math.exp(raw_ent * math.log(CONFIG['codebook_size']))
                vq_ema   = model.vq_loss_ema.item()

                print(f"Ep {epoch:04d} | "
                      f"Loss: {final_loss.item():.3f} | "
                      f"TextPPX: {text_ppx:.1f} | "
                      f"Depth: {stats['depth'].item() if hasattr(stats['depth'],'item') else float(stats['depth']):.2f} | "
                      f"Plast: {plast:.2f} | "
                      f"VQ_EMA: {vq_ema:.4f} | "
                      f"LR: {lr_val:.2e}")

                if history['grad_norms']:
                    gn      = history['grad_norms'][-1]
                    dominant = max(gn, key=gn.get)
                    print(f"         Dominant module: [{dominant}]  "
                          f"Confidence: {confidence.item():.3f}")

    except KeyboardInterrupt:
        print("\n  Training interrupted by user.")

    # ── Post-Training Evaluation ──────────────────────────────────────────
    model.eval()
    print("\n══════════════════════════════════════")
    print("  POST-TRAINING EVALUATION")
    print("══════════════════════════════════════")

    generate_text(model, tokenizer, seed="The", length=80,
                  temp=0.8, verbose_modules=False)
    generate_text(model, tokenizer, seed="Logic", length=60,
                  temp=1.0, verbose_modules=True)

    dream_graph_walk(transitions, steps=25, temp=1.2)
    extract_concept_rules(transitions, top_k=12)
    anomaly_detector(model, tokenizer)
    visualise_suite(model, history, transitions, meta_history)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
