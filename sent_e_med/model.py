"""
model.py
========
Full Sent-e-Med architecture (Section 4.2 + Section 7.2 of the paper).

Architecture overview (Figure 3):

    ICD codes  ──► text descriptions ──► SBERT ──► sentence embeddings (FROZEN)
                                                         │
    visit position ──────────────────────────────► visit embeddings (learnable)
                                                         │
                                              element-wise sum
                                                         │
                                             Transformer Encoder
                                           (4 layers, 4 heads, 384-dim)
                                                         │
                        ┌────────────────────┬───────────┴──────────────┐
                   Pretraining            Pretraining               Fine-tuning
                   MLM head          Next Visit Pred. head      Classification head
                (per-token)            (avg pool → FC)          (avg pool → FC → sigmoid)

Key design choices (paper):
  1. No [CLS] or [SEP] tokens (unlike vanilla BERT)
  2. No positional encodings (unlike vanilla BERT)
  3. SBERT embeddings are FROZEN — visit embeddings are the only learned code-level params
  4. [MASK] positions use a LEARNABLE embedding (not SBERT, since SBERT is frozen)
  5. Hidden dim = 384 matches SBERT "all-MiniLM-L6-v2" output exactly

Implementation details (Section 7.2):
  hidden_dim=384, num_layers=4, num_heads=4, ffn_dim=1536,
  linear_dim=64, max_seq_len=128, lr=1e-5, AdamW weight decay
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import MASK_TOKEN_ID


class SenteMed(nn.Module):
    """
    Sent-e-Med: BERT-style EHR encoder using frozen SBERT code embeddings.

    Usage:
        # Pretraining
        losses = model.pretrain_step(code_ids, visit_ids, attention_mask,
                                      is_masked, mlm_labels, nvp_labels)
        losses["total_loss"].backward()

        # Fine-tuning
        logits = model.finetune_step(code_ids, visit_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
    """

    def __init__(
        self,
        config,
        vocab_size: int,
        sbert_embeddings: torch.Tensor,
        # Shape: (vocab_size, encoder_dim) — FROZEN
        # encoder_dim = 384 for SBERT, 768 for Bio_ClinicalBERT
        phecode_embeddings: Optional[torch.Tensor] = None,
        # Shape: (vocab_size, 384) — FROZEN, always SBERT-encoded PheCode descriptions
        # Only used when config.use_phecode = True
    ):
        """
        Args:
            config:              SenteMedConfig instance
            vocab_size:          number of unique ICD codes in the vocabulary
            sbert_embeddings:    pre-computed frozen encoder vectors.
                                 Shape: (vocab_size, encoder_dim).
                                 encoder_dim = 384 for SBERT, 768 for Bio_ClinicalBERT.
            phecode_embeddings:  optional frozen PheCode description vectors.
                                 Shape: (vocab_size, 384). Always SBERT-encoded.
                                 Required when config.use_phecode=True.
        """
        super().__init__()
        self.config      = config
        self.vocab_size  = vocab_size
        H                = config.hidden_dim
        encoder_dim      = sbert_embeddings.shape[1]
        self.encoder_dim = encoder_dim

        # ── (A) Frozen ICD code embeddings ────────────────────────────────────
        # Registered as a buffer → moves to GPU with .to(device) but is NOT
        # a parameter, so gradients are never computed for it.
        # Shape: (vocab_size, encoder_dim)
        assert sbert_embeddings.shape[0] == vocab_size, (
            f"Embedding table has {sbert_embeddings.shape[0]} rows "
            f"but vocab_size={vocab_size}"
        )
        self.register_buffer("sbert_code_embeddings", sbert_embeddings.float())

        # ── (A2) Input projection (Bio_ClinicalBERT variant only) ─────────────
        # Bio_ClinicalBERT outputs 768-dim CLS embeddings. We project them down
        # to hidden_dim (384) so the transformer width stays the same as SBERT.
        # This projection IS trained during pretraining.
        # For SBERT (encoder_dim == hidden_dim), this is an identity (no-op).
        if encoder_dim != H:
            self.input_projection = nn.Linear(encoder_dim, H, bias=False)
            nn.init.xavier_uniform_(self.input_projection.weight)
            print(
                f"Bio_ClinicalBERT mode: added input_projection "
                f"Linear({encoder_dim} → {H})"
            )
        else:
            self.input_projection = None   # SBERT path — no projection needed

        # ── (A3) PheCode dual-embedding (optional extension) ──────────────────
        # When config.use_phecode=True, each token embedding is the fusion of:
        #   (a) its ICD-level embedding (fine-grained, from encoder_type)
        #   (b) its PheCode-level embedding (coarse-grained, always SBERT 384-dim)
        #
        # Masking order: ICD masked positions receive mask_embedding first,
        # then the masked code_emb is fused with the (unmasked) phecode_emb.
        # This means masked positions carry "I am masked, and my phenotype group
        # is X" — giving the MLM task coarse context without revealing the code.
        #
        # Two fusion modes (config.phecode_fusion):
        #   "concat" — Linear(2H → H) applied to cat([icd_H, phecode_H])
        #   "gate"   — gate = sigmoid(Linear(2H → H)); out = gate*icd + (1-gate)*phecode
        #              Gate weights init near 0 → sigmoid near 0.5 (equal mix at start)
        self.use_phecode = getattr(config, "use_phecode", False)

        if self.use_phecode:
            if phecode_embeddings is None:
                raise ValueError(
                    "config.use_phecode=True but phecode_embeddings was not provided. "
                    "Build them with phecode_utils.build_phecode_embeddings() and pass here."
                )
            assert phecode_embeddings.shape[0] == vocab_size, (
                f"PheCode table has {phecode_embeddings.shape[0]} rows "
                f"but vocab_size={vocab_size}"
            )
            assert phecode_embeddings.shape[1] == H, (
                f"PheCode embeddings must be {H}-dim (got {phecode_embeddings.shape[1]}). "
                f"PheCode always uses SBERT (384-dim); ensure config.hidden_dim=384."
            )
            self.register_buffer(
                "phecode_code_embeddings", phecode_embeddings.float()
            )

            fusion_mode = getattr(config, "phecode_fusion", "concat")
            self.phecode_fusion = fusion_mode

            if fusion_mode == "concat":
                # cat([icd_H; phecode_H]) → Linear(2H → H)
                self.phecode_fusion_proj = nn.Linear(H * 2, H, bias=True)
                nn.init.xavier_uniform_(self.phecode_fusion_proj.weight)
                nn.init.zeros_(self.phecode_fusion_proj.bias)
                print(f"PheCode fusion: concat  Linear({H * 2} → {H})")

            elif fusion_mode == "gate":
                # gate = sigmoid(Linear(2H → H))
                # fused = gate * icd_H + (1 - gate) * phecode_H
                # Init near zero so gate ≈ 0.5 at the start of training
                self.phecode_gate_proj = nn.Linear(H * 2, H, bias=True)
                nn.init.zeros_(self.phecode_gate_proj.weight)
                nn.init.zeros_(self.phecode_gate_proj.bias)
                print(f"PheCode fusion: gate    sigmoid(Linear({H * 2} → {H}))")

            else:
                raise ValueError(
                    f"Unknown phecode_fusion '{fusion_mode}'. "
                    "Choose 'concat' or 'gate'."
                )

        # ── (B) Learnable [MASK] embedding ───────────────────────────────────
        # The encoder is frozen, so masked positions need their own trainable vector.
        # Lives in hidden_dim space (after projection if applicable).
        # Shape: (H,)
        self.mask_embedding = nn.Parameter(torch.empty(H))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)

        # ── (C) Visit position embeddings (learnable) ─────────────────────────
        # "visit embeddings serve as unique identifiers for each visit,
        #  akin to segment embeddings in BERT, and are randomly initialized
        #  and updated during pre-training." — Section 4.2
        # Shape: (max_visits, H)
        self.visit_embeddings = nn.Embedding(config.max_visits, H)
        nn.init.normal_(self.visit_embeddings.weight, mean=0.0, std=0.02)

        # ── (D) Input normalization + dropout ────────────────────────────────
        self.input_ln      = nn.LayerNorm(H)
        self.input_dropout = nn.Dropout(config.dropout)

        # ── (E) Transformer encoder ───────────────────────────────────────────
        # Pre-LN (norm_first=True) improves stability on smaller datasets.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,   # (batch, seq, feat) convention
            norm_first=True,    # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(H),
        )

        # ── (F) MLM head ──────────────────────────────────────────────────────
        # Predicts original code at masked positions.
        # Linear(H→H) + GELU + LN + Linear(H→vocab_size)
        self.mlm_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.LayerNorm(H),
            nn.Linear(H, vocab_size),
        )

        # ── (G) Next Visit Prediction head ────────────────────────────────────
        # Multi-label: which codes appear in the next visit?
        # avg_pool(hidden) → Linear(H→H) → GELU → Linear(H→vocab_size) → sigmoid
        self.nvp_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, vocab_size),
        )

        # ── (H) Classification head (fine-tuning) ────────────────────────────
        # avg_pool(hidden) → Linear(H→64) → ReLU → Linear(64→1)
        self.classifier = nn.Sequential(
            nn.Linear(H, config.linear_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_dim, 1),
        )

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────
    # Weight initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in [self.mlm_head, self.nvp_head, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        # Note: phecode_fusion_proj is initialised in __init__ with xavier_uniform_,
        # and phecode_gate_proj with zeros (to start gate at sigmoid(0)=0.5).
        # Those initialisations are intentional and must NOT be overridden here.

    # ─────────────────────────────────────────────────────────────────────────
    # Core: input embedding construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_input_embeddings(
        self,
        code_ids:  torch.Tensor,   # (B, L)  — raw code indices; MASK_TOKEN_ID for masked
        visit_ids: torch.Tensor,   # (B, L)  — which visit each token belongs to
        is_masked: torch.Tensor,   # (B, L) bool — True at masked positions
    ) -> torch.Tensor:
        """
        Build the input embedding for each token:

            input_emb = code_emb  +  visit_emb

        Where:
            code_emb  = SBERT embedding  (frozen) for non-masked tokens
                      = learnable mask_embedding  for masked tokens
            visit_emb = learned visit position embedding

        "The final input embedding is generated by summing the sentence
         and visit embeddings." — Section 4.2
        """
        B, L = code_ids.shape
        H    = self.config.hidden_dim

        # ① Safe lookup from frozen ICD embedding table.
        #    Clamp sentinel MASK_TOKEN_ID to 0 for indexing safety —
        #    masked positions will be overwritten in step ②.
        safe_ids = code_ids.clamp(min=0, max=self.vocab_size - 1)   # (B, L)
        code_emb = self.sbert_code_embeddings[safe_ids]              # (B, L, encoder_dim)

        # ① (Bio_ClinicalBERT variant) Project encoder_dim → hidden_dim
        if self.input_projection is not None:
            code_emb = self.input_projection(code_emb)               # (B, L, H)

        # ② Replace masked positions with learnable [MASK] embedding.
        #    [MASK] lives in hidden_dim space (after projection if applicable).
        #    Note: masking is applied BEFORE PheCode fusion so that masked
        #    positions contribute "mask_embedding + phecode_context" — the
        #    coarse phenotype group remains visible but the specific ICD is hidden.
        mask_emb = self.mask_embedding.view(1, 1, H).expand(B, L, H)  # (B, L, H)
        code_emb = torch.where(
            is_masked.unsqueeze(-1).expand_as(code_emb),
            mask_emb,
            code_emb,
        )                                                              # (B, L, H)

        # ③ PheCode dual-embedding fusion (optional extension).
        #    Look up the PheCode description embedding for each token using
        #    safe_ids (unmasked) — masked positions still get their phecode
        #    context so the model knows the phenotype group while predicting
        #    the specific ICD code.
        if self.use_phecode:
            ph_emb = self.phecode_code_embeddings[safe_ids]          # (B, L, H)

            if self.phecode_fusion == "concat":
                # Concatenate ICD and PheCode embeddings, project back to H.
                # Linear(2H → H) learns how to weight each source.
                fused    = torch.cat([code_emb, ph_emb], dim=-1)     # (B, L, 2H)
                code_emb = self.phecode_fusion_proj(fused)            # (B, L, H)

            else:  # "gate"
                # Element-wise gate in H-dim space.
                # gate ≈ 0.5 at init (weights zeroed) → equal mix, differentiates
                # during training: rare codes learn to rely more on PheCode.
                gate_input = torch.cat([code_emb, ph_emb], dim=-1)   # (B, L, 2H)
                gate       = torch.sigmoid(
                    self.phecode_gate_proj(gate_input)
                )                                                      # (B, L, H)
                code_emb   = gate * code_emb + (1.0 - gate) * ph_emb # (B, L, H)

        # ④ Visit embeddings (learnable, randomly initialized)
        visit_emb = self.visit_embeddings(visit_ids)                  # (B, L, H)

        # ⑤ Sum and normalize
        x = code_emb + visit_emb                                      # (B, L, H)
        x = self.input_ln(x)
        x = self.input_dropout(x)
        return x

    # ─────────────────────────────────────────────────────────────────────────
    # Core: average pooling
    # ─────────────────────────────────────────────────────────────────────────

    def _avg_pool(
        self,
        hidden:         torch.Tensor,   # (B, L, H)
        attention_mask: torch.Tensor,   # (B, L) — 1=real, 0=padding
    ) -> torch.Tensor:
        """
        Average pool hidden states over non-padding positions → (B, H).

        "a feed-forward layer (FFL) was added to average the outputs
         from all of the visits to represent a sequence" — Section 4.2
        """
        mask_f  = attention_mask.float().unsqueeze(-1)   # (B, L, 1)
        summed  = (hidden * mask_f).sum(dim=1)            # (B, H)
        counts  = mask_f.sum(dim=1).clamp(min=1e-9)       # (B, 1)
        return summed / counts                             # (B, H)

    # ─────────────────────────────────────────────────────────────────────────
    # Core: transformer encoding
    # ─────────────────────────────────────────────────────────────────────────

    def encode(
        self,
        code_ids:       torch.Tensor,             # (B, L)
        visit_ids:      torch.Tensor,             # (B, L)
        attention_mask: torch.Tensor,             # (B, L)
        is_masked:      Optional[torch.Tensor] = None,  # (B, L) bool
    ) -> torch.Tensor:
        """
        Run the full forward pass through the transformer.

        Returns:
            hidden: (B, L, H) — contextual representations for all positions
        """
        if is_masked is None:
            is_masked = torch.zeros_like(code_ids, dtype=torch.bool)

        x = self._build_input_embeddings(code_ids, visit_ids, is_masked)

        # TransformerEncoder src_key_padding_mask: True = IGNORE (padding)
        pad_mask = attention_mask == 0   # (B, L) — True at padding positions

        hidden = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, L, H)
        return hidden

    # ─────────────────────────────────────────────────────────────────────────
    # Pretraining forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def pretrain_step(
        self,
        code_ids:       torch.Tensor,   # (B, L) — MASK_TOKEN_ID at masked positions
        visit_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        is_masked:      torch.Tensor,   # (B, L) bool — which positions to predict
        mlm_labels:     torch.Tensor,   # (B, L) — original code ids; -100 elsewhere
        nvp_labels:     torch.Tensor,   # (B, vocab_size) — multi-hot for next visit
        use_nvp:        bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pretraining losses.

        Returns dict with keys:
            "mlm_loss"   — cross-entropy over masked positions
            "nvp_loss"   — BCE multi-label over next-visit codes (0 if use_nvp=False)
            "total_loss" — sum of the above
        """
        hidden = self.encode(code_ids, visit_ids, attention_mask, is_masked)

        losses: Dict[str, torch.Tensor] = {}

        # ── MLM loss ──────────────────────────────────────────────────────────
        # Only compute at positions that are (a) masked AND (b) not padding.
        real_mask_positions = is_masked & (attention_mask == 1)  # (B, L)

        if real_mask_positions.any():
            masked_hidden  = hidden[real_mask_positions]       # (N_masked, H)
            mlm_logits     = self.mlm_head(masked_hidden)      # (N_masked, V)
            masked_labels  = mlm_labels[real_mask_positions]   # (N_masked,)
            losses["mlm_loss"] = F.cross_entropy(mlm_logits, masked_labels)
        else:
            losses["mlm_loss"] = hidden.new_tensor(0.0)

        # ── Next Visit Prediction loss ─────────────────────────────────────────
        if use_nvp:
            # Patient-level representation = average over all real tokens
            patient_repr = self._avg_pool(hidden, attention_mask)  # (B, H)
            nvp_logits   = self.nvp_head(patient_repr)              # (B, V)

            # Only include samples where nvp_labels are non-trivial
            # (i.e., patient had at least one code in their next visit)
            has_nvp = nvp_labels.sum(dim=1) > 0  # (B,)
            if has_nvp.any():
                losses["nvp_loss"] = F.binary_cross_entropy_with_logits(
                    nvp_logits[has_nvp],
                    nvp_labels[has_nvp],
                )
            else:
                losses["nvp_loss"] = hidden.new_tensor(0.0)
        else:
            losses["nvp_loss"] = hidden.new_tensor(0.0)

        losses["total_loss"] = losses["mlm_loss"] + losses["nvp_loss"]
        return losses

    # ─────────────────────────────────────────────────────────────────────────
    # Fine-tuning forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def finetune_step(
        self,
        code_ids:       torch.Tensor,   # (B, L)
        visit_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """
        Forward pass for binary risk classification.

        No masking is applied during fine-tuning.

        Returns:
            logits: (B,) — raw scores; apply torch.sigmoid() for probabilities
        """
        hidden       = self.encode(code_ids, visit_ids, attention_mask)
        patient_repr = self._avg_pool(hidden, attention_mask)      # (B, H)
        logits       = self.classifier(patient_repr).squeeze(-1)   # (B,)
        return logits

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        """Report trainable vs frozen parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        # The SBERT embeddings are buffers, not parameters, so count separately
        buffer_size = sum(b.numel() for b in self.buffers())
        return {
            "trainable_params": trainable,
            "frozen_params":    frozen,
            "sbert_buffer_elements": buffer_size,
        }
