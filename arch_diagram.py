"""
Code for Model architecture diagram
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_W, FIG_H = 18, 9
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

C = {
    "input"    : "#1B4F72",
    "input_lt" : "#AED6F1",
    "enc"      : "#145A32",
    "enc_lt"   : "#A9DFBF",
    "emb"      : "#4A235A",
    "emb_lt"   : "#D7BDE2",
    "trans"    : "#1A5276",
    "trans_lt" : "#AED6F1",
    "pre"      : "#7E5109",
    "pre_lt"   : "#FAD7A0",
    "fine"     : "#1D6A4A",
    "fine_lt"  : "#A9DFBF",
    "opt"      : "#AAB7B8",
    "opt_lt"   : "#F2F3F4",
    "arrow"    : "#2C3E50",
    "text_dark": "#1C2833",
    "frozen"   : "#5D6D7E",
    "learn"    : "#E74C3C",
    "sub_bg"   : "#EBF5FB",
}

def box(x, y, w, h, fc, ec, lw=1.5, radius=0.22, ls="-", zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))

def arrow(x0, y0, x1, y1, color=C["arrow"], lw=2.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                mutation_scale=12), zorder=5)

def line(x0, y0, x1, y1, color=C["arrow"], lw=2.0, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=4)

def txt(x, y, s, fs=9, color=C["text_dark"], ha="center", va="center",
        bold=False, italic=False, zorder=6):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", zorder=zorder)

# ═════════════════════════════════════════════════════════════════════════════
# COLUMN LAYOUT
# Sec 1: Patient Input        x=0.05–2.50   (w=2.45)
# Sec 2: Text Encoding        x=2.55–6.80   (w=4.25)
# Sec 3: Token Repr + Trans   x=6.85–10.00  (w=3.15)
# Sec 4: Output Heads         x=10.10–17.90 (w=7.80)
# ═════════════════════════════════════════════════════════════════════════════

# ── Phase background strips ───────────────────────────────────────────────────
phase_y0, phase_y1 = 0.45, 8.72
for px, pw, pc in [
    (0.05,  2.45, C["input_lt"]),
    (2.55,  4.25, C["enc_lt"]),
    (6.85,  3.15, C["emb_lt"]),    # wider: covers token repr + transformer
    (10.10, 7.80, C["pre_lt"]),    # output heads
]:
    ax.add_patch(FancyBboxPatch((px, phase_y0), pw, phase_y1 - phase_y0,
        boxstyle="round,pad=0,rounding_size=0.3",
        facecolor=pc, edgecolor="none", alpha=0.08, zorder=0))

for px, pw, label, color in [
    (0.05,  2.45, "① Patient Input",             C["input"]),
    (2.55,  4.25, "② Text Encoding",              C["enc"]),
    (6.85,  3.15, "③ Token Repr. + Transformer",  C["emb"]),
    (10.10, 7.80, "④ Output Heads",               C["pre"]),
]:
    txt(px + pw/2, 0.25, label, fs=8.5, color=color, bold=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Patient Input
# ═════════════════════════════════════════════════════════════════════════════
sec1_cx = 0.05 + 2.45/2

box(0.12, 5.55, 2.28, 2.90, C["input_lt"], C["input"], lw=2.0)
txt(sec1_cx, 8.22, "Patient Visits", fs=9.5, color=C["input"], bold=True)

for i, (vlabel, codes, masked) in enumerate([
    ("V₁",  ["I25.10", "E11.9", "Z87.39"], False),
    ("V₂",  ["I50.9", "N18.3", "I25.10"],  False),
    ("Vₖ",  ["[MASK]", "[MASK]", "[MASK]"], True),
]):
    vx, vy = 0.20, 7.88 - i * 0.78
    vw, vh = 2.08, 0.62
    box(vx, vy, vw, vh,
        "#FDEDEC" if masked else "#D6EAF8",
        C["learn"] if masked else C["input"],
        lw=2.0 if masked else 1.5)
    txt(vx + 0.26, vy + vh/2, vlabel, fs=8.5,
        color=C["learn"] if masked else C["input"], bold=True)
    txt(vx + 1.14, vy + vh/2, "  ".join(codes),
        fs=6.8 if masked else 7.0,
        color=C["learn"] if masked else C["text_dark"])

txt(sec1_cx, 5.72, "⋮", fs=14, color=C["input"])

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Text Encoding  (bottom-up, code_emb output at INTER_Y=3.73)
# ═════════════════════════════════════════════════════════════════════════════
sec2_x0 = 2.55
sec2_w   = 4.05
sec2_cx  = sec2_x0 + sec2_w/2
bx       = sec2_x0 + 0.10
bw       = sec2_w - 0.20

INTER_Y  = 3.73   # shared y for all inter-section horizontal arrows

CE_Y0, CE_H = INTER_Y - 0.21, 0.42    # code_emb centered at INTER_Y
box(sec2_x0 + 0.80, CE_Y0, 2.50, CE_H, "#D5F5E3", C["enc"], lw=1.5)
txt(sec2_cx, CE_Y0 + CE_H/2, "code_emb", fs=8.5, color=C["enc"], bold=True)

GAP = 0.14
PH_Y0, PH_H = CE_Y0 + CE_H + GAP, 0.54
box(bx, PH_Y0, bw, PH_H, C["opt_lt"], C["opt"], lw=1.5, ls="--")
txt(sec2_cx, PH_Y0 + PH_H - 0.20, "PheCode Fusion  (optional)", fs=8,
    color=C["frozen"], italic=True)
txt(sec2_cx, PH_Y0 + 0.18, "ICD → PheCode description → SBERT  ·  Gated Fusion",
    fs=7.0, color=C["frozen"])

TE_Y0, TE_H = PH_Y0 + PH_H + GAP, 0.54
box(bx, TE_Y0, bw, TE_H, "#D5F5E3", C["enc"], lw=2.0)
txt(sec2_cx, TE_Y0 + TE_H - 0.20, "Text Encoder  (Frozen)", fs=9,
    color=C["enc"], bold=True)
txt(sec2_cx, TE_Y0 + 0.18, "SBERT  all-MiniLM-L6-v2", fs=8.0, color=C["text_dark"])

GPT_Y0, GPT_H = TE_Y0 + TE_H + GAP, 0.52
box(bx, GPT_Y0, bw, GPT_H, C["opt_lt"], C["opt"], lw=1.5, ls="--")
txt(sec2_cx, GPT_Y0 + GPT_H - 0.20, "GPT-4o-mini Enrichment  (optional)", fs=8,
    color=C["frozen"], italic=True)
txt(sec2_cx, GPT_Y0 + 0.17, "100–120 word clinical summary per ICD code",
    fs=7.0, color=C["frozen"])

ICD_Y0, ICD_H = GPT_Y0 + GPT_H + GAP, 2.08
box(bx, ICD_Y0, bw, ICD_H, C["sub_bg"], C["enc"], lw=2.0)
txt(sec2_cx, ICD_Y0 + ICD_H - 0.18,
    "Offline ICD Description Encoding", fs=9, color=C["enc"], bold=True)

sb_x = bx + 0.10
sb_w = bw - 0.20

V1_TOP = ICD_Y0 + ICD_H - 0.28
V1_H   = 0.76
V1_Y0  = V1_TOP - V1_H
box(sb_x, V1_Y0, sb_w, V1_H, "#D6EAF8", C["input"], lw=1.0, radius=0.10, zorder=3)
txt(sb_x + 0.12, V1_TOP - 0.14, "V₁", fs=8, color=C["input"], bold=True, ha="left")
ax.text(sb_x + 0.12, V1_TOP - 0.29,
        "Atherosclerotic heart disease of native coronary artery\n"
        "without angina pectoris, Type 2 diabetes mellitus without\n"
        "complications, Personal history of other diseases of\n"
        "the musculoskeletal system and connective tissue.",
        fontsize=6.3, color=C["text_dark"], ha="left", va="top",
        linespacing=1.32, zorder=6)

V2_TOP = V1_Y0 - 0.10
V2_H   = 0.62
V2_Y0  = V2_TOP - V2_H
box(sb_x, V2_Y0, sb_w, V2_H, "#D6EAF8", C["input"], lw=1.0, radius=0.10, zorder=3)
txt(sb_x + 0.12, V2_TOP - 0.14, "V₂", fs=8, color=C["input"], bold=True, ha="left")
ax.text(sb_x + 0.12, V2_TOP - 0.29,
        "Heart failure, unspecified, Chronic kidney disease,\n"
        "stage 3 (moderate), Atherosclerotic heart disease of\n"
        "native coronary artery without angina pectoris.",
        fontsize=6.3, color=C["text_dark"], ha="left", va="top",
        linespacing=1.32, zorder=6)

txt(sec2_cx, ICD_Y0 + 0.20, "...", fs=13, color=C["enc"], bold=True)

# Down arrows inside sec 2
arrow(sec2_cx, ICD_Y0,  sec2_cx, GPT_Y0 + GPT_H, color=C["enc"])
arrow(sec2_cx, GPT_Y0,  sec2_cx, TE_Y0  + TE_H,  color=C["opt"], lw=1.4)
arrow(sec2_cx, TE_Y0,   sec2_cx, PH_Y0  + PH_H,  color=C["enc"])
arrow(sec2_cx, PH_Y0,   sec2_cx, CE_Y0  + CE_H,  color=C["enc"])

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Token Representation + Transformer (single vertical column)
# Designed so that contextual token repr. is centered at INTER_Y=3.73
# ═════════════════════════════════════════════════════════════════════════════
sec3_x0 = 6.85
sec3_cx  = sec3_x0 + 3.15/2    # 8.425
ebox_x   = sec3_x0 + 0.20
ebox_w   = 2.75

# Work bottom-up from contextual repr (centered at INTER_Y)
CONT_H   = 0.40
CONT_Y0  = INTER_Y - CONT_H/2  # 3.53

# Transformer block (above contextual repr, gap 0.28)
TRANS_H  = 0.82
TRANS_Y0 = CONT_Y0 + CONT_H + 0.28   # 4.21

# token_emb (above transformer, gap 0.26)
TK_H     = 0.42
TK_Y0    = TRANS_Y0 + TRANS_H + 0.26  # 5.29

# [MASK] emb box (above token_emb)
EMB_H    = 0.60
EMB_GAP  = 0.30    # gap between embedding boxes (plus sign lives here)
EMB_STEP = EMB_H + EMB_GAP   # 0.90

MASK_Y0  = TK_Y0 + TK_H + 0.14   # 5.85
VISIT_Y0 = MASK_Y0 + EMB_STEP     # 6.75
CODE_Y0  = VISIT_Y0 + EMB_STEP    # 7.65

# Header above the topmost box
txt(sec3_cx, 8.55, "Token Representation", fs=9.5, color=C["emb"], bold=True)

# ── Three embedding boxes ──────────────────────────────────────────────────────
for ey, label, efc, eec in [
    (CODE_Y0,  "code_emb",   C["emb_lt"], C["emb"]),
    (VISIT_Y0, "visit_emb",  "#D2B4DE",   "#7D3C98"),
    (MASK_Y0,  "[MASK] emb", "#FADBD8",   C["learn"]),
]:
    box(ebox_x, ey, ebox_w, EMB_H, efc, eec, lw=1.5)
    txt(ebox_x + ebox_w/2, ey + EMB_H/2, label, fs=8.5, color=eec, bold=True)

# Plus signs centered in each gap
for bot, top_next in [
    (CODE_Y0,  VISIT_Y0 + EMB_H),
    (VISIT_Y0, MASK_Y0  + EMB_H),
]:
    txt(sec3_cx, (bot + top_next)/2, "+", fs=14, color=C["emb"], bold=True)

# ── Arrow: [MASK] emb → token_emb ─────────────────────────────────────────────
arrow(sec3_cx, MASK_Y0, sec3_cx, TK_Y0 + TK_H, color=C["emb"], lw=1.8)

# ── token_emb box ─────────────────────────────────────────────────────────────
box(ebox_x, TK_Y0, ebox_w, TK_H, "#E8DAEF", C["emb"], lw=1.8)
txt(sec3_cx, TK_Y0 + TK_H/2, "token_emb", fs=8.5, color=C["emb"], bold=True)

# ── Arrow: token_emb → Transformer ───────────────────────────────────────────
arrow(sec3_cx, TK_Y0, sec3_cx, TRANS_Y0 + TRANS_H, color=C["trans"], lw=1.8)

# ── Transformer single block ──────────────────────────────────────────────────
box(ebox_x, TRANS_Y0, ebox_w, TRANS_H, "#D6EAF8", C["trans"], lw=2.0)
txt(sec3_cx, TRANS_Y0 + TRANS_H/2 + 0.12, "Transformer Encoder",
    fs=8.5, color=C["trans"], bold=True)
txt(sec3_cx, TRANS_Y0 + TRANS_H/2 - 0.14, "Multi-Head Attention + FFN",
    fs=7.2, color=C["trans"])

# ── Arrow: Transformer → contextual token repr. ───────────────────────────────
arrow(sec3_cx, TRANS_Y0, sec3_cx, CONT_Y0 + CONT_H, color=C["trans"], lw=1.8)

# ── Contextual token repr. ────────────────────────────────────────────────────
box(ebox_x, CONT_Y0, ebox_w, CONT_H, "#D6EAF8", C["trans"], lw=1.5)
txt(sec3_cx, CONT_Y0 + CONT_H/2, "contextual token repr.", fs=8,
    color=C["trans"], bold=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Output Heads  (x=10.10, w=7.80)
# ═════════════════════════════════════════════════════════════════════════════
sec5_x0 = 10.10
sec5_w   = 7.70
sec5_cx  = sec5_x0 + sec5_w/2   # 13.95

txt(sec5_cx, 8.55, "Output Heads", fs=9.5, color=C["pre"], bold=True)

# ── Pre-training box ──────────────────────────────────────────────────────────
pre_y0, pre_h = 5.62, 2.65
box(sec5_x0 + 0.15, pre_y0, sec5_w - 0.30, pre_h, C["pre_lt"], C["pre"], lw=2.0)
txt(sec5_cx, pre_y0 + pre_h - 0.22, "Pre-Training", fs=9, color=C["pre"], bold=True)

mlm_w = 3.10
box(sec5_x0 + 0.35, pre_y0 + 1.12, mlm_w, 0.90, "#FDEBD0", C["pre"], lw=1.5)
txt(sec5_x0 + 0.35 + mlm_w/2, pre_y0 + 1.57, "MLM Head",
    fs=8.5, color=C["pre"], bold=True)

nvp_x0 = sec5_x0 + 0.35 + mlm_w + 0.40
nvp_w  = 3.10
box(nvp_x0, pre_y0 + 1.12, nvp_w, 0.90, "#FDEBD0", C["pre"], lw=1.5)
txt(nvp_x0 + nvp_w/2, pre_y0 + 1.57, "Next-Visit\nPrediction Head",
    fs=8, color=C["pre"], bold=True)

# ── Fine-tuning box ───────────────────────────────────────────────────────────
ft_y0, ft_h = 0.82, 4.57
box(sec5_x0 + 0.15, ft_y0, sec5_w - 0.30, ft_h, C["fine_lt"], C["fine"], lw=2.0)
txt(sec5_cx, ft_y0 + ft_h - 0.22, "Fine-Tuning", fs=9, color=C["fine"], bold=True)

mp_y0, mp_h = ft_y0 + 3.22, 0.52
box(sec5_x0 + 1.50, mp_y0, 4.70, mp_h, "#D5F5E3", C["fine"], lw=1.5)
txt(sec5_cx, mp_y0 + mp_h/2, "Mean Pooling", fs=8.5, color=C["fine"], bold=True)

cl_y0, cl_h = ft_y0 + 2.35, 0.55
box(sec5_x0 + 1.50, cl_y0, 4.70, cl_h, "#D5F5E3", C["fine"], lw=1.5)
txt(sec5_cx, cl_y0 + cl_h/2, "Classifier Head", fs=8.5, color=C["fine"], bold=True)

rs_y0, rs_h = ft_y0 + 1.55, 0.52
box(sec5_x0 + 2.20, rs_y0, 3.30, rs_h, "#0E6655", "#0E6655", lw=2.0)
txt(sec5_cx, rs_y0 + rs_h/2, "Risk Score", fs=8.5, color="white", bold=True)

arrow(sec5_cx, mp_y0, sec5_cx, cl_y0 + cl_h, color=C["fine"])
arrow(sec5_cx, cl_y0, sec5_cx, rs_y0 + rs_h, color=C["fine"])

line(sec5_x0 + 0.15, 5.62, sec5_x0 + sec5_w - 0.15, 5.62,
     color=C["arrow"], lw=1.5)

# ═════════════════════════════════════════════════════════════════════════════
# INTER-SECTION ARROWS  (all at y=INTER_Y=3.73)
# ═════════════════════════════════════════════════════════════════════════════
# ① → ②  Patient Input → Text Encoding
arrow(2.28, INTER_Y, 2.53, INTER_Y, color=C["enc"], lw=2.5)

# ② → ③  Text Encoding → code_emb in Token Repr (L-shaped: right → up → right)
CODE_MID_Y = CODE_Y0 + EMB_H / 2   # center of code_emb box in sec3 (~7.95)
route_x    = 6.74                   # right of sec2 boxes (end ~6.50), left of sec3 strip (6.85)
line(5.87, INTER_Y, route_x, INTER_Y, color=C["emb"], lw=2.5)   # horizontal right
line(route_x, INTER_Y, route_x, CODE_MID_Y, color=C["emb"], lw=2.5)  # vertical up
arrow(route_x, CODE_MID_Y, ebox_x + 0.15, CODE_MID_Y, color=C["emb"], lw=2.5)  # into code_emb

# ③ → ④  Contextual repr → Output Heads (fork)
ctx_right = ebox_x + ebox_w    # ~9.80
fork_x    = 10.05

line(ctx_right, INTER_Y, fork_x, INTER_Y, color=C["arrow"], lw=2.0)

# Upper fork → Pre-training
pt_y = pre_y0 + pre_h/2
line(fork_x, INTER_Y, fork_x, pt_y, color=C["arrow"], lw=2.0)
arrow(fork_x, pt_y, sec5_x0 + 0.15, pt_y, color=C["pre"], lw=2.0)

# Lower fork → Fine-tuning
ft_y = ft_y0 + ft_h/2
line(fork_x, INTER_Y, fork_x, ft_y, color=C["arrow"], lw=2.0)
arrow(fork_x, ft_y, sec5_x0 + 0.15, ft_y, color=C["fine"], lw=2.0)

# ═════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═════════════════════════════════════════════════════════════════════════════
leg_x, leg_y = 0.10, 4.88
txt(leg_x, leg_y + 0.24, "Legend", fs=7.5, bold=True, ha="left")
for i, (marker, color, label) in enumerate([
    ("●", C["frozen"], "Frozen"),
    ("●", C["learn"],  "Learnable"),
    ("--", C["opt"],   "Optional"),
]):
    ly = leg_y - i * 0.38
    if marker == "--":
        ax.plot([leg_x, leg_x + 0.30], [ly, ly],
                color=color, lw=1.5, ls="--", zorder=6)
    else:
        ax.text(leg_x, ly, marker, color=color, fontsize=11,
                va="center", zorder=6)
    ax.text(leg_x + 0.36, ly, label, color=C["text_dark"],
            fontsize=7.5, va="center", zorder=6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FIG_W/2, FIG_H + 0.12,
        "Sent-e-Med: EHR Representation Learning via Masked Language Modeling on ICD Sequences",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=C["text_dark"], zorder=6, clip_on=False)

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = "/sessions/gifted-admiring-keller/mnt/LLM project code/sent_e_med_architecture.pdf"
out_png = "/sessions/gifted-admiring-keller/mnt/LLM project code/sent_e_med_architecture.png"
plt.savefig(out_pdf, bbox_inches="tight", dpi=200, facecolor="white")
plt.savefig(out_png, bbox_inches="tight", dpi=200, facecolor="white")
print("Saved:", out_pdf)
print("Saved:", out_png)
plt.close()
