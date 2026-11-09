"""Generates notebooks/kubaho_model.ipynb. Execute it afterwards with:

    .venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/kubaho_model.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md(r"""
# Kubaho — Model Notebook (initial version)

**Kubaho: a camera-free WiFi-sensing system for elderly fall and wellness monitoring in rural Rwandan homes,
using π RuView and a LoRA-adapted fall-detection model.**
BSc Software Engineering

This notebook covers the ML part of the MVP:

1. **Data engineering**: the CSI quality phase (packet-loss repair, phase sanitisation, Hampel filter,
   band-pass, windowing, session filtering).
2. **Data visualisation**: class distribution, CSI heatmaps, motion profiles, feature distributions and
   correlations, and the domain gap.
3. **Model architecture**: RuView-style CSI encoder + LoRA adapter (layers, activations, optimiser, parameter budget).
4. **Initial performance metrics**: models M0–M7 from proposal Table 5 on leave-one-layout-out splits with
   unseen subjects. Accuracy, precision, recall, F1, sensitivity and specificity, plus McNemar's test.
5. **Deployment artefact**: merged INT8 ONNX model + model card, served by the FastAPI `/predict` endpoint.

> **Data status: these results are preliminary.** The Kubaho-Lab recordings (20 volunteers in a mock rural
> room) and the pilot field data have not been collected yet. Every result below comes from a
> **physics-inspired synthetic CSI generator** (`src/kubaho/synthetic.py`). It reproduces the structure of
> the planned dataset: 12 activities, 2 room layouts, 20 subjects, ESP32-S3 noise, packet loss and phase
> offsets. The RuView pretrained encoder is replaced by a stand-in encoder pre-trained on synthetic
> *generic-room* data. The pipeline is complete and runs end to end; once the real data exists, it replaces
> the synthetic data through `kubaho.data`.
""")

code(r"""
import os, sys, json, time, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")   # local file store (./mlruns); no tracking server needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import mlflow

from kubaho import CLASSES, FLOOR_LEVEL
from kubaho.data import load_or_build
from kubaho.synthetic import (KUBAHO_ACTIVITIES, SOURCE_CLASSES, make_layout, make_subject, simulate_clip, SUB_IDX)
from kubaho.preprocess import (MAX_LOSS, FS, TARGET_FS, interpolate_lost, sanitize_phase, hampel, bandpass,
                               preprocess_window, motion_energy)
from kubaho.features import extract_features, FEATURE_NAMES
from kubaho.baselines import ThresholdDetector, threshold_scores
from kubaho.models import CSIClassifier, CNNLSTM, count_parameters
from kubaho.train import TrainConfig, fit, predict_proba, set_seed
from kubaho.metrics import evaluate, calibrate_threshold, fall_metrics, mcnemar, FALL

SEED = 42
set_seed(SEED)
torch.set_num_threads(os.cpu_count())
FIG = ROOT / "docs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
mlflow.set_experiment("kubaho")
print("torch", torch.__version__, "| CPU threads", torch.get_num_threads())
""")

code(r"""
# Chart style: one recessive grid, thin marks, fixed categorical order (blue, orange, aqua).
BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
GROUP_COLORS = {"Floor-level daily activity": BLUE, "Fall": ORANGE, "Other (walk, empty)": AQUA}
SEQ = mcolors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
DIV = mcolors.LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"])
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2,
    "ytick.color": INK2, "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "legend.frameon": False, "figure.dpi": 110,
})

def group_of(cls):
    return "Fall" if cls == "fall" else ("Floor-level daily activity" if cls in FLOOR_LEVEL else "Other (walk, empty)")

def save(name):
    plt.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
""")

md(r"""
## 1. Data definition and acquisition

| Source | Content | Volume | Status in this notebook |
|---|---|---|---|
| **Kubaho-Lab** (primary) | CSI from ESP32-S3 at 100 packets/s in a mock rural room (mats, low stools, low bed), 2 layouts; 4 fall types, 6 fall-like/daily activities, walk, empty room | 20 volunteers × 12 activities × 2 layouts × 5 reps = 2,400 events | **Synthetic stand-in** |
| Kubaho-Field | On-device event logs and short CSI windows from 8 pilot homes, caregiver/CHW diary as ground truth | 8 homes × 6 weeks | The API records acknowledgements as ground truth |
| MM-Fi (public) | WiFi CSI for 27 daily actions, 40 subjects | — | Replaced by a synthetic *generic-room* source domain for pre-training |
| RuView pretrained encoder | 128-d CSI embedding model | — | Replaced by a stand-in encoder with the same 128-d interface, pre-trained here |

Each event is a 3 s clip at 100 packets/s. Three raw 2 s windows (offsets 0, 0.5 and 1.0 s) are cut from it and
cleaned. The 0.5 s window is used for validation and testing; all three are used as training augmentation.
""")

code(r"""
t0 = time.time()
X_lab, meta = load_or_build("kubaho_lab")      # (N, 3 crops, 100 frames @ 50 Hz, 52 subcarriers)
X_src, meta_src = load_or_build("source")
print(f"built/loaded in {time.time()-t0:.0f}s")
print("Kubaho-Lab:", X_lab.shape, "| generic-room source:", X_src.shape)
meta.head()
""")

md(r"""
## 2. Data engineering: the CSI quality phase

The ESP32-S3 delivers complex CSI per packet. Before learning, we:

1. **Repair lost packets**: linear interpolation along time.
2. **Sanitise phase**: unwrap across subcarriers and remove the per-packet linear slope (sampling-time
   offset) and constant (carrier-frequency offset).
3. **Hampel filter**: replace outliers that are more than 3 robust standard deviations from the rolling median.
4. **Band-pass 0.2–20 Hz**: remove the static room response and high-frequency noise. Human motion is
   below 20 Hz, so we then decimate to 50 Hz.
5. **Window**: 2 s windows with a 0.25 s stride on streams (the node), 3 crops per clip for training.
6. **Drop sessions** with more than 10% packet loss.
""")

code(r"""
rng = np.random.default_rng(11)
demo_layout, demo_subject = make_layout("A", 8, rural=True), make_subject(rng)
raw = simulate_clip("fall_forward", demo_layout, demo_subject, rng, loss_rate=0.04)
raw_amp = np.abs(raw)
clean = preprocess_window(raw[50:250])

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
im = axes[0].imshow(raw_amp.T, aspect="auto", origin="lower", cmap=SEQ, extent=[0, 3, 0, 52])
axes[0].set(title="Raw CSI amplitude: forward fall (white gaps = lost packets)", xlabel="time (s)", ylabel="subcarrier")
axes[0].grid(False); fig.colorbar(im, ax=axes[0], fraction=0.03)
lim = np.abs(clean).max()
im = axes[1].imshow(clean.T, aspect="auto", origin="lower", cmap=DIV, vmin=-lim, vmax=lim, extent=[0.5, 2.5, 0, 52])
axes[1].set(title="After the quality phase (2 s window, 50 Hz)", xlabel="time (s)", ylabel="subcarrier")
axes[1].grid(False); fig.colorbar(im, ax=axes[1], fraction=0.03)
plt.tight_layout(); save("01_raw_vs_clean_heatmap"); plt.show()
""")

code(r"""
filled = interpolate_lost(raw)
packets = [10, 60, 110, 160, 210]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharex=True)
for p in packets:
    axes[0].plot(SUB_IDX, np.unwrap(np.angle(filled[p])), color=BLUE, lw=1.2, alpha=0.7)
    axes[1].plot(SUB_IDX, sanitize_phase(filled[p:p+1])[0], color=BLUE, lw=1.2, alpha=0.7)
axes[0].set(title="Raw unwrapped phase, 5 packets", xlabel="subcarrier index", ylabel="radians")
axes[1].set(title="Sanitised phase (slope and offset removed)", xlabel="subcarrier index")
plt.tight_layout(); save("02_phase_sanitisation"); plt.show()

k = 20
amp = np.abs(filled)[:, k]
t = np.arange(len(amp)) / FS
h = hampel(np.abs(filled))[:, k]
bp = bandpass(hampel(np.abs(filled)))[:, k]
fig, axes = plt.subplots(2, 1, figsize=(12, 4.4), sharex=True)
axes[0].plot(t, amp, color=GRAY, lw=1, label="raw")
axes[0].plot(t, h, color=BLUE, lw=1.5, label="after Hampel filter")
axes[0].set(title=f"Subcarrier {SUB_IDX[k]}: outlier removal", ylabel="amplitude"); axes[0].legend(loc="upper right")
axes[1].plot(t, bp, color=BLUE, lw=1.5)
axes[1].axvspan(0.5, 2.5, color=ORANGE, alpha=0.08, lw=0)
axes[1].text(0.52, bp.max()*0.85, "centre 2 s window", color=INK2)
axes[1].set(title="After 0.2–20 Hz band-pass (static room response removed)", xlabel="time (s)", ylabel="amplitude")
plt.tight_layout(); save("03_hampel_bandpass"); plt.show()
""")

code(r"""
loss = meta.groupby("session")["packet_loss"].mean().sort_values()
bad = loss[loss > MAX_LOSS].index.tolist()
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.bar(range(len(loss)), loss.values * 100, color=[ORANGE if s in bad else BLUE for s in loss.index], width=0.7)
ax.axhline(MAX_LOSS * 100, color=INK2, lw=1, ls="--"); ax.text(0.2, MAX_LOSS * 100 + 0.6, "10% drop threshold", color=INK2)
ax.set(title="Packet loss per recording session (orange = dropped)", ylabel="% packets lost", xlabel="session (sorted)")
ax.set_xticks([]); plt.tight_layout(); save("04_packet_loss"); plt.show()

keep = ~meta.session.isin(bad)
X_lab, meta = X_lab[keep.values], meta[keep].reset_index(drop=True)
y_lab = meta.label.values
print(f"dropped {len(bad)} sessions {bad} -> {len(meta)} events remain")
""")

md("## 3. Data visualisation")

code(r"""
counts = meta.class_name.value_counts().reindex(CLASSES)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
axes[0].barh(CLASSES[::-1], counts.values[::-1], color=[ORANGE if c == "fall" else BLUE for c in CLASSES[::-1]], height=0.6)
for i, v in enumerate(counts.values[::-1]):
    axes[0].text(v + 8, i, str(v), va="center", color=INK2)
axes[0].set(title="Events per class (8 model classes)", xlabel="events"); axes[0].grid(axis="y", visible=False)
act = meta.activity.value_counts().reindex(list(KUBAHO_ACTIVITIES))
axes[1].barh(act.index[::-1], act.values[::-1], color=[ORANGE if a.startswith("fall") else BLUE for a in act.index[::-1]], height=0.6)
axes[1].set(title="Events per recorded activity (12 protocol activities)", xlabel="events"); axes[1].grid(axis="y", visible=False)
plt.tight_layout(); save("05_class_distribution"); plt.show()
print(f"Imbalance: falls are {counts['fall']/counts.sum():.0%} of events; handled with class-weighted focal loss.")
""")

code(r"""
examples = ["fall_forward", "sit_on_mat", "kneel_pray", "squat_stool", "bend_sweep", "walk"]
fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), sharex=True, sharey=True)
wins = [X_lab[meta.index[(meta.subject == 3) & (meta.layout == "A") & (meta.rep == 0) & (meta.activity == a)][0], 1] for a in examples]
lim = np.percentile(np.abs(np.stack(wins)), 99.5)
for ax, a, w in zip(axes.flat, examples, wins):
    ax.imshow(w.T, aspect="auto", origin="lower", cmap=DIV, vmin=-lim, vmax=lim, extent=[0, 2, 0, 52])
    ax.set_title(a.replace("_", " "), color=ORANGE if a.startswith("fall") else INK); ax.grid(False)
for ax in axes[1]: ax.set_xlabel("time (s)")
for ax in axes[:, 0]: ax.set_ylabel("subcarrier")
fig.suptitle("Cleaned CSI amplitude: a fall vs. fall-like floor-level activities (same volunteer, layout A)", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); save("06_csi_heatmaps"); plt.show()
""")

code(r"""
E = motion_energy(X_lab[:, 1])                      # (N, 99)
tt = np.arange(E.shape[1]) / TARGET_FS
groups = meta.class_name.map(group_of)
fig, ax = plt.subplots(figsize=(12, 3.6))
for g, color in GROUP_COLORS.items():
    e = E[(groups == g).values]
    ax.plot(tt, np.median(e, 0), color=color, lw=2, label=g)
    ax.fill_between(tt, *np.percentile(e, [25, 75], 0), color=color, alpha=0.12, lw=0)
ax.set(title="Motion energy over the 2 s window (median and interquartile range)", xlabel="time (s)", ylabel="mean |Δ amplitude|")
ax.legend(loc="upper right"); plt.tight_layout(); save("07_motion_energy_profiles"); plt.show()
""")

code(r"""
t0 = time.time()
F_lab = extract_features(X_lab[:, 1])
F_df = pd.DataFrame(F_lab, columns=FEATURE_NAMES)
print(f"{F_lab.shape[1]} features for {len(F_lab)} windows in {time.time()-t0:.0f}s")

show = ["energy_max", "accel_max", "post_peak_stillness", "energy_peak_ratio"]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
for ax, f in zip(axes, show):
    data = [F_df.loc[meta.class_name == c, f].values for c in CLASSES]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS))
    for patch, c in zip(bp["boxes"], CLASSES):
        patch.set(facecolor=ORANGE if c == "fall" else BLUE, alpha=0.75, edgecolor="#fcfcfb")
    ax.set_title(f.replace("_", " ")); ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(1, len(CLASSES) + 1), CLASSES)
fig.suptitle("Feature distributions by class (falls in orange): post-peak stillness separates falls best", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); save("08_feature_distributions"); plt.show()
""")

code(r"""
corr = F_df.corr()
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr, cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson r")
ax.set_title("Correlation between hand-crafted CSI features")
plt.tight_layout(); save("09_feature_correlation"); plt.show()
top = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().abs().sort_values(ascending=False).head(5)
print("Most correlated pairs (candidates for pruning):"); print(top.round(2).to_string())
""")

md(r"""
### The domain gap (RQ1)

A generic detector learns that *"a big, sudden change followed by stillness"* means a fall. Its threshold is
calibrated on generic rooms with hard floors and chairs. In a rural room, falls land on mats and are softer,
while squatting onto a low stool or kneeling to pray are fast descents that end in stillness. The plot shows
the threshold detector's two decision variables on the Kubaho-Lab data. The dashed lines are the
thresholds calibrated on generic-room data.
""")

code(r"""
src_fall = (meta_src.class_name == "fall").values
m0 = ThresholdDetector().fit(X_src[:, 1], src_fall, target_sensitivity=0.95)
peak, still = threshold_scores(X_lab[:, 1])
fig, ax = plt.subplots(figsize=(9, 5))
for g, color in GROUP_COLORS.items():
    m = (groups == g).values
    ax.scatter(peak[m], still[m], s=14, color=color, alpha=0.55, label=g, edgecolors="#fcfcfb", linewidths=0.4)
ax.axvline(m0.accel_threshold, color=INK2, ls="--", lw=1); ax.axhline(m0.stillness_ratio, color=INK2, ls="--", lw=1)
ax.text(m0.accel_threshold * 1.03, ax.get_ylim()[1] * 0.95, "threshold from\ngeneric rooms", color=INK2, va="top")
ax.set_xscale("log")
ax.set(title="M0 threshold detector's view of Kubaho-Lab windows", xlabel="peak motion-energy acceleration (log)",
       ylabel="post-peak stillness ratio")
ax.legend(loc="upper left"); plt.tight_layout(); save("10_domain_gap_threshold"); plt.show()
alerts = m0.predict(X_lab[:, 1])
print(f"M0 alert fires on {alerts[y_lab == FALL].mean():.0%} of falls and on "
      f"{alerts[np.isin(y_lab, [CLASSES.index(c) for c in FLOOR_LEVEL])].mean():.0%} of floor-level daily activities")
""")

md(r"""
## 4. Evaluation protocol

**Leave-one-layout-out with unseen subjects.** For each fold, the model trains on one room layout
(subjects 1–13), validates on subjects 14–16 in the same layout (early stopping and threshold calibration),
and is **tested on the other layout with subjects 17–20**. The test room and test people are both unseen.
There are two folds (A→B, B→A), and the reported metrics are their mean.

The fall-alert threshold is calibrated on the validation set: the highest threshold that keeps sensitivity
at or above 95%, which gives the fewest false alarms.
""")

code(r"""
def split(train_layout):
    test_layout = "B" if train_layout == "A" else "A"
    tr = np.where((meta.layout == train_layout) & (meta.subject <= 13))[0]
    va = np.where((meta.layout == train_layout) & meta.subject.between(14, 16))[0]
    te = np.where((meta.layout == test_layout) & (meta.subject >= 17))[0]
    return tr, va, te

for L in "AB":
    tr, va, te = split(L)
    print(f"fold {L}->{'B' if L=='A' else 'A'}: train {len(tr)} events ({len(tr)*3} windows), val {len(va)}, test {len(te)}"
          f" (falls in test: {(y_lab[te]==FALL).sum()})")
""")

md(r"""
## 5. Pre-trained CSI encoder (RuView-encoder stand-in)

RuView ships a pretrained 128-d CSI encoder. We pre-train an encoder with the same interface on
**generic-room data** only (6 activities, 3 office layouts, hard floors, chairs). It has never seen mats,
low stools or floor-level postures. This reproduces the starting point the proposal assumes.
""")

code(r"""
y_src = meta_src.label.values
s_tr, s_va = np.where(meta_src.subject < 24)[0], np.where(meta_src.subject >= 24)[0]
encoder_model = CSIClassifier(n_classes=len(SOURCE_CLASSES))
flat = X_src[s_tr].reshape(-1, 52)
encoder_model.encoder.set_normalization(flat.mean(0), flat.std(0))
hist = fit(encoder_model, X_src[s_tr], y_src[s_tr], X_src[s_va], y_src[s_va], len(SOURCE_CLASSES),
           TrainConfig(epochs=20, lr=1e-3, seed=SEED), run_name="pretrain_encoder")
ENCODER_PATH = MODELS / "ruview_encoder_standin.pt"
torch.save(encoder_model.encoder.state_dict(), ENCODER_PATH)
print(f"source-domain val macro-F1 {hist['best_val_macro_f1']:.3f} after {hist['epochs_run']} epochs ({hist['seconds']:.0f}s)")
""")

md(r"""
## 6. Model architecture

```
CSI window (100 frames × 52 subcarriers, 50 Hz)
 └─ normalisation (per-subcarrier mean/std, stored in the model)
 └─ Conv1d 52→96, k=5, stride 2 + GELU
 └─ Conv1d 96→96, k=3, stride 2 + GELU        (temporal downsampling: 100 frames → 25 tokens)
 └─ + learned positional embedding (25 × 96)
 └─ 3 × pre-norm Transformer block
 │     LayerNorm → multi-head self-attention (4 heads; q, k, v, o Linear 96→96)  ← LoRA on q and v
 │     LayerNorm → FFN Linear 96→192 → GELU → Linear 192→96
 │     residual connections, dropout 0.1
 └─ LayerNorm → mean pooling → Linear 96→128 → 128-d embedding   (frozen RuView-style encoder)
 └─ classification head Linear 128→8 (trainable)
```

The stand-in encoder is deliberately small (~0.3 M parameters) so that the full benchmark trains on a
2-core laptop CPU. RuView's encoder takes its place when the weights are available.

**LoRA.** For each adapted weight $W \in \mathbb{R}^{d\times k}$ (frozen), we learn $\Delta W = BA$ with
$B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r \ll \min(d,k)$, scaled by $\alpha/r$ ($\alpha = 2r$).
$B$ starts at zero, so training begins exactly at the pretrained model.

**Training.** Class-weighted focal loss ($\gamma = 2$) for class imbalance, AdamW (weight decay 0.01), cosine
learning-rate schedule, early stopping on validation macro-F1 (patience 6), random-crop augmentation.
Experiments are logged to MLflow (`mlruns/`).
""")

code(r"""
from peft import LoraConfig, get_peft_model

def pretrained_classifier():
    m = CSIClassifier(n_classes=len(CLASSES))
    m.encoder.load_state_dict(torch.load(ENCODER_PATH))
    return m

def lora_model(r):
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=["q", "v"], modules_to_save=["head"])
    return get_peft_model(pretrained_classifier(), cfg)

def lora_param_count(m):
    return sum(p.numel() for n, p in m.named_parameters() if "lora_" in n)

print(pretrained_classifier())
enc_params = count_parameters(pretrained_classifier().encoder)
rows = []
for r in (4, 8, 16):
    m = lora_model(r)
    lp = lora_param_count(m)
    trainable = count_parameters(m, trainable_only=True)
    rows.append({"LoRA rank r": r, "LoRA params": lp, "% of encoder": 100 * lp / enc_params,
                 "trainable incl. head": trainable, "adapter size fp32 (KB)": lp * 4 / 1024, "adapter size int8 (KB)": lp / 1024})
print(f"Encoder parameters: {enc_params:,}")
pd.DataFrame(rows).round(2)
""")

md(r"""
## 7. Training and benchmarking M0–M7 (proposal Table 5)

| ID | Model | Input |
|---|---|---|
| M0 | RuView-style threshold detector (calibrated on generic rooms) | motion-energy acceleration |
| M1 | Random Forest | 23 statistical features |
| M2 | XGBoost | 23 statistical features |
| M3 | SVM (RBF) | 23 statistical features |
| M4 | CNN-LSTM from scratch | cleaned CSI windows |
| M5 | Pretrained encoder + linear probe | 128-d embeddings |
| M6 | Pretrained encoder, full fine-tune | cleaned CSI windows |
| **M7** | **Pretrained encoder + LoRA adapter (r = 4, proposed)** | cleaned CSI windows |
""")

code(r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

t0 = time.time()
F_all = extract_features(X_lab.reshape(-1, *X_lab.shape[2:])).reshape(len(X_lab), 3, -1)  # features for every crop
print(f"features for all crops in {time.time()-t0:.0f}s")

@torch.no_grad()
def embed(model, X, bs=256):
    model.eval()
    return np.concatenate([model.embed(torch.from_numpy(X[i:i+bs])).numpy() for i in range(0, len(X), bs)])

def classical(make):
    def run(tr, va, te):
        clf = make().fit(F_all[tr].reshape(-1, F_all.shape[-1]), np.repeat(y_lab[tr], 3))
        return clf.predict_proba(F_all[va, 1]), clf.predict_proba(F_all[te, 1]), clf
    return run

def torch_model(make, lr, epochs=25, name=None):
    def run(tr, va, te):
        m = make()
        if hasattr(m, "set_normalization"):          # models trained from scratch use target-domain stats
            flat = X_lab[tr].reshape(-1, 52); m.set_normalization(flat.mean(0), flat.std(0))
        h = fit(m, X_lab[tr], y_lab[tr], X_lab[va], y_lab[va], len(CLASSES),
                TrainConfig(epochs=epochs, lr=lr, seed=SEED), run_name=name)
        print(f"   {name}: {h['epochs_run']} epochs, {h['seconds']:.0f}s, val macro-F1 {h['best_val_macro_f1']:.3f}")
        return predict_proba(m, X_lab[va]), predict_proba(m, X_lab[te]), m
    return run

def linear_probe(tr, va, te):
    enc = pretrained_classifier()
    Z = lambda idx, crop: embed(enc, X_lab[idx, crop])
    Ztr = np.concatenate([Z(tr, c) for c in range(3)])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Ztr, np.tile(y_lab[tr], 3))
    return clf.predict_proba(Z(va, 1)), clf.predict_proba(Z(te, 1)), clf

MODELS_TO_RUN = {
    "M1 Random Forest": classical(lambda: RandomForestClassifier(400, class_weight="balanced", n_jobs=-1, random_state=SEED)),
    "M2 XGBoost": classical(lambda: XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9,
                                                  colsample_bytree=0.9, random_state=SEED, n_jobs=-1)),
    "M3 SVM (RBF)": classical(lambda: make_pipeline(StandardScaler(), SVC(C=10, probability=True, class_weight="balanced",
                                                                          random_state=SEED))),
    "M4 CNN-LSTM (scratch)": torch_model(lambda: CNNLSTM(len(CLASSES)), lr=1e-3, name="M4_cnn_lstm"),
    "M5 Encoder + linear probe": linear_probe,
    "M6 Encoder full fine-tune": torch_model(pretrained_classifier, lr=3e-4, name="M6_full_finetune"),
    "M7 Encoder + LoRA r=4 (proposed)": torch_model(lambda: lora_model(4), lr=2e-3, name="M7_lora_r4"),
}
""")

code(r"""
results, test_store = [], {}
for L in "AB":
    tr, va, te = split(L)
    fold = f"{L}->{'B' if L == 'A' else 'A'}"
    print(f"=== fold {fold}")
    m0_pred = m0.predict(X_lab[te, 1])
    results.append({"model": "M0 Threshold (RuView-style)", "fold": fold, **fall_metrics(y_lab[te], m0_pred)})
    test_store[("M0 Threshold (RuView-style)", fold)] = {"y": y_lab[te], "fall_pred": m0_pred, "idx": te}
    for name, run in MODELS_TO_RUN.items():
        t0 = time.time()
        p_va, p_te, fitted = run(tr, va, te)
        thr = calibrate_threshold(y_lab[va], p_va[:, FALL])
        res = evaluate(y_lab[te], p_te, thr)
        results.append({"model": name, "fold": fold, **res, "train_s": time.time() - t0})
        test_store[(name, fold)] = {"y": y_lab[te], "proba": p_te, "fall_pred": p_te[:, FALL] >= thr, "idx": te,
                                    "model": fitted}
        print(f"  {name:<34} fall F1 {res['fall_f1']:.3f}  sens {res['fall_sensitivity']:.3f}  "
              f"spec(floor) {res['specificity_floor_level']:.3f}  macro-F1 {res['macro_f1']:.3f}")
res_df = pd.DataFrame(results)
""")

code(r"""
cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "fall_precision", "fall_sensitivity", "fall_f1",
        "specificity_floor_level", "specificity_all_non_fall", "false_alarms"]
order = ["M0 Threshold (RuView-style)"] + list(MODELS_TO_RUN)
summary = res_df.groupby("model")[cols].mean().reindex(order)
summary.to_csv(ROOT / "docs" / "results_summary.csv")
summary.style.format("{:.3f}", subset=cols[:-1]).format("{:.1f}", subset=["false_alarms"]).highlight_max(
    subset=["fall_f1", "macro_f1", "specificity_floor_level"], color="#cde2fb")
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
names = summary.index.tolist()
for ax, (metric, title) in zip(axes, [("fall_f1", "Fall F1-score"), ("fall_sensitivity", "Fall sensitivity (recall)"),
                                      ("specificity_floor_level", "Specificity on floor-level activities")]):
    vals = summary[metric].values
    ax.barh(range(len(names)), vals, color=[ORANGE if n.startswith("M7") else BLUE for n in names], height=0.6)
    for i, v in enumerate(vals):
        ax.text(min(v, 1) + 0.01, i, f"{v:.2f}", va="center", color=INK2, fontsize=9)
    target = {"fall_f1": 0.90, "fall_sensitivity": 0.95, "specificity_floor_level": 0.98}[metric]
    ax.axvline(target, color=INK2, ls="--", lw=1); ax.text(target, -0.9, f"target {target:.2f}", color=INK2, ha="center", fontsize=8)
    ax.set(title=title, xlim=(0, 1.12)); ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(names)), names); axes[0].invert_yaxis()
fig.suptitle("Leave-one-layout-out, unseen subjects (mean of 2 folds)", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); save("11_model_comparison"); plt.show()
""")

md("### LoRA rank sweep (r ∈ {4, 8, 16}, fold A→B)")

code(r"""
sweep = []
tr, va, te = split("A")
for r in (4, 8, 16):
    if r == 4:
        row = res_df[(res_df.model.str.startswith("M7")) & (res_df.fold == "A->B")].iloc[0].to_dict()
    else:
        p_va, p_te, _ = torch_model(lambda: lora_model(r), lr=2e-3, name=f"M7_lora_r{r}")(tr, va, te)
        row = evaluate(y_lab[te], p_te, calibrate_threshold(y_lab[va], p_va[:, FALL]))
    sweep.append({"rank": r, "fall_f1": row["fall_f1"], "macro_f1": row["macro_f1"],
                  "specificity_floor_level": row["specificity_floor_level"]})
sweep_df = pd.DataFrame(sweep).set_index("rank")
sweep_df["% of encoder"] = [100 * lora_param_count(lora_model(r)) / enc_params for r in sweep_df.index]
sweep_df.round(3)
""")

md("## 8. Detailed metrics for the proposed model (M7, both test folds pooled)")

code(r"""
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

M7 = "M7 Encoder + LoRA r=4 (proposed)"
y7 = np.concatenate([test_store[(M7, f)]["y"] for f in ("A->B", "B->A")])
p7 = np.concatenate([test_store[(M7, f)]["proba"] for f in ("A->B", "B->A")])
print(classification_report(y7, p7.argmax(1), target_names=CLASSES, digits=3))

cm = confusion_matrix(y7, p7.argmax(1), normalize="true")
fig, ax = plt.subplots(figsize=(7.5, 6.2))
im = ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if cm[i, j] >= 0.01:
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#fcfcfb" if cm[i, j] > 0.55 else INK)
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right"); ax.set_yticks(range(len(CLASSES)), CLASSES)
ax.set(xlabel="predicted", ylabel="actual", title="M7 confusion matrix (row-normalised)"); ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.04); plt.tight_layout(); save("12_confusion_matrix"); plt.show()
""")

code(r"""
is_fall = y7 == FALL
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name, color in [("M5 Encoder + linear probe", BLUE), (M7, ORANGE)]:
    yy = np.concatenate([test_store[(name, f)]["y"] for f in ("A->B", "B->A")]) == FALL
    pp = np.concatenate([test_store[(name, f)]["proba"] for f in ("A->B", "B->A")])[:, FALL]
    fpr, tpr, _ = roc_curve(yy, pp); prec, rec, _ = precision_recall_curve(yy, pp)
    axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name.split(' ')[0]} AUC {auc(fpr, tpr):.3f}")
    axes[1].plot(rec, prec, color=color, lw=2, label=name.split(" ")[0])
axes[0].plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
axes[0].set(title="ROC: fall vs. everything else", xlabel="false positive rate", ylabel="true positive rate"); axes[0].legend()
axes[1].set(title="Precision–recall: fall class", xlabel="recall", ylabel="precision"); axes[1].legend()

thr_grid = np.linspace(0.01, 0.99, 99)
sens = [(p7[is_fall, FALL] >= t).mean() for t in thr_grid]
fa_rate = [(p7[~is_fall, FALL] >= t).mean() for t in thr_grid]
ax = axes[2]
ax.plot(thr_grid, sens, color=ORANGE, lw=2, label="falls detected (sensitivity)")
ax.plot(thr_grid, fa_rate, color=BLUE, lw=2, label="non-fall windows alerted")
ax.set(title="Threshold trade-off (M7)", xlabel="P(fall) threshold", ylabel="share of windows", ylim=(0, 1.02))
thr_used = res_df[res_df.model == M7].threshold.mean()
ax.axvline(thr_used, color=INK2, ls="--", lw=1); ax.text(thr_used + 0.01, 0.5, f"calibrated ≈ {thr_used:.2f}", color=INK2)
ax.legend(loc="upper right")
plt.tight_layout(); save("13_roc_pr_threshold"); plt.show()
""")

md("### Success criteria (proposal Table 6)")

code(r"""
m7 = summary.loc[M7]
lp4 = lora_param_count(lora_model(4))
targets = pd.DataFrame([
    ("Sensitivity (recall) for falls", "≥ 95%", f"{m7.fall_sensitivity:.1%}", m7.fall_sensitivity >= 0.95),
    ("Specificity on floor-level ADLs", "≥ 98%", f"{m7.specificity_floor_level:.1%}", m7.specificity_floor_level >= 0.98),
    ("F1-score (fall class)", "≥ 0.90", f"{m7.fall_f1:.3f}", m7.fall_f1 >= 0.90),
    ("Adapter size", "< 2% of encoder; < 100 KB", f"{100*lp4/enc_params:.2f}%; {lp4*4/1024:.1f} KB fp32", lp4/enc_params < 0.02 and lp4*4/1024 < 100),
    ("F1 gain over generic baselines (H1)", "≥ +15 pp over M0 and M5",
     f"+{100*(m7.fall_f1-summary.loc['M0 Threshold (RuView-style)'].fall_f1):.1f} pp vs M0, "
     f"+{100*(m7.fall_f1-summary.loc['M5 Encoder + linear probe'].fall_f1):.1f} pp vs M5",
     (m7.fall_f1 - max(summary.loc['M0 Threshold (RuView-style)'].fall_f1, summary.loc['M5 Encoder + linear probe'].fall_f1)) >= 0.15),
    ("Within 2 pp of full fine-tune (H1)", "|M7 − M6| ≤ 2 pp", f"{100*(m7.fall_f1-summary.loc['M6 Encoder full fine-tune'].fall_f1):+.1f} pp",
     abs(m7.fall_f1 - summary.loc['M6 Encoder full fine-tune'].fall_f1) <= 0.02),
    ("False alarms per household per week", "≤ 1", "pending pilot", None),
    ("On-device latency (ESP32-S3)", "< 200 ms", "pending hardware (CPU latency in §10)", None),
    ("Fall-to-SMS time", "< 60 s median", "pending pilot (MVP pipeline in API)", None),
], columns=["metric", "target", "result (synthetic data)", "met"])
targets
""")

md("### McNemar's test (H0: no difference between M7 and the baselines, α = 0.05)")

code(r"""
def correct(name):
    return np.concatenate([test_store[(name, f)]["fall_pred"] == (test_store[(name, f)]["y"] == FALL) for f in ("A->B", "B->A")])

rows = []
for other in ["M0 Threshold (RuView-style)", "M5 Encoder + linear probe", "M6 Encoder full fine-tune"]:
    t = mcnemar(correct(M7), correct(other))
    rows.append({"comparison": f"M7 vs {other}", "only M7 correct": t["a_only_correct"],
                 "only other correct": t["b_only_correct"], "p-value": t["p_value"],
                 "significant (α=0.05)": t["p_value"] < 0.05})
pd.DataFrame(rows)
""")

md(r"""
## 9. Error analysis (RQ3): which floor-level activities look like falls?

False-alarm rate per protocol activity (share of test windows that trigger a fall alert), pooled over both folds.
""")

code(r"""
acts = [a for a in KUBAHO_ACTIVITIES if not a.startswith("fall")]
cmp_models = [("M0 Threshold (RuView-style)", GRAY), ("M5 Encoder + linear probe", BLUE), (M7, ORANGE)]
rates = {}
for name, _ in cmp_models:
    idx = np.concatenate([test_store[(name, f)]["idx"] for f in ("A->B", "B->A")])
    pred = np.concatenate([test_store[(name, f)]["fall_pred"] for f in ("A->B", "B->A")])
    a = meta.activity.values[idx]
    rates[name] = [pred[a == act].mean() for act in acts]
fig, ax = plt.subplots(figsize=(12, 4))
w = 0.26
for k, (name, color) in enumerate(cmp_models):
    ax.bar(np.arange(len(acts)) + (k - 1) * w, rates[name], width=w - 0.03, color=color, label=name)
ax.set_xticks(range(len(acts)), [a.replace("_", " ") for a in acts])
ax.set(title="False fall alerts per activity (lower is better)", ylabel="share of windows alerted")
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
plt.tight_layout(); save("14_false_alarms_by_activity"); plt.show()
pd.DataFrame(rates, index=acts).round(3)
""")

code(r"""
from sklearn.manifold import TSNE

te = split("A")[2]
lora_fold = test_store[(M7, "A->B")]["model"]
Z_generic = embed(pretrained_classifier(), X_lab[te, 1])
Z_adapted = embed(lora_fold.base_model.model, X_lab[te, 1])   # LoRA layers active inside the PEFT wrapper
g = meta.class_name.values[te]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, Z, title in [(axes[0], Z_generic, "Generic pretrained encoder"), (axes[1], Z_adapted, "After LoRA adaptation")]:
    P = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(Z)
    for grp, color in GROUP_COLORS.items():
        m = np.array([group_of(c) == grp for c in g])
        ax.scatter(P[m, 0], P[m, 1], s=16, color=color, alpha=0.7, label=grp, edgecolors="#fcfcfb", linewidths=0.4)
    ax.set(title=f"t-SNE of 128-d embeddings: {title}", xticks=[], yticks=[])
axes[0].legend(loc="best")
plt.tight_layout(); save("15_tsne_embeddings"); plt.show()
""")

md(r"""
## 10. Final adapter, INT8 export and deployment artefact

The deployed adapter is trained on **both layouts** (subjects 1–17). Subjects 18–20 are kept for threshold
calibration. The validation metrics below are therefore *seen-room, unseen-people* numbers and are higher
than the unseen-room results in §7. They are reported only to compare FP32 with INT8. We then merge LoRA into the encoder ($W' = W + \frac{\alpha}{r}BA$), export to ONNX, quantise
weights to INT8 and check that INT8 agrees with FP32. Latency is measured on this laptop's CPU (1 thread).
The ESP32-S3 latency must be measured on hardware (ESP-DL / TFLite-Micro conversion is the next step).
""")

code(r"""
from kubaho.export import export_onnx, quantize_int8, onnx_predict_proba, benchmark_ms

f_tr = np.where(meta.subject <= 17)[0]; f_va = np.where(meta.subject >= 18)[0]
final = lora_model(4)
h = fit(final, X_lab[f_tr], y_lab[f_tr], X_lab[f_va], y_lab[f_va], len(CLASSES),
        TrainConfig(epochs=25, lr=2e-3, seed=SEED), run_name="M7_final_adapter")
p_va = predict_proba(final, X_lab[f_va])
final_thr = calibrate_threshold(y_lab[f_va], p_va[:, FALL])
print(f"final adapter: {h['epochs_run']} epochs, val macro-F1 {h['best_val_macro_f1']:.3f}, threshold {final_thr:.3f}")

final.save_pretrained(MODELS / "kubaho_lora_adapter")           # adapter only (OTA payload)
merged = final.merge_and_unload()
fp32_path = export_onnx(merged, MODELS / "kubaho_lora_fp32.onnx")
int8_path = quantize_int8(fp32_path, MODELS / "kubaho_lora_int8.onnx")

Xv = X_lab[f_va, 1]
p32, p8 = onnx_predict_proba(fp32_path, Xv), onnx_predict_proba(int8_path, Xv)
adapter_file = next((MODELS / "kubaho_lora_adapter").glob("adapter_model.*"))
export_rows = []
for label, path, p in [("FP32", fp32_path, p32), ("INT8", int8_path, p8)]:
    m = evaluate(y_lab[f_va], p, final_thr)
    export_rows.append({"model": label, "file size (KB)": path.stat().st_size / 1024, "CPU latency (ms, median)": benchmark_ms(path),
                        "macro-F1": m["macro_f1"], "fall F1": m["fall_f1"], "fall sensitivity": m["fall_sensitivity"]})
export_df = pd.DataFrame(export_rows)
print(f"INT8 vs FP32 argmax agreement: {(p32.argmax(1) == p8.argmax(1)).mean():.1%}")
print(f"adapter file {adapter_file.name}: {adapter_file.stat().st_size/1024:.1f} KB (includes the 8-class head)")
export_df.round(3)
""")

code(r"""
card = {
    "version": "kubaho-lora-r4-v0.1",
    "created": time.strftime("%Y-%m-%d"),
    "classes": CLASSES,
    "threshold": round(float(final_thr), 4),
    "input": {"raw_fs": FS, "model_fs": TARGET_FS, "window_s": 2.0, "subcarriers": 52,
              "preprocessing": "interpolate lost packets, Hampel(5, 3σ), Butterworth band-pass 0.2-20 Hz, decimate to 50 Hz"},
    "architecture": {"encoder": "Conv1d stem + 4 transformer blocks (d=128, 4 heads), mean-pooled 128-d embedding",
                     "adapter": "LoRA r=4, alpha=8 on attention q and v", "head": "Linear 128->8",
                     "encoder_params": int(enc_params), "lora_params": int(lp4)},
    "evaluation": {
        "protocol": "leave-one-layout-out with unseen subjects, mean of 2 folds",
        "data": "SYNTHETIC Kubaho-Lab stand-in - preliminary, not validated on real CSI",
        "M7": {k: round(float(v), 4) for k, v in summary.loc[M7, cols].items()},
        "M0_threshold": {k: round(float(v), 4) for k, v in summary.loc["M0 Threshold (RuView-style)", cols].dropna().items()},
    },
    "export": {r["model"]: {"size_kb": round(r["file size (KB)"], 1), "cpu_latency_ms": round(r["CPU latency (ms, median)"], 3)}
               for r in export_rows},
    "files": {"onnx_int8": int8_path.name, "onnx_fp32": fp32_path.name, "adapter": "kubaho_lora_adapter/"},
}
(MODELS / "model_card.json").write_text(json.dumps(card, indent=2))
print(json.dumps(card["evaluation"]["M7"], indent=2))
""")

md(r"""
## 11. Summary and next steps

**What this notebook demonstrates (on synthetic data):**
- A complete, reproducible pipeline from raw CSI to a deployable INT8 model: quality phase, features,
  eight compared models, calibrated alert threshold, statistical test, ONNX export and model card.
- The generic threshold detector (M0) fires on many floor-level activities (RQ1, RQ3). This is the false-alarm
  problem the proposal predicts.
- A LoRA adapter that trains under 2% of the encoder's parameters (RQ2) is compared with full fine-tuning
  and the generic baselines in the results table and the success-criteria table above.

**Next steps:**
1. Collect the Kubaho-Lab dataset (20 volunteers, 2 layouts) with the labelling app and swap it in for the
   synthetic generator in `kubaho.data`.
2. Replace the stand-in encoder with the RuView pretrained encoder weights and repeat the benchmark.
3. Add leave-one-subject-out cross-validation on the real data.
4. Convert the merged INT8 model for the ESP32-S3 (ESP-DL / TFLite-Micro) and measure on-device latency
   against the 200 ms budget.
5. Run the 6-week, 8-household pilot. Measure false alarms per household per week and fall-to-SMS time
   (already logged by the API).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (kubaho)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).with_name("kubaho_model.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
