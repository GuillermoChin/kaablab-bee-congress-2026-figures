# =============================================================================
# generate_figures.py — KAABLAB Article Figures (v2)
# =============================================================================

import zipfile, json, io, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from PIL import Image
from pathlib import Path
import os

#Línea válida solo para mi computador
#os.chdir(r"G:\My Drive\Laboratorio de Ambientes Inteligentes\Proyecto Abejas IA\Codigos y Datos\Gráficas del Artículo")
# Reemplazar por — rutas relativas al repositorio para fuera del computador
DATA_DIR    = Path("data/results")
SOURCE_DOCS = Path("data/source_docs")

# -----------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------
DPI        = 300
OUT_DIR    = Path("figures_output")
COLOR_MAIN = "#2E75B6"
COLOR_SEC  = "#ED7D31"
COLOR_3    = "#70AD47"
COLOR_GRID = "#E8E8E8"
FONT_TITLE = 11
FONT_LABEL = 9
FONT_TICK  = 8
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------------------------

def extract_images_from_docx(path):
    imgs = {}
    with zipfile.ZipFile(path, "r") as z:
        for name in z.namelist():
            if name.startswith("word/media/") and \
               name.lower().endswith((".png", ".jpg", ".jpeg")):
                with z.open(name) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGBA")
                    imgs[Path(name).name] = img
    return imgs

def ax_img(ax, img):
    """Muestra PIL image en eje sin bordes."""
    ax.imshow(img)
    ax.axis("off")

def panel_label(ax, letter, size=13):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="top", ha="left")

def bar_labels(ax, bars, vals, fmt="{:.3f}"):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                fmt.format(v), ha="center", va="bottom",
                fontsize=7, fontweight="bold")

def style_ax(ax):
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_TICK)

def caption(fig, text, y=0.01, chars=100):
    """Pie de figura centrado, con saltos de línea automáticos."""
    wrapped = textwrap.fill(text, width=chars)
    fig.text(0.5, y, wrapped, ha="center", va="bottom",
             fontsize=8.5, style="italic", linespacing=1.4)

def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓  {path}")

def equalize_heights(img_a, img_b):
    """Rellena con blanco la imagen más baja para igualar alturas."""
    h_a, h_b = img_a.height, img_b.height
    target   = max(h_a, h_b)
    def pad(img, target_h):
        if img.height == target_h:
            return img
        canvas = Image.new("RGBA", (img.width, target_h), (255, 255, 255, 255))
        canvas.paste(img, (0, (target_h - img.height) // 2))
        return canvas
    return pad(img_a, target), pad(img_b, target)

def plot_conf_matrix_normalized(ax, cm, labels, cmap="Blues", title=""):
    """
    Dibuja una matriz de confusión normalizada por fila (0-1).
    cm: np.array 2D con conteos absolutos.
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, linewidths=0.5, linecolor="#cccccc",
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Etiqueta predicha", fontsize=FONT_LABEL)
    ax.set_ylabel("Etiqueta real",     fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)

# -----------------------------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------------------------
print("Cargando archivos...")

imgs_a = extract_images_from_docx(SOURCE_DOCS / "Primeros resultados acustica.docx")
imgs_v = extract_images_from_docx(SOURCE_DOCS / "Primeros resultados fotos.docx")

def load(fname):
    with open(DATA_DIR / fname) as f:
        return json.load(f)

m_l1   = load("comparison.json")
m_eff  = load("metrics_efficientnet_b0.json")
m_mob  = load("metrics_mobilenet_v2.json")
m_res  = load("metrics_resnet50.json")
m_cnn  = load("metrics_varroa_cnn.json")
m_yolo = load("metrics_yolov8_classification.json")

df_raw = pd.read_csv(DATA_DIR / "results.csv")
df_raw.columns = df_raw.columns.str.strip()
starts  = df_raw.index[df_raw["epoch"] == 1].tolist()
df_yolo = df_raw.iloc[starts[-1]:].reset_index(drop=True)

print("  ✓  Todos los archivos cargados\n")

# =============================================================================
# FIGURA 1 — Bioacústica L1: métricas comparativas + features importantes
#   (A) Comparación de métricas  (B) Features más importantes
# =============================================================================
print("Generando Fig. 1...")

img_1a, img_1b = equalize_heights(imgs_a["image1.png"], imgs_a["image2.png"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.subplots_adjust(wspace=0.04, bottom=0.18)

ax_img(axes[0], img_1a)
ax_img(axes[1], img_1b)
panel_label(axes[0], "A")
panel_label(axes[1], "B")

caption(fig,
    "Figura 1. Resultados del módulo bioacústico — Línea 1 (clasificación multiclase del estado de salud). "
    "(A) Comparación de métricas de desempeño por modelo. "
    "(B) Importancia relativa de las 15 características más relevantes según XGBoost.",
    y=0.02, chars=115)
save_fig(fig, "Fig1_bioacustica_L1_metricas_features.png")

# =============================================================================
# FIGURA 2 — Bioacústica L1: matrices de confusión normalizadas
#   Se reconstruyen programáticamente desde los datos de los reportes
#   para permitir normalización por fila.
# =============================================================================
print("Generando Fig. 2...")

# Matrices de confusión L1 bioacústica (extraídas de los reportes de clasificación)
# Orden de clases: 0, 1, 2, 3, 4, 5
cms_bio = {
    "Random Forest":        np.array([[13,4,0,0,5,0],[0,14,2,0,0,0],[0,2,15,1,0,0],[0,0,1,72,0,0],[0,0,0,0,32,2],[1,0,0,0,0,91]]),
    "SVM":                  np.array([[4,4,0,0,6,8],[0,15,1,0,0,0],[0,2,12,4,0,0],[0,0,1,72,0,0],[0,0,0,0,30,4],[1,0,0,0,0,91]]),
    "Logistic Regression":  np.array([[11,3,0,0,6,2],[1,14,0,0,1,0],[0,2,14,2,0,0],[0,0,0,73,0,0],[2,0,0,0,26,6],[3,0,0,0,4,85]]),
    "XGBoost":              np.array([[14,3,0,0,5,0],[0,14,2,0,0,0],[0,1,16,1,0,0],[0,1,0,72,0,0],[1,0,0,0,30,3],[3,0,0,0,2,87]]),
    "KNN":                  np.array([[8,2,0,0,4,8],[1,13,1,1,0,0],[0,2,10,5,1,0],[0,2,4,67,0,0],[2,0,0,1,24,7],[1,0,1,1,3,87]]),
    "Gradient Boosting":    np.array([[15,3,0,0,4,0],[1,14,0,0,0,0],[0,2,15,1,0,0],[0,1,0,72,0,0],[1,0,0,0,31,2],[1,0,0,0,0,91]]),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.subplots_adjust(hspace=0.35, wspace=0.35, bottom=0.12)

labels_bio = ["0","1","2","3","4","5"]
for ax, (nombre, cm) in zip(axes.flatten(), cms_bio.items()):
    plot_conf_matrix_normalized(ax, cm, labels_bio, title=nombre)
    ax.set_title(nombre, fontsize=FONT_LABEL, pad=4)

caption(fig,
    "Figura 2. Matrices de confusión normalizadas (proporción por fila) para los seis clasificadores "
    "evaluados en la Línea 1 del módulo bioacústico "
    "(clasificación multiclase del estado de salud de la colmena, clases 0–5).",
    y=0.01, chars=115)
save_fig(fig, "Fig2_bioacustica_L1_matrices_confusion.png")

# =============================================================================
# FIGURA 3 — Bioacústica: curvas ROC L2 + correlaciones Spearman L3
#   (A) Curvas ROC — detección de reina
#   (B) Correlaciones de Spearman significativas
# =============================================================================
print("Generando Fig. 3...")

img_3a, img_3b = equalize_heights(imgs_a["image4.png"], imgs_a["image7.png"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
fig.subplots_adjust(wspace=0.04, bottom=0.18)

ax_img(axes[0], img_3a)
ax_img(axes[1], img_3b)
panel_label(axes[0], "A")
panel_label(axes[1], "B")

caption(fig,
    "Figura 3. (A) Curvas ROC de los seis modelos para la detección binaria de presencia de reina "
    "(Línea 2, módulo bioacústico). "
    "(B) Correlaciones de Spearman significativas (p < 0.05) entre variables ambientales "
    "e indicadores de estado de colmena (Línea 3, módulo bioacústico).",
    y=0.02, chars=115)
save_fig(fig, "Fig3_bioacustica_L2_ROC_L3_correlaciones.png")

# =============================================================================
# FIGURA 4 — Varroa: comparación global L1–L4
#   (A) F1 ponderado   (B) AUC-ROC
# =============================================================================
print("Generando Fig. 4...")

lineas   = ["L1\nML Clásico\n(SVM)", "L2\nTransfer\n(MobileNetV2)",
            "L3\nCNN\nLigera",       "L4\nYOLOv8s\nClasif."]
f1_vals  = [m_l1["svm"]["f1"],       m_mob["f1"],
            m_cnn["f1"],              m_yolo["f1"]]
auc_vals = [m_l1["svm"]["auc_roc"],  m_mob["auc_roc"],
            m_cnn["auc_roc"],         m_yolo["auc_roc"]]

def bar_color(v):
    if v < 0.60: return COLOR_MAIN
    if v < 0.80: return COLOR_SEC
    return COLOR_3

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.35, bottom=0.22)

for ax, vals, ylabel, title in zip(
    axes,
    [f1_vals,  auc_vals],
    ["F1 ponderado", "AUC-ROC"],
    ["F1 ponderado por línea de análisis", "AUC-ROC por línea de análisis"]
):
    colors = [bar_color(v) for v in vals]
    bars = ax.bar(lineas, vals, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title,   fontsize=FONT_TITLE, pad=8)
    ax.axhline(0.80, color="#C00000", linestyle="--",
               linewidth=1.0, alpha=0.75, label="Umbral 0.80", zorder=4)
    ax.legend(fontsize=7, loc="lower right")
    bar_labels(ax, bars, vals)
    style_ax(ax)

panel_label(axes[0], "A")
panel_label(axes[1], "B")

from matplotlib.patches import Patch
leg_els = [Patch(facecolor=COLOR_MAIN, label="< 0.60"),
           Patch(facecolor=COLOR_SEC,  label="0.60 – 0.79"),
           Patch(facecolor=COLOR_3,    label="≥ 0.80")]
fig.legend(handles=leg_els, title="Nivel de desempeño", fontsize=7.5,
           title_fontsize=8, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.01), frameon=False)

caption(fig,
    "Figura 4. Comparación del desempeño entre las cuatro líneas de análisis del módulo de visión computacional. "
    "(A) F1 ponderado. (B) AUC-ROC. "
    "La línea discontinua roja indica el umbral de referencia de 0.80.",
    y=0.02, chars=115)
save_fig(fig, "Fig4_varroa_comparacion_L1_L4.png")

# =============================================================================
# FIGURA 5 — Varroa: entrenamiento YOLOv8 + comparación L2 + confusión L2
#   (A) Curvas entrenamiento YOLOv8s
#   (B) Comparación F1 y AUC arquitecturas L2
#   (C) Matriz de confusión ResNet50 normalizada
# =============================================================================
print("Generando Fig. 5...")

fig = plt.figure(figsize=(17, 6))
gs  = gridspec.GridSpec(1, 3, figure=fig,
                        wspace=0.45,           # más espacio entre paneles
                        width_ratios=[1.3, 1.2, 0.9],
                        left=0.06, right=0.97,
                        bottom=0.20, top=0.92)

# ── Panel A: curvas entrenamiento YOLO ──
ax_a      = fig.add_subplot(gs[0])
ax_a_twin = ax_a.twinx()

epochs     = df_yolo["epoch"].values
train_loss = df_yolo["train/loss"].values
val_acc    = df_yolo["metrics/accuracy_top1"].values

ax_a.plot(epochs, train_loss, color=COLOR_MAIN, linewidth=1.8,
          label="Pérdida (entrenamiento)", zorder=3)
ax_a_twin.plot(epochs, val_acc, color=COLOR_SEC, linewidth=1.8,
               linestyle="--", label="Accuracy (validación)", zorder=3)

ax_a.set_xlabel("Época", fontsize=FONT_LABEL)
ax_a.set_ylabel("Pérdida de entrenamiento", color=COLOR_MAIN, fontsize=FONT_LABEL)
ax_a_twin.set_ylabel("Accuracy de validación", color=COLOR_SEC, fontsize=FONT_LABEL)
ax_a.tick_params(axis="y", labelcolor=COLOR_MAIN, labelsize=FONT_TICK)
ax_a_twin.tick_params(axis="y", labelcolor=COLOR_SEC, labelsize=FONT_TICK)
ax_a.tick_params(axis="x", labelsize=FONT_TICK)
ax_a.set_title("YOLOv8s — Entrenamiento (L4)", fontsize=FONT_TITLE, pad=6)
style_ax(ax_a)
ax_a.spines["top"].set_visible(False)
ax_a_twin.spines["top"].set_visible(False)

lines_  = ax_a.get_lines() + ax_a_twin.get_lines()
labels_ = [l.get_label() for l in lines_]
ax_a.legend(lines_, labels_, fontsize=7, loc="center right")
panel_label(ax_a, "A")

# ── Panel B: comparación arquitecturas L2 ──
ax_b = fig.add_subplot(gs[1])

arch_names = ["EfficientNetB0", "MobileNetV2", "ResNet50"]
f1_l2  = [m_eff["f1"],      m_mob["f1"],      m_res["f1"]]
auc_l2 = [m_eff["auc_roc"], m_mob["auc_roc"], m_res["auc_roc"]]

x  = np.arange(len(arch_names))
w  = 0.35
b1 = ax_b.bar(x - w/2, f1_l2,  w, label="F1",
              color=COLOR_MAIN, alpha=0.88, edgecolor="white", zorder=3)
b2 = ax_b.bar(x + w/2, auc_l2, w, label="AUC-ROC",
              color=COLOR_SEC,  alpha=0.88, edgecolor="white", zorder=3)
bar_labels(ax_b, b1, f1_l2)
bar_labels(ax_b, b2, auc_l2)
ax_b.set_xticks(x)
ax_b.set_xticklabels(arch_names, fontsize=7, rotation=15, ha="right")
ax_b.set_ylim(0, 1.05)
ax_b.set_ylabel("Valor de métrica", fontsize=FONT_LABEL)
ax_b.set_title("Arquitecturas L2 — Transfer Learning", fontsize=FONT_TITLE, pad=6)
style_ax(ax_b)
ax_b.legend(fontsize=8, loc="upper left")
panel_label(ax_b, "B")

# ── Panel C: matriz de confusión ResNet50 normalizada ──
ax_c = fig.add_subplot(gs[2])

# Valores reales de la matriz ResNet50
cm_resnet = np.array([[2263, 203],
                       [541,  401]])
plot_conf_matrix_normalized(ax_c, cm_resnet,
                             labels=["Sana", "Varroa"],
                             cmap="Blues")
ax_c.set_title("Conf. Matrix norm.\nResNet50 (L2)", fontsize=FONT_TITLE, pad=6)
panel_label(ax_c, "C")

caption(fig,
    "Figura 5. (A) Curvas de pérdida y accuracy durante el entrenamiento de YOLOv8s (Línea 4). "
    "(B) Comparación de F1 y AUC-ROC entre las tres arquitecturas de transferencia de aprendizaje (Línea 2). "
    "(C) Matriz de confusión normalizada de ResNet50 (proporción por fila).",
    y=0.01, chars=115)
save_fig(fig, "Fig5_varroa_YOLO_L2_comparacion.png")

# =============================================================================
# FIGURA 6 — Varroa L1: panel 4 curvas ROC
#   (A) SVM  (B) Random Forest  (C) XGBoost  (D) LDA
# =============================================================================
print("Generando Fig. 6...")

roc_map = [
    ("image1.png",  "SVM"),
    ("image2.png",  "Random Forest"),
    ("image15.png", "XGBoost"),
    ("image3.png",  "LDA"),
]

# Igualar tamaños en pares para el panel 2×2
img_A = imgs_v["image1.png"]
img_B = imgs_v["image2.png"]
img_C = imgs_v["image15.png"]
img_D = imgs_v["image3.png"]
img_A, img_B = equalize_heights(img_A, img_B)
img_C, img_D = equalize_heights(img_C, img_D)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.06, wspace=0.04, bottom=0.14)

for ax, img, letter in zip(axes.flatten(),
                            [img_A, img_B, img_C, img_D],
                            ["A","B","C","D"]):
    ax_img(ax, img)
    panel_label(ax, letter)

caption(fig,
    "Figura 6. Curvas ROC de los cuatro clasificadores evaluados en la Línea 1 del módulo de visión "
    "computacional (aprendizaje automático clásico con descriptores HOG+LBP+HSV+PCA). "
    "(A) SVM. (B) Random Forest. (C) XGBoost. (D) LDA.",
    y=0.02, chars=115)
save_fig(fig, "Fig6_varroa_L1_ROC_panel.png")

# =============================================================================
print(f"\n✅ Listo. Figuras en: {OUT_DIR.resolve()}")