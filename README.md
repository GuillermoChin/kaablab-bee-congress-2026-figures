# KAABLAB — Bee Congress 2025: Figure Reproduction Package

**Code Author:** Guillermo Adrián Chin Canché with ClaudeIA.  
**Institution:** Instituto Tecnológico Superior de Calkiní (ITESCAM) — Laboratorio de Ambientes Inteligentes  
**Congress:** “II Congreso sobre Abejas, Biodiversidad y Soberanía Alimentaria” 
**Year:** 2026
**Data:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19685877.svg)]((https://doi.org/10.5281/zenodo.19685877))

---

## Description

This repository contains the figure-generation script, quantitative results,
and supporting data associated with the article:

> **KAABLAB: SISTEMA AIOT PARA EL MONITOREO Y DE-TECCIÓN TEMPRANA DE ANOMALÍAS EN COLONIAS DE ABEJAS [APIS MELLIFERA]**  
> Chin Canché, G.A. et al. (2026). *Ciencia y Tecnología ITESCAM Calkiní*.

The complete source code for the `bee_audio_analysis` and `varroa_vision`
pipelines is available from the corresponding author upon reasonable request.

---

## Repository Structure

    data/
    ├── results/                # Metrics used to generate figures
    ├── source_docs/            # Source documents with embedded plots
    └── supporting_results/     # Additional results supporting the article text
    generate_figures.py         # Reproduces all article figures
    requirements.txt            # Python dependencies

---

## How to Reproduce the Figures

### 1. Clone this repository

    git clone https://github.com/guillermochin/kaablab-bee-congress-2025-figures.git
    cd kaablab-bee-congress-2025-figures

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Run the script

    python generate_figures.py

Output figures (300 dpi) will be saved to the `figures_output/` directory.

---

## Data Description

### `data/results/`
Quantitative outputs from both analytical pipelines used directly to generate figures:

| File | Description |
|---|---|
| `comparison.json` | Classical ML metrics — Varroa vision L1 |
| `metrics_efficientnet_b0.json` | EfficientNetB0 metrics — Transfer learning L2 |
| `metrics_mobilenet_v2.json` | MobileNetV2 metrics — Transfer learning L2 |
| `metrics_resnet50.json` | ResNet50 metrics — Transfer learning L2 |
| `metrics_varroa_cnn.json` | Custom CNN metrics — L3 |
| `metrics_yolov8_classification.json` | YOLOv8s metrics — L4 |
| `results.csv` | YOLOv8s training log (loss, accuracy per epoch) |

### `data/supporting_results/`
Additional results that support claims in the article text
but are not directly used for figure generation.

### `data/source_docs/`
Source documents containing intermediate plots extracted
programmatically by `generate_figures.py`.

---

## External Datasets

This study used the following publicly available datasets:

- **BeeAudio Dataset** — European honey bee hive audio recordings [public domain] https://www.kaggle.com/api/v1/datasets/download/annajyang/beehive-sounds
- **VarroaDataset** — Bee hive photos. Zenodo. https://doi.org/10.5281/zenodo.4085044 (CC BY 4.0)

---

## License

The code in this repository is licensed under the **MIT License**.  
Result data is shared under **CC BY 4.0**.

---

## Contact

For access to the complete pipeline source code or for questions regarding
this work, please contact the corresponding author:

**Guillermo Adrián Chin Canché**  
Instituto Tecnológico Superior de Calkiní — ITESCAM  
Laboratorio de Ambientes Inteligentes  
📧 gcchin@itescam.edu.mx
