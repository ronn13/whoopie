> **Course:** 18-786 Intro to Deep Learning · Carnegie Mellon University Africa  
> **Team:** Whoopie Wanjiru (MSEAI27) · Theophilus Owiti (MSEAI27) · Ronnie Delyon (MSEAI26)

---

## 🧩 Project Overview

Civil aviation authorities worldwide struggle to manually classify safety reports under the ICAO **ADREP** (Accident/Incident Reporting) taxonomy, leading to inconsistent labeling, processing delays, and missed trends.

A prior system — the **CMU-Africa SDCPS pipeline** — automated this using a hybrid transformer + multi-LLM approach, achieving **92.96% accuracy** on 3,600 reports. However, **26.94% of reports** were conservatively labelled **"OTHER"** — cases where flat text classification failed to resolve implicit, multi-factor causal narratives.

This project focuses specifically on classifying these "OTHER" hard cases. While we initially theorized a graph-based reasoning approach, we ultimately resolved these complex incident reports using a **Dual-Layer SafeAeroBERT NLP Pipeline**: marrying a dedicated NER extractor (Layer 1) with an end-to-end text classifier (Layer 3) mapped precisely to the official ICAO ECCAIRS Taxonomy.

---

## 🏗️ System Architecture (2-Layer SafeAeroBERT Pipeline)

```
Raw Narrative (classified "OTHER" by SDCPS)
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 1 — Event Extraction (NER)     │  ← complete
│  Extract: ACTOR, SYSTEM, PHASE,       │
│           TRIGGER, OUTCOME            │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 3 — ADREP Classification       │  ← complete
│  SafeAeroBERT text-classifier         │
│  Predicts ICAO Taxonomy (CFIT, MAC..) │
└───────────────────────────────────────┘
```

---

## 🔭 Future Work

- **Continuous Learning Loop:** Develop a feedback mechanism in the Web Demo UI allowing domain experts to submit corrections, driving automated periodic re-training of the Layer 3 RoBERTa classifier.
- **Enhanced Entity Graphing:** Build interactive, visual representations of the Layer 1 NER extracted tuples to quickly isolate trigger conditions in the UI.
- **Broader Taxonomy Support:** Expand the models to perform multi-label classification on secondary ICAO ECCAIRS attributes beyond the primary Occurrence Category.

---

## ✅ Current Progress: Layer 1 — Event Extraction

### What We Extract

For each aviation safety narrative, we extract a structured event tuple:

| Field       | Source                               | Method                         |
| ----------- | ------------------------------------ | ------------------------------ |
| **ACTOR**   | Narrative text                       | Fine-tuned BERT NER            |
| **SYSTEM**  | Narrative text                       | Fine-tuned BERT NER            |
| **TRIGGER** | Narrative text                       | Fine-tuned BERT NER            |
| **PHASE**   | Existing `phase` column              | Direct mapping (~98% coverage) |
| **OUTCOME** | Existing `events`/`events_6` columns | Direct mapping (100% coverage) |

### Example Output

```
Report #73
  Synopsis: During approach, crew received TCAS RA advisory...

  ACTOR:   ['Captain', 'First Officer']
  SYSTEM:  ['TCAS', 'autopilot']
  PHASE:   Approach
  TRIGGER: ['loss of situational awareness', 'communication breakdown']
  OUTCOME: ['AIRPROX', 'Near Miss']

  Causal Edges:
    Captain ──[caused]──▶ loss of situational awareness
    loss of situational awareness ──[led_to]──▶ AIRPROX
    TCAS ──[involved_in]──▶ loss of situational awareness
```

---

## 🧠 Approach: Silver-Label Fine-tuning

Since no gold-annotated NER labels exist for our dataset, we use a **silver-labeling** strategy:

1. **Build rule-based extractors** — regex + aviation domain dictionaries across 3 entity types
2. **Align spans to BERT tokens** — character-level span → BIO (Begin-Inside-Outside) tags
3. **Fine-tune BERT** on silver labels — model learns contextual patterns beyond the rules
4. **Run inference** on all 173 reports → export structured event tuples

#### Entity Pattern Categories

| Entity      | Pattern Types                                                                                                                                                         |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ACTOR**   | Flight crew roles (Captain, FO, PIC, SIC), ATC roles, pronominal pilot references                                                                                     |
| **SYSTEM**  | Navigation (GPS, ILS, RNAV), safety systems (TCAS, GPWS), autopilot, comms, aircraft components                                                                       |
| **TRIGGER** | Causal language (due to, failed to, resulted from), human factors (fatigue, distraction), environmental (turbulence, icing), mechanical (engine failure, malfunction) |

---

## 📊 Results

### Model Comparison: BERT-base vs SafeAeroBERT

We compared standard `bert-base-uncased` against `NASA-AIML/MIKA_SafeAeroBERT`, a domain-specific BERT pre-trained on aviation safety corpora.

| Metric          | BERT-base       | SafeAeroBERT         | Winner          |
| --------------- | --------------- | -------------------- | --------------- |
| **Test F1**     | 0.741           | **0.830**            | ✅ SafeAeroBERT |
| **Precision**   | 0.655           | **0.786**            | ✅ SafeAeroBERT |
| **Recall**      | 0.857           | **0.881**            | ✅ SafeAeroBERT |
| **Test Loss**   | 0.690           | **0.495**            | ✅ SafeAeroBERT |
| **Best Val F1** | 0.731 (epoch 9) | **0.758** (epoch 10) | ✅ SafeAeroBERT |
| Parameters      | 108.9M          | 108.9M               | —               |

**SafeAeroBERT wins** — domain pre-training on aviation text gives it a meaningful edge on this task (~+9 F1 points).

### Manual Validation (3 reports, human-annotated)

| Entity      | Precision | Recall   | F1       |
| ----------- | --------- | -------- | -------- |
| ACTOR       | 1.00      | 0.50     | 0.67     |
| SYSTEM      | 1.00      | 1.00     | **1.00** |
| TRIGGER     | 1.00      | 1.00     | **1.00** |
| **Overall** | **1.00**  | **0.67** | **0.80** |

> Quality rating: **EXCELLENT** — zero false positives; missed some ACTOR mentions (known limitation of implicit pronoun references).

### Training Configuration

```
Model:         bert-base-uncased / NASA-AIML/MIKA_SafeAeroBERT
Max Seq Len:   256 tokens (with overlapping chunking for longer texts)
Batch Size:    8
Learning Rate: 3e-5
Epochs:        10 (with early stopping, patience=3)
Warmup:        10% of total steps
Loss:          Weighted CrossEntropy (to handle O-class imbalance)
Optimizer:     AdamW (weight decay=0.01)
Data Split:    70% train / 15% val / 15% test
Dataset Size:  173 aviation safety reports
```

---

## 📁 Repository Structure

```
.
├── app/                            # Web Demo Application
│   ├── static/                     # UI Assets (app.js, index.html, index.css)
│   ├── inference.py                # Fast inference logic for NER & Classification
│   └── server.py                   # Flask API Server
│
├── src/                            # Training & Source code
│   └── layer3_reasoning/
│       └── train.py                # SafeAeroBERT ADREP classifier training script
│
├── data/                           # Datasets & Scripts
│   ├── raw/data_aviation.csv       # Scraped NASA ASRS unlabelled data 
│   ├── reference/                  # Label datasets & Taxonomy specifications
│   └── align_labels.py             # Script to map raw text to ICAO ADREP tags
│
├── event_extraction_model.py       # Main Layer 1 NER Training Script
├── manual_validation.py            # Human-annotation validation script
│
├── entity_distribution.png         # Top extracted entities visualization
│
├── DeepLearning_Project_Proposal.pdf
├── multi_llm_aviation_incident_classification.pdf  # Prior SDCPS paper
├── adrep_project_critique.md
└── proposal_text.txt
```

---

## 🚀 Running the Project

### Requirements

```bash
pip install torch transformers seqeval pandas numpy matplotlib flask flask-cors
```

### 1. Run the Web Demo API & UI (New)

The project includes an interactive web interface to demonstrate both the NER extraction and ADREP incident classification simultaneously.

```bash
python app/server.py
```
* **API Endpoints**: Localhost runs on `http://127.0.0.1:5000`
* **Demo UI**: Open your browser and navigate to `http://127.0.0.1:5000/app` to use the comprehensive visualization dashboard.

### 2. Retraining the Models

If you need to retrain the underlying models from scratch:

**Layer 1 (NER):**
```bash
python event_extraction_model.py
```

**Layer 3 (Occurrence Classification):**
```bash
python src/layer3_reasoning/train.py
```
*(Note: Requires `data/reference/labeled_hard_cases.csv` to be properly mapped via `data/align_labels.py` first).*

---

## 📚 References

1. Aviation Safety Network Database — https://aviation-safety.net/database/
2. Delyon, Manyara, Iliya, Gachomba. _Intelligent Aviation Safety Data Analysis using Transformer and Multi-LLM Consensus_. CMU-Africa Capstone, 2026.
3. FAA Aviation Safety Information Analysis — https://www.asias.faa.gov
4. NASA ASRS Database — https://asrs.arc.nasa.gov/search/database.html
5. New & Wallace. _Classifying Aviation Safety Reports using Supervised NLP_. Safety, 11(1):7, 2025.
6. NTSB Aviation Data — https://data.ntsb.gov/avdata
7. Wang et al. _Self-Consistency Improves Chain of Thought Reasoning in Language Models_. ICLR 2023.
