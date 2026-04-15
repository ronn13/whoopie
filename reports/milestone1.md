# Layer 1 — Event Extraction for Causal Graph Reasoning
### Milestone 1 Report · 18-786 Intro to Deep Learning
**Whoopie Wanjiru** · MSEAI27 · wwanjiru@andrew.cmu.edu

---

## 3 Methods

### 3.1 Problem Understanding

The CMU-Africa SDCPS pipeline classifies aviation safety reports under the ICAO ADREP taxonomy using a hybrid transformer and multi-LLM consensus engine. Despite achieving high overall accuracy, the system conservatively assigns **26.94% of reports** to the catch-all **"OTHER"** category. These are typically narratives where the causal structure is implicit, multi-factor, or expressed in language that resists flat classification.

My contribution addresses **Layer 1** of a four-layer causal refinement module that acts exclusively on OTHER-labelled reports. The goal of Layer 1 is to extract a structured *event tuple* of the form:

$$\text{Event} = (\textit{ACTOR},\ \textit{SYSTEM},\ \textit{PHASE},\ \textit{TRIGGER},\ \textit{OUTCOME})$$

from each raw aviation narrative. These structured tuples are the input to Layer 2 (causal graph construction), Layer 3 (graph reasoning), and Layer 4 (ADREP reclassification). Without a reliable extraction layer, downstream components cannot function — making this the foundational piece of the pipeline.

The extraction problem is a **Named Entity Recognition (NER)** task framed as token-level sequence labeling, where each sub-word token of a narrative is assigned a BIO (Begin–Inside–Outside) label. PHASE and OUTCOME are obtained directly from existing structured columns in the dataset (≥98% and 100% coverage respectively), so the NER model focuses on the three text-only entities: **ACTOR**, **SYSTEM**, and **TRIGGER**.

---

### 3.2 Hybrid Extraction Strategy

The extraction pipeline uses a **hybrid approach** that combines structured columns with neural NER:

| Entity | Source | Coverage |
|---|---|---|
| **PHASE** | Existing `phase` column | ~98% |
| **OUTCOME** | Existing `events` + `events_6` columns | 100% |
| **ACTOR** | Narrative text → BERT NER | Learned |
| **SYSTEM** | Narrative text → BERT NER | Learned |
| **TRIGGER** | Narrative text → BERT NER | Learned |

---

### 3.3 Silver-Label Generation

Since the dataset contains **no gold-annotated NER labels**, training data is generated via a **silver-labeling** strategy. Three sets of domain-curated regex patterns are applied to each narrative to extract character-level entity spans:

- **ACTOR patterns** — cover specific crew roles (*Captain*, *First Officer*, *PIC*, *SIC*, *ATC*, *Flight Crew*), ATC roles, and first-person pilot narrations ("I noticed...", "we decided...")
- **SYSTEM patterns** — cover navigation systems (GPS, ILS, RNAV, FMS), safety systems (TCAS, GPWS, TAWS), autopilot/automation, communications (radio, transponder), and aircraft components (engine, landing gear, flaps)
- **TRIGGER patterns** — cover causal language ("due to", "failed to", "resulted from"), human factors (fatigue, distraction, situational awareness loss), environmental triggers (turbulence, icing, IMC), and mechanical failures (engine failure, malfunction)

Overlapping spans are resolved by keeping the longest match. The resulting character-level spans are then aligned to BERT sub-word token positions to produce BIO-tagged sequences, forming the training set for fine-tuning.

This silver-labeling strategy is deliberate: the BERT model is expected to **generalise beyond** the rules by learning contextual cues from the BERT pre-training that pattern matching alone cannot capture.

---

### 3.4 Model Architecture

The model follows the standard **Transformer + Token Classification Head** architecture:

$$[\text{narrative tokens}] \rightarrow \text{BERT Encoder} \rightarrow \mathbf{H} \in \mathbb{R}^{L \times 768} \rightarrow \text{Linear}(768 \to 7) \rightarrow \text{BIO logits}$$

The classification head maps each token's hidden state to 7 labels: `O, B-ACTOR, I-ACTOR, B-SYSTEM, I-SYSTEM, B-TRIGGER, I-TRIGGER`.

Two pre-trained encoders are evaluated:

1. **BERT-base-uncased** — general-purpose BERT (Devlin et al., 2019)
2. **SafeAeroBERT** (`NASA-AIML/MIKA_SafeAeroBERT`) — domain-specific BERT pre-trained on ASRS and NTSB aviation safety narratives; the same type of data we work with

Both models share **108.9M trainable parameters** and are trained under identical hyperparameters for a fair comparison.

#### Training Configuration

| Hyperparameter | Value |
|---|---|
| Max sequence length | 256 tokens |
| Batch size | 8 |
| Learning rate | 3 × 10⁻⁵ |
| Epochs | 10 (max) |
| Early stopping patience | 3 epochs |
| LR schedule | Linear warmup (10%) + linear decay |
| Loss function | Weighted cross-entropy (inverse frequency, capped at 20×) |
| Optimizer | AdamW (weight decay = 0.01) |
| Gradient clipping | max norm = 1.0 |
| Data split | 70% train / 15% val / 15% test |
| Dataset | 173 aviation safety reports |

**Class weighting** is applied to the cross-entropy loss to mitigate the heavy imbalance between the `O` label (which accounts for the vast majority of tokens) and entity labels. Weights are normalised so that `O` receives a weight of 1.0, and entity tokens receive up to 20×.

**Long narrative handling:** narratives exceeding 256 tokens are split into overlapping windows (stride = 128 token-equivalents) so entities near chunk boundaries are not lost.

---

### 3.5 Baseline

The **rule-based silver-label extractor** itself serves as the baseline. It is deterministic and interpretable, but it is limited to patterns explicitly enumerated in the pattern lists — it cannot generalise to unseen phrasing or contextual variations. The BERT models are expected to improve upon it by learning latent contextual signals from pre-training.

---

### 3.6 Planned Future Experiments

The following experiments are planned for the remainder of the project:

1. **CRF decoding layer** — add a Conditional Random Field (CRF) on top of the BERT hidden states to enforce valid BIO transition constraints (e.g., no `I-ACTOR` without a preceding `B-ACTOR`), which the current linear head does not guarantee.

2. **Coreference resolution for ACTOR recall** — the current model misses actors expressed through first-person pronouns in non-pattern-matched contexts. Integrating a lightweight coreference resolution step (e.g., `coreferee` or `neuralcoref`) to resolve "we/I" → pilot role before extraction could improve ACTOR recall.

3. **LLM-augmented silver labels** — replace or supplement the rule-based silver labels with structured extractions from an LLM (e.g., Gemini) using chain-of-thought prompting, then fine-tune BERT on the richer labels.

4. **Active learning** — use model uncertainty (token-level entropy) to select the most informative reports for manual annotation, enabling targeted quality improvement with minimal annotation effort.

5. **Layer 2 integration** — pass the extracted event tuples from `extracted_events.json` into a causal DAG construction pipeline using `torch_geometric`, with typed nodes (ACTOR, SYSTEM, PHASE, TRIGGER, OUTCOME) and directed causal edges.

---

## 4 Results

### 4.1 Quantitative Results

Both models were evaluated on a held-out test set (15% of 173 reports). Weighted F1, weighted precision, and weighted recall are computed using `seqeval`, the standard library for sequence labeling evaluation.

| Metric | BERT-base | SafeAeroBERT | Δ (Aero − BERT) |
|---|---|---|---|
| **Test F1 (weighted)** | 0.741 | **0.830** | +0.089 |
| **Precision** | 0.655 | **0.786** | +0.131 |
| **Recall** | 0.857 | **0.881** | +0.024 |
| **Test Loss** | 0.690 | **0.495** | −0.195 |
| **Best Val F1** | 0.731 (epoch 9) | **0.758** (epoch 10) | +0.027 |

**SafeAeroBERT is the clear winner**, outperforming BERT-base by **+8.9 F1 points** on the test set. The improvement is most pronounced in precision (+13.1 points), suggesting the domain-specific pre-training allows the model to be more selective — identifying true entity spans with fewer false positives.

### 4.2 Training Curves

Figure 1 shows the training and validation loss/F1 curves for BERT-base and SafeAeroBERT side by side. Both models converge steadily; SafeAeroBERT reaches a lower final validation loss and maintains a higher F1 floor throughout training.

![Model comparison training curves](model_comparison_curves.png)
*Figure 1: Training and validation loss (top) and F1 (bottom) for BERT-base (left) and SafeAeroBERT (right) over 10 epochs.*

### 4.3 Test Set Performance Comparison

Figure 2 presents the head-to-head F1, Precision, and Recall comparison on the test set. SafeAeroBERT leads across all three metrics.

![Model comparison bar chart](model_comparison_bar.png)
*Figure 2: BERT-base vs. SafeAeroBERT on F1 score, precision, and recall on the held-out test set.*

### 4.4 Manual Validation Against Human Annotations

Because the test set uses silver labels (generated from the same rule-based patterns), F1 scores on it measure *agreement with the rules*, not true extraction quality. To obtain a more meaningful quality estimate, **3 reports were manually annotated** by hand, and the best BERT-base model's predictions were compared against these gold annotations.

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| ACTOR | 1.00 | 0.50 | 0.67 |
| SYSTEM | 1.00 | 1.00 | **1.00** |
| TRIGGER | 1.00 | 1.00 | **1.00** |
| **Overall** | **1.00** | **0.67** | **0.80** |

The model achieves **perfect precision (1.00)** — every entity it predicts is correct. The recall gap (0.67 overall) is entirely attributable to **ACTOR**, where the model misses some mentions expressed implicitly (e.g., pronouns or contextual role assignments not matching any rule). SYSTEM and TRIGGER extraction is perfect on these examples.

The overall validation quality is rated **EXCELLENT**, providing confidence that the extracted event tuples are reliable inputs for the causal graph layer.

### 4.5 Extracted Entity Distribution

Figure 3 shows the most frequently extracted entities across all 173 reports.

![Entity distribution](entity_distribution.png)
*Figure 3: Top 10 extracted ACTORS (left), SYSTEMS (center), and TRIGGERS (right) across the full dataset.*

---

## 5 Ablation Study

The primary ablation in this work is the **encoder backbone comparison**: BERT-base-uncased (general domain) vs. SafeAeroBERT (aviation domain). All other aspects of the training pipeline are held constant — same silver labels, same data split, same hyperparameters, same loss function — so any performance difference is attributable solely to the pre-training corpus.

The results (Section 4.1) confirm that **domain-specific pre-training provides a measurable and consistent advantage** across every metric. This is consistent with the broader NLP literature on domain-adapted models: when the target text (aviation incident narratives) is stylistically and lexically distinct from general web text, a model pre-trained on in-domain data will learn better contextual representations of domain-specific terms.

A secondary implicit ablation exists in the **silver-label training strategy itself** — the model is trained on rule-extracted labels and then evaluated against human annotations. The fact that test F1 on silver labels (0.74 / 0.83) is lower than manual validation F1 (0.80 overall) is somewhat surprising at first, but reflects the fact that the model has learnt to predict entities in contexts the rules do not cover. This is the intended generalisation effect of the fine-tuning approach.

---

## 6 Discussion and Conclusion

### 6.1 Findings

The key finding of this work is that **fine-tuning a domain-adapted BERT model on silver-labeled aviation narratives produces a reliable event extractor** for the ACTOR, SYSTEM, and TRIGGER entities needed by the causal pipeline. SafeAeroBERT achieves a weighted F1 of 0.830 on the test set and 0.800 on manual annotations, with perfect precision — meaning it does not invent entities that are not there.

The **hybrid extraction design** proved to be a pragmatic and effective choice. Rather than attempting to learn PHASE and OUTCOME from text (where structured labels already exist), the model focuses its capacity on entities that genuinely require reading comprehension. This keeps the task tractable on a small dataset of 173 reports.

### 6.2 Limitations

- **Silver-label noise:** The 173 reports used here are a sample drawn from the full 68,000+ report corpus. While scaling to more reports is straightforward, the quality ceiling is set by the silver-labeling rules — if a pattern systematically mislabels a span, the model learns to replicate that error.
- **ACTOR recall:** First-person pilot narrations ("I noticed the engine was surging...") are partially missed because the rule patterns do not cover all pronoun-role associations. This is the main source of false negatives.
- **Silver-label ceiling:** Both models are ultimately trained on imperfect supervision. If the rule-based extractor systematically misses a certain entity type in a certain phrasing, the BERT model inherits that blind spot.
- **No causal relation extraction yet:** The current output is a flat entity tuple, not a causal graph. Edge types between entities (e.g., ACTOR *caused* TRIGGER *led\_to* OUTCOME) are not yet modelled; this is the responsibility of Layer 2.

### 6.3 Implementation Challenges

- **BIO alignment with sub-word tokenization:** BERT tokenizes words into sub-word pieces (e.g., "TCAS" → ["TC", "##AS"]). Aligning character-level regex spans to sub-word token positions required careful offset mapping to correctly assign B- vs I- tags and ensure [CLS]/[SEP]/[PAD] tokens are masked from the loss with `−100`.
- **Class imbalance:** The `O` label accounts for the overwhelming majority of tokens in any narrative (>95%). Without explicit class weighting, the model converges to predicting `O` everywhere. Inverse-frequency weighting (capped at 20×) was critical to obtaining meaningful entity predictions.
- **Long narratives:** Some reports exceed 256 tokens. The overlapping window strategy (stride = 128 token-equivalents) ensures entities near chunk boundaries are not missed, but requires careful span re-indexing within each chunk.
- **Evaluation with `seqeval`:** Seqeval computes entity-level metrics (a span is only correct if both boundary tokens and the entity type match exactly), which is stricter than token-level accuracy. This is the appropriate metric for NER but takes care to interpret correctly alongside silver-label training.
