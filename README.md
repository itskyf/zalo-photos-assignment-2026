# Zalo Photos Assignment 2026

## 0. Executive memo: what the data says, what I chose, and why it is the safest 3-day decision

**The Problem**: While the benchmark expects a clean single-label classification, the real-world dataset is tiny (~300 samples) and inherently constrained by semantic ambiguity and label noise.

**The Decision**: I implemented a data-centric pipeline utilizing a frozen CLIP-family encoder (SigLIP2) combined with an offline noise-weighted linear head, handling the "other" class as a reject-style residual outcome.

**Why It Matters**: For a tiny dataset with highly ambiguous boundaries, complex modeling risks overfitting to label noise. A data-centric approach with frozen priors maximizes 3-day ROI by establishing a stable, reproducible baseline while exposing underlying data ambiguities.

**Key Metrics & Data Realities:**

- Dataset Constraints: ~261 training samples and 50 clean evaluation samples.
- Data Quality Issues: Identified and handled 5 duplicate/near-duplicate images and 10 leaky samples across the train/test splits.
- Final Performance: Achieved a 0.8213 Macro-F1 and 0.8611 Balanced Accuracy.
- Core Confusion: The primary source of error is the "other" class; because it acts as a heterogeneous catch-all, it frequently overlaps with specific semantic clusters (e.g., visually festive "other" images confidently misclassified as lunar_new_year).

## Act I: Let the data speak first

The benchmark looks simple on the surface, but the underlying data tells a different story. Initial explorations revealed that the true challenge is not model capacity, but rather label ambiguity, residual classes, data leakage, and a tiny sample size.

### 1. What the benchmark asks

At first glance, the task is straightforward:

- Task: Single-label, 6-class image classification (`baby_playing`, `lunar_new_year`, `nature`, `trekking`, `gathering`, and `other`).

- Constraint: A strict 3-day timeline; heavy LLMs or VLMs are prohibited.

- Metric: Macro-F1 (primary) to account for class representation, and Balanced Accuracy (secondary).

### 2. Patterns, outliers, and data quality issues

Prioritizing data over model capacity, EDA reveals this isn't a simple classification task. We are navigating a data-constrained regime (~320 images) riddled with collection artifacts and composition quirks.

**Table 1:** train/test class distribution

| Category | train | test | total | train_pct | test_pct |
| --- | --- | --- | --- | --- | --- |
| **baby_playing** | 39 | 9 | 48 | 14.9% | 15.0% |
| **gathering** | 24 | 5 | 29 | 9.2% | 8.3% |
| **lunar_new_year** | 32 | 7 | 39 | 12.3% | 11.7% |
| **nature** | 60 | 14 | 74 | 23.0% | 23.3% |
| **other** | 66 | 16 | 82 | 25.3% | 26.7% |
| **trekking** | 40 | 9 | 49 | 15.3% | 15.0% |

Beyond imbalance, the dataset contains critical structural flaws:

- Duplicates & Leakage: [C-RADIOv4](https://arxiv.org/abs/2601.17237) embeddings revealed 5 duplicates and 10 leaky samples across the train-test split. Without correction, this leakage would artificially inflate scores, creating a false sense of generalization.
- Semantic Overlap: There is severe semantic overlap between classes like trekking vs. nature (thiên nhiên), and lunar_new_year (ngày_tết) vs. gathering (tụ_họp).

**Table 2:** Short summary table of exact dup / near dup / leaky test

| Category | Count | Percentage | Note |
| --- | --- | --- | --- |
| Exact Duplicates | 14 | 4.4% | File hash match (MD5/SHA) |
| Near Duplicates | 17 | 5.3% | C-RADIOv4, threshold=0.1 |
| Leaky Test Samples | 10 | 16.7%* | Test samples too similar to Train (Data Leakage) |

**Image 1:** Demonstrating a Leaky Split. This example also illustrates Semantic Ambiguity (e.g., an image perfectly fitting both `gathering` and `lunar_new_year`)

| `data_train/tụ_họp_verified/0346_d79805bc0361.png` | `data_test/ngày_tết_verified/0346_d79805bc0361.png` |
| :---: | :---: |
| ![Train Image](./lfs/images/leaky/test_lunar_new_year_0346_d79805bc0361.png) | ![Test Image](lfs/images/leaky/train_gathering_0346_d79805bc0361.png) |

These patterns change the problem. The goal is no longer about picking the strongest classifier, but designing the safest decision rule for messy, overlapping data.

### 3. Why the obvious closed-set mental model breaks

Treating this as a mutually exclusive 6-class problem optimizes for the wrong outcome. A strict single-label classifier breaks down here for two core reasons:

- **The `other` class is a residual:** It is not a compact semantic cluster but a heterogeneous catch-all. Forcing standard softmax classification forces the model to guess, often confidently misclassifying visually festive `other` images as `lunar_new_year`.
- **Forced single-choice is unrealistic:** In real-world products, an image often belongs to multiple categories at once. For example, a photo of a family dinner is both gathering and lunar_new_year. Forcing a standard model to pick only one label forces it to make arbitrary, confusing guesses.

> **The Takeaway:** The real problem is not "training the strongest model," but "designing a decision rule that safely handles noisy, ambiguous data and a heterogeneous residual class."

## Act II: Reframing the problem in product context

### 4. Task reframing: from clean classification to ambiguity-aware decision-making

- While the final output must remain strictly single-label to satisfy the benchmark, training and error analysis must embrace the dataset's inherent ambiguity. The problem is practically **semantic retrieval + calibrated decision-making** rather than plain closed-set classification.
- `other` is a residual, not a concept: Instead of treating `other` as a compact semantic cluster, it is modeled as a reject-style outcome for out-of-distribution or ambiguous samples.

### 5. Design principles under a 3-day constraint

- Prefer strong pretrained representations over end-to-end tuning. In a severely constrained dataset, establishing a stable feature space for photo tagging is much safer than risking overfitting.
- Let actual data realities (semantic overlap between tags) dictate model and loss design. Data-centric fixes yield a higher ROI than architectural complexity.
- Treat uncertainty, and failure analysis as core features. The 3-day goal is to deliver a reproducible, robust tagging logic, rather than a brittle model over-engineered solely for a leaderboard.

### 6. Why this method fits the data, the constraint, and the product context

| Component | Rationale for Selection |
| --- | --- |
| [SigLIP2 (so400m-patch16-384)](https://huggingface.co/google/siglip2-so400m-patch16-384) | Chosen over DINOv3, MetaCLIP2, and NAVER ProLIP because this task is driven more by semantic overlap and multilingual label cues than by purely vision-only dense features. The So400m variant provides the best quality-to-cost ratio for a strict 3-day timeline. |
| [CLIPCleaner](https://github.com/MrChenFeng/CLIPCleaner_ACMMM2024) | Hard filtering is too destructive for a ~300-sample dataset. Soft weighting mitigates label noise without sacrificing critical data mass. |
| **K-fold / OOF** | A single validation split on tiny data yields highly unstable metrics. Out-of-fold scoring ensures robust, reliable model selection. |
| **Linear head** | Stable, low-variance decision boundary on top of frozen embeddings, maximizing the 3-day ROI while remaining fully reproducible and easy to calibrate. |

## Act III: The chosen system

### 8. Representation choice

- Vì sao SigLIP2 là backbone chính.
- Vì sao chọn biến thể cụ thể (ví dụ So400m) trong 3 ngày.

### 9. Data-centric supervision

- CLIPCleaner-style weighting
- ambiguity-aware supervision
- K-fold / OOF logic nếu dùng

### 10. Why `other` is handled as a reject-style outcome

- Không coi `other` là một semantic cluster compact.
- Quyết định `other` dựa trên uncertainty / margin / weak similarity.

### 11. Final pipeline in one diagram

- Data audit → denoise/weight → embedding extraction → decision rule / head → calibration → failure analysis

## Act IV: Evidence for the decision

### 12. Evidence block A — Data evidence

- 1–3 visual/bảng mạnh nhất: counts, duplicates/leakage, embedding overlap

### 13. Evidence block B — Decision evidence

- baseline rất đơn giản
- model chính
- 1–3 ablation thật sự trả lời câu hỏi quan trọng

### 14. Failure modes, limitations, and evaluation context

- failure archetypes
- expected non-use / caution cases
- giới hạn của kết luận do dataset nhỏ, noisy, ambiguous

### 15. Implementation and analysis discipline

- FiftyOne cho audit / neighborhood / error triage
- Transformers cho implementation path
- reproducibility / fallback / calibration discipline

## Act V: What I would ship in 3 days

### 16. What I would ship — and where I would be cautious

- đúng 1 pipeline tôi sẽ ship
- vì sao nó hợp bài test và hợp production mindset

### 17. Rejected but reviewed options

- chỉ 2–4 hướng đáng nhắc
- mỗi hướng 1 câu: vì sao không chọn trong 3 ngày

### 18. Highest-ROI next steps

- targeted relabeling
- hard negatives / data expansion
- projector-only adaptation / complementary encoder nếu có thêm thời gian

## Appendix A — Execution under a 3-day constraint

- Day 0 / Day 1–2 / Day 3 ở mức rất ngắn

## Appendix B — AI-assisted workflow and reproducibility note

- AI agents hỗ trợ research/code scaffolding
- mọi quyết định, kiểm chứng, kết luận đều được tự kiểm và chịu trách nhiệm
