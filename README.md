# Uber Support AI Agent

An AI support agent for **Uber_Support** (rides only), built using the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) Kaggle dataset.

Given an incoming customer message, the agent:

1. Classifies the customer's intent.
2. Retrieves historically similar Uber support conversations.
3. Drafts a reply grounded in those historical responses.
4. Decides whether the request should be **auto-handled** or **escalated to a human**.
5. Provides a reason for the escalation decision.

The system is evaluated on a **199-example hand-labelled Golden Set**, with automated evaluation, LLM-as-judge scoring, human agreement validation, baseline comparisons, and failure analysis.

---

## Project Structure

```text
Hiver/
│
├── src/
│   ├── classify_intent.py
│   ├── src_retrieval.py
│   ├── reply_generation.py
│   ├── escalation.py
│   └── pipeline.py
│
├── eval/
│   ├── explore.py
│   ├── build_threads.py
│   ├── clean_data.py
│   ├── filter_riders.py
│   ├── create_golden_set.py
│   ├── evaluate_intent.py
│   ├── baseline_trivial.py
│   ├── baseline_simple.py
│   ├── run_full_eval.py
│   ├── extract_false_negatives.py
│   ├── failure_analysis.py
│   ├── llm_judge.py
│   ├── clean_judge_results.py
│   └── human_gemini_agreement.py
│
├── submission_data/
│   ├── golden_set_TO_LABEL.csv
│   ├── full_pipeline_results.csv
│   ├── baseline_results.csv
│   ├── intent_eval_results.csv
│   ├── false_negative_cases.csv
│   └── human_gemini_agreement_summary.csv
│
├── report.md
├── decision_log.md
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

* Python 3.11+
* Kaggle account
* Groq API key

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your-groq-api-key
```

### Kaggle API setup

The dataset is downloaded using `kagglehub`.

Configure Kaggle API access once:

```bash
mkdir ~/.kaggle
echo YOUR_KAGGLE_TOKEN > ~/.kaggle/access_token
```

---

## Reproducing the Evaluation (~15 minutes)

### 1. Prepare the dataset

```bash
python eval/explore.py
python eval/build_threads.py
python eval/clean_data.py
python eval/filter_riders.py
```

Downloads and prepares the Twitter support data, isolating Uber rider-support conversations.

### 2. Run the baselines

```bash
python eval/baseline_trivial.py
python eval/baseline_simple.py
```

* **Trivial baseline** — predicts the majority class for both intent and escalation decision.
* **Simple baseline** — TF-IDF + Logistic Regression for intent, keyword-based escalation rule.

### 3. Run the full pipeline

```bash
python eval/run_full_eval.py
```

```text
Customer message
       ↓
Intent classification
       ↓
Historical example retrieval
       ↓
Grounded reply generation
       ↓
Escalation decision
       ↓
Reason
```

Results saved to `submission_data/full_pipeline_results.csv`.

### 4. Run failure analysis

```bash
python eval/failure_analysis.py
```

Analyzes cases where the system predicted `auto_handle` but the Golden Set labelled the case for escalation. Saved to `submission_data/false_negative_cases.csv`.

### 5. Evaluate generated replies

```bash
python eval/llm_judge.py
python eval/human_gemini_agreement.py
```

The LLM judge scores replies on relevance, grounding, helpfulness, and safety. A stratified sample of 30 cases was independently reviewed by a human and compared against the judge.

**On rate limits:** this runs on Groq's free tier (200,000 tokens/day). Evaluation scripts save progress incrementally and skip already-completed rows — if you hit the daily quota mid-run, re-run the same command after it resets to resume rather than restart.

---

## Golden Set

199 hand-labelled customer messages, manually labelled for intent, expected escalation decision, and reason for escalation where applicable. Kept separate from model-development decisions as much as possible. Full labelling methodology in [`report.md`](report.md).

---

## Results

| System | Intent Accuracy | Decision Accuracy |
|---|---:|---:|
| Trivial baseline | 25.13% | 73.87% |
| TF-IDF + Logistic Regression + rules | 25.00% | 76.38% |
| **AI support pipeline** | **55.28%** | **81.91%** |

**Escalation performance:** precision 86.36%, recall 36.54%, false negatives 33/199.

```text
                    Predicted
                auto_handle  escalate

True auto_handle     144          3
True escalate         33         19
```

### What's misleading about the headline number

81.91% decision accuracy should not be read as reliable escalation performance. The Golden Set has more auto-handle cases than escalation cases, so a trivial "always auto_handle" system already scores 73.87%. More importantly, the pipeline correctly catches only 19/52 (36.54%) of cases that should have been escalated — decision accuracy alone hides this under-escalation weakness. Full discussion in [`report.md`](report.md).

---

## LLM-as-Judge Validation

| Metric | Value |
|---|---:|
| Overall agreement | 96.67% |
| Cohen's κ | 0.933 |
| Relevance agreement | 90% |
| Grounding agreement | 90% |
| Helpfulness agreement | 100% |
| Safety agreement | 100% |

Based on a human reviewer independently scoring a stratified sample of 30 cases against the LLM judge (n=30 — a limitation noted in `report.md`).

---

## Failure Analysis (summary)

33 false-negative escalation cases. Of these: 18/33 had correct intent but wrong escalation decision; 15/33 also had incorrect intent; 8/33 involved model-output parsing failures defaulting to `auto_handle` (since fixed — see `decision_log.md`).

Key patterns: safety incidents read as general complaints, security/fraud cases under-escalated, serious billing disputes treated as routine, repeated/unresolved complaints auto-handled. Full examples and hypotheses in [`report.md`](report.md).

---

## Documentation

* [`report.md`](report.md) — problem framing, methodology, baselines, failure analysis, misleading-metric discussion, judge validation, next-steps plan
* [`decision_log.md`](decision_log.md) — non-obvious engineering and evaluation decisions
* [`submission_data/`](submission_data/) — golden set and all evaluation output CSVs

---

## Dataset

[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle). This project uses the Uber_Support subset, scoped to rider/customer-support interactions (rides only).

---

## Notes on Reproducibility

The full pipeline and LLM judge depend on API availability and rate limits. Evaluation scripts save results incrementally and resume from existing progress rather than restarting if a quota is hit. Results may vary slightly across runs since LLM-generated classifications and replies are not fully deterministic.
