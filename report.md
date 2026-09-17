# Report — Uber Support AI Agent

**Hiver SDE Intern Take-Home Assignment**
Brand: **Uber_Support** (rides only) · Dataset: Customer Support on Twitter (Kaggle) · Golden Set: 199 hand-labelled examples

---

## 1. Problem Framing

### What "good" means for this brand

Uber_Support's Twitter traffic is high-volume, public, and dominated by a recognizable set of recurring rider issues — fare disputes, cancellation fees, driver behavior, wait times, account access, and lost items. For this brand, "good" for an automated triage agent means:

- **Correct, stable intent classification**, so downstream handling and routing is consistent rather than arbitrary.
- **Grounded, non-hallucinated replies.** A plausible-sounding but invented promise (a refund amount, a policy claim) is worse than a generic acknowledgment, because it can commit the brand to something it never agreed to.
- **Conservative escalation.** Money, safety, and account-security issues should default to human review when uncertain — silently auto-closing a safety report is the single worst failure mode this system could produce, worse by orders of magnitude than an unnecessary escalation.
- **Honest uncertainty.** When the system doesn't have strong signal, it should escalate rather than guess.

### Scope: what I chose not to build

- **Multi-turn dialogue management.** The raw dataset contains full reply threads, but I reduced each conversation to (first customer message → response), treating this as single-message triage. This matches the assignment's literal framing ("classify each incoming message... draft a reply... decide whether to escalate") and avoids a materially larger system (conversation state tracking, turn-taking logic) that the assignment doesn't ask for.
- **UberEats and driver/partner support.** `Uber_Support` handles both riders and Eats customers, and both riders and driver-partners. I scoped tightly to **rider-side, rides-only** traffic via keyword filtering. This filtering is imperfect — see Failure Analysis — but keeping scope narrow kept the intent taxonomy coherent rather than trying to cover three structurally different support domains at once.
- **Real backend/account lookups.** No order/account/trip database exists for this public dataset. Where a real agent would check account state, this system asks for details via DM — which is, not coincidentally, what the real brand's actual replies do.
- **A learned (ML) escalation classifier.** Escalation combines deterministic keyword rules (safety/security/discrimination language) with LLM reasoning, rather than a trained classifier. This was a deliberate choice: escalation is a safety boundary, and I wanted it auditable and not subject to silent drift if retrained on different data.

---

## 2. System Design

```
customer message
       │
       ▼
intent classification (LLM, few-shot, 9 categories)
       │
       ▼
retrieval over historical Uber_Support complaints (embeddings + ChromaDB, top-k similar)
       │
       ▼
grounded reply drafting (LLM, conditioned on retrieved examples)
       │
       ▼
escalation decision (keyword rules + LLM reasoning)
       │
       ▼
{intent, drafted_reply, decision, reason}
```

**Intent taxonomy (9 categories):** `fare_billing_dispute`, `driver_safety_incident`, `wait_time_cancellation`, `account_access`, `app_technical_issue`, `lost_item`, `general_complaint`, `spam_irrelevant`, `compliment_other`. This taxonomy was not designed up front — it was derived empirically by reading real complaints during labeling, and `lost_item` was added mid-labeling after being repeatedly forced into `general_complaint` (see `decision_log.md`).

**Escalation policy:** escalate when a message shows safety/physical-harm language, security/fraud signals, a discrimination pattern, a large fare discrepancy, or an explicitly-stated pattern of unresolved prior complaints. Everything else defaults to `auto_handle`.

**Two baselines implemented** for comparison:
- **Trivial:** always predicts the majority intent class, always predicts `auto_handle`.
- **Simple:** TF-IDF + Logistic Regression for intent, plus a keyword-only escalation rule (no LLM involved anywhere).

---

## 3. Results vs. Baselines

All numbers are from the **full pipeline**, run end-to-end against the 199-example golden set (`eval/run_full_eval.py`).

| System | Intent Accuracy | Decision Accuracy |
|---|---:|---:|
| Trivial baseline | 25.13% | 73.87% |
| TF-IDF + Logistic Regression + keyword rules | 25.00% | 76.38% |
| **Full AI pipeline** | **55.28%** | **81.91%** |

**Escalation performance (full pipeline):**

| Metric | Value |
|---|---:|
| Precision | 86.36% |
| Recall | 36.54% |
| False negatives | 33 / 199 |

```
                    Predicted
                auto_handle   escalate
True auto_handle    144           3
True escalate        33          19
```

**What this shows:** the pipeline clearly beats both baselines on intent (more than doubling trivial, more than doubling the ML baseline) and modestly beats them on decision accuracy. But decision accuracy is a weak signal here — see §5. Escalation precision is strong (when the system escalates, it's usually right to), but recall is weak: it misses roughly two out of three cases that should have gone to a human.

**Reply quality — LLM-as-judge validated against a human:**

| Metric | Value |
|---|---:|
| Overall judge/human agreement | 96.67% |
| Cohen's κ | 0.933 |
| Relevance agreement | 90% |
| Grounding agreement | 90% |
| Helpfulness agreement | 100% |
| Safety agreement | 100% |

Based on 30 stratified examples independently scored by a human and compared to the LLM judge. The single disagreement (of 30) is instructive: the judge rated a reply "GOOD" despite it asserting prepaid Mastercard support with no basis in the retrieved grounding context — the judge rewarded fluency and topical relevance over strict factual grounding. This is a real, specific judge blind spot, not random noise (see §5).

---

## 4. Failure Analysis — Top Failure Modes

### 1. Under-escalation driven by missing "pattern" and "severity" context
The single largest driver of the 33 false negatives. The model correctly reads individual messages but has no memory of prior interactions and no explicit severity threshold, so it systematically under-escalates:
- **Repeated/unresolved complaints** ("I raised a complaint yesterday and you're ignoring me") get read as one-off routine complaints, because the pipeline has no conversation history.
- **Large financial discrepancies** (a fare quote doubling, a currency mischarge, a driver demanding cash beyond the quoted fare) get classified correctly as billing disputes but not flagged as *severe enough* to escalate — the escalation prompt doesn't give the model a concrete magnitude threshold.
- *Hypothesis:* both are architectural gaps, not classifier weaknesses — the system simply isn't given the signals (history, magnitude) it would need to make this call correctly. *Fix:* add explicit numeric-discrepancy detection and a "prior complaint" flag as structured inputs to the escalation prompt, rather than relying on the LLM to infer severity from prose alone.

### 2. Third-party / non-account-holder requests are not recognized as a risk category
Example: a spouse asking about their partner's ride status, someone recovering a lost phone that has 2FA tied to someone else's identity. These get treated as routine account-access or lost-item requests. *Hypothesis:* the taxonomy and escalation rules never explicitly named "request made by someone other than the account holder" as a trigger. *Fix:* add an identity/authorization check as an explicit escalation signal.

### 3. Intent misclassification concentrates in `driver_safety_incident` → `general_complaint`
15 of the 33 false negatives also had the wrong predicted intent, and the dominant error pattern is safety incidents being swallowed into the catch-all `general_complaint` bucket (e.g., "driver using phone while driving," "driver essentially extorted $50 cash"). *Hypothesis:* `general_complaint` is a large, semantically loose catch-all, and messages with implicit rather than explicit safety language ("this is going to make me throw up" for erratic driving) don't trip the model's safety framing. *Fix:* tighten `general_complaint`'s definition and add more implicit-safety-language examples to the classification few-shot prompt.

### 4. A structured-output parsing bug silently defaulted risky cases to `auto_handle`
8 of the 33 false negatives were not reasoning failures at all — the escalation LLM's response didn't match the expected `DECISION:`/`REASON:` format, and the parser silently fell back to `auto_handle` on any unparseable output. **This is the most operationally dangerous failure mode found**, because it fails in the wrong direction: an uncertain or malformed model response should never default to the less-safe outcome. Structured-output parsing failures silently defaulted to `auto_handle`; eight of the 33 false negatives were caused by this fallback behavior. A fail-safe change — routing parsing failures to human review instead of auto-handling — is a planned improvement (see §6), but was **not** included in the reported evaluation above; the 33/199 false-negative count and the 36.54% recall figure both reflect the unpatched behavior.

### 5. The judge rewards fluency and topical relevance over strict factual grounding
From the human-agreement check: a reply was rated "GOOD" by the judge despite asserting a specific factual claim (prepaid card support) that wasn't present anywhere in the retrieved grounding examples. *Hypothesis:* the judge rubric weights relevance/helpfulness/tone more heavily than a strict "is every claim traceable to a retrieved source" check. This is a narrow-sample finding (n=1 disagreement out of 30) but it's the *kind* of error that matters most for this use case, since a hallucinated policy claim sent to a real customer carries real risk. *Fix:* add an explicit "grounding audit" step to the judge rubric — flag any factual claim in the reply that cannot be traced to a specific retrieved example.

---

## 5. What's Misleading About My Headline Number

**81.91% decision accuracy is not a reliable measure of escalation performance, and reporting it alone would be misleading.**

- **It is barely above a trivial baseline.** The golden set is imbalanced (147 true `auto_handle` vs. 52 true `escalate` — from the confusion matrix: 144 true negatives + 3 false positives = 147 true `auto_handle` cases), so a system that predicts `auto_handle` for everything already scores 73.87%. The full pipeline's 81.91% represents only an ~8-point lift over doing nothing — a much smaller improvement than the headline number alone suggests.
- **Recall (36.54%) tells the real story, and it's weak.** Of the cases a human labeler flagged for escalation, the system caught roughly one in three. For a safety-relevant decision, recall — not accuracy — is the metric that should be read first, and it should be reported alongside, not instead of, the accuracy figure.
- **A meaningful share of the recall failure is a fixable bug, not a modeling limitation — and the reported numbers do not yet reflect the fix.** 8 of 33 false negatives were parsing failures that silently defaulted to the unsafe outcome, meaning the true underlying reasoning-based miss rate is 24/33, not 33/33 — but I want to be explicit that the 36.54% recall figure reported in §3 includes these 8 cases as-is, since re-running the full evaluation after the fix was not completed in time for this report. The bug itself is a legitimate, serious finding regardless: a production system defaulting to the unsafe branch on any malformed output is a real operational risk, independent of how the headline recall number should ultimately be corrected.
- **An earlier, isolated run of the intent classifier reported a higher accuracy (~74.5%) than the 55.28% shown in the full-pipeline results above.** This discrepancy has **not yet been verified** against the current codebase and should not be treated as an established finding — `classify_intent.py` was modified multiple times during development (including a temporary switch to a different API provider and back), so the 74.5% figure may reflect a different version of the classifier than the one used to produce the full-pipeline run in §3. Before this number is cited anywhere, it needs to be reproduced by re-running `eval/evaluate_intent.py` against the current, final version of `src/classify_intent.py`. Until that verification happens, **the only accuracy figure this report treats as established is the 55.28% full-pipeline result**, since that is the number directly tied to a complete, reproducible end-to-end run against the golden set.
- **The golden set and the escalation policy were developed together, not independently.** The taxonomy (including the late addition of `lost_item`) and the escalation rule itself were both derived from reading these same 199 examples during labeling. This means the golden set is not a fully held-out, independent test of the system's design choices — there is real overlap between "what informed the rules" and "what the rules are tested against." A more rigorous setup would develop the policy on one sample and evaluate on a separate, disjoint sample.
- **Judge/human agreement (96.67%, n=30) is a small sample.** It's a genuinely strong result, but 30 examples is not enough to rule out the judge having systematic blind spots that simply didn't surface in this particular subsample (the one disagreement found — a grounding miss — is a plausible candidate for a more general pattern, not necessarily a one-off).

---

## 6. What I'd Do With One More Week

1. **Fix the escalation parser's fail-safe behavior and re-run the full evaluation.** Route parsing/malformed-output failures to escalation or explicit human review instead of the current silent fallback to `auto_handle`, then re-generate the headline recall/precision numbers so they reflect the corrected behavior rather than the as-reported 36.54% figure. More broadly, audit every LLM call in the pipeline for what happens on malformed/empty output and apply the same fail-safe default throughout.
2. **Add explicit structured signals the LLM currently has to infer from prose:** a numeric fare-discrepancy threshold, a "prior complaint" flag (would require reconstructing conversation history, currently out of scope), and an "authorization mismatch" flag for third-party requests.
3. **Separate policy-development data from evaluation data.** Re-derive the taxonomy and escalation rules on one sample, then hand-label a fresh, disjoint sample purely for evaluation, to get an honest read on generalization rather than a number partly measuring the system's fit to the examples that shaped its own design.
4. **Verify (or discard) the ~74.5% classifier-only accuracy figure** by re-running `eval/evaluate_intent.py` against the exact current version of `src/classify_intent.py`, and if it reproduces, investigate why it diverges from the 55.28% full-pipeline result — likely candidates are prompt/context differences or code changes made to the classifier between the two runs, but this is not yet root-caused.
5. **Expand and stress-test the LLM-judge rubric**, specifically adding an explicit grounding-audit step (does every factual claim in the reply trace to a retrieved example), and grow the human-agreement sample past 30 with deliberate oversampling of edge cases (safety, fraud, third-party requests) rather than pure random sampling.
6. **Tighten intent boundaries, especially around `general_complaint`**, which is currently absorbing a disproportionate share of driver-safety misclassifications — likely needs either a narrower definition or more few-shot examples covering implicit (non-explicit-keyword) safety language.
