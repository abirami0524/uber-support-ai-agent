# Decision Log — Uber Support AI Agent

Non-obvious decisions made during this project, and why. Not every decision made is listed — only ones where a reasonable alternative existed and the choice wasn't forced by the assignment.

---

**1. Scoped to Uber_Support, rides only — not the highest-volume brand.**
AmazonHelp and AppleSupport both had higher raw tweet volume, but their issue spaces are broader and harder to bound into a small, coherent intent taxonomy. Uber_Support's rider-side traffic clusters into a naturally small set of well-defined issue types, which made intent design and escalation-policy design more tractable given the time available.

**2. Filtered out UberEats and driver/partner traffic, even though `Uber_Support` handles both.**
Mixing rider issues (fare disputes, driver behavior) with driver-partner issues (onboarding, background checks, vehicle inspection) would have produced an incoherent taxonomy — these are structurally different support domains with no shared resolution patterns. Filtering was done via keyword matching, which is imperfect (see Failure Analysis / Decision 10 below) but sufficient given the scope.

**3. The intent taxonomy was built empirically from reading real data, not designed up front.**
Started with a hypothesis taxonomy, then revised it while reading ~100+ real complaints before finalizing. `lost_item` was not in the original taxonomy — it was added mid-labeling after the same pattern (item left in vehicle, driver unreachable) kept appearing and being force-fit into `general_complaint` at least 5 times.

**4. Reduced multi-turn threads to single-message triage.**
The raw dataset preserves full reply threads, but the assignment's own framing ("classify each incoming message... draft a reply... decide whether to escalate") describes single-message triage, not conversation management. Modeling multi-turn state would be a materially larger system than what's being asked for.

**5. Escalation policy is rule-based + LLM reasoning, not a trained classifier.**
Escalation is a safety boundary. A rule-based component (explicit keyword signals for safety/security/discrimination language) was kept as a hard floor specifically so escalation behavior is auditable and doesn't silently drift if the system were retrained on different data later. The LLM adds judgment on top for cases the keyword rules don't catch, but the keyword layer cannot be overridden toward `auto_handle` — only toward `escalate`.

**6. The escalation policy itself was derived from labeling the golden set, not independently designed beforehand.**
This is a real methodological limitation, documented explicitly in the report's "misleading headline number" section: the golden set and the system's rules were developed in the same pass, so evaluation against the golden set is not a fully independent test. A more rigorous setup would separate policy-development data from held-out evaluation data.

**7. Grounding uses retrieval over historical Uber replies, but ~95% of those replies are generic templates ("send us a DM"), not real resolutions.**
The actual resolution of any given complaint happens off-platform (private DM), invisible in this dataset. This was discovered early during data exploration and reshaped what "grounding" honestly means for this project: the system is grounded in Uber's historical *triage/acknowledgment pattern*, not in real resolution content. This is stated explicitly rather than implied to be something it isn't.

**8. Chose Groq (and briefly OpenRouter) over OpenAI, due to budget constraints, not technical preference.**
OpenAI's API requires paid credits with no meaningful free tier; Groq offers a genuine free daily token allowance. This introduced real friction (a 200,000 tokens/day cap was hit mid-evaluation, requiring incremental/resumable evaluation scripts) but was the only viable option without a project budget.

**9. Evaluation scripts (`run_full_eval.py`, `llm_judge.py`) save progress incrementally and skip completed rows on re-run.**
A direct consequence of Decision 8 — hitting a daily token cap mid-run is a real risk on a free tier, and losing completed work to a crash or rate limit would have been costly. Built resumability in rather than re-running from scratch each time.

**10. Rider/rides-only filtering is keyword-based and imperfect — not fixed retroactively after being caught.**
During golden-set labeling, several UberEats-related and driver-partner-related messages were found to have slipped through the filter (at least 7 confirmed instances). Rather than iterating the filter until it caught 100% of these, the decision was to document the leakage rate as an honest, quantified limitation — a perfect filter would have consumed disproportionate time relative to what it would improve in the final system's core behavior.

**11. Escalation recall (36.54%) is reported as-measured, including 8 false negatives caused by a since-identified parsing bug.**
The escalation parser originally defaulted to `auto_handle` on any unparseable LLM response — the wrong direction for a safety-relevant fallback. This bug was identified during failure analysis. The decision was to report the recall number as actually measured (including the bug's effect) rather than quietly re-running only the affected rows and presenting a cleaner number, since the evaluation run as submitted reflects the unpatched behavior and should be described as such.

**12. Used 5,000-example subsample (not the full ~35,000 cleaned rider complaints) for the retrieval index.**
Per the assignment's explicit guidance that a subsample is expected and encouraged, and to keep embedding/indexing time reasonable during iterative development. `random_state=42` fixed for reproducibility.

**13. LLM-judge rubric scores on relevance, grounding, helpfulness, and safety — four dimensions, not a single overall score.**
A single quality score would hide which dimension is actually weak. Separating dimensions surfaced a real, specific finding during human-agreement validation: relevance/grounding agreement (90%) was lower than helpfulness/safety agreement (100%), which would have been invisible behind one blended number.

**14. Human-agreement sample size (n=30) was fixed pragmatically, not statistically derived.**
Given time and token-budget constraints, 30 stratified examples was chosen as a size large enough to compute a meaningful Cohen's κ while remaining feasible to hand-score in one sitting. This is explicitly noted as a limitation in the report rather than treated as a fully sufficient sample.

**15. Repo reorganized into `src/` (core pipeline) and `eval/` (data prep + evaluation) after initial flat development.**
Development started with all scripts in one flat directory for speed of iteration. Reorganized once the pipeline was functionally complete, separating "the product" (`src/`) from "how it was built and tested" (`eval/`) for repo clarity — done via `git mv` where possible to preserve file history.
