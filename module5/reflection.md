# Reflection: From California Housing to the Texas Home Price Predictor

## Where we stopped on California Housing
We stopped at **Iteration 2 — Scale up / regularize**, but honestly, we
overshot it. The script kept training past the turnaround epoch — the
point where validation MAE stops improving and the model starts
memorizing the training data instead of learning general patterns
(the "confidently wrong" / screen burn-in effect from the Module 4
lecture). We never applied early stopping or `restore_best_weights=True`
to actually catch that point, so the final model we ended up with was
past its best version, not at it.

## What's different this time
- **We're defining "good enough" before building, on purpose.** With
  California Housing, the turnaround epoch was something we discovered
  after the fact, as a lesson. This time, the charter already states the
  definition of good enough up front: validation MAE beats the naive
  baseline, and validation MAE stops improving. That second condition
  is exactly the thing we missed catching last time — so this project
  is a direct chance to catch it on purpose, likely using early stopping.
- **New dataset, same shape.** We're reskinning the same process onto
  the Texas Residential Real Estate Intelligence 2026 dataset, predicting
  home price instead of California district median value. Same tools
  (Keras, k-fold validation), new data.
- **Scope is guarded up front.** Last time scope wasn't really a
  question — it was a guided class script. This time we explicitly
  ruled out feature engineering beyond what's provided and deployment,
  before writing any code, so we don't drift into over-scoping.
- **Process is track-appropriate.** As Prompt Masters, I'm committing
  directly to `main` with Issues for tracking (instructor-confirmed),
  rather than the full Sacred Flow. My AI partner's guardrails are
  documented separately so it stays scoped to one Issue at a time
  regardless of the commit process.
