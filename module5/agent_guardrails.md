# Agent Guardrails: Texas Home Price Predictor

These are the rules that keep my AI partner (Claude) on task for this
project. The goal is to get help without letting the AI wander off and
do side quests I didn't ask for.

## What the AI is allowed to do
- Explain concepts from the course material (regression, MAE, k-fold,
  overfitting, etc.) when I'm confused.
- Help draft or review code for one Issue at a time — data loading,
  preprocessing, the naive baseline, the Keras model, or k-fold validation.
- Suggest fixes for errors I run into, with an explanation of *why* the
  fix works, not just the fix itself.
- Act as a rubber-duck reviewer on my pull requests when I'm working solo,
  as long as I document that review in the PR itself.

## What the AI must never do
- Make changes across multiple Issues at once. One Issue, one branch,
  one change at a time (the One Change Rule).
- Add features, models, or techniques that aren't part of the current
  Issue — no unscoped feature engineering, no deployment code, no NLP
  work on the description field. That's covered by the scope guard in
  charter.md, and it applies to AI-generated code too.
- Commit or push its own changes directly. As Prompt Masters, I commit
  directly to `main` myself (confirmed valid for this track), tracked
  by Issues — but any code the AI writes on my behalf still needs to
  go through me and get reviewed before it lands, not be pushed
  autonomously.
- Rewrite or restructure code I didn't ask it to touch.
- Present unverified claims about the dataset or model performance as
  fact — if it's a guess or an assumption, it should be labeled as one.

## How I'll check the AI's work
- Every piece of AI-assisted work gets a real review before it's
  committed, not a rubber stamp: I quote at least one specific part
  of the change and explain why it's right (or why I changed it),
  documented against the relevant Issue.
- Before committing to `main`, I run the code myself and check that
  the output matches what the Issue asked for.
- If the AI's suggestion touches something outside the current Issue's
  scope, I flag it and either reject it or open a new Issue for it
  instead of committing it in.

## Note on track
I'm on the Prompt Masters track. My instructor confirmed I commit
directly to `main`, tracked by Issues, rather than using the full
Sacred Flow (Issue → Branch → PR → Review → Merge). The AI itself can
still use branches/PRs internally if useful, but that's not required
of me for this project.
