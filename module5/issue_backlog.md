# Module 5 — Issue Backlog (Walking Skeleton for Module 6)

These 3–5 issues describe the *first working version* of the Texas Home Price
Predictor: an end-to-end pipeline that runs, not yet a good model. Tuning and
improvement happen in Module 7 (Iteration 2).

---

### Issue 1: Load and inspect the Texas housing dataset
Read the dataset CSV into pandas. Check the shape (rows/columns), the data
types of each column, and how many missing values exist. Goal: understand
what we're working with before touching it.

### Issue 2: Clean and preprocess the data
Drop unusable columns (IDs, free-text description fields, image references).
Handle missing values. Encode categorical features (e.g., home type) into a
numeric form. Split into training and test sets.

### Issue 3: Build the naive baseline
Implement a "predict the mean price" baseline. This is the ruler we compare
everything else against, per our Definition of Good Enough.

### Issue 4: Build and train a first Keras model (end-to-end)
A simple dense network. The goal here is not accuracy — it's proving the
full pipeline runs from raw data to a prediction without breaking.

### Issue 5: Add k-fold validation and report MAE
Wire up k-fold cross-validation. Compute validation MAE and compare it
against the naive baseline from Issue 3. This is the first real read on
whether we're anywhere near "good enough."
