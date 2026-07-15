## Definition of "good enough"
Before we build, we agree this project is good enough when:
- Held-out test accuracy beats the naive baseline (~33%, always guessing
  the majority price category) by a meaningful margin.
- 5-fold cross-validation mean accuracy is stable (low fold-to-fold
  variance), confirming the held-out score isn't a lucky split.
- Target achieved: ~68% held-out accuracy, k-fold mean 0.684 (std dev 0.033) —
  consistent across folds and well above the ~33% naive baseline.


Goal: Predict whether a Texas home is low-priced, mid-priced, or high-priced based on its physical characteristics — not the exact dollar amount, just which price bucket it falls into.
Steps, in order:

Load and clean the data — pulls in the Texas housing CSV, keeps only 9 relevant columns (type, sub-type, size, stories, beds, baths, garage, year built, and price), and drops any rows with missing values.
Create the target categories — splits listPrice into three equal-sized groups (tertiles) using pd.qcut, labeled low/mid/high-priced. This is what turns it into a classification problem instead of a regression problem — the model predicts a category, not a number.
Split the data — 80% train, 20% held-out test, with stratification so all three price categories are proportionally represented in both sets.
Preprocess the features — numeric columns (sqft, beds, baths, etc.) get standardized (scaled to a consistent range); categorical columns (type, sub_type) get one-hot encoded (turned into 0/1 columns per category).
Train a logistic regression classifier — a LogisticRegression model learns to map the preprocessed features to one of the three price categories.
Validate with 5-fold cross-validation — before trusting the result, the training data gets split 5 different ways to check the model's accuracy is stable and not just a lucky split (mean accuracy ~68%, std dev ~3.3%).
Evaluate on the held-out test set — the final, untouched 20% is used to get one honest accuracy number, plus precision/recall/F1 per category.

Bottom line: it's a 3-class classifier, not a price predictor — it tells you "this house is probably mid-priced," not "this house is probably $412,000."
