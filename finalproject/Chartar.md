## Definition of "good enough"
Before we build, we agree this project is good enough when:
- Held-out test accuracy beats the naive baseline (~33%, always guessing
  the majority price category) by a meaningful margin.
- 5-fold cross-validation mean accuracy is stable (low fold-to-fold
  variance), confirming the held-out score isn't a lucky split.
- Target achieved: ~68% held-out accuracy, k-fold mean 0.684 (std dev 0.033) —
  consistent across folds and well above the ~33% naive baseline.
