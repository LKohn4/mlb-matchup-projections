# Model scorecard (2026 MLB regular season)

Walk-forward backtest, 3 folds, 86,197 test plate appearances. Best: **Gradient boosting**.

| Model | Log loss | Brier | Skill vs league % | Skill vs log5 % |
| --- | --- | --- | --- | --- |
| League average | 1.4549 | 0.7051 | 0.0000 | -1.2153 |
| Log5 | 1.4373 | 0.6967 | 1.2153 | 0.0000 |
| Random forest | 1.4344 | 0.6958 | 1.4085 | 0.1932 |
| Gradient boosting | 1.4336 | 0.6956 | 1.4677 | 0.2524 |
| Neural network | 1.4381 | 0.6969 | 1.1549 | -0.0604 |
