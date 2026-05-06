| Method | Costs in training | Costs at inference | Frequency | Adversarial | Special output | Decision |
|---|---|---|---|---|---|---|
| CE | no | no | no | no | no | argmax |
| CE + cost-min inference | no | yes | no | no | no | cost-min |
| Menon logit adjustment | no | no | yes | no | no | argmax |
| Cost-weighted CE | row mean | no | no | no | no | argmax |
| cost-sensitive regularized CE | expected cost | no | no | no | no | argmax |
| CSADA | target selection | no | no | yes | no | argmax |
| NICME | pairwise margin + expected cost | no | no | no | no | argmax |
