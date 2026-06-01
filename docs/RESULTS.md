# Results

## Local Official-like Validation

The following values were obtained from the local full validation run:

```text
judge_full_accuracy_20260516/medai_summary_fixed.json
```

| Method | Cases | Semantic cases | Instance cases | Semantic DSC | Semantic NSD | Instance F1_50 | Instance DSC-TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BiomedParse v2 official baseline | - | - | - | 0.8000 | 0.8829 | 0.4609 | 0.6882 |
| Ours | 2144 | 1458 | 686 | 0.8272 | 0.8925 | 0.4904 | 0.7152 |

Notes:

- The validation run used an official-like evaluator.
- The BiomedParse v2 row reports the official all-data baseline released by the challenge organizers.
- Some evaluator compatibility handling was needed locally for NumPy 2.0 `np.Inf` behavior.
- The final selected inference/refinement stack used the processed state-dict-only fine-tuned checkpoint.
- Final hidden-test scores may differ from local validation scores.
