# Functional Significance Verdict

- task: `push`
- chunk_len: `8`
- split counts: `{"eligible_total": 1500, "central_total": 589, "ood_total": 911, "train": 412, "val": 88, "iid_test": 89}`
- figure: `/home/solee/physrepa_tasks/artifacts/figures/functional_significance_push_fusion_ood_l1_iid_vs_ood.png`

| Representation | Model | Layer | IID $R^2$ | OOD $R^2$ | OOD gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `vjepa_fusion` | `large` | `11+23` | 0.9174 ± 0.0015 | 0.8923 ± 0.0007 | 0.0250 ± 0.0019 |
| `vjepa_last` | `large` | `23` | 0.9193 ± 0.0002 | 0.8916 ± 0.0005 | 0.0277 ± 0.0007 |
| `videomae_best` | `videomae_large` | `23` | 0.9181 ± 0.0015 | 0.8903 ± 0.0004 | 0.0278 ± 0.0019 |
| `vjepa_pez` | `large` | `11` | 0.9071 ± 0.0010 | 0.8794 ± 0.0004 | 0.0277 ± 0.0014 |
| `dino_mid` | `dinov2_large` | `15` | 0.8276 ± 0.0035 | 0.7955 ± 0.0041 | 0.0321 ± 0.0006 |

## Current ranking

- best OOD representation: `vjepa_fusion` (`large`, layer `[11, 23]`)
- claim under test: `PEZ-aligned representations are more control-relevant under hidden physics variation`.
