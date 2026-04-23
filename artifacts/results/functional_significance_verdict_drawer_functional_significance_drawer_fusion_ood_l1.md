# Functional Significance Verdict

- task: `drawer`
- chunk_len: `8`
- split counts: `{"eligible_total": 996, "central_total": 696, "ood_total": 300, "train": 487, "val": 104, "iid_test": 105}`
- figure: `/home/solee/physrepa_tasks/artifacts/figures/functional_significance_drawer_fusion_ood_l1_iid_vs_ood.png`

| Representation | Model | Layer | IID $R^2$ | OOD $R^2$ | OOD gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `videomae_best` | `videomae_large` | `23` | 0.9057 ± 0.0004 | 0.9197 ± 0.0009 | -0.0140 ± 0.0005 |
| `vjepa_fusion` | `large` | `11+23` | 0.8968 ± 0.0006 | 0.9125 ± 0.0002 | -0.0157 ± 0.0005 |
| `vjepa_last` | `large` | `23` | 0.8922 ± 0.0006 | 0.9076 ± 0.0004 | -0.0154 ± 0.0004 |
| `vjepa_pez` | `large` | `11` | 0.8911 ± 0.0009 | 0.9059 ± 0.0003 | -0.0148 ± 0.0008 |
| `dino_best` | `dinov2_large` | `15` | 0.8735 ± 0.0000 | 0.8893 ± 0.0007 | -0.0158 ± 0.0006 |

## Current ranking

- best OOD representation: `videomae_best` (`videomae_large`, layer `23`)
- claim under test: `PEZ-aligned representations are more control-relevant under hidden physics variation`.
