# Functional Significance Verdict

- task: `drawer`
- chunk_len: `8`
- split counts: `{"eligible_total": 996, "central_total": 696, "ood_total": 300, "train": 487, "val": 104, "iid_test": 105}`
- figure: `/home/solee/physrepa_tasks/artifacts/figures/functional_significance_drawer_ood_iid_vs_ood.png`

| Representation | Model | Layer | IID $R^2$ | OOD $R^2$ | OOD gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `videomae_best` | `videomae_large` | `23` | 0.9057 ± 0.0004 | 0.9197 ± 0.0009 | -0.0140 ± 0.0005 |
| `vjepa_last` | `large` | `23` | 0.8922 ± 0.0006 | 0.9076 ± 0.0004 | -0.0154 ± 0.0004 |
| `vjepa_pez` | `large` | `11` | 0.8911 ± 0.0009 | 0.9059 ± 0.0003 | -0.0148 ± 0.0008 |
| `dino_best` | `dinov2_large` | `15` | 0.8735 ± 0.0000 | 0.8893 ± 0.0007 | -0.0158 ± 0.0006 |

## Current ranking

- best OOD representation: `videomae_best` (`videomae_large`, layer `23`)
- claim under test: `PEZ-aligned representations are more control-relevant under hidden physics variation`.

## Interpretation

- supported: video-pretrained features remain more control-relevant than the
  static-image baseline under a drawer-damping OOD split
- not supported: neither the V-JEPA PEZ layer nor V-JEPA itself is uniformly
  best for offline action prediction on Drawer
- nuance: all four representations score slightly higher on the damping-OOD
  subset than on the central IID subset, which means this split is
  distribution-shifted but not intrinsically harder under the current offline
  action metric

The functional-significance story therefore generalizes across tasks in a
limited but useful way. Push and Drawer both show that video-pretrained
representations are more control-relevant than DINOv2 under hidden-physics
variation. However, the best offline control layer is late rather than PEZ-like,
and VideoMAE-L can surpass V-JEPA~2 Large on Drawer even though V-JEPA is
stronger on the paper's kinematic and force-proxy analysis targets.
