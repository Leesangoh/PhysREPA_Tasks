# Functional Significance Verdict

- task: `push`
- chunk_len: `8`
- split counts: `{"central_total": 589, "ood_total": 911, "train": 412, "val": 88, "iid_test": 89}`
- figure: `/home/solee/physrepa_tasks/artifacts/figures/functional_significance_push_ood_iid_vs_ood.png`

| Representation | Model | Layer | IID $R^2$ | OOD $R^2$ | OOD gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `vjepa_last` | `large` | `23` | 0.9193 ± 0.0002 | 0.8916 ± 0.0005 | 0.0277 ± 0.0007 |
| `videomae_best` | `videomae_large` | `23` | 0.9181 ± 0.0015 | 0.8903 ± 0.0004 | 0.0278 ± 0.0019 |
| `vjepa_pez` | `large` | `11` | 0.9071 ± 0.0010 | 0.8794 ± 0.0004 | 0.0277 ± 0.0014 |
| `dino_mid` | `dinov2_large` | `15` | 0.8276 ± 0.0035 | 0.7955 ± 0.0041 | 0.0321 ± 0.0006 |
| `dino_last` | `dinov2_large` | `23` | 0.8016 ± 0.0018 | 0.7586 ± 0.0008 | 0.0430 ± 0.0019 |

## Current ranking

- best OOD representation: `vjepa_last` (`large`, layer `23`)
- claim under test: `PEZ-aligned representations are more control-relevant under hidden physics variation`.

## Interpretation

- Supported: video-pretrained backbones are substantially more control-relevant than the static-image baseline under hidden-physics shift.
- Not supported: the mid-depth V-JEPA PEZ layer is not the best layer for Push action-chunk prediction. Within V-JEPA, the last layer slightly outperforms the PEZ layer on both IID and OOD splits.
- Nuance: `vjepa_last` and `videomae_best` are effectively tied on OOD action-chunk prediction (`0.8916` vs `0.8903`), while both are clearly stronger than `dino_mid` (`0.7955`) and `dino_last` (`0.7586`).
- OOD robustness pattern: `vjepa_last`, `videomae_best`, and `vjepa_pez` have nearly identical OOD gaps (`~0.0277`), whereas DINOv2 shows larger degradation, especially at the last layer (`0.0430`).

## Bottom line

The functional-significance experiment partially closes the reviewer's
``so-what'' objection, but in a narrower way than originally hoped. It shows
that video pretraining yields more control-relevant Push representations than a
static-image backbone under hidden-physics variation, yet it does not show that
the PEZ-aligned intermediate layer is itself the best control layer. The most
defensible conclusion is therefore:

> PEZ-aligned layers are strongly diagnostic for manipulation kinematics and
> interaction magnitude, but downstream action relevance in Push is maximized at
> late video-model layers rather than at the PEZ layer itself.
