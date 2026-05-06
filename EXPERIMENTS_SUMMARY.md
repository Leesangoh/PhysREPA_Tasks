# PhysProbe ICLR 2027 — 실험 요약

작성: 2026-05-06

이 문서는 지금까지 진행한 모든 실험을 **목적**, **방법**, **결과** 중심으로 쉽게 정리한 것입니다.

---

## 큰 그림: 무엇을 묻는가

V-JEPA 2 같은 **video self-supervised model** 이 manipulation video 를 보고 "물리" 를 어떻게 표현하는지를 알고 싶다.

- **Dataset**: Isaac Lab 6 manipulation tasks (push / strike / drawer / reach / peg_insert / nut_thread), 12,100 episodes
- **Method**: Layer-wise linear probing — 모델의 각 layer 에서 다양한 물리량을 readout
- **Compared models**: V-JEPA 2 (video SSL), VideoMAE-L (video SSL, 다른 objective), DINOv2-L (image SSL only)

---

## Part 1: 첫 번째 paper round (Per-timestep probing) — 13개 실험

원래 매 frame 단위로 물리량을 직접 readout 하는 방식. ICLR 26 submission level 까지 갔으나 advisor feedback 으로 paradigm 바꿈.

### 1. PEZ 식별 (Physics Encoding Zone)
- **목적**: V-JEPA 가 physics 정보를 담나? 어디 layer 에?
- **방법**: 24 layer × 6 task × 여러 target (속도/접촉/힘 등) 로 linear probe
- **결과**: Mid/late layer (L17-21) 에 contact-physics 집중. R² 0.7-0.9. → "**PEZ** 존재"

### 2. F1-a Contact-Event Alignment
- **목적**: PEZ 가 단순 평균 정보인가, 아니면 **event-driven** 인가?
- **방법**: 접촉 발생 순간 (τ=0) 기준 ±8 frame 의 latent 속도 변화 측정
- **결과**: τ=0 직후 latent 속도 급증. 모델이 contact event 시간적으로 인식.

### 3. F1-b Phase-space Geometry
- **목적**: latent 자체 (z) + 변화 (Δz) 의 기하적 모양
- **방법**: 각 layer 의 latent 궤적 + 1차 차이 분석
- **결과**: 깊은 layer 일수록 빠르게 움직이고 smooth. 접촉 중 phase-space 더 넓게 퍼짐.

### 4. F2 Phase-conditional Probing (pre/during/post contact)
- **목적**: 접촉 전/중/후 representation 다르게 작동하나?
- **방법**: window 를 3 phase 로 나눠 각각 probe
- **결과**: Drawer contact=0 일 때 force R²=-0.06 (당연), contact=1 R²=0.83. Push/Strike velocity 는 contact 무관 robust.

### 5. F3 Cross-task CKA
- **목적**: PEZ 가 task-general 인가 task-specific 인가?
- **방법**: 6 task pairwise representation similarity
- **결과**: L3 까지 task-shared (CKA=0.90), L23 task-specific (CKA=0.36). 깊은 layer 가 task-aware.

### 6. F4-A Physical-condition Split ⭐
- **목적**: friction, mass 같은 randomized parameter 가 contact decodability 를 modulate 하나?
- **방법**: parameter 를 Low/Med/High 로 나눠 각 구간 내에서 probe, ΔR² 측정
- **결과**: **Strike static_friction L2 ΔR²=+0.565** (강한 modulation). 22/22 (task,param) coverage.
- **의의**: V-JEPA 가 friction 자체는 못 읽지만, 그 **downstream effect** 는 강하게 반영.

### 7. F4-B Physics-as-target (Negative Result)
- **목적**: Physics parameter (friction, mass) 자체가 latent 에서 직접 decode 되나?
- **방법**: parameter 를 target 으로 ridge regression
- **결과**: 모든 task R² ≤ 0. **직접 decode 불가**. F4-A 의 미묘함 강화.

### 8. F5 Frame Shuffle ⭐
- **목적**: V-JEPA 가 input temporal coherence 사용하는가? (causal-style intervention)
- **방법**: 16-frame window 를 임의 순서로 섞어서 forward → 같은 probe
- **결과** (push + strike):
  - velocity ΔR² = -0.30 (대대적 망가짐)
  - acceleration ΔR² = -0.11
  - contact ΔR² ≈ -0.05 (거의 안 망가짐)
- **의의**: V-JEPA 가 **temporal coherence 를 actively 사용**. 동적 정보엔 필수, 접촉엔 거의 무관.

### 9. M1 R3M Baseline
- **목적**: image-only SSL (R3M ResNet50) 대비 V-JEPA 우위 있나?
- **방법**: R3M 5-stage feature 로 같은 probe
- **결과**: V-JEPA 가 9 cell (3 task × 3 target) 모두 우위. 가장 큰 차이 drawer ee_acceleration -0.293.

### 10. M3 Embedding Trajectory (대량 분석)
- **목적**: 다각도 representation 구조
- **방법**: 26 stats (path length, intrinsic dim, CKA, RSA, Koopman 등)
- **결과**: 깊을수록 path 더 길고, position 은 모든 layer 에서 perfect, velocity L8 saturate, acceleration L20 까지 정제.

### 11. Time-only Baseline
- **목적**: 시간 leakage 검증
- **방법**: feature = [t/T] (1차원 시간) 로 probe
- **결과**: Push/Strike phase R²~0.41 (leakage 큼), Reach phase ~0.03 (leakage 없음).

### 12. Cross-task Transfer
- **목적**: A task 에서 학습한 probe 가 B task 에 가나?
- **방법**: 6×6 transfer matrix
- **결과**: Push ↔ Strike 잘 transfer, **Drawer outlier** (transfer collapse).

### 13. Bootstrap CI (statistical defense)
- **목적**: Point estimate 만으로 reviewer 통계 비판 차단
- **방법**: Episode-level bootstrap 1000 회
- **결과**: 3,120 행 bootstrap CI. F5/F4-A/transfer/F2 모두 95% CI 0 제외.

---

## Paradigm Shift (2026-05-02): "매 timestep 예측 자체가 한계"

**Advisor Feedback**: "매 timestep 예측 잘 못해도, 전체 동작이나 미래 예측엔 정보 충분할 수 있다. timestep 예측이 좋은 방법 아닐 수도."

**문제 인식**:
- 기존 per-timestep probe → V-JEPA pretraining objective (future latent self-prediction) 와 circular
- "V-JEPA 가 V-JEPA trained-for 를 잘 함" 이상의 의미가 없음

**해결 방향** (Codex 와 3 round 논의 후):
- **분석 단위 자체** 를 frame → segment-level summary 로 바꿈
- 매 frame readout 이 아니라 **window 내 일어난 일의 요약** (mean/integral/peak/duration 등) 을 readout
- "PEZ 가 어느 time scale 의 어떤 functional family 의 sufficient statistic 인가?" 로 질문 재정의

---

## Part 2: 새 paradigm — Segment-level Functional Probing

### 14. Functional Target Infrastructure
- **목적**: per-frame target 을 여러 시간 스케일의 **functional summary** 로 변환
- **방법**: 20 functional 정의 (continuous: mean/max/integral/peak_to_peak/std/total_variation/peak_time_frac; binary: positive_fraction/event_count/duration_above; progress: net_change/monotonic_score; distance: net_decrease 등)
- **Multi-scale**: short (16f) / med (32f) / long (64f) / prefix (full causal)
- **Effect-class targets**: 4 categorical regimes (approach_speed, contact_involvement, peak_force, force_integral)

### 15. V-JEPA Variant A 전체 6 task functional sweep ✅
- **방법**: 60 functional × 24 layer × 6 task = ~7,500 cells
- **핵심 발견**:

| Family | Push | Strike | Reach | Drawer | Peg | Nut |
|---|---|---|---|---|---|---|
| Magnitude | 0.69 | 0.81 | 0.90 | 0.94 | 0.77 | 0.52 |
| Variability | 0.64 | 0.76 | 0.87 | 0.92 | 0.77 | 0.48 |
| **Timing** | **0.51** | **0.71** | **0.28** | **0.54** | **0.16** | **0.07** |
| Occupancy | 0.73 | 0.87 | — | 0.84 | — | — |
| Progress | 0.13 | 0.42 | 0.76 | **0.96** | 0.80 | 0.65 |

**5 paper-quality observations**:
1. Magnitude family 가장 일관 (모든 task L17-18 PEZ peak)
2. **Timing family 일관되게 worst** (R²=0.07-0.71) — V-JEPA 는 "WHAT 일어났나" 잘, "WHEN" 약함
3. **Push progress 0.125 vs Drawer progress 0.963** — drawer 의 articulated state 매우 강하게 인코딩
4. Progress family layer-of-peak task-specific (push L10 / strike L12 / nut L20)
5. Nut_thread 모든 family 약함 (always-in-contact + 미세 angular motion 어려움)

---

## Part 3: Cross-model 비교 (Phase C, 진행 중)

### 16. VideoMAE-L 추출 + probe (video SSL family replication)
- **목적**: V-JEPA 결과가 V-JEPA-specific 인지, 더 일반적 video pretraining 효과인지
- **현재까지**: push (Variant A + B 모두 완료), drawer/strike 진행 중

### 17. DINOv2-L 추출 + probe (image-only baseline)
- **목적**: Image SSL 과 비교해 video pretraining 의 진짜 advantage
- **완료**: push, drawer, strike, reach, peg_insert (5/6) — nut_thread OOM partial

### 18. 핵심 cross-model 발견 (paper main contribution)

#### 18-1. **5-model push 비교** (모든 5 model push 완료):

| Target | V-JEPA A | V-JEPA B | VideoMAE A | VideoMAE B | **DINOv2 A** |
|---|---|---|---|---|---|
| ee_velocity__mean | 0.980 | 0.980 | 0.977 | 0.981 | **0.894** |
| **ee_acceleration peak2peak** | 0.802 | 0.820 | 0.774 | 0.800 | **0.451** |
| obj_velocity__mean | 0.933 | 0.924 | 0.904 | 0.902 | **0.763** |

→ **VideoMAE ≈ V-JEPA**, **DINOv2 만 outlier**.
→ **"Gap 은 V-JEPA-specific 아니라 video vs image pretraining"** — paper main contribution.

#### 18-2. **5-task DINOv2 vs V-JEPA gap matrix**:

Acceleration peak_to_peak gap ordering:
- **reach -0.40** (largest!) ← surprise
- push -0.35
- peg_insert -0.23
- strike -0.09
- drawer -0.02

→ **Gap 이 task 의 visual-abstraction 난이도와 inverse**.
- **Reach** (free motion, 시각 cue 없음) → **video advantage 가장 큼**
- **Drawer** (visually direct articulated state) → image SSL 도 충분

#### 18-3. **Variant A ≈ Variant B** (V-JEPA push 기준):
- Per-temporal-slot factorization (8192-d) 가 spatiotemporal mean (1024-d) 보다 추가 readout gain 없음
- → "Slot factorization 은 not necessary for segment-summary readout"

---

## Paper Main Contribution (현재)

> **"Video SSL pretraining (V-JEPA + VideoMAE) selectively linearizes higher-order dynamics summaries**; image SSL (DINOv2) saturates at a lower ceiling. The advantage scales with task's **visual abstraction difficulty**: reach (free motion, gap 0.40) > push (3D rigid, 0.35) > peg (constrained, 0.23) > strike (visually-direct impact, 0.09) > drawer (visually-direct articulated state, 0.02). First-order motion / contact occupancy / timing show consistently small gaps because they are appearance-readable. Per-temporal-slot factorization does not provide additional readout gain over spatiotemporal mean."

---

## 통계적 안전장치

- Bootstrap CI: episode-level resampling 1000 회 — 3120 row CI library
- Per-fold consistency check
- GroupKFold disjointness assertion
- Cross-review (Claude ↔ Codex) 모든 신규 script
- 12a-12d integrity gates (NaN frac, fold leak, etc)

---

## 진행 상황 (2026-05-06 기준)

### 완료된 probe sweeps (12개):
- V-JEPA A 6/6 ✅
- DINOv2 A 5/6 (nut OOM partial)
- V-JEPA B push 95% (L0-L21 saved)
- VideoMAE A push, VideoMAE B push

### 진행 중 (3 GPU):
- VideoMAE A drawer, VideoMAE B drawer, V-JEPA VarB strike

### 예상 ETA:
- VideoMAE A 5 tasks: ~50h
- VideoMAE B 5 tasks: ~150h (long pole, 6일)
- V-JEPA VarB 5 tasks: ~150h

**Disk 261G stable**.

---

## Out of scope (의도적 제외)

- **Killer-app**: 사용자 결정으로 제외 (ICLR 2027 submission window 내 timing 기준)
- **Drawer 2nd articulated env**: 사용자 결정으로 future work memo
- **F1-c DTW / F1-d Fréchet**: methodology fit X (Codex consensus)
- **Real-video sanity**: P2, optional
- **VC-1 / 다른 video SSL**: 시간 부족

---

## ICLR 27 Oral 도달도 (Codex 자체평가)

- 현재 (5-task DINOv2 + 5-model push): **45-55%**
- 6-task × 5-model 풀 matrix 완성 시 (~6-7일 더): **55-70%**
- 추가 cross-task transfer + permutation null + multi-seed 까지: **70-80%**

(Killer-app 추가 시 +10%, 별도 Drawer 2nd env 시 +5%)
