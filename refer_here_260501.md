# 260413 — Meeting with Prof. Han & Prof. Mo

> 날짜: 2026-04-13 (월)
> 참석: 이상오, 한욱신 교수님, 모상우 교수님
> 주제: PhysProbe 진행상황 발표 — PhysREPA에서 PhysProbe로의 전환, 벤치마크 구성, 현재 상태
> 발표자료: [[Presentations/PhysProbe_Seminar_260412.pptx]]

---

## 사전 논의: VAP 특허

미팅 시작 전 모상우 교수님이 VAP 논문으로 특허 출원 의향을 밝힘.
- 모상우 교수님: 과제에서 특허를 내야 하는 상황. 신임 교원 3천만 원 연구비 소진 필요 + 향후 과제 제안서에 특허 실적이 유리.
- 한욱신 교수님: 지분 등 세부사항 상의 필요하지만 기본적으로 동의.
- **Action**: 두 교수님이 별도로 지분/비용 관련 협의 예정.

---

## 발표 요약

PhysREPA → PhysProbe 전환 경위 및 현재 진행 상황을 발표. 다음 순서로 진행:

1. **PhysREPA Recap** — V-JEPA 2의 Physics Emergence Zone을 GR00T Action DiT에 REPA-style alignment하는 proposal (3월 16일 발표 내용)
2. **BridgeData V2 실험 결과** — 4가지 alignment variant(MeanPool/TS-Align × ViT-L/ViT-G) 모두 alignment은 수렴했으나(cosine sim 0.93-0.97), downstream MSE는 baseline과 동일(0.200-0.236). Alignment이 학습되었지만 action prediction에 도움이 안 됨
3. **Root Cause Analysis** — (1) BridgeData V2는 evaluation이 sim인데 학습은 real-world라 gap 존재, (2) BridgeData task 자체가 physics understanding을 크게 요구하지 않음(pick & place 수준), (3) PEZ 논문은 free-body motion만 다뤘고 manipulation contact dynamics에 PEZ가 존재하는지 검증된 바 없음
4. **PhysProbe 전환** — Alignment(Step 1) 전에 probing study(Step 0)가 먼저 필요. 이것만으로도 독립적 contribution 가능
5. **PEZ 논문 방법론** — V-JEPA 2 frozen representation에서 layer별 linear probe로 R²를 찍어 PEZ를 찾는 방식. 기존 논문은 Kubric 합성 공 영상에서 speed/acceleration/direction만 probing
6. **PhysProbe 벤치마크** — IsaacLab에서 6개 manipulation task(Push, Strike, PegInsert, NutThread, Drawer, Reach) 구성, 총 ~10,800 episodes, 27개 physics GT field 로깅 완료
7. **현재 상태** — 데이터 수집 완료, V-JEPA 2 feature extraction 진행 중

---

## 교수님 질문 & 피드백

### 한욱신 교수님

**Q1. PEZ 원 논문은 로봇과 관계가 있는 건가?**
- "그전 논문은 어떤 물리량인가? 로봇하고는 상관없는 거네."
- → 답변: 맞습니다. PEZ 논문은 단순히 공이 굴러가는 free-body motion에서의 speed, direction, acceleration만 분석. 로봇 manipulation과는 직접 관련 없음.
- 교수님 반응: "저게 꼭 로봇에 쓰려고 만든 모델은 아니니까, 그냥 월드 모델의 속성으로 이렇게 한 거라고 보면 되겠네."

**Q2. IsaacLab 시뮬레이터가 물리 법칙을 잘 이해하고 있는 건지?**
- → 답변: IsaacLab은 물리 엔진 기반 시뮬레이터라서 물리 법칙에 따라 동작. 시뮬레이션이기 때문에 friction, mass 등 ground-truth를 전부 수집할 수 있어서 이 환경을 선택함.

**Q3. 모델이 물리 법칙을 잘 이해한다면 사람보다 더 잘할 수 있는 건가?**
- "된다면 물리법칙이… 우리가 정밀하게 언제 요게 된다 이렇게 예측하기 되게 사람 어려울 것 같은데, 이 모델은 그렇게 예측할 수가 있을 것 같기도 하고."
- → 답변: 네, 만약 probing이 잘 된다면 사람이 implicit하게만 이해하는 물리량을 모델은 더 정밀하게 encode하고 있을 수 있음.

**Q4. 어느 정도까지 물리 법칙을 잘 캡처해야 의미가 있는 건지? 평가 metric에 대한 우려**
- "밸류 예측한 다음에 그냥 디스턴스 계산해서 그렇다, 이걸로 하면 되는 건가?"
- "시간 타임시리즈로 쭉 나올 것 같은데, 완전히 align돼서 매칭하면 계속 다 틀릴 텐데, 시간을 약간 오차를 주면 거의 잘 될 수도 있는 움직임일 수도 있잖아."
- → 핵심 우려: per-frame MSE/R²로 평가하면 시간축 약간의 shift만으로도 에러가 크게 나올 수 있음. 실제로는 trajectory 패턴이 비슷한데 metric 상으로는 나쁘게 나올 위험.
- **제안: Dynamic Time Warping (DTW)** 같은 시간축에 유연한 metric도 함께 사용할 것.
- → 답변: 일단 MSE/R²로 시작하되, 잘 안 나오면 DTW 등을 적용해보겠음.

**Q5. VLM(월드 모델)이 생성한 비디오와 시뮬레이터 결과를 비교 검증하는 건 없나?**
- "월드 모델에서 만약에 생산한 비디오 있으면, 그거하고 아이작 시뮬레이터 거 두 개 사이에 체크하는 건 없냐?"
- → 답변: 그건 test-time adaptation 느낌. 현재 scope 밖이지만 관련 연구는 있음 (시뮬레이션과 WM output을 비교해서 physically plausible한 scene을 생성하는 방법론).

### 모상우 교수님

**Q1. Ego-centric view vs. Exo view 비교가 필요하지 않나?**
- "비디오 월드 모델이 에고센트릭 뷰가 있을 것 같고 엑소 뷰가 있을 것 같은데… 우리가 하고 있는 매니플레이션은 에고 비디오 모델로 해야 되는 게 아닌가."
- "아마 V-JEPA는 저런 매니플레이션을 못 잡는다. 근데 에고 월드 모델이 잡을 수 있다. 뭐 그런 식의 결과가 나올 수 있을 것 같고요."
- → 흥미로운 포인트. V-JEPA 2는 인터넷 영상(주로 exo view) 학습이라 manipulation table-top view에서 약할 수 있음. Ego-centric world model과의 비교는 추가 분석 방향으로 고려 가능.

**Q2. 분석 결과의 killer application — TTA(Test-Time Adaptation)로 연결**
- "REPA까지 안 가도 이 논문의 scope에서 할 수 있는 거는 이걸 월드 모델을 가지고 verify해서 test-time adaptation 하려는 것들이 많잖아요. 이 physics layer를 통해서 detect하면 더 잘 된다, 막 그런 식의 결론이 나오면 좋을 것 같거든요."
- "사람들이 분명히 이런 분석 논문에 대해서 항상 물어보는 게 '이거 분석해서 어디다 쓸 거냐.' Killer app으로 뭔가 보여줄 수 있으면 좋을 것 같아요."
- → **핵심 피드백**: 순수 분석만으로는 약할 수 있으니, downstream application (특히 TTA)과 연결되는 실험이 있으면 논문 impact가 훨씬 좋아질 것.

**Q3. Embedding space trajectory 분석 제안**
- "비디오 trajectory가 쭉 있으니까, 거기서 layer별로 embedding을 뽑을 거잖아요. 그 embedding space의 trajectory를 보고서 이게 물리적으로 맞는 건지 틀린 건지 판단할 수 있으면…"
- "저 latent trajectory를 보면서 그걸 가지고 guidance를 주는 게 도움이 될 수도 있겠다는 생각이 들거든요."
- → **새로운 분석 방향**: Layer별 embedding trajectory를 시각화/분석하여, physics-aware layer에서의 trajectory가 물리적으로 consistent한 패턴을 보이는지 확인. 이걸 guidance로 활용하는 것까지 연결 가능.

---

## Key Takeaways & Action Items

1. **방향성 긍정적** — 두 교수님 모두 PhysProbe 전환에 동의. 분석 자체의 가치를 인정하되, downstream application 연결을 강조.
2. **Killer App 필요 (모상우 교수님)** — 순수 probing 분석만으로는 "어디다 쓸 거냐" 질문에 약함. TTA나 physics-aware verification 같은 downstream application을 scope에 포함할 것을 권고.
3. **Metric 주의 (한욱신 교수님)** — Per-frame MSE/R²만으로는 시간축 shift에 취약. DTW 같은 time-warping metric도 함께 사용할 것.
4. **Embedding trajectory 분석 (모상우 교수님)** — Layer별 latent trajectory가 물리적으로 meaningful한 패턴을 보이는지 시각화/분석. 새로운 분석 축으로 추가.
5. **Ego vs. Exo view 비교 (모상우 교수님)** — V-JEPA 2(exo 학습)가 manipulation table-top에서 약할 수 있다는 가설. Ego-centric world model과의 비교를 추가 분석으로 고려.
6. **VAP 특허** — 모상우 교수님 주도로 진행, 한욱신 교수님과 지분 협의 예정.
7. **다음 단계** — Feature extraction 완료 후 Tier A probing 실행. MSE/R² 결과를 먼저 보고, 필요 시 DTW 적용 및 embedding trajectory 분석 추가.

---

## Slack 후속 피드백 (한욱신 교수님, 미팅 후 15:16~20:25)

미팅 종료 후 한욱신 교수님이 `#robotics` 채널에 구체적인 실험 설계 피드백을 올려주심. Pseudo code와 다이어그램까지 포함된 매우 상세한 피드백.

### F1. DTW의 함정과 대안 metric (14:45)

미팅 중 제안했던 DTW에 대해 상세 분석글을 작성해주심. 핵심: **물리 법칙 검증 시 DTW를 무턱대고 쓰면 시간 왜곡이 물리 법칙을 파괴함.** 4가지 대안 제시:
1. **Contact-Event Alignment** ★추천 — 접촉 순간을 t=0으로 강제 동기화, 시간을 늘리지 않으므로 충격량(FΔt)과 마찰 감속 곡선을 정밀 비교 가능
2. **Phase-Space Analysis** — 시간 축 자체를 제거, (위치 x, 속도 v) 위상 공간에서 궤적 비교
3. **Windowed DTW (Sakoe-Chiba Band)** — 시간 왜곡을 최대 1-2프레임 이내로 제한
4. **Fréchet / Chamfer Distance** — 순수 공간 거리 측정, penetration 등 공간 오차 검출용

### F2. 접촉 전/중/후 구간별 PEZ 분석 (15:16)

> 에피소드의 "접촉 전 → 접촉 중 → 접촉 후" 구간별로 PEZ가 달라지는가?

```
기존 클립 (N_clips, 16, 384²)     contact_flag로 3구간 분류        각 구간 별도 Probe
+ contact_flag GT          →     Pre: flag=0 연속           →    R²_pre
                                  During: flag=1                  R²_during
                                  Post: flag=0 (접촉 후)          R²_post
```
![[image 1.png|674]]
- `contact_flag` GT를 이용해 에피소드를 Pre-Contact / During-Contact / Post-Contact 클립으로 분리
- 각 구간에서 독립적으로 probing → R²_pre, R²_during, R²_post 비교
- **만약 R²_during만 높고 R²_pre/R²_post가 낮으면 → contact-specific PEZ의 강력한 evidence**

### F3. CKA로 cross-task representation similarity (15:18)

> Push와 Strike의 CKA가 PEZ 레이어에서 특별히 높고, non-PEZ에서는 낮다면??

```
Push Layer l (N_push, d)  ─┐
                           ├─→  Linear CKA  →  CKA score [0,1]  →  3D Heatmap
Strike Layer l (N_strike, d) ─┘                per (task₁, task₂, layer)      task×task×layer
```
![[image (1).png]]
- **Centered Kernel Alignment (CKA)** 을 사용하여 두 task의 같은 layer feature 간 representation similarity 측정
- Layer별 CKA score를 task×task×layer 3D heatmap으로 시각화
- GitHub: `ryusudol/Centered-Kernel-Alignment`
- 기존 계획(weight cosine similarity / principal angle)보다 더 직접적인 분석 방법

### F4. 물리 파라미터가 PEZ에 미치는 영향 (15:19)

> 에피소드별로 랜덤화된 물리 파라미터(mass, friction 등)가 PEZ 인코딩에 영향을 주는가?

**실험 A: 물리 조건별 분할 Probing**
```
episodes.jsonl              에피소드를 friction 구간별 분할
(에피소드별 랜덤화된     →   Low / Med / High              →   각 구간 별도 Probe   →  R² vs friction
friction, mass, ...)                                           R²_low, R²_med, R²_high     물리 조건에 체계적 변화?
```

**실험 B: 물리 파라미터 자체를 Probe Target으로**
```
V-JEPA 2 Layer l Feature    새 label: friction
(N, 1408)               +   (N, 1) — 에피소드별 상수    →   Ridge Probe   →   R² (영상에서 마찰계수를 decode 가능?)
```
![[image 2.png]]

- 실험 B는 우리 **Tier C (latent physics)** 실험과 정확히 일치

### F5. Frame Shuffle 실험 (15:23)

> 같은 에피소드 내에서 프레임 순서를 무작위로 섞은 뒤 probing → R²가 유지되면 시간적 인과가 아닌 정적 상관에 의존?

- 16프레임의 순서를 랜덤으로 뒤섞은 뒤 동일하게 probing
- R²가 유지되면: 모델이 시간적 dynamics가 아닌 **정적 visual 상관**에 의존하고 있다는 뜻 → PEZ의 실질적 의미에 의문
- R²가 크게 떨어지면: 모델이 **시간적 인과 관계(temporal causality)**를 실제로 인코딩하고 있다는 증거

### F6. Probing Pseudo Code (20:19)

교수님이 직접 실험 파이프라인 pseudo code를 작성해주심:
```python
for task in ["push", "strike", "peg", "nut", "drawer", "reach"]:
    for layer in range(1, 41):   # ViT-G: 40 layers
        X = load_features(task, layer)    # (N, 1408)
        for variable in ["kinematics", "distance_timing"]:
            y = load_physics_gt(task, variable)  # (N, 1)
            groups = load_episode_ids(task)       # (N,)
            cv = GroupKFold(n_splits=5, groups=groups)
            r2_scores = []
            for train_idx, test_idx in cv.split(X, y, groups):
                probe = Ridge(alpha=1.0)
                probe.fit(X[train_idx], y[train_idx])
                r2 = probe.score(X[test_idx], y[test_idx])
                r2_scores.append(r2)
            save_result(task, layer, variable, mean(r2_scores))
```

---

*260413 미팅 기록 + Slack 후속 피드백*
