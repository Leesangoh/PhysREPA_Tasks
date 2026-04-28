# PhysProbe — Latent Trajectory Analysis Report

All analyses use Variant A features (V-JEPA 2 ViT-L, 1024-d spatiotemporal-mean pool, 24 residual layers). Per-task we sample 30 episodes (seed=42) and pool windows. Whitening: PCA-whitened per (task, layer) on inner-pool with var_keep=0.99 to remove anisotropy/scale artifacts before geometric measures.

## 1. Trajectory geometry per (task, layer, episode)

Path length, mean speed, tortuosity (path/direct), and curvature (mean angular change between consecutive direction unit vectors). Per-task means + SD across sampled episodes.

### drawer

| layer | path_length | mean_speed | tortuosity | curvature |
|---|---|---|---|---|
| 0 | 62.68±8.19 | 0.225±0.029 | 11.98±2.48 | 2.286±0.187 |
| 1 | 280.38±92.23 | 1.005±0.331 | 30.03±15.74 | 2.736±0.088 |
| 2 | 383.40±143.08 | 1.374±0.513 | 26.13±18.95 | 2.783±0.064 |
| 3 | 343.00±120.11 | 1.229±0.430 | 26.47±17.93 | 2.753±0.059 |
| 4 | 571.23±108.45 | 2.047±0.389 | 42.58±9.30 | 2.596±0.050 |
| 5 | 541.16±112.54 | 1.940±0.403 | 46.29±9.13 | 2.555±0.048 |
| 6 | 552.68±103.68 | 1.981±0.372 | 45.73±8.73 | 2.440±0.042 |
| 7 | 474.04±73.56 | 1.699±0.264 | 41.96±8.56 | 2.416±0.040 |
| 8 | 476.94±64.11 | 1.709±0.230 | 40.68±8.55 | 2.391±0.042 |
| 9 | 562.73±59.38 | 2.017±0.213 | 44.94±9.42 | 2.294±0.051 |
| 10 | 516.41±45.22 | 1.851±0.162 | 45.17±9.23 | 2.264±0.048 |
| 11 | 447.73±33.97 | 1.605±0.122 | 42.55±8.84 | 2.244±0.046 |
| 12 | 430.90±27.95 | 1.544±0.100 | 41.10±7.99 | 2.245±0.047 |
| 13 | 404.28±20.19 | 1.449±0.072 | 40.22±7.90 | 2.209±0.045 |
| 14 | 364.60±15.10 | 1.307±0.054 | 39.47±7.64 | 2.180±0.040 |
| 15 | 364.09±13.49 | 1.305±0.048 | 39.65±7.44 | 2.161±0.041 |
| 16 | 368.89±13.30 | 1.322±0.048 | 40.65±8.19 | 2.124±0.036 |
| 17 | 373.09±13.26 | 1.337±0.048 | 39.32±8.17 | 2.112±0.035 |
| 18 | 401.97±17.52 | 1.441±0.063 | 40.18±8.67 | 2.094±0.034 |
| 19 | 423.72±19.45 | 1.519±0.070 | 39.63±9.37 | 2.088±0.034 |
| 20 | 458.93±23.26 | 1.645±0.083 | 38.84±8.96 | 2.083±0.033 |
| 21 | 504.45±25.31 | 1.808±0.091 | 39.98±9.04 | 2.075±0.032 |
| 22 | 617.43±33.14 | 2.213±0.119 | 41.46±8.54 | 2.060±0.031 |
| 23 | 760.00±40.58 | 2.724±0.145 | 43.52±9.00 | 2.053±0.028 |

### nut_thread

| layer | path_length | mean_speed | tortuosity | curvature |
|---|---|---|---|---|
| 0 | 14.79±2.53 | 0.116±0.020 | 8.00±1.84 | 1.687±0.294 |
| 1 | 44.73±22.15 | 0.349±0.173 | 12.27±6.02 | 2.240±0.195 |
| 2 | 82.95±31.92 | 0.648±0.249 | 8.58±3.52 | 2.444±0.098 |
| 3 | 91.12±27.46 | 0.712±0.215 | 10.56±3.50 | 2.440±0.090 |
| 4 | 180.51±19.13 | 1.410±0.149 | 23.70±7.47 | 2.386±0.079 |
| 5 | 174.80±17.04 | 1.366±0.133 | 25.31±6.75 | 2.360±0.076 |
| 6 | 179.72±15.53 | 1.404±0.121 | 25.82±5.85 | 2.334±0.063 |
| 7 | 153.04±11.75 | 1.196±0.092 | 23.85±5.07 | 2.312±0.056 |
| 8 | 158.14±9.21 | 1.235±0.072 | 23.56±3.86 | 2.251±0.045 |
| 9 | 254.11±22.09 | 1.985±0.173 | 27.06±4.13 | 2.095±0.043 |
| 10 | 234.72±17.28 | 1.834±0.135 | 27.98±4.33 | 2.105±0.036 |
| 11 | 212.51±14.85 | 1.660±0.116 | 28.01±4.56 | 2.111±0.037 |
| 12 | 228.30±14.42 | 1.784±0.113 | 29.08±4.50 | 2.093±0.036 |
| 13 | 217.56±12.84 | 1.700±0.100 | 28.46±3.83 | 2.092±0.036 |
| 14 | 191.30±10.15 | 1.494±0.079 | 29.24±3.81 | 2.103±0.032 |
| 15 | 192.72±9.83 | 1.506±0.077 | 29.84±4.35 | 2.115±0.033 |
| 16 | 186.09±9.84 | 1.454±0.077 | 30.31±4.43 | 2.102±0.033 |
| 17 | 187.33±9.99 | 1.464±0.078 | 28.96±4.30 | 2.089±0.034 |
| 18 | 196.73±10.29 | 1.537±0.080 | 27.36±3.98 | 2.088±0.034 |
| 19 | 200.98±10.41 | 1.570±0.081 | 26.36±4.26 | 2.083±0.034 |
| 20 | 225.04±12.47 | 1.758±0.097 | 26.17±4.44 | 2.077±0.034 |
| 21 | 246.23±13.50 | 1.924±0.105 | 25.41±4.35 | 2.066±0.037 |
| 22 | 305.44±16.20 | 2.386±0.127 | 23.66±3.88 | 2.047±0.039 |
| 23 | 379.67±18.97 | 2.966±0.148 | 23.23±3.67 | 2.022±0.040 |

### peg_insert

| layer | path_length | mean_speed | tortuosity | curvature |
|---|---|---|---|---|
| 0 | 18.37±3.09 | 0.144±0.024 | 9.37±1.78 | 1.625±0.277 |
| 1 | 155.70±57.44 | 1.216±0.449 | 46.68±23.87 | 2.516±0.204 |
| 2 | 281.76±104.36 | 2.201±0.815 | 46.28±14.72 | 2.618±0.129 |
| 3 | 246.37±86.48 | 1.925±0.676 | 41.30±11.55 | 2.606±0.119 |
| 4 | 254.45±54.67 | 1.988±0.427 | 40.26±8.63 | 2.544±0.094 |
| 5 | 232.30±45.26 | 1.815±0.354 | 34.49±7.73 | 2.499±0.090 |
| 6 | 227.74±40.42 | 1.779±0.316 | 30.01±6.61 | 2.432±0.093 |
| 7 | 195.95±34.05 | 1.531±0.266 | 28.96±6.33 | 2.401±0.091 |
| 8 | 199.60±27.46 | 1.559±0.215 | 26.29±5.57 | 2.326±0.083 |
| 9 | 324.99±24.03 | 2.539±0.188 | 29.82±4.05 | 2.235±0.065 |
| 10 | 295.74±20.03 | 2.310±0.157 | 29.87±3.83 | 2.218±0.053 |
| 11 | 258.83±16.37 | 2.022±0.128 | 29.07±3.88 | 2.201±0.050 |
| 12 | 253.61±13.57 | 1.981±0.106 | 23.85±5.17 | 2.139±0.040 |
| 13 | 237.79±10.51 | 1.858±0.082 | 23.92±4.88 | 2.115±0.036 |
| 14 | 206.01±8.84 | 1.609±0.069 | 24.27±4.40 | 2.112±0.037 |
| 15 | 201.18±8.36 | 1.572±0.065 | 23.74±3.81 | 2.098±0.035 |
| 16 | 194.70±8.30 | 1.521±0.065 | 24.09±3.89 | 2.085±0.035 |
| 17 | 196.88±8.82 | 1.538±0.069 | 22.11±3.61 | 2.067±0.037 |
| 18 | 204.53±8.94 | 1.598±0.070 | 21.42±3.37 | 2.062±0.036 |
| 19 | 209.27±9.40 | 1.635±0.073 | 21.36±3.11 | 2.057±0.036 |
| 20 | 233.54±10.65 | 1.825±0.083 | 21.30±3.15 | 2.051±0.035 |
| 21 | 257.76±11.82 | 2.014±0.092 | 21.05±3.05 | 2.037±0.034 |
| 22 | 319.85±14.55 | 2.499±0.114 | 20.79±3.02 | 2.018±0.033 |
| 23 | 389.06±19.78 | 3.039±0.155 | 21.11±2.99 | 2.004±0.032 |

### push

| layer | path_length | mean_speed | tortuosity | curvature |
|---|---|---|---|---|
| 0 | 38.55±5.51 | 0.172±0.015 | 23.70±6.79 | 1.722±0.137 |
| 1 | 231.88±40.33 | 1.046±0.178 | 88.92±43.05 | 2.701±0.085 |
| 2 | 382.11±64.15 | 1.718±0.255 | 42.05±14.15 | 2.749±0.048 |
| 3 | 319.08±54.39 | 1.433±0.206 | 37.76±13.22 | 2.695±0.048 |
| 4 | 377.08±128.50 | 1.683±0.525 | 29.59±15.78 | 2.539±0.053 |
| 5 | 339.56±123.14 | 1.514±0.506 | 27.16±15.41 | 2.472±0.063 |
| 6 | 471.27±105.36 | 2.088±0.376 | 36.40±8.98 | 2.430±0.050 |
| 7 | 420.17±87.63 | 1.861±0.301 | 35.81±7.83 | 2.413±0.049 |
| 8 | 440.91±89.71 | 1.952±0.304 | 34.88±6.53 | 2.383±0.044 |
| 9 | 542.27±99.71 | 2.403±0.311 | 37.08±6.28 | 2.262±0.051 |
| 10 | 467.94±82.16 | 2.075±0.243 | 37.49±5.76 | 2.265±0.053 |
| 11 | 404.17±68.83 | 1.792±0.196 | 37.70±5.97 | 2.252±0.054 |
| 12 | 408.58±66.35 | 1.812±0.176 | 37.66±6.22 | 2.258±0.052 |
| 13 | 379.66±58.96 | 1.684±0.143 | 38.85±6.48 | 2.221±0.051 |
| 14 | 323.28±48.19 | 1.436±0.105 | 37.89±6.33 | 2.205±0.047 |
| 15 | 323.26±47.32 | 1.437±0.098 | 34.06±5.75 | 2.172±0.044 |
| 16 | 320.11±45.86 | 1.423±0.089 | 35.25±6.04 | 2.133±0.044 |
| 17 | 323.06±45.68 | 1.436±0.083 | 35.68±6.00 | 2.104±0.043 |
| 18 | 336.55±46.26 | 1.497±0.074 | 35.28±6.18 | 2.078±0.045 |
| 19 | 335.61±45.79 | 1.493±0.070 | 34.55±6.52 | 2.071±0.047 |
| 20 | 362.98±48.91 | 1.615±0.070 | 31.72±5.54 | 2.056±0.046 |
| 21 | 399.85±52.91 | 1.780±0.069 | 31.31±5.39 | 2.036±0.044 |
| 22 | 508.02±67.52 | 2.263±0.096 | 29.91±5.13 | 2.005±0.039 |
| 23 | 628.84±81.90 | 2.804±0.118 | 28.15±5.73 | 1.987±0.037 |

### reach

| layer | path_length | mean_speed | tortuosity | curvature |
|---|---|---|---|---|
| 0 | 24.84±2.26 | 0.106±0.010 | 23.56±5.88 | 2.700±0.067 |
| 1 | 220.55±50.27 | 0.943±0.215 | 84.72±40.10 | 2.967±0.019 |
| 2 | 286.65±53.62 | 1.225±0.229 | 49.26±24.91 | 2.898±0.028 |
| 3 | 247.27±47.72 | 1.057±0.204 | 48.16±24.43 | 2.807±0.025 |
| 4 | 358.76±117.56 | 1.533±0.502 | 68.52±35.13 | 2.444±0.060 |
| 5 | 338.90±114.60 | 1.448±0.490 | 70.76±34.00 | 2.318±0.079 |
| 6 | 718.37±73.73 | 3.070±0.315 | 152.87±50.45 | 2.377±0.061 |
| 7 | 639.66±60.01 | 2.734±0.256 | 150.37±48.27 | 2.410±0.060 |
| 8 | 756.11±74.21 | 3.231±0.317 | 173.00±52.78 | 2.458±0.065 |
| 9 | 747.63±65.96 | 3.195±0.282 | 119.81±32.38 | 2.352±0.060 |
| 10 | 599.03±48.03 | 2.560±0.205 | 113.68±30.32 | 2.338±0.057 |
| 11 | 521.71±34.13 | 2.230±0.146 | 112.97±31.59 | 2.339±0.050 |
| 12 | 503.20±28.01 | 2.150±0.120 | 105.57±29.09 | 2.343±0.051 |
| 13 | 438.03±21.40 | 1.872±0.091 | 95.17±21.91 | 2.305±0.043 |
| 14 | 359.69±16.67 | 1.537±0.071 | 91.09±19.92 | 2.295±0.040 |
| 15 | 349.28±18.46 | 1.493±0.079 | 81.86±19.28 | 2.288±0.037 |
| 16 | 348.39±17.00 | 1.489±0.073 | 85.10±17.67 | 2.276±0.036 |
| 17 | 355.36±16.07 | 1.519±0.069 | 83.08±17.46 | 2.255±0.036 |
| 18 | 351.14±16.98 | 1.501±0.073 | 78.35±17.17 | 2.240±0.038 |
| 19 | 344.99±18.00 | 1.474±0.077 | 73.55±16.10 | 2.241±0.040 |
| 20 | 365.36±19.11 | 1.561±0.082 | 66.92±15.33 | 2.225±0.039 |
| 21 | 392.28±21.50 | 1.676±0.092 | 62.74±15.00 | 2.197±0.036 |
| 22 | 471.92±29.42 | 2.017±0.126 | 55.08±14.14 | 2.164±0.035 |
| 23 | 566.29±38.62 | 2.420±0.165 | 52.27±15.58 | 2.140±0.035 |

### strike

| layer | path_length | mean_speed | tortuosity | curvature |
|---|---|---|---|---|
| 0 | 30.49±4.27 | 0.180±0.043 | 20.01±3.86 | 2.275±0.244 |
| 1 | 185.90±28.41 | 1.105±0.290 | 30.95±8.39 | 2.849±0.080 |
| 2 | 230.67±22.24 | 1.383±0.364 | 16.18±2.61 | 2.809±0.071 |
| 3 | 194.88±18.85 | 1.167±0.302 | 15.44±2.31 | 2.737±0.060 |
| 4 | 234.91±41.35 | 1.382±0.332 | 15.22±2.84 | 2.535±0.037 |
| 5 | 215.34±43.22 | 1.260±0.304 | 14.30±3.00 | 2.439±0.046 |
| 6 | 451.12±172.40 | 2.455±0.324 | 29.62±11.40 | 2.424±0.050 |
| 7 | 403.93±154.25 | 2.198±0.286 | 29.39±11.27 | 2.415±0.053 |
| 8 | 437.61±175.65 | 2.367±0.351 | 29.93±12.13 | 2.385±0.062 |
| 9 | 498.03±176.36 | 2.729±0.260 | 32.29±11.89 | 2.276±0.055 |
| 10 | 420.24±143.37 | 2.312±0.190 | 32.64±11.55 | 2.273±0.051 |
| 11 | 363.12±122.63 | 1.999±0.153 | 33.23±11.59 | 2.264±0.050 |
| 12 | 360.73±113.82 | 1.999±0.115 | 33.85±11.31 | 2.253±0.050 |
| 13 | 326.75±94.96 | 1.824±0.090 | 33.85±9.79 | 2.227±0.053 |
| 14 | 274.17±77.53 | 1.534±0.068 | 32.65±8.89 | 2.211±0.052 |
| 15 | 266.22±69.66 | 1.499±0.077 | 29.05±7.33 | 2.188±0.056 |
| 16 | 262.88±69.10 | 1.479±0.070 | 30.30±7.64 | 2.154±0.057 |
| 17 | 265.71±69.38 | 1.496±0.068 | 30.31±7.29 | 2.128±0.055 |
| 18 | 270.73±67.77 | 1.529±0.081 | 29.15±6.37 | 2.112±0.057 |
| 19 | 267.99±64.97 | 1.517±0.092 | 27.91±5.85 | 2.105±0.058 |
| 20 | 289.15±67.57 | 1.641±0.113 | 25.84±5.27 | 2.092±0.057 |
| 21 | 316.76±71.38 | 1.802±0.140 | 26.04±5.12 | 2.072±0.053 |
| 22 | 392.84±84.41 | 2.242±0.191 | 24.77±4.77 | 2.044±0.049 |
| 23 | 481.23±98.90 | 2.754±0.263 | 24.11±4.72 | 2.028±0.045 |

## 2. PCA(2) latent trajectories per layer

Per-layer PCA(2) fit on pooled windows of that task. Each panel shows 5–8 sample episode trajectories on the same PC1-PC2 axes. Start = circle, end = square.

### push

![per-layer PCA — push](plots/pca_push.png)

### strike

![per-layer PCA — strike](plots/pca_strike.png)

### reach

![per-layer PCA — reach](plots/pca_reach.png)

### drawer

![per-layer PCA — drawer](plots/pca_drawer.png)

### peg_insert

![per-layer PCA — peg_insert](plots/pca_peg_insert.png)

### nut_thread

![per-layer PCA — nut_thread](plots/pca_nut_thread.png)

## 3. Shared-PCA(2) trajectory across layers (per task)

Per-layer whitening then a single PCA(2) basis fit on the concatenated whitened pool across all 24 layers. Same PC1-PC2 axes for every panel — directly comparable layer geometry.

### push

![shared-basis PCA — push](plots/shared_pca_push.png)

### strike

![shared-basis PCA — strike](plots/shared_pca_strike.png)

### reach

![shared-basis PCA — reach](plots/shared_pca_reach.png)

### drawer

![shared-basis PCA — drawer](plots/shared_pca_drawer.png)

### peg_insert

![shared-basis PCA — peg_insert](plots/shared_pca_peg_insert.png)

### nut_thread

![shared-basis PCA — nut_thread](plots/shared_pca_nut_thread.png)

## 4. Intrinsic dimensionality per layer

![intrinsic dim per layer](plots/intrinsic_dim_per_layer.png)
![EVR spectrum — push](plots/evr_spectrum_push.png)
![EVR spectrum — strike](plots/evr_spectrum_strike.png)
![EVR spectrum — reach](plots/evr_spectrum_reach.png)
![EVR spectrum — drawer](plots/evr_spectrum_drawer.png)
![EVR spectrum — peg_insert](plots/evr_spectrum_peg_insert.png)
![EVR spectrum — nut_thread](plots/evr_spectrum_nut_thread.png)

## 5. Cross-layer CKA per task

![cross-layer CKA — push](plots/cross_layer_cka_push.png)
![cross-layer CKA — strike](plots/cross_layer_cka_strike.png)
![cross-layer CKA — reach](plots/cross_layer_cka_reach.png)
![cross-layer CKA — drawer](plots/cross_layer_cka_drawer.png)
![cross-layer CKA — peg_insert](plots/cross_layer_cka_peg_insert.png)
![cross-layer CKA — nut_thread](plots/cross_layer_cka_nut_thread.png)

## 6. Cross-task feature CKA per layer

![cross-task CKA evolution](plots/cross_task_cka_evolution.png)
![cross-task CKA L00](plots/cross_task_cka_layer00.png)
![cross-task CKA L06](plots/cross_task_cka_layer06.png)
![cross-task CKA L12](plots/cross_task_cka_layer12.png)
![cross-task CKA L18](plots/cross_task_cka_layer18.png)
![cross-task CKA L23](plots/cross_task_cka_layer23.png)

## 7. Event-locked latent geometry (push, strike, drawer)

Trajectories aligned to contact onset (push, strike) or sustained motion onset (drawer). Each task: 4 metrics × layer × τ heatmaps. Speed/curvature/tortuosity/PR. τ=0 marks the event; PR is participation ratio on local 2w+1 = 9 window covariance.

![event-locked push](plots/event_locked_push.png)
![event-locked strike](plots/event_locked_strike.png)
![event-locked drawer](plots/event_locked_drawer.png)

## 8. Partial RSA — latent geometry vs physics, controlling for pos+vel

Vectorized RDMs computed on whitened latent and standardized physics groups. Both rank-transformed; residualized against [r_pos, r_vel]; Pearson on residuals = partial Spearman correlation.

![partial RSA acceleration](plots/rsa_partial_heatmap_acc.png)
![partial RSA contact](plots/rsa_partial_heatmap_ct.png)
![partial RSA — push](plots/rsa_partial_push.png)
![partial RSA — strike](plots/rsa_partial_strike.png)
![partial RSA — reach](plots/rsa_partial_reach.png)
![partial RSA — drawer](plots/rsa_partial_drawer.png)
![partial RSA — peg_insert](plots/rsa_partial_peg_insert.png)
![partial RSA — nut_thread](plots/rsa_partial_nut_thread.png)

## 9. Tangent RSA — latent dynamics vs physical dynamics

U_l[t] = Z_white[t+1] − Z_white[t]; A_l[t] = U_l[t+1] − U_l[t]. Compare pdist(U) vs pdist(physical velocity); pdist(A) vs pdist(physical acc); and partial version controlling for pos+vel.

![tangent RSA U vs V](plots/rsa_vel_tan.png)
![tangent RSA A vs G](plots/rsa_acc_tan.png)
![tangent RSA A vs G | pos,vel](plots/rsa_acc_tan_partial.png)

## 10. CCA — canonical subspace alignment between latent and physics groups

PCA-reduce whitened latent to top-k (95% variance, ≤128). Fit linear CCA between Ẑ_l and X_g for each physics group g ∈ {pos, vel, acc, ct}. Report ρ1 (top canonical correlation), CCA energy (Σρ²), rank90 (subspace dim for 90% energy).

![CCA rho1 — pos](plots/cca_rho1_pos.png)
![CCA rho1 — vel](plots/cca_rho1_vel.png)
![CCA rho1 — acc](plots/cca_rho1_acc.png)
![CCA rho1 — ct](plots/cca_rho1_ct.png)
![CCA energy — pos](plots/cca_energy_pos.png)
![CCA energy — vel](plots/cca_energy_vel.png)
![CCA energy — acc](plots/cca_energy_acc.png)
![CCA energy — ct](plots/cca_energy_ct.png)
![CCA rank90 — pos](plots/cca_rank90_pos.png)
![CCA rank90 — vel](plots/cca_rank90_vel.png)
![CCA rank90 — acc](plots/cca_rank90_acc.png)
![CCA rank90 — ct](plots/cca_rank90_ct.png)

## 11. Koopman-style linear dynamics scores

A. Self-predictability ΔR² (z_{t+1} ≈ A_l z_t, fit Ridge, report delta over persistence baseline z_{t+1}=z_t).
B. Next-step physics predictability from latent: ee_pos / ee_vel / ee_acc / obj_vel one-step ahead.
Episode-level 80/20 train/test split, episode-aware boundary masking. PCA-whitened latent (var_keep=0.99, K_cap=128).

![Koopman ΔR² self](plots/koopman_r2_self_delta.png)
![Next-step ee_pos R²](plots/koopman_ee_pos_next.png)
![Next-step ee_vel R²](plots/koopman_ee_vel_next.png)
![Next-step ee_acc R²](plots/koopman_ee_acc_next.png)
![Next-step obj_vel R²](plots/koopman_obj_vel_next.png)

## 11. Methodology notes

- **Whitening**: per-(task, layer) PCA-whitening on the pooled-windows feature matrix, var_keep=0.99. This removes layer-norm-induced anisotropy that would otherwise confound geometric measures.
- **RSA bootstrap**: episode-level resampling planned (B=200) but the heavy pdist recomputation is left at point-estimate in this run; CIs are an extension.
- **Sub-sampling**: 1500 windows for partial-RSA / tangent-RSA, 4000 for CCA, 8000 for shared-PCA fit. Stratified by episode × normalized-time bin where applicable.
- **Drawer event-lock**: contact_flag is sparse in drawer; we use sustained motion-onset on object speed (threshold = 30% of episode-max for k=3 consecutive windows) per Codex guidance.
- **Acceleration target**: finite-diff(velocity) per the Variant-A/B probing pipeline (Isaac-Lab body acceleration is not d/dt of stored velocity; finite-diff used uniformly to avoid distribution shift).

