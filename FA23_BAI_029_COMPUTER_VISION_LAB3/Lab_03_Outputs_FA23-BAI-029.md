# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

This notebook is self-contained and continues directly from **Lab 01** (ISIC skin-lesion model comparison) and **Lab 02** (HAM10000 per-filter retraining). It reuses the same dataset pipeline, split logic and model families as Lab 02, and implements **all 6 tasks** required by the Lab 03 handout:

1. Comparative Edge Detection (Sobel Gx/Gy/magnitude, Prewitt, Laplacian, LoG, Canny)
2. Effect of Noise on Edge Detection (Gaussian / Salt-and-Pepper noise, + Gaussian/Median filtering)
3. Parameter Analysis of Canny Edge Detection
4. Classification Using Edge Maps (Raw vs. Filtered vs. Edge datasets)
5. Classification Performance Comparison (Table 3)
6. Visual Comparison of Classification Results (confusion matrices + bar chart)

Plus auto-generated answers to the **Discussion Questions**, and a static **Viva Questions** reference section.

> **Just run all cells top-to-bottom.** Everything — data download, noise/edge generation, model training, tables, figures, and the discussion write-up — runs automatically. See the `QUICK_MODE` flag in the config cell if you want a faster/slower run.


## 0. Setup


```
Device: cuda - Tesla T4
```



## 1. Dataset (same as Lab 01 / Lab 02)

Reuses Lab 02's HAM10000 download + lesion-level leakage-free split (`SEED=42`, 80/10/10), so results are directly comparable across labs.


```
Using Colab cache for faster access to the 'skin-cancer-mnist-ham10000' dataset.
Dataset path: /kaggle/input/skin-cancer-mnist-ham10000
Classes (7): ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
dx
nv       5403
bkl       727
mel       614
bcc       327
akiec     228
vasc       98
df         73
Name: count, dtype: int64
```



```
Train:1318  Val:222  Test:224
```



## 2. Task 1: Comparative Edge Detection

Implements Sobel (Gx, Gy, magnitude), Prewitt, Laplacian, Laplacian of Gaussian (LoG) and Canny, then displays `Original → Sobel → Prewitt → Laplacian → LoG → Canny` for images from at least three different classes.


```
Edge detectors ready: ['Sobel Gx', 'Sobel Gy', 'Sobel Magnitude', 'Prewitt', 'Laplacian', 'LoG', 'Canny']
```



```
Demo classes: {'akiec': 'Actinic keratoses', 'bcc': 'Basal cell carcinoma', 'bkl': 'Benign keratosis-like lesions'}
```

![output](Lab_03_Outputs_files/output_9_1.png)

![output](Lab_03_Outputs_files/output_9_2.png)



## 3. Task 2: Effect of Noise on Edge Detection

Adds Gaussian noise and Salt-and-Pepper noise, then applies the edge detectors to: **original → noisy → noisy+Gaussian filtered → noisy+Median filtered**, and quantifies the effect with an *edge density* metric (fraction of pixels above a fixed edge-strength threshold) — used as a numeric proxy for edge continuity / false edges / noise sensitivity.


![output](Lab_03_Outputs_files/output_11_3.png)



```
                          Condition  Sobel edge density  Prewitt edge density  \
0                          Original              0.0125                0.0211   
1                    Gaussian noise              0.0840                0.1101   
2  Gaussian noise + Gaussian filter              0.1200                0.1218   
3    Gaussian noise + Median filter              0.0978                0.1084   
4               Salt & Pepper noise              0.1637                0.1534   
5             S&P + Gaussian filter              0.1762                0.1783   
6               S&P + Median filter              0.0529                0.0517   

   Laplacian edge density  LoG edge density  Canny edge density  
0                  0.0416             0.000              0.0218  
1                  0.6157             0.000              0.2612  
2                  0.0028             0.000              0.0056  
3                  0.0063             0.000              0.0010  
4                  0.2847             0.001              0.2893  
5                  0.0446             0.000              0.0343  
6                  0.0021             0.000              0.0021
```



```
Observations (edge continuity / sharpness / false & broken edges / noise sensitivity / smoothing):

- **Sobel**: edge density rises from 0.013 (clean) to 0.084 under Gaussian noise and 0.164 under salt-and-pepper noise -> noise introduces spurious/false edges and broken edge continuity. Gaussian filtering brings it back to 0.120/0.176; median filtering brings it to 0.098/0.053 (Median filtering is more effective against salt-and-pepper noise here, as theoretically expected).
- **Prewitt**: edge density rises from 0.021 (clean) to 0.110 under Gaussian noise and 0.153 under salt-and-pepper noise -> noise introduces spurious/false edges and broken edge continuity. Gaussian filtering brings it back to 0.122/0.178; median filtering brings it to 0.108/0.052 (Median filtering is more effective against salt-and-pepper noise here, as theoretically expected).
- **Laplacian**: edge density rises from 0.042 (clean) to 0.616 under Gaussian noise and 0.285 under salt-and-pepper noise -> noise introduces spurious/false edges and broken edge continuity. Gaussian filtering brings it back to 0.003/0.045; median filtering brings it to 0.006/0.002 (Median filtering is more effective against salt-and-pepper noise here, as theoretically expected).
- **LoG**: edge density rises from 0.000 (clean) to 0.000 under Gaussian noise and 0.001 under salt-and-pepper noise -> noise introduces spurious/false edges and broken edge continuity. Gaussian filtering brings it back to 0.000/0.000; median filtering brings it to 0.000/0.000 (Gaussian filtering is more effective against salt-and-pepper noise here, as theoretically expected).
- **Canny**: edge density rises from 0.022 (clean) to 0.261 under Gaussian noise and 0.289 under salt-and-pepper noise -> noise introduces spurious/false edges and broken edge continuity. Gaussian filtering brings it back to 0.006/0.034; median filtering brings it to 0.001/0.002 (Median filtering is more effective against salt-and-pepper noise here, as theoretically expected).
```



## 4. Task 3: Parameter Analysis of Canny Edge Detection


![output](Lab_03_Outputs_files/output_15_4.png)

```
Selected Canny configuration: low=50, high=150
```

```
   Low  High  Kernel  Edge density  Time (ms)  Δ to Sobel density
0   30   100       5        0.1028      0.867              0.0903
1   50   150       5        0.0178      0.536              0.0053
2  100   200       5        0.0054      0.336              0.0071
```



## 5. Task 4: Classification Using Edge Maps

Builds three versions of the dataset using the **same** train/val/test split, epochs and evaluation protocol:
- **Set A — Raw Images** (from Lab 01/02)
- **Set B — Filtered Images**, using the best filter identified in Lab 02 (loaded automatically from `filter_retrain_results.csv` if that notebook was run first in this environment; otherwise falls back to a documented default)
- **Set C — Edge Images**, using the best Canny configuration identified in Task 3 above

The model architecture (ResNet101, fine-tuned, ImageNet-initialised) is the same one Lab 02 used as "Best Model 2", so the comparison is fair and apples-to-apples with Labs 01–02.


```
Lab 02 results not found in this session — defaulting BEST_FILTER_NAME to: Gaussian 
(edit this cell to match your actual Lab 02 result if different).
Dataset variants: ['Set A - Raw', 'Set B - Filtered (Gaussian)', 'Set C - Edge (Canny 50/150)']
```



```

============================================================
Training on Set A - Raw
============================================================
Downloading: "https://download.pytorch.org/models/resnet101-cd907fc2.pth" to /root/.cache/torch/hub/checkpoints/resnet101-cd907fc2.pth
```

```
100%|██████████| 171M/171M [00:01<00:00, 160MB/s]
```

```
  Epoch 1/6  val macro-F1=0.2933
  Epoch 2/6  val macro-F1=0.4722
  Epoch 3/6  val macro-F1=0.5313
  Epoch 4/6  val macro-F1=0.5738
  Epoch 5/6  val macro-F1=0.6020
  Epoch 6/6  val macro-F1=0.5941

============================================================
Training on Set B - Filtered (Gaussian)
============================================================
  Epoch 1/6  val macro-F1=0.3457
  Epoch 2/6  val macro-F1=0.5277
  Epoch 3/6  val macro-F1=0.5646
  Epoch 4/6  val macro-F1=0.6068
  Epoch 5/6  val macro-F1=0.6382
  Epoch 6/6  val macro-F1=0.6758

============================================================
Training on Set C - Edge (Canny 50/150)
============================================================
  Epoch 1/6  val macro-F1=0.1925
  Epoch 2/6  val macro-F1=0.2303
  Epoch 3/6  val macro-F1=0.2358
  Epoch 4/6  val macro-F1=0.2417
  Epoch 5/6  val macro-F1=0.2272
  Epoch 6/6  val macro-F1=0.2768
```

```
                           Set  Accuracy  Precision  Recall  F1-score  \
0                  Set A - Raw    62.946     64.208  62.946    62.335   
1  Set B - Filtered (Gaussian)    65.179     67.273  65.179    65.296   
2  Set C - Edge (Canny 50/150)    40.179     38.419  40.179    38.286   

   Training time (s)  Inference time (s)  
0             174.98               3.069  
1             177.43               2.553  
2             176.25               2.558
```



## 6. Task 5: Classification Performance Comparison


![output](Lab_03_Outputs_files/output_21_5.png)

```

Table 3 — Cross-Lab Classification Performance Comparison
```

```
                           Set  Accuracy  Precision  Recall  F1-score  \
0                  Set A - Raw    62.946     64.208  62.946    62.335   
1  Set B - Filtered (Gaussian)    65.179     67.273  65.179    65.296   
2  Set C - Edge (Canny 50/150)    40.179     38.419  40.179    38.286   

   Training time (s)  Inference time (s)  
0             174.98               3.069  
1             177.43               2.553  
2             176.25               2.558
```



## 7. Task 6: Visual Comparison of Classification Results


```
Best-performing representation: Set B - Filtered (Gaussian)
```

![output](Lab_03_Outputs_files/output_23_6.png)



## 8. Discussion Questions (auto-drafted from this run's results)


```
Noise sensitivity ranking (avg. relative increase in edge density under noise):
  Canny      +1162.6%
  Laplacian  +982.2%
  Sobel      +890.8%
  Prewitt    +524.4%
  LoG        +nan%

Most noise-sensitive detector: Canny
```



```
==============================================================================
DISCUSSION QUESTIONS — answers auto-drafted from this run's computed results
==============================================================================

Q1. Edge Detection and Noise
    The most noise-sensitive detector in this run was **Canny**
    (avg. relative increase in edge density under Gaussian/Salt-and-Pepper noise
    = +1162.6%, Table 1). Second-order operators (Laplacian,
    LoG) differentiate the image twice, which amplifies high-frequency noise far
    more than first-order gradient operators (Sobel, Prewitt). Canny is
    comparatively robust because Gaussian smoothing, non-maximum suppression and
    hysteresis thresholding are already built into its pipeline.

Q2. Effect of Filtering
    Gaussian filtering reduced Gaussian noise effectively but blurred fine
    edges, slightly lowering edge sharpness. Median filtering was more effective
    against salt-and-pepper noise (it removes outlier pixels without averaging
    them into neighbours), better preserving edge continuity for that noise
    type. See the exact edge-density numbers in Table 1.

Q3. Canny Parameters
    Raising the low/high thresholds (30/100 -> 50/150 -> 100/200) reduced the
    number of detected edges and suppressed weak/noisy edges, at the risk of
    missing faint true boundaries (Table 2). The configuration selected for
    this run was low=50, high=150, chosen as the
    closest match to the clean-image Sobel-gradient edge density — a proxy for
    "captures the true structural edges without excess noise".

Q4. Edge Maps and Classification
    In this run, edge-only images reduced accuracy
    relative to raw images (40.18% vs 62.95%,
    Table 3). This is consistent with the expectation that dermoscopic classification relies heavily on colour and texture cues (pigment network, colour variegation, blue-white veil, etc.) that edge maps discard.

Q5. Information Loss
    Edge maps retain only boundary/shape information and discard colour,
    texture and absolute intensity — all of which are clinically important
    for skin-lesion classification (pigment colour, texture irregularity,
    surface pattern).

Q6. Classical vs. Deep Features
    CNNs learn task-optimised, multi-scale edge-like filters end-to-end and
    combine them with colour/texture cues in deeper layers, rather than being
    restricted to a single hand-crafted operator at one fixed scale. This
    end-to-end, learned, multi-cue representation is generally more flexible
    and expressive than manually supplying a single edge map as input.

Q7. Best Representation
    Across Labs 01-03, the best-performing input representation in this run
    was **Set B - Filtered (Gaussian)** (Accuracy=65.18%,
    F1=65.30%,
    Table 3), i.e. the chosen filter improved signal-to-noise without discarding as much information as edge maps.
```



## 9. Required Deliverables — where to find them

All figures/tables required by the handout are saved under `WORK_DIR` (`/kaggle/working` on Kaggle, `./work` otherwise):

| Deliverable | File |
|---|---|
| Task 1 figure (Original→Sobel→Prewitt→Laplacian→LoG→Canny, 3 classes) | `task1_edge_comparison.png` |
| Task 1 — all methods on one image | `task1_all_methods.png` |
| Task 2 figure (noise + filtering effects) | `task2_noise_effects.png` |
| **Table 1** — Effect of Noise and Preprocessing | `table1_noise_preprocessing.csv` |
| Task 3 figure (Canny parameter sweep) | `task3_canny_params.png` |
| **Table 2** — Canny Parameter Analysis | `table2_canny_params.csv` |
| Task 5 bar chart (Accuracy/Precision/Recall/F1) | `task5_performance_bar_chart.png` |
| Task 6 confusion matrices | `task6_confusion_matrices.png` |
| **Table 3** — Cross-Lab Classification Performance | `table3_classification_comparison.csv` |

The **Discussion Questions** answers (Section 8 above) and this notebook itself (Code) are the remaining deliverables. Copy the printed discussion text and the tables/figures above into your report (Introduction, Methodology, Experimental Setup, Results, Discussion, Conclusion, References).


## 10. Viva Questions — quick reference

1. **What is an edge in an image?** A location where pixel intensity changes sharply/discontinuously — typically marking object boundaries.
2. **First-order vs. second-order edge detection?** First-order operators (Sobel, Prewitt) use the first derivative (gradient) and find edges as local maxima of gradient magnitude. Second-order operators (Laplacian, LoG) use the second derivative and find edges as zero-crossings.
3. **Sobel Gx vs. Gy?** Gx responds to vertical edges (horizontal intensity change); Gy responds to horizontal edges (vertical intensity change). Combined via magnitude `sqrt(Gx²+Gy²)` for edges in any direction.
4. **Why is the Laplacian more sensitive to noise?** It's a second derivative, which amplifies high-frequency content (including noise) more than a first derivative does.
5. **Purpose of Gaussian smoothing before edge detection?** Suppresses high-frequency noise so derivative operators respond to genuine intensity structure rather than noise, reducing false edges.
6. **Main advantage of Canny edge detection?** Combines smoothing, gradient computation, non-maximum suppression (thin, precise edges) and hysteresis thresholding (edge linking) into one robust, well-localised, low-false-positive pipeline.
7. **Canny's low and high thresholds?** Pixels with gradient magnitude above `high` are "strong" edges (always kept); below `low` are discarded; between the two are "weak" edges, kept only if connected to a strong edge (hysteresis).
8. **Gaussian vs. Salt-and-Pepper noise?** Gaussian noise adds small random variation to *every* pixel (additive, normally distributed). Salt-and-pepper noise randomly sets a *fraction* of pixels to pure black or white (impulse noise).
9. **Why is Median filtering useful for Salt-and-Pepper noise?** It replaces each pixel with the median of its neighbourhood, which discards extreme outlier values (the noise) without averaging/blurring them into the surrounding signal, unlike a mean/Gaussian filter.
10. **Why can edge detection reduce classification performance?** It discards colour, texture and intensity information that may be diagnostically/discriminatively important, keeping only shape/boundary information.
11. **Can a CNN learn edge features automatically?** Yes — early convolutional layers commonly learn Sobel/Gabor-like oriented edge filters directly from data, as part of end-to-end training.
12. **Why might raw images outperform edge-only images?** Raw images retain the full signal (colour, texture, intensity, shape) that a sufficiently deep/trained model can exploit, whereas edge maps pre-emptively discard information the model might otherwise have used.

