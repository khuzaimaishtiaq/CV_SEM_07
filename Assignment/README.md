# Skin Lesion Boundary Detection Using Canny Edge Detection

A simple computer vision lab that finds the boundary of a skin lesion in a dermoscopy image using image filtering, Canny edge detection, and contour detection. The goal is to see how well edge detection can separate a lesion from the surrounding skin.

Everything runs from one Google Colab notebook. **No dataset upload is needed.**

---

## Quick Start

1. Open [Google Colab](https://colab.research.google.com) and choose **File → Upload notebook**.
2. Upload `Skin_Lesion_Canny_Lab.ipynb`.
3. Click **Runtime → Run all**.

The notebook needs internet access in Colab so it can download the images. All required libraries (OpenCV, NumPy, pandas, SciPy, Matplotlib) are already installed in Colab.

---

## Dataset

The notebook uses **5 real dermoscopy images** from the public **ISIC** skin-lesion archive (ISIC 2017/2018 challenge data):

| Image   | ISIC ID        |
|---------|----------------|
| Image 1 | ISIC_0000012   |
| Image 2 | ISIC_0000042   |
| Image 3 | ISIC_0000059   |
| Image 4 | ISIC_0000119   |
| Image 5 | ISIC_0000163   |

The images are downloaded automatically from a public GitHub copy of ISIC data, at 256 × 256 pixels. Each image comes with an expert-drawn lesion mask. The masks are used **only to evaluate** the result and are never used to find the boundary.

---

## Pipeline

```
Original → Grayscale → Gaussian Filter → Canny Edges → Lesion Boundary → Area & Perimeter
```

| Step | What happens |
|------|--------------|
| Load | Download 5 ISIC images and display them |
| Preprocess | Convert to grayscale, then apply a Gaussian filter (7×7, σ = 1.5) |
| Canny | Run Canny with three threshold settings and show all edge maps |
| Select best | Score each setting by overlap (IoU) with the true lesion mask and pick the best |
| Boundary | Close gaps in the edge ring, fill it, remove hair spurs, keep the largest central contour, draw it on the original |
| Measure | Area = pixels inside the boundary (`cv2.countNonZero`), perimeter = contour length (`cv2.arcLength`) |
| Compare | Evaluate 8 filter + edge-detector combinations |
| Answer | Print answers to the 6 lab questions |

### Threshold note

The assignment's example thresholds (50–100, 100–200, 150–250) were too strict for these soft-bordered 256 × 256 images. They gave almost empty edge maps, with only about 59 edge pixels per image at 150–250. The notebook prints these counts and uses proportionally lower thresholds with the same low / medium / high idea:

| Setting | Thresholds |
|---------|-----------|
| Low     | 10–30     |
| Medium  | 20–60     |
| High    | 40–100    |

---

## Results

Results from a test run. Image, code, and parameters are fixed, so your numbers should match.

### Threshold selection (IoU with the true lesion)

| Image   | 10–30 | 20–60 | 40–100 | Best  |
|---------|-------|-------|--------|-------|
| Image 1 | 0.152 | 0.857 | 0.000  | 20–60 |
| Image 2 | 0.811 | 0.154 | 0.016  | 10–30 |
| Image 3 | 0.619 | 0.022 | 0.000  | 10–30 |
| Image 4 | 0.885 | 0.690 | 0.063  | 10–30 |
| Image 5 | 0.628 | 0.757 | 0.132  | 20–60 |

Overall, **10–30** gave the best average overlap (0.62), ahead of 20–60 (0.50) and 40–100 (0.04).

### Lesion area and perimeter

| Image   | Best Filter           | Edge Method  | Area (pixels) | Perimeter (pixels) |
|---------|-----------------------|--------------|---------------|--------------------|
| Image 1 | Gaussian (7×7, σ=1.5) | Canny 20–60  | 2901          | 208.6              |
| Image 2 | Gaussian (7×7, σ=1.5) | Canny 10–30  | 24854         | 885.3              |
| Image 3 | Gaussian (7×7, σ=1.5) | Canny 10–30  | 17341         | 579.8              |
| Image 4 | Gaussian (7×7, σ=1.5) | Canny 10–30  | 10456         | 456.2              |
| Image 5 | Gaussian (7×7, σ=1.5) | Canny 20–60  | 14656         | 586.3              |

Values are in pixels of the 256 × 256 images.

### Final comparison (average over 5 images)

| Method            | Noise Handling         | Edge Quality         | Boundary Detection   | Overall            |
|-------------------|------------------------|----------------------|----------------------|--------------------|
| Original + Sobel  | Poor (5.63)            | Poor (F1 0.12)       | Poor (IoU 0.36)      | Poor (0.24)        |
| Original + Canny  | Poor (5.63)            | Fair (F1 0.13)       | Poor (IoU 0.36)      | Poor (0.25)        |
| Average + Sobel   | Excellent (1.48)       | Excellent (F1 0.17)  | Fair (IoU 0.40)      | Fair (0.28)        |
| Average + Canny   | Excellent (1.48)       | Good (F1 0.14)       | Excellent (IoU 0.64) | Excellent (0.39)   |
| Gaussian + Sobel  | Good (1.78)            | Good (F1 0.17)       | Good (IoU 0.41)      | Good (0.29)        |
| Gaussian + Canny  | Good (1.78)            | Fair (F1 0.13)       | Excellent (IoU 0.62) | Good (0.38)        |
| Median + Sobel    | Fair (1.78)            | Poor (F1 0.13)       | Fair (IoU 0.38)      | Fair (0.25)        |
| Median + Canny    | Fair (1.78)            | Excellent (F1 0.17)  | Good (IoU 0.60)      | Excellent (0.39)   |

The three Canny combinations with a smoothing filter score almost the same (0.38–0.39). Canny clearly beats Sobel on boundary detection. Unfiltered images perform worst because noise and hair create false edges.

**How the comparison is scored:**
- **Noise handling** is the residual fine-grain noise in lesion-free skin (lower is better).
- **Edge quality** is the F1 score of edge pixels against the true lesion border, with a 3-pixel tolerance.
- **Boundary detection** is the IoU of the detected lesion region against the expert mask.
- **Overall** is the mean of edge F1 and boundary IoU.
- The rating labels (Poor / Fair / Good / Excellent) are ranks among the 8 methods.

---

## Questions Answered in the Notebook

1. Why is Gaussian filtering applied before Canny detection?
2. How did the three Canny threshold settings affect the result?
3. Which threshold produced the best lesion boundary?
4. Why are edges useful for detecting skin lesions?
5. What problems were observed in detecting the lesion boundary?
6. How could the method be improved?

The answers print in the last notebook cell. Answers 2 and 3 use the numbers measured in your run.

---

## Known Limitations

- **Hairs** produce strong edges that can attach to the lesion contour. In Image 2 this inflates the area.
- **Soft, low-contrast borders** give weak edges and gaps in the contour, so higher thresholds fail.
- **Noise and skin texture** create false edges at low thresholds.
- **Morphological closing** bridges gaps but slightly shifts the boundary, which affects area and perimeter.
- Only 5 images are used, so averages are indicative rather than statistically strong.

## Possible Improvements

- Choose Canny thresholds automatically (Otsu or median-based) for each image.
- Remove hair first (DullRazor-style black-hat filtering plus inpainting) and correct uneven illumination.
- Use colour channels (e.g. Lab) instead of plain grayscale, or try bilateral / non-local-means filtering.
- Combine edges with Otsu thresholding, active contours, or GrabCut to get smooth closed borders.
- Use deep-learning segmentation (e.g. U-Net) trained on ISIC for the best accuracy.

---

## Files

| File | Description |
|------|-------------|
| `Skin_Lesion_Canny_Lab.ipynb` | Complete lab notebook (8 code cells, run top to bottom) |
| `README.md` | This file |

## Tech Stack

Python · OpenCV · NumPy · pandas · SciPy · Matplotlib

## Acknowledgements

Images are from the [ISIC Archive](https://www.isic-archive.com/) (International Skin Imaging Collaboration), accessed through a public GitHub copy of the ISIC challenge data. They are used here for educational purposes only. This project is **not** a medical diagnostic tool.
