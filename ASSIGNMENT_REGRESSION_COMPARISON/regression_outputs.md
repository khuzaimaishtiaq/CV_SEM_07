# Comparative Implementation of ML Regression Algorithms
**Dataset:** California Housing (20,640 districts), target = `median_house_value`

**Models:** Linear Regression, Decision Tree, Random Forest, Neural Network (MLP), XGBoost, KNN

**Setup:** 80:20 split (`random_state=42`), median imputation and scaling fitted on training data only.

**How to run:** Runtime > Run all. The dataset downloads automatically, so no upload is needed.

```
Libraries loaded
```

## 1. Load the dataset automatically
The `housing.csv` file is downloaded from the Hands-On ML project on GitHub and read directly into pandas.

```
Dataset shape: (20640, 10)
```

|    |   longitude |   latitude |   housing_median_age |   total_rooms |   total_bedrooms |   population |   households |   median_income |   median_house_value | ocean_proximity   |
|---:|------------:|-----------:|---------------------:|--------------:|-----------------:|-------------:|-------------:|----------------:|---------------------:|:------------------|
|  0 |     -122.23 |      37.88 |                   41 |           880 |              129 |          322 |          126 |          8.3252 |               452600 | NEAR BAY          |
|  1 |     -122.22 |      37.86 |                   21 |          7099 |             1106 |         2401 |         1138 |          8.3014 |               358500 | NEAR BAY          |
|  2 |     -122.24 |      37.85 |                   52 |          1467 |              190 |          496 |          177 |          7.2574 |               352100 | NEAR BAY          |
|  3 |     -122.25 |      37.85 |                   52 |          1274 |              235 |          558 |          219 |          5.6431 |               341300 | NEAR BAY          |
|  4 |     -122.25 |      37.85 |                   52 |          1627 |              280 |          565 |          259 |          3.8462 |               342200 | NEAR BAY          |

## 2. Exploratory data analysis

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   longitude           20640 non-null  float64
 1   latitude            20640 non-null  float64
 2   housing_median_age  20640 non-null  float64
 3   total_rooms         20640 non-null  float64
 4   total_bedrooms      20433 non-null  float64
 5   population          20640 non-null  float64
 6   households          20640 non-null  float64
 7   median_income       20640 non-null  float64
 8   median_house_value  20640 non-null  float64
 9   ocean_proximity     20640 non-null  object 
dtypes: float64(9), object(1)
memory usage: 1.6+ MB

Missing values per column:
longitude               0
latitude                0
housing_median_age      0
total_rooms             0
total_bedrooms        207
population              0
households              0
median_income           0
median_house_value      0
ocean_proximity         0
dtype: int64

ocean_proximity categories:
ocean_proximity
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: count, dtype: int64
```

|       |   longitude |    latitude |   housing_median_age |   total_rooms |   total_bedrooms |   population |   households |   median_income |   median_house_value |
|:------|------------:|------------:|---------------------:|--------------:|-----------------:|-------------:|-------------:|----------------:|---------------------:|
| count | 20640       | 20640       |           20640      |      20640    |        20433     |     20640    |     20640    |     20640       |                20640 |
| mean  |  -119.57    |    35.6319  |              28.6395 |       2635.76 |          537.871 |      1425.48 |       499.54 |         3.87067 |               206856 |
| std   |     2.00353 |     2.13595 |              12.5856 |       2181.62 |          421.385 |      1132.46 |       382.33 |         1.89982 |               115396 |
| min   |  -124.35    |    32.54    |               1      |          2    |            1     |         3    |         1    |         0.4999  |                14999 |
| 25%   |  -121.8     |    33.93    |              18      |       1447.75 |          296     |       787    |       280    |         2.5634  |               119600 |
| 50%   |  -118.49    |    34.26    |              29      |       2127    |          435     |      1166    |       409    |         3.5348  |               179700 |
| 75%   |  -118.01    |    37.71    |              37      |       3148    |          647     |      1725    |       605    |         4.74325 |               264725 |
| max   |  -114.31    |    41.95    |              52      |      39320    |         6445     |     35682    |      6082    |        15.0001  |               500001 |

## 3. Preprocessing and 80:20 split
- One-hot encode `ocean_proximity`
- Split 80:20 with `random_state=42`
- Fill missing `total_bedrooms` with the **training** median
- Create scaled copies for Linear Regression, KNN and Neural Network

```
Train: (16512, 13) | Test: (4128, 13)
Missing values left in training set: 0
```

## 4. Define the six models
Each entry is `(model, needs_scaled_features)`. The Neural Network also uses a scaled target, which is inverse-transformed automatically when predicting.

```
Models defined: ['Linear Regression', 'Decision Tree', 'Random Forest', 'Neural Network', 'XGBoost', 'KNN']
```

## 5. Train, predict and evaluate
Metrics: MAE, MSE, RMSE and R².

```
Linear Regression finished
Decision Tree finished
Random Forest finished
Neural Network finished
XGBoost finished
KNN finished
```

| Rank | Model | MAE | MSE | RMSE | R2 |
|---:|:---|---:|---:|---:|---:|
|  1 | XGBoost                           |                           30508 |                      2127612566 |                            46126 |                         0.8376 |
|  2 | Random Forest                     |                           31548 |                      2392580046 |                            48914 |                         0.8174 |
|  3 | Neural Network                    |                           35851 |                      2799353935 |                            52909 |                         0.7864 |
|  4 | KNN                               |                           40870 |                      3769707700 |                            61398 |                         0.7123 |
|  5 | Decision Tree                     |                           42909 |                      3929477514 |                            62686 |                         0.7001 |
|  6 | Linear Regression                 |                           50670 |                      4908290571 |                            70059 |                         0.6254 |

## 6. Actual vs predicted plots

![Plot 1](img/plot_1.png)

## 7. RMSE comparison chart

![Plot 2](img/plot_2.png)

## 8. Best model and export

```
Best model: XGBoost
RMSE (USD): 46,126
R2: 0.8376
```
