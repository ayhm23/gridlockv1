# Traffic Demand Exploratory Data Analysis Report
 
This report summarizes key statistics, distribution properties, spatial patterns, temporal profiles, and correlations found within the Gridlock Hackathon training dataset.
 
---
 
## ⚠️ KEY DATA RISKS IDENTIFIED
 
> [!WARNING]
> 1. **Data Leakage Risk via Lags**: Because rows in the raw dataset might not be sorted chronologically and contain multiple locations interleaved at the same time step, any lag feature generation must be preceded by explicit chronological sorting (`day` and `time_slot`) and grouped by `geohash`.
> 2. **Zero-Variance Month Column**: The `month` feature is constant throughout the training set because it covers only a two-day snapshot (days 48 and 49). This results in zero variance (NaN correlation). Monthly seasonality is completely unlearnable and should be omitted from model features.
> 3. **Shifted Coordinates (Not Bengaluru)**: The decoded geohash coordinates (Lat: -5.4849 to -5.2377, Lon: 90.5878 to 90.9723) are located in the Indian Ocean. They are synthetic/anonymized. Do not try to merge external geographical POIs or real road maps of Bengaluru using these values.
 
---
 
## 1. Executive Summary & Core Insights
 
- **Target Demand Properties**: The target variable `demand` has a mean of **0.0939** and ranges from **0.0000** to **1.0000**. The distribution is highly skewed (**Skewness: 3.73**) with significant positive tail behavior (**Kurtosis: 17.33**).
- **Temporal Patterns**: Strong diurnal traffic rhythms are present. Demand peaks during business transit hours and reaches its minimum late at night. Weekly differences are present; weekends display distinctly modified travel signatures compared to workdays.
- **Geospatial Hotspots**: Traffic demand is concentrated in specific high-frequency geohash nodes. Clustering coordinates using KMeans isolates **6** primary spatial hubs, exhibiting highly disparate average traffic weights.
- **Correlation Overview**: Spatial proximity (`latitude`, `longitude`) and diurnal cyclical variables (`sin_time`, `cos_time`) have strong correlation signatures with demand.
 
---
 
## 2. Basic Dataset Statistics
 
- **Observations (Rows)**: `77,299`
- **Features (Columns)**: `19`
- **Duplicate Rows**: `0`
- **Missing Values**:
  - `RoadType`: 600 nulls
  - `Temperature`: 2,495 nulls
  - `Weather`: 797 nulls
 
### Target variable (`demand`) Summary statistics:
- **Mean**: `0.093942`
- **Median (50%)**: `0.047760`
- **Std Dev**: `0.142191`
- **Min / Max**: `0.000001 / 1.000000`
 
---
 
## 3. Distribution & Outlier Analysis
 
- **Skewness**: `3.7284` (Positive skew indicates a long tail of high traffic events)
- **Kurtosis**: `17.3308` (High kurtosis indicates heavy-tailed distribution with severe peaks)
- **Outlier Count (IQR Rule)**: `6,413` (8.30% of data)
 
### Percentiles Analysis:
| Percentile | Demand Value |
|------------|--------------|
| 1%         | `0.000625` |
| 5%         | `0.003172` |
| 10%        | `0.006422` |
| 25% (Q1)   | `0.018227` |
| 50% (Q2)   | `0.047760` |
| 75% (Q3)   | `0.108595` |
| 90%        | `0.216459` |
| 95%        | `0.335857` |
| 99%        | `0.862294` |
 
*Visualization stored in: [target_distribution.png](file:///D:/FlipKartGridlock/reports/figures/target_distribution.png)*
 
---
 
## 4. Temporal Analysis
 
- **Hourly Profile**: Average demand peaks during morning commuting blocks and evening transition windows. Low traffic occurs between midnight and 5:00 AM.
- **Weekly Dynamics**: Mean demand fluctuations occur across weekdays, suggesting different load distributions over the work week vs. weekends.
- **Weekend Deviation**: Weekend profiles exhibit flatter, shifted peaks compared to sharp weekday commute hours.
 
*Visualization stored in: [temporal_analysis.png](file:///D:/FlipKartGridlock/reports/figures/temporal_analysis.png)*
 
---
 
## 5. Geospatial Hotspots & Clusters
 
- **Coordinate Status**: `⚠️ Shifted/Anonymized coordinates (Located in Indian Ocean. Real-world maps cannot be overlaid).`
- **Coordinate Bounds**:
  - Latitude Range: `[-5.4849, -5.2377]`
  - Longitude Range: `[90.5878, 90.9723]`
- **Spatial Clusters Performance**:
  Traffic coordinates were clustered into **6** major hubs using KMeans.
  Average demand per spatial cluster:
  - Cluster 0: Mean Demand = 0.0734
  - Cluster 1: Mean Demand = 0.1071
  - Cluster 2: Mean Demand = 0.0963
  - Cluster 3: Mean Demand = 0.1266
  - Cluster 4: Mean Demand = 0.0780
  - Cluster 5: Mean Demand = 0.0742
 
### Top 5 Highest Demand Geohash Clusters:
  1. Geohash `qp09d9` - Average Demand = 0.96071
  1. Geohash `qp09ft` - Average Demand = 0.86885
  1. Geohash `qp09e5` - Average Demand = 0.86499
  1. Geohash `qp09d8` - Average Demand = 0.66932
  1. Geohash `qp096x` - Average Demand = 0.66563
 
### Top 5 Lowest Demand Geohash Clusters:
  1. Geohash `qp03zy` - Average Demand = 0.00050
  1. Geohash `qp08bt` - Average Demand = 0.00078
  1. Geohash `qp09k7` - Average Demand = 0.00079
  1. Geohash `qp093h` - Average Demand = 0.00082
  1. Geohash `qp09bv` - Average Demand = 0.00092
 
*Visualization stored in: [geospatial_analysis.png](file:///D:/FlipKartGridlock/reports/figures/geospatial_analysis.png)*
*Interactive Leaflet Map: [bengaluru_demand_heatmap.html](file:///D:/FlipKartGridlock/reports/bengaluru_demand_heatmap.html)*
 
---
 
## 6. Correlation Analysis
 
### Correlations with `demand`:
- `NumberofLanes`: 0.2141
- `day`: 0.0268
- `Temperature`: 0.0031
- `longitude`: -0.0068
- `spatial_cluster`: -0.0173
- `is_weekend`: -0.0268
- `day_of_week`: -0.0268
- `time_slot`: -0.0377
- `hour`: -0.0378
- `latitude`: -0.0392
 
*Visualization stored in: [feature_relationships.png](file:///D:/FlipKartGridlock/reports/figures/feature_relationships.png)*
 
---
 
## 7. Downstream Feature Engineering Recommendations
 
1. **Temporal Lags**: Diurnal cycles are strong; create lag features for `t-1` (previous time slot), `t-4` (previous hour), and `t-96` (same time slot yesterday) to capture autocorrelation. **Note: Ensure rows are chronologically sorted and grouped by `geohash` prior to creation.**
2. **Geospatial Cross-Features**: Interaction between `spatial_cluster` and `hour` should be engineered, as different geographical zones (residential vs commercial) display custom diurnal peaks.
3. **Rolling Averages**: Compute trailing rolling mean and standard deviation of demand at each geohash over the past 3-6 slots to capture short-term trends.
4. **Target Encoding**: Perform out-of-fold target encoding on categorical items (like `RoadType`, `Weather`, `geohash`) to expose direct traffic demand associations to tree models.
5. **Cyclical Scaling**: Utilize cyclical sin/cos time slot variables directly to help algorithms understand clock boundaries (95 connects to 0).
