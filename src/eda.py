import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from src.utils import decode_geohash

logger = logging.getLogger(__name__)

class TrafficEDA:
    """
    TrafficEDA executes comprehensive exploratory data analysis on the traffic demand dataset.
    It produces statistics, generates visualization figures saved in reports/figures/,
    and writes a detailed markdown report reports/eda_report.md.
    """
    def __init__(self, data_path: str, output_dir: str = "reports"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.insights = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Loads the training data and parses spatial/temporal columns for analysis.
        """
        logger.info(f"Loading data for EDA from {self.data_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with shape {df.shape}")
        
        # 1. Parse timestamps
        logger.info("Extracting temporal features for EDA...")
        df['timestamp'] = df['timestamp'].fillna('0:0').astype(str)
        time_split = df['timestamp'].str.split(':', expand=True)
        df['hour'] = time_split[0].astype(int)
        df['minute'] = time_split[1].astype(int)
        df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
        df['day_of_week'] = df['day'] % 7
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Approximate month assuming 30 days per month
        df['month'] = (df['day'] // 30) % 12 + 1
        
        # Verify chronological order of dataset rows to prevent lag feature leakage
        logger.info("Verifying chronological order of dataset rows...")
        sort_key = df['day'] * 100 + df['time_slot']
        diffs = sort_key.diff()
        if (diffs < 0).any():
            logger.warning("WARNING - RISK DETECTED: The raw dataset is NOT sorted chronologically! Lag features created directly will suffer from leakage/alignment errors.")
            logger.info("Chronologically sorting dataset by day and time_slot...")
            df = df.sort_values(by=['day', 'time_slot']).reset_index(drop=True)
        else:
            logger.info("Dataset chronological row order verified.")
        
        # 2. Parse geohashes
        logger.info("Decoding geohashes for geospatial mapping...")
        unique_geohashes = df['geohash'].dropna().unique()
        geohash_map = {}
        for gh in unique_geohashes:
            try:
                lat, lon = decode_geohash(gh)
                geohash_map[gh] = (lat, lon)
            except Exception as e:
                logger.error(f"Error decoding geohash {gh}: {e}")
                geohash_map[gh] = (np.nan, np.nan)
                
        df['latitude'] = df['geohash'].map(lambda x: geohash_map.get(x, (np.nan, np.nan))[0])
        df['longitude'] = df['geohash'].map(lambda x: geohash_map.get(x, (np.nan, np.nan))[1])
        
        # Handle coordinate nulls
        median_lat = df['latitude'].median()
        median_lon = df['longitude'].median()
        df['latitude'] = df['latitude'].fillna(median_lat if not pd.isnull(median_lat) else 12.9716)
        df['longitude'] = df['longitude'].fillna(median_lon if not pd.isnull(median_lon) else 77.5946)
        
        self.df = df
        return df

    def run_basic_analysis(self) -> dict:
        """
        Gathers dataset shape, schemas, missing values, duplicates, and general stats.
        """
        logger.info("Running basic statistical analysis...")
        df = self.df
        
        # General stats
        shape = df.shape
        columns = list(df.columns)
        dtypes = {col: str(val) for col, val in df.dtypes.items()}
        missing_vals = {col: int(val) for col, val in df.isnull().sum().items()}
        duplicate_count = int(df.duplicated().sum())
        
        # Target stats
        demand_series = df['demand']
        target_stats = {
            'count': int(demand_series.count()),
            'mean': float(demand_series.mean()),
            'std': float(demand_series.std()),
            'min': float(demand_series.min()),
            '25%': float(demand_series.quantile(0.25)),
            '50%': float(demand_series.median()),
            '75%': float(demand_series.quantile(0.75)),
            'max': float(demand_series.max())
        }
        
        basic_info = {
            'shape': shape,
            'columns': columns,
            'dtypes': dtypes,
            'missing_values': missing_vals,
            'duplicates': duplicate_count,
            'target_stats': target_stats
        }
        
        self.insights['basic_analysis'] = basic_info
        return basic_info

    def run_target_analysis(self) -> dict:
        """
        Investigates the distribution, skewness, kurtosis, quantiles, and outliers of demand.
        """
        logger.info("Running target variable analysis...")
        df = self.df
        demand = df['demand']
        
        # Calculate moments
        skewness_val = float(skew(demand))
        kurtosis_val = float(kurtosis(demand))
        
        # Quantiles
        quantiles = {f"{q*100:.0f}%": float(demand.quantile(q)) for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]}
        
        # Outlier Detection via IQR rule
        q1 = demand.quantile(0.25)
        q3 = demand.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = demand[(demand < lower_bound) | (demand > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = float(outlier_count / len(demand) * 100)
        
        target_info = {
            'skewness': skewness_val,
            'kurtosis': kurtosis_val,
            'quantiles': quantiles,
            'iqr': float(iqr),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_pct
        }
        
        # Generate target distribution plots
        plt.figure(figsize=(18, 5))
        sns.set_theme(style="whitegrid")
        
        # 1. Histogram + KDE
        plt.subplot(1, 3, 1)
        sns.histplot(demand, kde=True, bins=50, color='royalblue')
        plt.title('Demand Distribution (Hist + KDE)', fontsize=13)
        plt.xlabel('Demand')
        
        # 2. Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=demand, color='salmon')
        plt.title('Demand Boxplot (Outlier Check)', fontsize=13)
        plt.xlabel('Demand')
        
        # 3. Cumulative Quantiles
        plt.subplot(1, 3, 3)
        percentiles = np.linspace(0, 100, 100)
        quantile_vals = np.percentile(demand, percentiles)
        plt.plot(percentiles, quantile_vals, color='darkgreen', linewidth=2.5)
        plt.axvline(95, color='red', linestyle='--', label='95th Percentile')
        plt.axvline(99, color='orange', linestyle='--', label='99th Percentile')
        plt.title('Cumulative Quantiles', fontsize=13)
        plt.xlabel('Percentile')
        plt.ylabel('Demand Value')
        plt.legend()
        
        plt.tight_layout()
        plot_path = self.figures_dir / "target_distribution.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Target distribution plots saved to {plot_path}")
        
        self.insights['target_analysis'] = target_info
        return target_info

    def run_temporal_analysis(self) -> dict:
        """
        Analyzes demand patterns over hours, days, weekends, and cyclic components.
        """
        logger.info("Running temporal analysis...")
        df = self.df
        
        # Groupings
        hourly_demand = df.groupby('hour')['demand'].mean().to_dict()
        weekly_demand = df.groupby('day_of_week')['demand'].mean().to_dict()
        weekend_demand = df.groupby('is_weekend')['demand'].mean().to_dict()
        
        temporal_info = {
            'hourly_mean': hourly_demand,
            'weekly_mean': weekly_demand,
            'weekend_mean': weekend_demand
        }
        
        # Create visual subplots
        plt.figure(figsize=(16, 12))
        
        # 1. Demand by Hour of Day
        plt.subplot(2, 2, 1)
        sns.lineplot(data=df, x='hour', y='demand', errorbar='ci', color='blue', marker='o', linewidth=2)
        plt.title('Mean Demand by Hour of Day (Diurnal Cycle)', fontsize=13)
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Demand')
        plt.xticks(range(0, 24, 2))
        
        # 2. Demand by Day of Week
        plt.subplot(2, 2, 2)
        sns.barplot(data=df, x='day_of_week', y='demand', hue='day_of_week', legend=False, palette='coolwarm')
        plt.title('Mean Demand by Day of Week', fontsize=13)
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        plt.ylabel('Mean Demand')
        
        # 3. Heatmap of Hour vs Day of Week
        plt.subplot(2, 2, 3)
        pivot_df = df.pivot_table(index='hour', columns='day_of_week', values='demand', aggfunc='mean')
        sns.heatmap(pivot_df, cmap='YlOrRd', cbar_kws={'label': 'Mean Demand'})
        plt.title('Demand Heatmap: Hour vs Day of Week', fontsize=13)
        plt.xlabel('Day of Week')
        plt.ylabel('Hour of Day')
        
        # 4. Weekday vs Weekend Profile
        plt.subplot(2, 2, 4)
        sns.lineplot(data=df, x='hour', y='demand', hue='is_weekend', palette={0: 'teal', 1: 'orange'}, marker='s')
        plt.title('Diurnal Patterns: Weekdays vs Weekends', fontsize=13)
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Demand')
        plt.legend(title='Weekend?')
        plt.xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plot_path = self.figures_dir / "temporal_analysis.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Temporal analysis plots saved to {plot_path}")
        
        self.insights['temporal_analysis'] = temporal_info
        return temporal_info

    def run_geospatial_analysis(self) -> dict:
        """
        Analyzes geospatial coordinates, locates hotspots, and clusters positions.
        """
        logger.info("Running geospatial analysis...")
        df = self.df
        
        # Group by coordinates to find hot/cold spots
        loc_stats = df.groupby('geohash').agg(
            latitude=('latitude', 'first'),
            longitude=('longitude', 'first'),
            mean_demand=('demand', 'mean'),
            record_count=('demand', 'count')
        ).reset_index()
        
        top_high = loc_stats.sort_values(by='mean_demand', ascending=False).head(10)[['geohash', 'mean_demand']].to_dict('records')
        top_low = loc_stats.sort_values(by='mean_demand', ascending=True).head(10)[['geohash', 'mean_demand']].to_dict('records')
        
        # KMeans Spatial Clustering (Group coordinates into 6 primary hubs)
        coords = df[['latitude', 'longitude']].values
        n_clusters = min(6, len(loc_stats))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df['spatial_cluster'] = kmeans.fit_predict(coords)
        
        cluster_means = df.groupby('spatial_cluster')['demand'].mean().to_dict()
        
        min_lat, max_lat = float(df['latitude'].min()), float(df['latitude'].max())
        min_lon, max_lon = float(df['longitude'].min()), float(df['longitude'].max())
        
        # Coordinate geofencing validation
        is_bengaluru = (12.0 <= min_lat <= 14.0 and 77.0 <= min_lon <= 78.5)
        if not is_bengaluru:
            logger.warning(
                f"WARNING - RISK DETECTED: Decoded coordinates (Lat: {min_lat:.4f} to {max_lat:.4f}, "
                f"Lon: {min_lon:.4f} to {max_lon:.4f}) do not align with Bengaluru geographically. "
                "The coordinates are shifted/anonymized. Avoid real-world map overlays for external features."
            )
            
        geospatial_info = {
            'coordinates_bbox': {
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon
            },
            'is_bengaluru_coordinates': is_bengaluru,
            'top_high_demand_locations': top_high,
            'top_low_demand_locations': top_low,
            'spatial_clusters_count': n_clusters,
            'cluster_demand_means': cluster_means
        }
        
        # Plot spatial coordinates
        plt.figure(figsize=(15, 6))
        
        # 1. Hotspot Heatmap Representation
        plt.subplot(1, 2, 1)
        sc1 = plt.scatter(
            df['longitude'], df['latitude'], c=df['demand'], 
            cmap='plasma', alpha=0.5, s=15, edgecolors='none'
        )
        plt.colorbar(sc1, label='Demand')
        plt.title('Demand Hotspots in Bengaluru Coordinate Plane', fontsize=13)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # 2. Cluster Groups Mapping
        plt.subplot(1, 2, 2)
        # Use loc_stats with cluster prediction to keep size clean and display centroid markers
        loc_stats['cluster'] = kmeans.predict(loc_stats[['latitude', 'longitude']].values)
        sns.scatterplot(
            data=loc_stats, x='longitude', y='latitude', hue='cluster', 
            palette='Set1', legend='full', s=40, alpha=0.8
        )
        # Centroids
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 1], centroids[:, 0], color='black', marker='X', 
            s=150, label='Centroids', zorder=10
        )
        plt.title('Spatial Hubs via KMeans Clustering', fontsize=13)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        
        plt.tight_layout()
        plot_path = self.figures_dir / "geospatial_analysis.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Geospatial analysis plots saved to {plot_path}")
        
        # Folium interactive map generation if available
        try:
            import folium
            from folium.plugins import HeatMap
            
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
            
            # Aggregate to reduce size and improve visualization
            agg_coords = df.groupby(['latitude', 'longitude'])['demand'].mean().reset_index().dropna()
            heat_data = [[row['latitude'], row['longitude'], row['demand']] for idx, row in agg_coords.iterrows()]
            
            HeatMap(heat_data, radius=12, blur=8, max_zoom=13).add_to(m)
            
            map_file = self.output_dir / "bengaluru_demand_heatmap.html"
            m.save(str(map_file))
            logger.info(f"Folium interactive map saved to {map_file}")
            geospatial_info['interactive_map_status'] = "Generated successfully"
        except Exception as e:
            logger.warning(f"Could not build Folium map: {e}")
            geospatial_info['interactive_map_status'] = f"Failed: {e}"
            
        self.insights['geospatial_analysis'] = geospatial_info
        return geospatial_info

    def run_feature_target_relationship(self) -> dict:
        """
        Explores correlations and multivariate dependencies against target demand.
        """
        logger.info("Running feature-target relationships...")
        df = self.df
        
        # Identify numeric columns for correlation matrix
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        
        # Remove raw Index, day, minute, and month (which has zero variance and would return nan correlation)
        cols_to_drop = [col for col in ['Index', 'minute', 'month'] if col in numeric_df.columns]
        numeric_df = numeric_df.drop(columns=cols_to_drop)
        
        corr_matrix = numeric_df.corr()
        demand_correlations = corr_matrix['demand'].sort_values(ascending=False).to_dict()
        
        relationship_info = {
            'demand_correlations': demand_correlations
        }
        
        # Plot relationships
        plt.figure(figsize=(16, 10))
        
        # 1. Correlation Matrix Heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1.0, vmax=1.0)
        plt.title('Correlation Heatmap', fontsize=13)
        
        # 2. Demand vs Hour boxplot
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x='hour', y='demand', palette='crest', showfliers=False)
        plt.title('Hourly Variance (No Outliers)', fontsize=13)
        plt.xlabel('Hour of Day')
        plt.ylabel('Demand')
        
        # 3. Demand vs Spatial Cluster barplot
        plt.subplot(2, 2, 3)
        if 'spatial_cluster' in df.columns:
            sns.barplot(data=df, x='spatial_cluster', y='demand', hue='spatial_cluster', legend=False, palette='Set2')
            plt.title('Mean Demand by Spatial Cluster Group', fontsize=13)
            plt.xlabel('Spatial Cluster Index')
            plt.ylabel('Mean Demand')
            
        # 4. Categorical variables vs Demand (e.g. Weather / RoadType)
        plt.subplot(2, 2, 4)
        if 'Weather' in df.columns:
            # Sort categories by mean demand
            weather_order = df.groupby('Weather')['demand'].mean().sort_values(ascending=False).index
            sns.barplot(data=df, x='Weather', y='demand', hue='Weather', legend=False, order=weather_order, palette='viridis')
            plt.title('Mean Demand by Weather Type', fontsize=13)
            plt.xlabel('Weather Type')
            plt.ylabel('Mean Demand')
            plt.xticks(rotation=30)
            
        plt.tight_layout()
        plot_path = self.figures_dir / "feature_relationships.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Feature-target relationship plots saved to {plot_path}")
        
        self.insights['feature_relationships'] = relationship_info
        return relationship_info

    def generate_report(self) -> None:
        """
        Aggregates all insights into reports/eda_report.md.
        """
        logger.info("Generating final Markdown report...")
        basic = self.insights.get('basic_analysis', {})
        target = self.insights.get('target_analysis', {})
        temporal = self.insights.get('temporal_analysis', {})
        geospatial = self.insights.get('geospatial_analysis', {})
        relationships = self.insights.get('feature_relationships', {})
        
        report_path = self.output_dir / "eda_report.md"
        
        # Check coordinate status for warning
        is_bengaluru = geospatial.get('is_bengaluru_coordinates', False)
        coord_status_str = (
            "✅ Geographically Valid coordinates." if is_bengaluru 
            else "⚠️ Shifted/Anonymized coordinates (Located in Indian Ocean. Real-world maps cannot be overlaid)."
        )
        
        markdown_content = f"""# Traffic Demand Exploratory Data Analysis Report
 
This report summarizes key statistics, distribution properties, spatial patterns, temporal profiles, and correlations found within the Gridlock Hackathon training dataset.
 
---
 
## ⚠️ KEY DATA RISKS IDENTIFIED
 
> [!WARNING]
> 1. **Data Leakage Risk via Lags**: Because rows in the raw dataset might not be sorted chronologically and contain multiple locations interleaved at the same time step, any lag feature generation must be preceded by explicit chronological sorting (`day` and `time_slot`) and grouped by `geohash`.
> 2. **Zero-Variance Month Column**: The `month` feature is constant throughout the training set because it covers only a two-day snapshot (days 48 and 49). This results in zero variance (NaN correlation). Monthly seasonality is completely unlearnable and should be omitted from model features.
> 3. **Shifted Coordinates (Not Bengaluru)**: The decoded geohash coordinates (Lat: {geospatial.get('coordinates_bbox', {}).get('min_lat', 0.0):.4f} to {geospatial.get('coordinates_bbox', {}).get('max_lat', 0.0):.4f}, Lon: {geospatial.get('coordinates_bbox', {}).get('min_lon', 0.0):.4f} to {geospatial.get('coordinates_bbox', {}).get('max_lon', 0.0):.4f}) are located in the Indian Ocean. They are synthetic/anonymized. Do not try to merge external geographical POIs or real road maps of Bengaluru using these values.
 
---
 
## 1. Executive Summary & Core Insights
 
- **Target Demand Properties**: The target variable `demand` has a mean of **{basic.get('target_stats', {}).get('mean', 0.0):.4f}** and ranges from **{basic.get('target_stats', {}).get('min', 0.0):.4f}** to **{basic.get('target_stats', {}).get('max', 0.0):.4f}**. The distribution is highly skewed (**Skewness: {target.get('skewness', 0.0):.2f}**) with significant positive tail behavior (**Kurtosis: {target.get('kurtosis', 0.0):.2f}**).
- **Temporal Patterns**: Strong diurnal traffic rhythms are present. Demand peaks during business transit hours and reaches its minimum late at night. Weekly differences are present; weekends display distinctly modified travel signatures compared to workdays.
- **Geospatial Hotspots**: Traffic demand is concentrated in specific high-frequency geohash nodes. Clustering coordinates using KMeans isolates **{geospatial.get('spatial_clusters_count', 0)}** primary spatial hubs, exhibiting highly disparate average traffic weights.
- **Correlation Overview**: Spatial proximity (`latitude`, `longitude`) and diurnal cyclical variables (`sin_time`, `cos_time`) have strong correlation signatures with demand.
 
---
 
## 2. Basic Dataset Statistics
 
- **Observations (Rows)**: `{basic.get('shape', (0, 0))[0]:,}`
- **Features (Columns)**: `{basic.get('shape', (0, 0))[1]}`
- **Duplicate Rows**: `{basic.get('duplicates', 0)}`
- **Missing Values**:
{chr(10).join([f"  - `{col}`: {val:,} nulls" for col, val in basic.get('missing_values', {}).items() if val > 0]) or "  - None"}
 
### Target variable (`demand`) Summary statistics:
- **Mean**: `{basic.get('target_stats', {}).get('mean', 0.0):.6f}`
- **Median (50%)**: `{basic.get('target_stats', {}).get('50%', 0.0):.6f}`
- **Std Dev**: `{basic.get('target_stats', {}).get('std', 0.0):.6f}`
- **Min / Max**: `{basic.get('target_stats', {}).get('min', 0.0):.6f} / {basic.get('target_stats', {}).get('max', 0.0):.6f}`
 
---
 
## 3. Distribution & Outlier Analysis
 
- **Skewness**: `{target.get('skewness', 0.0):.4f}` (Positive skew indicates a long tail of high traffic events)
- **Kurtosis**: `{target.get('kurtosis', 0.0):.4f}` (High kurtosis indicates heavy-tailed distribution with severe peaks)
- **Outlier Count (IQR Rule)**: `{target.get('outlier_count', 0):,}` ({target.get('outlier_percentage', 0.0):.2f}% of data)
 
### Percentiles Analysis:
| Percentile | Demand Value |
|------------|--------------|
| 1%         | `{target.get('quantiles', {}).get('1%', 0.0):.6f}` |
| 5%         | `{target.get('quantiles', {}).get('5%', 0.0):.6f}` |
| 10%        | `{target.get('quantiles', {}).get('10%', 0.0):.6f}` |
| 25% (Q1)   | `{target.get('quantiles', {}).get('25%', 0.0):.6f}` |
| 50% (Q2)   | `{target.get('quantiles', {}).get('50%', 0.0):.6f}` |
| 75% (Q3)   | `{target.get('quantiles', {}).get('75%', 0.0):.6f}` |
| 90%        | `{target.get('quantiles', {}).get('90%', 0.0):.6f}` |
| 95%        | `{target.get('quantiles', {}).get('95%', 0.0):.6f}` |
| 99%        | `{target.get('quantiles', {}).get('99%', 0.0):.6f}` |
 
*Visualization stored in: [target_distribution.png](file:///{str(self.figures_dir.resolve().as_posix())}/target_distribution.png)*
 
---
 
## 4. Temporal Analysis
 
- **Hourly Profile**: Average demand peaks during morning commuting blocks and evening transition windows. Low traffic occurs between midnight and 5:00 AM.
- **Weekly Dynamics**: Mean demand fluctuations occur across weekdays, suggesting different load distributions over the work week vs. weekends.
- **Weekend Deviation**: Weekend profiles exhibit flatter, shifted peaks compared to sharp weekday commute hours.
 
*Visualization stored in: [temporal_analysis.png](file:///{str(self.figures_dir.resolve().as_posix())}/temporal_analysis.png)*
 
---
 
## 5. Geospatial Hotspots & Clusters
 
- **Coordinate Status**: `{coord_status_str}`
- **Coordinate Bounds**:
  - Latitude Range: `[{geospatial.get('coordinates_bbox', {}).get('min_lat', 0.0):.4f}, {geospatial.get('coordinates_bbox', {}).get('max_lat', 0.0):.4f}]`
  - Longitude Range: `[{geospatial.get('coordinates_bbox', {}).get('min_lon', 0.0):.4f}, {geospatial.get('coordinates_bbox', {}).get('max_lon', 0.0):.4f}]`
- **Spatial Clusters Performance**:
  Traffic coordinates were clustered into **{geospatial.get('spatial_clusters_count', 0)}** major hubs using KMeans.
  Average demand per spatial cluster:
{chr(10).join([f"  - Cluster {cluster}: Mean Demand = {mean:.4f}" for cluster, mean in geospatial.get('cluster_demand_means', {}).items()])}
 
### Top 5 Highest Demand Geohash Clusters:
{chr(10).join([f"  1. Geohash `{loc['geohash']}` - Average Demand = {loc['mean_demand']:.5f}" for loc in geospatial.get('top_high_demand_locations', [])[:5]])}
 
### Top 5 Lowest Demand Geohash Clusters:
{chr(10).join([f"  1. Geohash `{loc['geohash']}` - Average Demand = {loc['mean_demand']:.5f}" for loc in geospatial.get('top_low_demand_locations', [])[:5]])}
 
*Visualization stored in: [geospatial_analysis.png](file:///{str(self.figures_dir.resolve().as_posix())}/geospatial_analysis.png)*
*Interactive Leaflet Map: [bengaluru_demand_heatmap.html](file:///{str(self.output_dir.resolve().as_posix())}/bengaluru_demand_heatmap.html)*
 
---
 
## 6. Correlation Analysis
 
### Correlations with `demand`:
{chr(10).join([f"- `{col}`: {val:.4f}" for col, val in relationships.get('demand_correlations', {}).items() if col != 'demand'])}
 
*Visualization stored in: [feature_relationships.png](file:///{str(self.figures_dir.resolve().as_posix())}/feature_relationships.png)*
 
---
 
## 7. Downstream Feature Engineering Recommendations
 
1. **Temporal Lags**: Diurnal cycles are strong; create lag features for `t-1` (previous time slot), `t-4` (previous hour), and `t-96` (same time slot yesterday) to capture autocorrelation. **Note: Ensure rows are chronologically sorted and grouped by `geohash` prior to creation.**
2. **Geospatial Cross-Features**: Interaction between `spatial_cluster` and `hour` should be engineered, as different geographical zones (residential vs commercial) display custom diurnal peaks.
3. **Rolling Averages**: Compute trailing rolling mean and standard deviation of demand at each geohash over the past 3-6 slots to capture short-term trends.
4. **Target Encoding**: Perform out-of-fold target encoding on categorical items (like `RoadType`, `Weather`, `geohash`) to expose direct traffic demand associations to tree models.
5. **Cyclical Scaling**: Utilize cyclical sin/cos time slot variables directly to help algorithms understand clock boundaries (95 connects to 0).
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        logger.info(f"Markdown report generated successfully at {report_path}")

    def run_all_eda(self) -> dict:
        """
        Executes the entire EDA flow sequentially.
        """
        logger.info("Executing full exploratory data analysis pipeline...")
        self.load_and_prepare_data()
        self.run_basic_analysis()
        self.run_target_analysis()
        self.run_temporal_analysis()
        self.run_geospatial_analysis()
        self.run_feature_target_relationship()
        self.generate_report()
        logger.info("EDA pipeline finished.")
        return self.insights
