"""
feature_engineering.py
========================
Produces the full, competition-grade feature set for the Gridlock Traffic
Demand Prediction challenge.

Feature categories:
  1.  Temporal lags         — previous demand at the same geohash
  2.  Rolling statistics    — short-window mean/std/max per geohash
  3.  Geohash target enc.   — per-geohash mean demand (OOF-safe version)
  4.  Categorical target enc.— per-category mean demand
  5.  Spatial clusters      — KMeans on lat/lon
  6.  Interaction features  — lanes×hour, cluster×dow, peak-hour flags
  7.  Label encoding        — low-cardinality string → integer

All lag/rolling operations are performed after sorting by (day, time_slot)
and grouping by geohash to prevent data leakage.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from src.utils import reduce_mem_usage

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS  = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
SAFE_LAG_SLOTS    = [94, 95, 96, 97, 98]       # 1-day lags (safe from test leakage)
REGIONAL_LAG_SLOTS = [95, 96, 97]              # regional yesterday-lags
KNN_K             = 5                          # number of spatial neighbors
N_SPATIAL_CLUSTERS = 6
PEAK_MORNING      = (7, 10)                    # inclusive hour range
PEAK_EVENING      = (17, 20)


class FeatureEngineer:
    """
    Stateful feature engineering class.
    Call .fit_transform(train_df) on training data, then .transform(test_df).

    Parameters
    ----------
    target_col : str
        Name of the demand target column.
    n_folds : int
        Number of CV folds to use for out-of-fold target encoding.
        Set to 1 to use global mean (no OOF split — useful for transform-only).
    """

    def __init__(self, target_col: str = "demand", n_folds: int = 5):
        self.target_col   = target_col
        self.n_folds      = n_folds
        self.is_fitted    = False

        # Fitted artefacts
        self.kmeans            : KMeans | None    = None
        self.label_encoders    : dict             = {}
        self.geohash_means     : pd.Series | None = None
        self.cat_means         : dict             = {}   # col → {cat → mean}
        self.global_mean       : float            = 0.0
        # RoadType geohash-level modal imputation (fitted on train, applied to test)
        self.roadtype_lookup   : dict             = {}   # geohash → modal RoadType
        self.knn_neighbors     : dict             = {}   # geohash → [neighbor_geohash, ...]

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits all encoders/scalers on df and returns the enriched dataframe.
        OOF target encoding is performed to avoid leakage.
        """
        logger.info("FeatureEngineer.fit_transform — starting on training data")
        df = df.copy()

        # Step 0: ensure chronological order
        df = self._sort_chronologically(df)

        # Step 0b: impute RoadType NaN via geohash modal lookup (fit + apply)
        # RoadType is the highest-signal feature (corr=0.86). All 600 NaN rows
        # in train and 324 in test can be filled this way.
        df = self._fit_impute_roadtype(df)

        # Step 1: core temporal + cyclic (already done in preprocessing,
        #         but we re-derive to be safe if running standalone)
        df = self._ensure_temporal_cols(df)

        # Step 2: spatial cluster (fit KMeans here)
        df = self._fit_spatial_clusters(df)

        # Step 3: label-encode categoricals
        df = self._fit_label_encode(df)

        # Step 4: lag & rolling features (train-only, no leakage issue for train)
        df = self._add_lag_features(df, is_train=True)
        df = self._add_rolling_features(df, is_train=True)
        df = self._fit_knn_spatial_lags(df)
        df = self._add_today_lags(df, df)
        df = self._add_diurnal_rolling_features(df, df)

        # Step 5: OOF target encoding  (must come AFTER sort, BEFORE return)
        self.global_mean = float(df[self.target_col].mean())
        df = self._oof_target_encode(df)

        # Step 6: interaction features
        df = self._add_interaction_features(df)

        # Step 7: store geohash means for test transform
        self.geohash_means = (
            df.groupby("geohash", observed=True)[self.target_col].mean()
            if self.target_col in df.columns
            else pd.Series(dtype=float)
        )
        for col in CATEGORICAL_COLS:
            if col in df.columns and f"{col}_enc" in df.columns:
                self.cat_means[col] = (
                    df.groupby(col, observed=True)[self.target_col].mean().to_dict()
                )

        self.is_fitted = True
        df = reduce_mem_usage(df, verbose=False)
        logger.info(f"fit_transform complete — final shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame, train_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Transforms the test set using artefacts fitted on training data.

        Parameters
        ----------
        df        : test DataFrame (no demand column)
        train_df  : the enriched training DataFrame (used to derive cross-day lags)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform(train_df) before transform(test_df).")
        logger.info("FeatureEngineer.transform — starting on test data")
        df = df.copy()

        df = self._sort_chronologically(df)
        df = self._impute_roadtype(df)    # apply fitted RoadType lookup to test
        df = self._ensure_temporal_cols(df)
        df = self._apply_spatial_clusters(df)
        df = self._apply_label_encode(df)

        # For lag/rolling on test: concatenate train tail with test, compute, then slice
        df = self._add_lag_features(df, is_train=False, train_df=train_df)
        df = self._add_rolling_features(df, is_train=False, train_df=train_df)
        df = self._apply_knn_spatial_lags(df, train_df=train_df)
        df = self._add_today_lags(df, ref_df=train_df)
        df = self._add_diurnal_rolling_features(df, ref_df=train_df)

        # Apply (global) target encoding using fitted means
        df = self._apply_target_encode(df)
        df = self._add_interaction_features(df)

        df = reduce_mem_usage(df, verbose=False)
        logger.info(f"transform complete — final shape: {df.shape}")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_impute_roadtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits geohash → RoadType modal lookup from known rows, then fills NaN."""
        if "RoadType" not in df.columns or "geohash" not in df.columns:
            return df
        known = df.dropna(subset=["RoadType"])
        if len(known) > 0:
            self.roadtype_lookup = (
                known.groupby("geohash", observed=True)["RoadType"]
                .agg(lambda x: x.mode()[0])
                .to_dict()
            )
            n_before = df["RoadType"].isna().sum()
            mask = df["RoadType"].isna()
            df.loc[mask, "RoadType"] = df.loc[mask, "geohash"].map(self.roadtype_lookup)
            n_after = df["RoadType"].isna().sum()
            logger.info(f"RoadType imputation (train): {n_before} NaN → {n_after} NaN")
        return df

    def _impute_roadtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted roadtype_lookup to test data."""
        if not self.roadtype_lookup or "RoadType" not in df.columns:
            return df
        n_before = df["RoadType"].isna().sum()
        mask = df["RoadType"].isna()
        df.loc[mask, "RoadType"] = df.loc[mask, "geohash"].map(self.roadtype_lookup)
        n_after = df["RoadType"].isna().sum()
        logger.info(f"RoadType imputation (test): {n_before} NaN → {n_after} NaN")
        return df

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        sort_cols = [c for c in ["day", "time_slot"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        return df

    def _ensure_temporal_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive hour/minute/time_slot/day_of_week if not already present."""
        if "timestamp" in df.columns and "hour" not in df.columns:
            df["timestamp"] = df["timestamp"].fillna("0:0").astype(str)
            ts = df["timestamp"].str.split(":", expand=True)
            df["hour"]      = ts[0].astype(int)
            df["minute"]    = ts[1].astype(int)
            df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
            df["sin_time"]  = np.sin(2 * np.pi * df["time_slot"] / 96.0)
            df["cos_time"]  = np.cos(2 * np.pi * df["time_slot"] / 96.0)
        if "day" in df.columns and "day_of_week" not in df.columns:
            df["day_of_week"] = df["day"] % 7
        if "day_of_week" in df.columns and "is_weekend" not in df.columns:
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)
        return df

    # ── Spatial clustering ────────────────────────────────────────────────────

    def _fit_spatial_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Fitting KMeans with {N_SPATIAL_CLUSTERS} clusters on coordinates")
        coords = df[["latitude", "longitude"]].values
        self.kmeans = KMeans(
            n_clusters=N_SPATIAL_CLUSTERS, random_state=42, n_init="auto"
        )
        df["spatial_cluster"] = self.kmeans.fit_predict(coords).astype(np.int8)

        # Cluster centroids as additional numeric features
        cluster_lats = self.kmeans.cluster_centers_[:, 0]
        cluster_lons = self.kmeans.cluster_centers_[:, 1]
        df["dist_to_centroid"] = np.sqrt(
            (df["latitude"]  - df["spatial_cluster"].map(lambda c: cluster_lats[c])) ** 2 +
            (df["longitude"] - df["spatial_cluster"].map(lambda c: cluster_lons[c])) ** 2
        ).astype(np.float32)
        return df

    def _apply_spatial_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[["latitude", "longitude"]].values
        df["spatial_cluster"] = self.kmeans.predict(coords).astype(np.int8)
        cluster_lats = self.kmeans.cluster_centers_[:, 0]
        cluster_lons = self.kmeans.cluster_centers_[:, 1]
        df["dist_to_centroid"] = np.sqrt(
            (df["latitude"]  - df["spatial_cluster"].map(lambda c: cluster_lats[c])) ** 2 +
            (df["longitude"] - df["spatial_cluster"].map(lambda c: cluster_lons[c])) ** 2
        ).astype(np.float32)
        return df

    # ── KNN spatial lags ─────────────────────────────────────────────────────

    def _fit_knn_spatial_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each unique geohash, finds the KNN_K nearest geohashes by Euclidean distance
        and pre-computes their yesterday demand (lag 96) averages and stds.
        Stores the neighbor lookup for later use on test.
        """
        target = self.target_col
        if target not in df.columns:
            return df

        logger.info(f"Building KNN spatial lag features (k={KNN_K})")

        # Build geohash → (lat, lon) mapping from unique pairs
        gh_coords = (
            df.groupby("geohash", observed=True)[["latitude", "longitude"]]
            .mean()
        )
        geohashes = gh_coords.index.tolist()
        coords    = gh_coords.values  # (N, 2)

        # For each geohash, find k nearest by Euclidean distance
        from sklearn.neighbors import BallTree
        import numpy as np
        ball = BallTree(np.deg2rad(coords), metric="haversine")
        dists, idxs = ball.query(np.deg2rad(coords), k=KNN_K + 1)  # +1 includes self

        self.knn_neighbors = {}
        for i, gh in enumerate(geohashes):
            # Skip index 0 (self), take next KNN_K
            neighbor_ghs = [geohashes[j] for j in idxs[i, 1:]]
            self.knn_neighbors[gh] = neighbor_ghs

        # Compute knn-based yesterday lags
        df = self._compute_knn_features(df, df)
        return df

    def _apply_knn_spatial_lags(self, df: pd.DataFrame, train_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Apply pre-fitted KNN neighbor lookup to test set."""
        if not self.knn_neighbors:
            return df
        ref_df = train_df if train_df is not None else df
        df = self._compute_knn_features(df, ref_df)
        return df

    def _compute_knn_features(self, df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each row in df, look up each of its KNN_K neighbor geohashes in ref_df
        at lag 96 (yesterday, same slot) and compute mean and std of their demand.
        Vectorized: pre-builds a ((day, slot) x n_geohashes) pivot, then does fast array lookups.
        """
        target   = self.target_col
        fill_val = self.global_mean

        if target not in ref_df.columns:
            df["knn_lag96_mean"] = fill_val
            df["knn_lag96_std"]  = 0.0
            return df

        # Build pivot: (day, time_slot) -> geohash -> demand
        ref_slice = ref_df[["day", "time_slot", "geohash", target]].copy()
        pivot = ref_slice.pivot_table(
            index=["day", "time_slot"], columns="geohash", values=target, aggfunc="mean"
        )

        geohash_to_col = {gh: i for i, gh in enumerate(pivot.columns)}
        pivot_arr = pivot.values  # shape: (n_rows, n_geohashes)
        idx_to_row = {idx: i for i, idx in enumerate(pivot.index)}

        knn_means = np.full(len(df), fill_val, dtype=np.float32)
        knn_stds  = np.zeros(len(df), dtype=np.float32)

        # Vectorize per unique geohash
        for gh, neighbors in self.knn_neighbors.items():
            mask = (df["geohash"] == gh).values
            if not mask.any():
                continue
            neighbor_cols = [geohash_to_col[n] for n in neighbors if n in geohash_to_col]
            if not neighbor_cols:
                continue

            rows_in_df = np.where(mask)[0]
            days       = df["day"].iloc[rows_in_df].values
            slots      = df["time_slot"].iloc[rows_in_df].values

            for r_idx, d, s in zip(rows_in_df, days, slots):
                # Lookup day - 1 (yesterday) at the same slot
                pivot_row = idx_to_row.get((int(d) - 1, int(s)))
                if pivot_row is None:
                    continue
                vals = pivot_arr[pivot_row, neighbor_cols]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    knn_means[r_idx] = float(np.mean(vals))
                    knn_stds[r_idx]  = float(np.std(vals))

        df["knn_lag96_mean"] = knn_means
        df["knn_lag96_std"]  = knn_stds
        return df

    # ── Label encoding ────────────────────────────────────────────────────────

    def _fit_label_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(
                df[col].astype(str).fillna("Unknown")
            ).astype(np.int16)
            self.label_encoders[col] = le
            logger.debug(f"Label-encoded '{col}' → {len(le.classes_)} classes")

        # Geohash integer ID (frequency-stable)
        if "geohash" in df.columns:
            le_gh = LabelEncoder()
            df["geohash_id"] = le_gh.fit_transform(df["geohash"]).astype(np.int16)
            self.label_encoders["geohash"] = le_gh

        return df

    def _apply_label_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, le in self.label_encoders.items():
            src = col if col != "geohash" else "geohash"
            dst = f"{col}_enc" if col != "geohash" else "geohash_id"
            if src not in df.columns:
                continue
            known   = set(le.classes_)
            series  = df[src].astype(str).fillna("Unknown")
            series  = series.where(series.isin(known), other="Unknown")
            # "Unknown" might not be in train classes; handle gracefully
            if "Unknown" not in known:
                series = series.where(series != "Unknown", other=le.classes_[0])
            df[dst] = le.transform(series).astype(np.int16)
        return df

    # ── Lag features ─────────────────────────────────────────────────────────

    def _add_lag_features(
        self,
        df: pd.DataFrame,
        is_train: bool,
        train_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        target = self.target_col
        if target not in df.columns and is_train:
            logger.warning("No target column found — skipping lag features")
            return df

        if is_train:
            logger.info(f"Creating safe lag features for train: {SAFE_LAG_SLOTS}")
            for lag in SAFE_LAG_SLOTS:
                df[f"demand_lag_{lag}"] = (
                    df.groupby("geohash", observed=True)[target]
                    .shift(lag)
                    .astype(np.float32)
                )
        else:
            if train_df is not None and target in train_df.columns:
                logger.info("Creating safe lag features for test using train tail")
                combined = _safe_concat_for_lags(train_df, df)
                for lag in SAFE_LAG_SLOTS:
                    combined[f"demand_lag_{lag}"] = (
                        combined.groupby("geohash", observed=True)[target]
                        .shift(lag)
                        .astype(np.float32)
                    )
                for lag in SAFE_LAG_SLOTS:
                    col = f"demand_lag_{lag}"
                    df[col] = combined[col].iloc[len(train_df):].values
            else:
                logger.warning("No train_df provided — lag features will be NaN for test")
                for lag in SAFE_LAG_SLOTS:
                    df[f"demand_lag_{lag}"] = np.nan

        # Yesterday trend statistics (mean and std of the 5 lags)
        lag_cols = [f"demand_lag_{l}" for l in SAFE_LAG_SLOTS]
        df["demand_yesterday_mean_5"] = df[lag_cols].mean(axis=1).astype(np.float32)
        df["demand_yesterday_std_5"] = df[lag_cols].std(axis=1).astype(np.float32)

        # Fill NaNs with global mean
        fill_val = self.global_mean if self.global_mean != 0.0 else 0.0
        cols_to_fill = lag_cols + ["demand_yesterday_mean_5", "demand_yesterday_std_5"]
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna(fill_val).astype(np.float32)
        return df

    # ── Rolling features ──────────────────────────────────────────────────────

    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        is_train: bool,
        train_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Actually adds regional lags (geohash_5 and geohash_4 shifted by 95, 96, 97)
        to capture historical spatial context from yesterday.
        """
        target = self.target_col
        if target not in df.columns and is_train:
            return df

        def _compute_regional(frame: pd.DataFrame) -> pd.DataFrame:
            frame["geohash_5"] = frame["geohash"].str[:-1]
            frame["geohash_4"] = frame["geohash"].str[:-2]
            
            # Regional means
            reg5_mean = frame.groupby(["day", "time_slot", "geohash_5"], observed=True)[target].transform("mean")
            reg4_mean = frame.groupby(["day", "time_slot", "geohash_4"], observed=True)[target].transform("mean")
            
            frame["reg5_demand"] = reg5_mean
            frame["reg4_demand"] = reg4_mean
            
            # Shifts
            for lag in REGIONAL_LAG_SLOTS:
                frame[f"reg5_demand_lag_{lag}"] = frame.groupby("geohash", observed=True)["reg5_demand"].shift(lag).astype(np.float32)
                frame[f"reg4_demand_lag_{lag}"] = frame.groupby("geohash", observed=True)["reg4_demand"].shift(lag).astype(np.float32)
            
            # Cleanup intermediate columns
            frame.drop(columns=["geohash_5", "geohash_4", "reg5_demand", "reg4_demand"], inplace=True, errors="ignore")
            return frame

        if is_train:
            logger.info("Creating regional lag features for train")
            df = _compute_regional(df)
        else:
            if train_df is not None and target in train_df.columns:
                logger.info("Creating regional lag features for test using train tail")
                combined = _safe_concat_for_lags(train_df, df)
                combined = _compute_regional(combined)
                
                regional_cols = []
                for lag in REGIONAL_LAG_SLOTS:
                    regional_cols.extend([f"reg5_demand_lag_{lag}", f"reg4_demand_lag_{lag}"])
                    
                for col in regional_cols:
                    df[col] = combined[col].iloc[len(train_df):].values
            else:
                for lag in REGIONAL_LAG_SLOTS:
                    df[f"reg5_demand_lag_{lag}"] = np.nan
                    df[f"reg4_demand_lag_{lag}"] = np.nan

        # Fill NaNs with global mean
        fill_val = self.global_mean if self.global_mean != 0.0 else 0.0
        regional_cols = []
        for lag in REGIONAL_LAG_SLOTS:
            regional_cols.extend([f"reg5_demand_lag_{lag}", f"reg4_demand_lag_{lag}"])
            
        for col in regional_cols:
            if col in df.columns:
                df[col] = df[col].fillna(fill_val).astype(np.float32)
        return df

    # ── Target encoding (OOF for train) ──────────────────────────────────────

    def _oof_target_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Out-of-fold mean target encoding for geohash and categorical columns.
        For each fold, encodes using the OTHER folds' statistics to prevent leakage.
        """
        from sklearn.model_selection import KFold
        logger.info(f"Running OOF target encoding with {self.n_folds} folds")

        target       = self.target_col
        encode_cols  = ["geohash"] + [c for c in CATEGORICAL_COLS if c in df.columns]
        kf           = KFold(n_splits=self.n_folds, shuffle=False)

        for col in encode_cols:
            out_col = f"{col}_te"
            df[out_col] = self.global_mean  # default fill

            for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(df)):
                fold_means = df.iloc[trn_idx].groupby(col, observed=True)[target].mean()
                # Cast to str to avoid Categorical dtype fillna crash
                mapped = (
                    df.iloc[val_idx][col]
                    .astype(str)
                    .map(fold_means)
                    .fillna(self.global_mean)
                    .astype(np.float32)
                )
                df.loc[df.index[val_idx], out_col] = mapped.values
            df[out_col] = df[out_col].astype(np.float32)

        # Also compute global geohash means for test-time application
        for col in encode_cols:
            self.cat_means[col] = df.groupby(col, observed=True)[target].mean().to_dict()

        return df

    def _apply_target_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fitted global target means to test set."""
        for col, means_dict in self.cat_means.items():
            out_col = f"{col}_te"
            if col in df.columns:
                df[out_col] = (
                    df[col].astype(str).map(means_dict).fillna(self.global_mean).astype(np.float32)
                )
        return df

    def _add_today_lags(self, df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds demand_today_lag_k for k in 0..8.
        Uses actual demand from the same day at slot k, if time_slot > k.
        """
        target = self.target_col
        # Build pivot of actual demands on ref_df at slots 0..8
        ref_slice = ref_df[ref_df["time_slot"] <= 8][["day", "geohash", "time_slot", target]].copy()
        if ref_slice.empty:
            for k in range(9):
                df[f"demand_today_lag_{k}"] = np.nan
            return df
            
        pivot = ref_slice.pivot_table(
            index=["day", "geohash"], columns="time_slot", values=target, aggfunc="mean"
        )
        # Re-index pivot columns to ensure 0..8 all exist
        for k in range(9):
            if k not in pivot.columns:
                pivot[k] = np.nan
        
        pivot_dict = pivot.to_dict(orient="index") # (day, geohash) -> {0: val, 1: val, ...}
        
        # Build features
        lag_data = {k: np.full(len(df), np.nan, dtype=np.float32) for k in range(9)}
        
        # Group df by (day, geohash) to make dictionary lookups fast
        for (d, gh), group_idx in df.groupby(["day", "geohash"], observed=True).groups.items():
            slots_map = pivot_dict.get((d, gh))
            if slots_map is None:
                continue
            
            group_slots = df["time_slot"].iloc[group_idx].values
            for k in range(9):
                val = slots_map.get(k)
                if pd.isna(val):
                    continue
                # only apply if time_slot > k
                mask = group_slots > k
                if mask.any():
                    lag_data[k][group_idx[mask]] = val
                    
        for k in range(9):
            df[f"demand_today_lag_{k}"] = lag_data[k]
            
        return df

    def _add_diurnal_rolling_features(self, df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds demand_yesterday_rolling_mean_9 and demand_yesterday_rolling_std_9:
        For each row, looks at yesterday's (day - 1) demand in [time_slot - 4, time_slot + 4] (inclusive).
        Also adds yesterday_peak_morning_mean (slots 28-40, i.e. hours 7-10)
        and yesterday_peak_evening_mean (slots 68-80, i.e. hours 17-20).
        """
        target = self.target_col
        fill_val = self.global_mean
        
        # Build pivot on ref_df: (day, geohash) -> time_slot -> demand
        ref_slice = ref_df[["day", "geohash", "time_slot", target]].copy()
        if ref_slice.empty:
            df["demand_yesterday_rolling_mean_9"] = fill_val
            df["demand_yesterday_rolling_std_9"]  = 0.0
            df["yesterday_peak_morning_mean"]     = fill_val
            df["yesterday_peak_evening_mean"]     = fill_val
            return df
            
        pivot = ref_slice.pivot_table(
            index=["day", "geohash"], columns="time_slot", values=target, aggfunc="mean"
        )
        for s in range(96):
            if s not in pivot.columns:
                pivot[s] = np.nan
                
        pivot_dict = pivot.to_dict(orient="index") # (day, geohash) -> {slot: val}
        
        rolling_means = np.full(len(df), fill_val, dtype=np.float32)
        rolling_stds  = np.zeros(len(df), dtype=np.float32)
        peak_morns    = np.full(len(df), fill_val, dtype=np.float32)
        peak_eves     = np.full(len(df), fill_val, dtype=np.float32)
        
        # Group df by (day, geohash) for fast lookups
        for (d, gh), group_idx in df.groupby(["day", "geohash"], observed=True).groups.items():
            # Look up day - 1
            slots_map = pivot_dict.get((d - 1, gh))
            if slots_map is None:
                continue
                
            # Compute peak slot averages on yesterday
            # morning peak: slots 28 to 40 inclusive (hours 7 to 10)
            morn_vals = [slots_map.get(s) for s in range(28, 41) if not pd.isna(slots_map.get(s))]
            if morn_vals:
                peak_morns[group_idx] = float(np.mean(morn_vals))
                
            # evening peak: slots 68 to 80 inclusive (hours 17 to 20)
            eve_vals = [slots_map.get(s) for s in range(68, 81) if not pd.isna(slots_map.get(s))]
            if eve_vals:
                peak_eves[group_idx] = float(np.mean(eve_vals))
                
            # Compute rolling features per row in this group
            group_slots = df["time_slot"].iloc[group_idx].values
            for r_idx, slot in zip(group_idx, group_slots):
                start_s = max(0, slot - 4)
                end_s   = min(95, slot + 4)
                window_vals = [slots_map.get(s) for s in range(start_s, end_s + 1) if not pd.isna(slots_map.get(s))]
                if window_vals:
                    rolling_means[r_idx] = float(np.mean(window_vals))
                    rolling_stds[r_idx]  = float(np.std(window_vals))
                    
        df["demand_yesterday_rolling_mean_9"] = rolling_means
        df["demand_yesterday_rolling_std_9"]  = rolling_stds
        df["yesterday_peak_morning_mean"]     = peak_morns
        df["yesterday_peak_evening_mean"]     = peak_eves
        return df

    # ── Interaction features ─────────────────────────────────────────────────

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Peak-hour binary flags
        if "hour" in df.columns:
            h = df["hour"]
            df["is_peak_morning"] = (
                (h >= PEAK_MORNING[0]) & (h <= PEAK_MORNING[1])
            ).astype(np.int8)
            df["is_peak_evening"] = (
                (h >= PEAK_EVENING[0]) & (h <= PEAK_EVENING[1])
            ).astype(np.int8)
            df["is_night"] = (
                (h >= 23) | (h <= 5)
            ).astype(np.int8)

            # Lanes × hour
            if "NumberofLanes" in df.columns:
                df["lanes_x_hour"] = (
                    df["NumberofLanes"].astype(np.float32) * h.astype(np.float32)
                )

        # Cluster × day-of-week and cluster × time_slot interactions
        if "spatial_cluster" in df.columns:
            if "day_of_week" in df.columns:
                df["cluster_x_dow"] = (
                    df["spatial_cluster"].astype(np.int16) * 7 +
                    df["day_of_week"].astype(np.int16)
                ).astype(np.int16)
            if "time_slot" in df.columns:
                df["cluster_x_slot"] = (
                    df["spatial_cluster"].astype(np.int32) * 96 +
                    df["time_slot"].astype(np.int32)
                ).astype(np.int32)

        # Geohash-hour cross-feature (encode the unique intra-day pattern per location)
        if "geohash_id" in df.columns and "time_slot" in df.columns:
            df["gh_x_slot"] = (
                df["geohash_id"].astype(np.int32) * 96 +
                df["time_slot"].astype(np.int32)
            ).astype(np.int32)

        # Highway flag: 9× demand difference vs non-highway (analysis finding #2)
        # is_highway = True if RoadType==Highway OR NumberofLanes >= 4
        if "RoadType" in df.columns or "NumberofLanes" in df.columns:
            is_hw = pd.Series(False, index=df.index)
            if "RoadType" in df.columns:
                is_hw = is_hw | (df["RoadType"] == "Highway")
            if "NumberofLanes" in df.columns:
                is_hw = is_hw | (df["NumberofLanes"] >= 4)
            df["is_highway"] = is_hw.astype(np.int8)

            # Highway × time_slot: captures rush-hour highway surge
            if "time_slot" in df.columns:
                df["highway_x_slot"] = (
                    df["is_highway"].astype(np.int32) * df["time_slot"].astype(np.int32)
                ).astype(np.int32)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper (not a class method)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_concat_for_lags(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenates train and test along rows for lag/rolling computation.
    The test DataFrame must NOT contain the target column.
    We temporarily fill the target column with NaN for test rows so that
    groupby/shift/rolling operations work on a uniform schema.
    """
    target_cols = [c for c in train_df.columns if c not in test_df.columns]
    test_copy   = test_df.copy()
    for c in target_cols:
        test_copy[c] = np.nan

    combined = pd.concat(
        [train_df, test_copy], axis=0, ignore_index=True
    )
    # Re-sort chronologically
    sort_cols = [c for c in ["day", "time_slot"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    return combined


def get_feature_cols(df: pd.DataFrame, target_col: str = "demand") -> list[str]:
    """
    Returns the list of feature columns to use for model training,
    excluding raw ID/string/target columns.
    """
    drop_always = {
        target_col, "Index", "geohash", "timestamp",
        "RoadType", "LargeVehicles", "Landmarks", "Weather",
    }
    feature_cols = [c for c in df.columns if c not in drop_always]
    return feature_cols
