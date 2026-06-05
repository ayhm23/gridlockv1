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
LAG_SLOTS         = [1, 2, 4, 8, 96]          # 15-min slots
ROLLING_WINDOWS   = [3, 6, 12]                 # trailing windows
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
        df = self._ensure_temporal_cols(df)
        df = self._apply_spatial_clusters(df)
        df = self._apply_label_encode(df)

        # For lag/rolling on test: concatenate train tail with test, compute, then slice
        df = self._add_lag_features(df, is_train=False, train_df=train_df)
        df = self._add_rolling_features(df, is_train=False, train_df=train_df)

        # Apply (global) target encoding using fitted means
        df = self._apply_target_encode(df)
        df = self._add_interaction_features(df)

        df = reduce_mem_usage(df, verbose=False)
        logger.info(f"transform complete — final shape: {df.shape}")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

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
            logger.info(f"Creating lag features for train: {LAG_SLOTS}")
            for lag in LAG_SLOTS:
                df[f"demand_lag_{lag}"] = (
                    df.groupby("geohash", observed=True)[target]
                    .shift(lag)
                    .astype(np.float32)
                )
        else:
            if train_df is not None and target in train_df.columns:
                logger.info("Creating lag features for test using train tail")
                # Concatenate, compute lags, then extract test rows
                combined = _safe_concat_for_lags(train_df, df)
                for lag in LAG_SLOTS:
                    combined[f"demand_lag_{lag}"] = (
                        combined.groupby("geohash", observed=True)[target]
                        .shift(lag)
                        .astype(np.float32)
                    )
                test_idx = df.index if not df.index.duplicated().any() else range(len(train_df), len(combined))
                for lag in LAG_SLOTS:
                    col = f"demand_lag_{lag}"
                    df[col] = combined[col].iloc[len(train_df):].values
            else:
                logger.warning("No train_df provided — lag features will be NaN for test")
                for lag in LAG_SLOTS:
                    df[f"demand_lag_{lag}"] = np.nan

        # Fill NaN lags with global mean (or 0 if no target)
        fill_val = self.global_mean if self.global_mean != 0.0 else 0.0
        for lag in LAG_SLOTS:
            col = f"demand_lag_{lag}"
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
        target = self.target_col
        if target not in df.columns and is_train:
            return df

        def _compute_rolling(frame: pd.DataFrame) -> pd.DataFrame:
            grp = frame.groupby("geohash", observed=True)[target]
            for w in ROLLING_WINDOWS:
                frame[f"demand_roll_mean_{w}"] = (
                    grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                    .astype(np.float32)
                )
            frame["demand_roll_std_6"] = (
                grp.transform(lambda x: x.shift(1).rolling(6, min_periods=2).std())
                .astype(np.float32)
            )
            frame["demand_roll_max_6"] = (
                grp.transform(lambda x: x.shift(1).rolling(6, min_periods=1).max())
                .astype(np.float32)
            )
            return frame

        if is_train:
            logger.info(f"Creating rolling features for train — windows: {ROLLING_WINDOWS}")
            df = _compute_rolling(df)
        else:
            if train_df is not None and target in train_df.columns:
                logger.info("Creating rolling features for test using train tail")
                combined = _safe_concat_for_lags(train_df, df)
                combined = _compute_rolling(combined)
                roll_cols = (
                    [f"demand_roll_mean_{w}" for w in ROLLING_WINDOWS]
                    + ["demand_roll_std_6", "demand_roll_max_6"]
                )
                for col in roll_cols:
                    df[col] = combined[col].iloc[len(train_df):].values
            else:
                for w in ROLLING_WINDOWS:
                    df[f"demand_roll_mean_{w}"] = np.nan
                df["demand_roll_std_6"] = np.nan
                df["demand_roll_max_6"] = np.nan

        # Fill NaN rolling with global mean
        fill_val = self.global_mean if self.global_mean != 0.0 else 0.0
        roll_cols = (
            [f"demand_roll_mean_{w}" for w in ROLLING_WINDOWS]
            + ["demand_roll_std_6", "demand_roll_max_6"]
        )
        for col in roll_cols:
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
