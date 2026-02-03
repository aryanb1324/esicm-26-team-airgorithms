from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone


# -----------------------------
# Configuration and outputs
# -----------------------------

@dataclass(frozen=True)
class UnitConversionConfig:
    """Unit normalization settings for common labs."""
    creatinine_umolL_to_mgdl: float = 1 / 88.4
    glucose_mmolL_to_mgdl: float = 18.0


@dataclass(frozen=True)
class EligibilityConfig:
    """Eligibility for decision points."""
    max_hours: int = 72
    fio2_threshold: float = 0.40
    sf_ratio_threshold: float = 235.0   # looser early-capture threshold
    horizon_hours: int = 1              # decision horizon for A_intub
    # MV proxy thresholds (replace with true invasive airway flag if available)
    peep_invasive_gt: float = 5.0
    peak_invasive_gt: float = 25.0


@dataclass(frozen=True)
class RecommendationConfig:
    """Decision policy from ARD + uncertainty."""
    ard_intubate_threshold: float = 0.02
    p_harm_threshold: float = 0.75
    ard_wait_threshold: float = -0.01
    p_harm_wait_threshold: float = 0.25


@dataclass
class DecisionOutput:
    visit_occurrence_id: Any
    measure_time: float
    risk_intub_now: float
    risk_wait_1h: float
    ard_wait_minus_intub: float
    ard_ci95: Optional[Tuple[float, float]]
    p_harm_waiting: Optional[float]
    recommendation: str
    rationale: str
    top_drivers: List[Tuple[str, float]]
    action: Dict[str, Any]


# -----------------------------
# Core agent
# -----------------------------

class IntubationTimingAgent:
    """
    Agent 1: Intubation timing decision support using a T-learner:
      - model_intub: outcome model fit on A=1 rows
      - model_wait:  outcome model fit on A=0 rows
    Counterfactual risks are derived by scoring the same state under both models.

    Notes:
      - This is NOT a causal estimator by itself unless weights w are valid
        (e.g., IPW from a correctly specified treatment model under an emulated trial).
      - The code assumes an hourly "long" table with a visit identifier.
    """

    def __init__(
        self,
        features: List[str],
        outcome_col: str = "Y_30d",
        treatment_col: str = "A_intub",
        weight_col: str = "w",
        visit_col: str = "visit_occurrence_id",
        time_col: str = "measure_time",
        unit_cfg: UnitConversionConfig = UnitConversionConfig(),
        elig_cfg: EligibilityConfig = EligibilityConfig(),
        rec_cfg: RecommendationConfig = RecommendationConfig(),
        random_state: int = 42,
    ):
        self.features = features
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.weight_col = weight_col
        self.visit_col = visit_col
        self.time_col = time_col

        self.unit_cfg = unit_cfg
        self.elig_cfg = elig_cfg
        self.rec_cfg = rec_cfg
        self.random_state = random_state

        self.model_intub: Optional[CalibratedClassifierCV] = None
        self.model_wait: Optional[CalibratedClassifierCV] = None

        self._fitted = False

    # -----------------------------
    # Preprocessing utilities
    # -----------------------------

    @staticmethod
    def normalize_oxygen_saturation(series: pd.Series) -> pd.Series:
        """
        Normalize SaO2/SpO2 to percent (0–100).
        Mixed fraction (0–1.1) and percent (1.1–100). Else NaN.
        """
        s = series.copy()
        out = pd.Series(np.nan, index=s.index, dtype="float")
        frac_mask = (s >= 0) & (s <= 1.1)
        out.loc[frac_mask] = s.loc[frac_mask] * 100.0
        pct_mask = (s > 1.1) & (s <= 100)
        out.loc[pct_mask] = s.loc[pct_mask]
        return out

    @staticmethod
    def normalize_temperature_to_celsius(series: pd.Series) -> pd.Series:
        """
        Celsius (30–45) or Fahrenheit (86–113) -> Celsius. Else NaN.
        """
        s = series.copy()
        out = pd.Series(np.nan, index=s.index, dtype="float")
        c_mask = (s >= 30) & (s <= 45)
        out.loc[c_mask] = s.loc[c_mask]
        f_mask = (s >= 86) & (s <= 113)
        out.loc[f_mask] = (s.loc[f_mask] - 32) * 5.0 / 9.0
        return out

    @staticmethod
    def enforce_feature_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        df = df.copy()
        for col, (low, high) in bounds.items():
            if col not in df.columns:
                continue
            df.loc[(df[col] < low) | (df[col] > high), col] = np.nan
        return df

    def apply_unit_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        creatinine_cols = [c for c in df.columns if "creatinine" in c.lower()]
        if creatinine_cols:
            df[creatinine_cols] = df[creatinine_cols] * self.unit_cfg.creatinine_umolL_to_mgdl

        glucose_cols = [c for c in df.columns if "glucose" in c.lower()]
        if glucose_cols:
            df[glucose_cols] = df[glucose_cols] * self.unit_cfg.glucose_mmolL_to_mgdl

        sao2_cols = [c for c in df.columns if "sao2" in c.lower()]
        for c in sao2_cols:
            df[c] = self.normalize_oxygen_saturation(df[c])

        spo2_cols = [c for c in df.columns if "spo2" in c.lower()]
        for c in spo2_cols:
            df[c] = self.normalize_oxygen_saturation(df[c])

        # Convert FiO2 percent to fraction if needed
        if "fio2_mean" in df.columns:
            # safeguard: if already 0.21-1, keep; if 21-100, convert
            df["fio2_mean"] = np.where(df["fio2_mean"] > 1.5, df["fio2_mean"] / 100.0, df["fio2_mean"])

        temp_cols = [c for c in df.columns if c.lower().startswith("temp")]
        for c in temp_cols:
            df[c] = self.normalize_temperature_to_celsius(df[c])

        return df

    def add_outcome_and_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds:
          - hour_ts (optional if visit_start_datetime exists)
          - Y_30d using death_hours <= 30 days
        """
        df = df.copy()

        if "visit_start_datetime" in df.columns:
            df["visit_start_datetime"] = pd.to_datetime(df["visit_start_datetime"])
            if "measure_time" in df.columns:
                df["hour_ts"] = df["visit_start_datetime"] + pd.to_timedelta(df["measure_time"], unit="h")

        # 30-day mortality label
        if "death_hours" in df.columns:
            H_HOURS = 24 * 30
            df["Y_30d"] = ((df["death_hours"].notna()) & (df["death_hours"] <= H_HOURS)).astype(int)

        return df

    def truncate_to_observation_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep 0 <= measure_time < MAX_HOURS,
        and remove hours beyond LOS, and after death.
        """
        df = df.copy()
        max_hours = self.elig_cfg.max_hours

        df = df[(df[self.time_col] >= 0) & (df[self.time_col] < max_hours)].copy()

        if "length_of_stay_hours" in df.columns:
            df = df[df[self.time_col] <= df["length_of_stay_hours"]].copy()

        if "death_hours" in df.columns:
            df = df[(df["death_hours"].isna()) | (df[self.time_col] < df["death_hours"])].copy()

        return df

    def add_sf_ratio_and_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "spo2_mean" in df.columns and "fio2_mean" in df.columns:
            df["sf_ratio"] = df["spo2_mean"] / df["fio2_mean"]
            df["sf_ratio"] = df["sf_ratio"].replace([np.inf, -np.inf], np.nan)

        df = df.sort_values([self.visit_col, self.time_col]).copy()

        def add_lagged_deltas(col: str) -> None:
            prev = f"{col}_prev"
            d1 = f"d_{col}_1h"
            df[prev] = df.groupby(self.visit_col)[col].shift(1)
            df[d1] = df[col] - df[prev]

        for c in ["sf_ratio", "spo2_mean", "fio2_mean", "map_mean", "sbp_mean", "dbp_mean",
                  "creatinine_mean", "crp_mean"]:
            if c in df.columns:
                add_lagged_deltas(c)

        return df

    # -----------------------------
    # Eligibility + treatment construction
    # -----------------------------

    def build_cohort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds eligible decision points and constructs:
          - invasive_proxy
          - t_intubation (first hour where invasive_proxy=1)
          - already_intubated
          - A_intub (intubation within next horizon)
          - Y_30d (ensures not missing)
        """
        df = df.copy()

        # Invasive proxy (replace with true flag if present)
        if "is_invasive_vent" in df.columns:
            df["invasive_proxy"] = df["is_invasive_vent"].fillna(0).astype(int)
        else:
            peep_ok = df.get("peep_mean", pd.Series(False, index=df.index)) > self.elig_cfg.peep_invasive_gt
            peak_ok = df.get("peak_mean", pd.Series(False, index=df.index)) > self.elig_cfg.peak_invasive_gt
            df["invasive_proxy"] = (peep_ok & peak_ok).fillna(False).astype(int)

        # Eligibility: FiO2 and hypoxemia via SF ratio
        df["fio2_qualified"] = df["fio2_mean"] >= self.elig_cfg.fio2_threshold
        df["hypoxemia_qualified"] = df["sf_ratio"] <= self.elig_cfg.sf_ratio_threshold

        # First intubation time
        mv_start_times = df[df["invasive_proxy"] == 1].groupby(self.visit_col)[self.time_col].min()
        df = df.merge(mv_start_times.rename("t_intubation"), on=self.visit_col, how="left")

        # Exclusion: already intubated at/after first intub time
        df["already_intubated"] = df["t_intubation"].notna() & (df[self.time_col] >= df["t_intubation"])

        cohort = df[
            (df["hypoxemia_qualified"]) &
            (df["fio2_qualified"]) &
            (~df["already_intubated"])
        ].copy()

        # Treatment: intubate within next horizon
        h = self.elig_cfg.horizon_hours
        cohort[self.treatment_col] = (
            cohort["t_intubation"].notna() &
            (cohort["t_intubation"] > cohort[self.time_col]) &
            (cohort["t_intubation"] <= cohort[self.time_col] + h)
        ).astype(int)

        # Outcome: ensure present
        if self.outcome_col in cohort.columns:
            cohort[self.outcome_col] = cohort[self.outcome_col].fillna(0).astype(int)

        return cohort

    # -----------------------------
    # Model fitting (T-learner)
    # -----------------------------

    def _make_base_model(self) -> CalibratedClassifierCV:
        base = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=self.random_state,
        )
        # isotonic calibration is reasonable if sample sizes are adequate; otherwise sigmoid
        return CalibratedClassifierCV(base, method="isotonic", cv=3)

    def fit(self, df_hourly: pd.DataFrame, bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> "IntubationTimingAgent":
        """
        Full pipeline:
          1) unit conversions
          2) time/outcome
          3) truncate observation window
          4) SF + deltas
          5) bounds (optional)
          6) eligibility + A
          7) fit two calibrated models (A=1 and A=0)
        """
        df = df_hourly.copy()
        df = self.apply_unit_conversions(df)
        df = self.add_outcome_and_time(df)
        df = self.truncate_to_observation_window(df)
        df = self.add_sf_ratio_and_deltas(df)
        if bounds is not None:
            df = self.enforce_feature_bounds(df, bounds)

        cohort = self.build_cohort(df)

        # Feature availability check
        missing = [f for f in self.features if f not in cohort.columns]
        if missing:
            raise ValueError(f"Missing required features in cohort: {missing}")

        # Fit A=1 model
        df_1 = cohort[cohort[self.treatment_col] == 1]
        df_0 = cohort[cohort[self.treatment_col] == 0]
        if df_1.empty or df_0.empty:
            raise ValueError("Cohort has empty A=1 or A=0 strata; cannot fit T-learner.")

        self.model_intub = self._make_base_model()
        self.model_wait = self._make_base_model()

        X1 = df_1[self.features]
        y1 = df_1[self.outcome_col]
        X0 = df_0[self.features]
        y0 = df_0[self.outcome_col]

        # Sample weights are optional; if present, use. Otherwise train unweighted.
        if self.weight_col in cohort.columns:
            w1 = df_1[self.weight_col]
            w0 = df_0[self.weight_col]
            self.model_intub.fit(X1, y1, sample_weight=w1)
            self.model_wait.fit(X0, y0, sample_weight=w0)
        else:
            self.model_intub.fit(X1, y1)
            self.model_wait.fit(X0, y0)

        self._fitted = True
        return self

    # -----------------------------
    # Counterfactual prediction
    # -----------------------------

    def _predict_counterfactuals(self, row: pd.Series) -> Tuple[float, float]:
        if not self._fitted or self.model_intub is None or self.model_wait is None:
            raise RuntimeError("Agent not fitted. Call fit() first.")

        X = pd.DataFrame([row])[self.features]
        r_intub = float(self.model_intub.predict_proba(X)[:, 1][0])
        r_wait = float(self.model_wait.predict_proba(X)[:, 1][0])
        return r_intub, r_wait

    # -----------------------------
    # Bootstrap CI (cluster bootstrap by visit)
    # -----------------------------

    def bootstrap_ci_for_row(
        self,
        row: pd.Series,
        df_train: pd.DataFrame,
        n_boot: int = 200,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Cluster bootstrap by visit_occurrence_id.
        Refit both models per bootstrap draw, then score the row.
        """
        rng = np.random.default_rng(random_state)
        visits = df_train[self.visit_col].unique()

        r1_list, r0_list, ard_list = [], [], []

        for _ in range(n_boot):
            boot_visits = rng.choice(visits, size=len(visits), replace=True)
            boot_df = df_train[df_train[self.visit_col].isin(boot_visits)].copy()

            df_1 = boot_df[boot_df[self.treatment_col] == 1]
            df_0 = boot_df[boot_df[self.treatment_col] == 0]
            if df_1.empty or df_0.empty:
                continue

            m1 = self._make_base_model()
            m0 = self._make_base_model()

            if self.weight_col in boot_df.columns:
                m1.fit(df_1[self.features], df_1[self.outcome_col], sample_weight=df_1[self.weight_col])
                m0.fit(df_0[self.features], df_0[self.outcome_col], sample_weight=df_0[self.weight_col])
            else:
                m1.fit(df_1[self.features], df_1[self.outcome_col])
                m0.fit(df_0[self.features], df_0[self.outcome_col])

            X = pd.DataFrame([row])[self.features]
            r1 = float(m1.predict_proba(X)[:, 1][0])
            r0 = float(m0.predict_proba(X)[:, 1][0])
            ard = r0 - r1

            r1_list.append(r1)
            r0_list.append(r0)
            ard_list.append(ard)

        if len(ard_list) < max(30, int(0.3 * n_boot)):
            return {
                "risk_intub_now_ci95": None,
                "risk_wait_1h_ci95": None,
                "ard_ci95": None,
                "p_harm": None,
                "boot_draws": len(ard_list),
            }

        def ci(a: np.ndarray) -> Tuple[float, float]:
            return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

        r1_arr = np.array(r1_list)
        r0_arr = np.array(r0_list)
        ard_arr = np.array(ard_list)

        return {
            "risk_intub_now_ci95": ci(r1_arr),
            "risk_wait_1h_ci95": ci(r0_arr),
            "ard_ci95": ci(ard_arr),
            "p_harm": float(np.mean(ard_arr > 0.0)),  # P(waiting increases risk)
            "boot_draws": int(len(ard_arr)),
        }

    # -----------------------------
    # Driver explanation (perturbation)
    # -----------------------------

    def explain_ard_drivers(self, row: pd.Series, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perturbation-based attribution:
          impact(f) = ARD_base - ARD_after_improving_feature_f
        """
        r_intub_base, r_wait_base = self._predict_counterfactuals(row)
        ard_base = r_wait_base - r_intub_base

        drivers: List[Tuple[str, float]] = []
        for f in self.features:
            if f in [self.time_col, "age", self.visit_col]:
                continue
            if pd.isna(row.get(f, np.nan)):
                continue

            row_imp = row.copy()

            # Heuristic: improve oxygenation/hemodynamics/pH/Hgb; reduce "bad" labs otherwise
            if any(token in f for token in ["spo2", "sf_ratio", "map", "ph", "hemoglobin"]):
                row_imp[f] = row[f] * 1.10
            else:
                row_imp[f] = row[f] * 0.90

            r_i_new, r_w_new = self._predict_counterfactuals(row_imp)
            ard_new = r_w_new - r_i_new
            impact = ard_base - ard_new
            drivers.append((f, float(impact)))

        drivers.sort(key=lambda x: x[1], reverse=True)
        return drivers[:top_k]

    # -----------------------------
    # Decision logic
    # -----------------------------

    def _make_recommendation(self, ard: float, p_harm: Optional[float]) -> Tuple[str, str]:
        """
        Maps (ARD, p_harm) -> recommendation.
        If p_harm is None, fall back to ARD-only thresholds.
        """
        cfg = self.rec_cfg
        if p_harm is None:
            if ard > cfg.ard_intubate_threshold:
                return "intubate_now", "ARD exceeds threshold (uncertainty not computed)."
            if ard < cfg.ard_wait_threshold:
                return "wait_monitor", "ARD suggests benefit to defer (uncertainty not computed)."
            return "equivocal", "ARD near zero (uncertainty not computed)."

        if (ard > cfg.ard_intubate_threshold) and (p_harm > cfg.p_harm_threshold):
            return "intubate_now", "Estimated harm of waiting exceeds threshold with high posterior probability."
        if (ard < cfg.ard_wait_threshold) and (p_harm < cfg.p_harm_wait_threshold):
            return "wait_monitor", "Estimated benefit of waiting with low probability that waiting is harmful."
        return "equivocal", "Counterfactual difference is small and/or uncertainty overlaps zero."

    def predict_one(
        self,
        row: pd.Series,
        compute_ci: bool = False,
        df_train_for_ci: Optional[pd.DataFrame] = None,
        n_boot: int = 200,
    ) -> DecisionOutput:
        """
        Returns orchestrator-ready decision support for a single row.
        """
        r_intub, r_wait = self._predict_counterfactuals(row)
        ard = r_wait - r_intub

        ard_ci = None
        p_harm = None
        if compute_ci:
            if df_train_for_ci is None:
                raise ValueError("df_train_for_ci must be provided when compute_ci=True.")
            ci_obj = self.bootstrap_ci_for_row(row=row, df_train=df_train_for_ci, n_boot=n_boot)
            ard_ci = ci_obj["ard_ci95"]
            p_harm = ci_obj["p_harm"]

        rec, rationale = self._make_recommendation(ard, p_harm)
        drivers = self.explain_ard_drivers(row)

        action = {
            "recommendation": rec,
            "horizon_hours": self.elig_cfg.horizon_hours,
        }

        return DecisionOutput(
            visit_occurrence_id=row.get(self.visit_col, None),
            measure_time=float(row.get(self.time_col, np.nan)),
            risk_intub_now=float(r_intub),
            risk_wait_1h=float(r_wait),
            ard_wait_minus_intub=float(ard),
            ard_ci95=ard_ci,
            p_harm_waiting=p_harm,
            recommendation=rec,
            rationale=rationale,
            top_drivers=drivers,
            action=action,
        )
