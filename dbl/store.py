"""MomentStore: the analyst-facing data layer (load, derive, filter, aggregate, evidence).

Everything the app and the LLM can do goes through this class, so answers are
always computed by pandas on the real tables -- the LLM never invents numbers.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from .config import BUILD_DIR, MATCHES
from .framework import FINDINGS

TABLES = ("duels", "passes", "runs", "recoveries")
FORWARD_RUNS = {"run_ahead_of_the_ball", "behind", "cross_receiver", "overlap", "underlap", "pulling_half_space"}


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def _derive(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    def fb(cond, valid):  # boolean as float with NaN where undefined
        return np.where(valid, cond.astype(float), np.nan)

    if name == "passes":
        df["receiver_free"] = fb(df.receiver_pressure_id <= 1, df.receiver_pressure_id.notna())
        df["receiver_low_pressure"] = fb(df.receiver_pressure_id <= 2, df.receiver_pressure_id.notna())
        df["checked"] = fb(df.shoulder_checks > 0, df.shoulder_checks.notna())
        df["dual_vision"] = fb(df.vision == "ball_and_man", df.vision.notna())
        df["marking_band"] = pd.cut(df.marking_distance, [0, 3, 6, 9, 99], labels=["<3m", "3-6m", "6-9m", ">9m"]).astype(str)
    if name == "runs":
        df["forward_run"] = df.subtype.isin(FORWARD_RUNS)
        df["square_body"] = fb(df.body_shape == "square", df.body_shape.notna())
        for c in ("blind_side", "runner_in_view", "beaten_in_behind", "reacted"):
            if c in df:
                df[c] = pd.to_numeric(df[c].map({True: 1.0, False: 0.0}), errors="coerce")
    if name == "duels":
        df["side_on"] = fb(df.body_shape == "side_on", df.body_shape.notna())
        for c in ("loaded_stance", "eyes_on_ball"):
            if c in df:
                df[c] = pd.to_numeric(df[c].map({True: 1.0, False: 0.0}), errors="coerce")
        df["distance_band"] = pd.cut(df.distance_m, [0, 3, 6, 99], labels=["<3m", "3-6m", ">6m"]).astype(str)
    if name == "recoveries":
        df["turned_within_1s"] = pd.to_numeric(df["turned_within_1s"].map({True: 1.0, False: 0.0}), errors="coerce")
    df["time"] = df.minute.astype(int).astype(str).str.zfill(2) + ":" + df.second.astype(int).astype(str).str.zfill(2)
    return df


class MomentStore:
    def __init__(self, build_dir=BUILD_DIR, matches=tuple(MATCHES)):
        self.tables: dict[str, pd.DataFrame] = {}
        for t in TABLES:
            parts = [pd.read_parquet(p) for m in matches if (p := build_dir / str(m) / f"{t}.parquet").exists()]
            self.tables[t] = _derive(t, pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()
        pl = [pd.read_parquet(p) for m in matches if (p := build_dir / str(m) / "players.parquet").exists()]
        self.players = pd.concat(pl, ignore_index=True) if pl else pd.DataFrame()

    # ----------------------------------------------------------- basics
    @cached_property
    def teams(self) -> list[str]:
        return sorted(set().union(*[set(t.defending_team.dropna()) for t in self.tables.values() if len(t)]))

    def schema(self) -> dict[str, list[str]]:
        return {k: list(v.columns) for k, v in self.tables.items()}

    def filter(self, table: str, where: dict | None = None, query: str | None = None) -> pd.DataFrame:
        """`where`: {column: value | [values] | {"min": a, "max": b}}; `query`: pandas query string."""
        df = self.tables[table]
        for col, v in (where or {}).items():
            if col not in df.columns:
                raise KeyError(f"unknown column '{col}' in {table}")
            if isinstance(v, dict):
                if "min" in v:
                    df = df[df[col] >= v["min"]]
                if "max" in v:
                    df = df[df[col] <= v["max"]]
            elif isinstance(v, (list, tuple, set)):
                df = df[df[col].isin(list(v))]
            elif isinstance(v, str) and (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
                df = df[df[col].astype(str).str.lower() == v.lower()]
            else:
                df = df[df[col] == v]
        if query:
            df = df.query(query, engine="python")
        return df

    def aggregate(self, table: str, metrics: list[str], group_by: list[str] | None = None,
                  where: dict | None = None, query: str | None = None, min_n: int = 1) -> pd.DataFrame:
        df = self.filter(table, where, query)
        cols = [m for m in metrics if m in df.columns]
        if not cols:
            raise KeyError(f"none of {metrics} in {table}")
        num = df[cols].apply(lambda s: s.astype(float) if s.dtype == bool or s.dtype == object and set(s.dropna().unique()) <= {True, False} else s)
        num = num.apply(pd.to_numeric, errors="coerce")
        if group_by:
            g = num.join(df[group_by]).groupby(group_by, dropna=False)
            out = g[cols].mean().round(3)
            out["n"] = g.size()
            out = out[out.n >= min_n].sort_values("n", ascending=False)
            return out.reset_index()
        out = num.mean().round(3).to_frame("value").T
        out["n"] = len(df)
        return out

    # ------------------------------------------------------- profiles
    def player_profile(self, min_moments: int = 5) -> pd.DataFrame:
        d, p, r, rc = (self.tables[t] for t in TABLES)
        parts = [
            d.groupby(["defender_id", "defender", "defending_team"]).agg(
                duels=("event_id", "size"), side_on=("side_on", "mean"), loaded=("loaded_stance", "mean"),
                regain=("regain", "mean"), carrier_progressed=("carrier_progressed", "mean")),
            p.dropna(subset=["defender_id"]).groupby(["defender_id", "defender", "defending_team"]).agg(
                passes_marked=("event_id", "size"), dual_vision=("dual_vision", "mean"),
                checks_per_pass=("shoulder_checks", "mean"), receiver_free=("receiver_free", "mean")),
            r.groupby(["defender_id", "defender", "defending_team"]).agg(
                runs_tracked=("event_id", "size"), blind_side=("blind_side", "mean"),
                square_at_run=("square_body", "mean"), reaction_lag_s=("reaction_lag_s", "mean"),
                beaten_in_behind=("beaten_in_behind", "mean")),
        ]
        for i, x in enumerate(parts):
            x.index = x.index.set_levels(x.index.levels[0].astype("int64"), level=0)
            parts[i] = x
        out = pd.concat(parts, axis=1).reset_index()
        out.columns = ["defender_id", "defender", "team", *out.columns[3:]]
        out = out.merge(self.players.drop_duplicates("player_id")[["player_id", "role"]],
                        left_on="defender_id", right_on="player_id", how="left").drop(columns="player_id")
        tot = out[["duels", "passes_marked", "runs_tracked"]].fillna(0).sum(1)
        return out[tot >= min_moments].round(3).sort_values("team")

    def team_profile(self) -> pd.DataFrame:
        d, p, r = self.tables["duels"], self.tables["passes"], self.tables["runs"]
        rf = r[r.forward_run]
        return pd.concat([
            d.groupby("defending_team").agg(duels=("event_id", "size"), side_on=("side_on", "mean"),
                                            loaded=("loaded_stance", "mean"), regain=("regain", "mean")),
            p.groupby("defending_team").agg(dual_vision=("dual_vision", "mean"), checks_per_pass=("shoulder_checks", "mean"),
                                            receiver_free=("receiver_free", "mean")),
            rf.groupby("defending_team").agg(blind_side_runs=("blind_side", "mean"), square_at_run=("square_body", "mean"),
                                             beaten_in_behind=("beaten_in_behind", "mean"),
                                             reaction_lag_s=("reaction_lag_s", "mean")),
        ], axis=1).round(3).reset_index().rename(columns={"defending_team": "team"})

    # ------------------------------------------------------- evidence
    def evidence(self, finding: dict, n_boot: int = 1000, seed: int = 7) -> dict:
        df = self.tables[finding["table"]]
        flt = finding.get("filter")
        if flt == "forward_run":
            df = df[df.forward_run]
        elif flt:
            df = df.query(flt)
        e, o = finding["exposure"], finding["outcome"]
        df = df.dropna(subset=[e, o]).copy()
        df[e] = df[e].astype(bool)
        df[o] = df[o].astype(float)
        strata = [s for s in finding.get("strata", []) if s in df.columns]
        key = df[strata].astype(str).agg("|".join, axis=1) if strata else pd.Series("all", index=df.index)
        df["_s"] = key.to_numpy()

        def strat_rd(x: pd.DataFrame) -> float:
            num = den = 0.0
            for _, g in x.groupby("_s"):
                a, b = g[g[e]], g[~g[e]]
                if len(a) and len(b):
                    w = len(a) * len(b) / len(g)
                    num += w * (a[o].mean() - b[o].mean())
                    den += w
            return num / den if den else np.nan

        rd = strat_rd(df)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        boots = [strat_rd(df.iloc[rng.choice(idx, len(idx))]) for _ in range(n_boot)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        ex, un = df[df[e]], df[~df[e]]
        return {
            **{k: finding[k] for k in ("id", "principle", "cue", "exposure_label", "outcome", "better")},
            "n_exposed": len(ex), "n_other": len(un),
            "rate_exposed": ex[o].mean(), "rate_other": un[o].mean(),
            "ci_exposed": wilson(ex[o].sum(), len(ex)), "ci_other": wilson(un[o].sum(), len(un)),
            "adj_diff": rd, "adj_diff_ci": (lo, hi), "strata": strata,
        }

    def all_evidence(self, n_boot: int = 1000, use_cache: bool = True) -> pd.DataFrame:
        """All pre-registered findings. Cached in data/build/evidence.json (written by `dbl build`)."""
        cache = BUILD_DIR / "evidence.json"
        if use_cache and cache.exists():
            return pd.read_json(cache, orient="records")
        rows = [self.evidence(f, n_boot) for f in FINDINGS]
        df = pd.DataFrame(rows)
        for c in ("ci_exposed", "ci_other", "adj_diff_ci"):
            df[c] = df[c].apply(lambda t: [round(float(x), 4) for x in t])
        df.to_json(cache, orient="records", indent=1)
        return df

    # ------------------------------------------------------- clips
    def moment(self, table: str, event_id, match_id: int | None = None) -> dict:
        df = self.tables[table]
        sel = df[df.event_id == event_id]
        if match_id is not None:
            sel = sel[sel.match_id == match_id]
        if sel.empty:
            raise KeyError(f"{table} moment {event_id} not found")
        return sel.iloc[0].dropna().to_dict()
