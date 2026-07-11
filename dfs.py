import argparse
import csv
import heapq
import itertools
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


def dk_points(pts, fg3, trb, ast, stl, blk, tov) -> float:
    score = (pts * 1.0 + fg3 * 0.5 + trb * 1.25 + ast * 1.5
             + stl * 2.0 + blk * 2.0 - tov * 0.5)
    cats10 = sum(1 for v in (pts, trb, ast, stl, blk) if v >= 10)
    if cats10 >= 2:
        score += 1.5
    if cats10 >= 3:
        score += 3.0
    return score


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


TEAM_ALIASES = {

    "MINNESOTA LYNX": "MIN",
    "GOLDEN STATE VALKYRIES": "GSV",
    "NEW YORK LIBERTY": "NYL",
    "DALLAS WINGS": "DAL",
    "INDIANA FEVER": "IND",
    "ATLANTA DREAM": "ATL",
    "LAS VEGAS ACES": "LVA",
    "TORONTO TEMPO": "TOR",
    "PHOENIX MERCURY": "PHO",
    "CHICAGO SKY": "CHI",
    "WASHINGTON MYSTICS": "WAS",
    "SEATTLE STORM": "SEA",
    "LOS ANGELES SPARKS": "LAS",
    "CONNECTICUT SUN": "CON",
    "PORTLAND FIRE": "POR",


    "PHX": "PHO", "PHO": "PHO",
    "LV": "LVA", "LVA": "LVA", "LVS": "LVA", "LAS VEGAS": "LVA",
    "NY": "NYL", "NYL": "NYL",
    "CONN": "CON", "CON": "CON",
    "WSH": "WAS", "WAS": "WAS",
    "GS": "GSV", "GSV": "GSV",
    "LA": "LAS", "LAS": "LAS", "LAX": "LAS",
    "PDX": "POR", "POR": "POR",
}


def norm_team(code: str) -> str:
    c = str(code).strip().upper()
    return TEAM_ALIASES.get(c, c)


def classify_pos(pos: str) -> str:
    """Collapse any position string to 'G' or 'F' (centers count as F)."""
    p = str(pos).strip().upper()
    return "G" if p.startswith("G") else "F"


def load_gamelogs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = ["pts", "fg3", "trb", "ast", "stl", "blk", "tov",
                "orb", "drb", "fg", "fga", "ft", "fta"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["pts"])
    df["date"] = pd.to_datetime(df["date_game"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    def mp_to_float(v):
        try:
            if isinstance(v, str) and ":" in v:
                m, s = v.split(":")
                return int(m) + int(s) / 60.0
            return float(v)
        except Exception:
            return math.nan
    df["minutes"] = df["mp"].map(mp_to_float)
    df["team"] = df["team_id"].map(norm_team)
    df["opp"] = df["opp_id"].map(norm_team)
    df["is_home"] = df["game_location"].fillna("") != "@"
    df["dkpts"] = [
        dk_points(r.pts, r.fg3 or 0, r.trb or 0, r.ast or 0,
                  r.stl or 0, r.blk or 0, r.tov or 0)
        for r in df.itertuples()
    ]
    df["name_key"] = df["player_name"].map(norm_name)
    return df


def load_positions(rosters_path: Path | None) -> dict:
    """name_key -> 'G'/'F' from your scraped rosters (used for defense-vs-position)."""
    if not rosters_path or not rosters_path.exists():
        return {}
    r = pd.read_csv(rosters_path)
    r = r.sort_values("season")
    out = {}
    for row in r.itertuples():
        out[norm_name(row.player_name)] = classify_pos(row.pos)
    return out


def load_salaries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["team"] = df["TeamAbbrev"].map(norm_team)
    df["slot"] = df["Roster Position"].map(lambda s: "G" if "G" in str(s).split("/") else "F")

    opps, games, homes = [], [], []
    for gi, tm in zip(df["Game Info"], df["team"]):
        m = re.match(r"(\w+)@(\w+)", str(gi))
        if m:
            away, home = norm_team(m.group(1)), norm_team(m.group(2))
            games.append(f"{away}@{home}")
            if tm == home:
                opps.append(away); homes.append(True)
            else:
                opps.append(home); homes.append(False)
        else:
            games.append(str(gi)); opps.append(""); homes.append(True)
    df["game"] = games
    df["opp"] = opps
    df["is_home"] = homes
    df["name_key"] = df["Name"].map(norm_name)
    return df


def slate_date_from_salaries(sal: pd.DataFrame):
    """Pull the slate date out of DK's 'Game Info' column (e.g. 'DAL@NYL 07/07/2026 08:00PM ET')."""
    for gi in sal["Game Info"]:
        m = re.search(r"(\d{2}/\d{2}/\d{4})", str(gi))
        if m:
            return pd.to_datetime(m.group(1), format="%m/%d/%Y")
    return None


def fetch_live_injuries(timeout: int = 20) -> tuple[set, dict]:
    """ESPN's live WNBA injury feed. Returns ({name_key OUT}, {name_key: status} for GTD).

    This catches late scratches and coach's decisions that gamelogs can never see —
    the single biggest source of 0-point roster spots.
    """
    import json
    import urllib.request
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"
    out, gtd = set(), {}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        print(f"\n!! Live injury fetch FAILED ({e}) — relying on injuries.txt only.")
        return out, gtd
    for team in data.get("injuries", []):
        for inj in team.get("injuries", []):
            name = inj.get("athlete", {}).get("displayName", "")
            if not name:
                continue
            status = str(inj.get("status", "")).strip().lower()
            if status == "out":
                out.add(norm_name(name))
            else:
                gtd[norm_name(name)] = inj.get("status", "?")
    return out, gtd


def load_injuries(path: Path | None) -> set:
    if not path or not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(norm_name(line))
    return names


def build_name_matcher(gamelog_keys: set):
    import difflib

    def match(dk_key: str) -> str | None:
        if dk_key in gamelog_keys:
            return dk_key

        cands = [k for k in gamelog_keys if dk_key in k or k in dk_key]
        if len(cands) == 1:
            return cands[0]

        parts = dk_key.split()
        if parts:
            last, fi = parts[-1], parts[0][:1]
            cands = [k for k in gamelog_keys
                     if k.split()[-1] == last and k.split()[0][:1] == fi]
            if len(cands) == 1:
                return cands[0]
        close = difflib.get_close_matches(dk_key, list(gamelog_keys), n=1, cutoff=0.85)
        return close[0] if close else None

    return match


def defense_vs_position(logs: pd.DataFrame, pos_map: dict) -> dict:
    logs = logs.copy()
    logs["pos"] = logs["name_key"].map(lambda k: pos_map.get(k, ""))
    logs = logs[logs["pos"].isin(["G", "F"])]
    if logs.empty:
        return {}
    max_season = logs["season"].max()
    logs["w"] = logs["season"].map(lambda s: 1.0 if s == max_season else 0.35)

    grp = logs.groupby(["opp", "pos", "season", "date"]).agg(
        dk=("dkpts", "sum"), w=("w", "first")).reset_index()
    league = {}
    for pos in ("G", "F"):
        sub = grp[grp["pos"] == pos]
        league[pos] = (sub["dk"] * sub["w"]).sum() / sub["w"].sum()

    factors = {}
    for (team, pos), sub in grp.groupby(["opp", "pos"]):
        allowed = (sub["dk"] * sub["w"]).sum() / sub["w"].sum()
        raw = allowed / league[pos] if league[pos] else 1.0
        factors[(team, pos)] = 1.0 + 0.5 * (raw - 1.0)
    return factors


def load_advanced_stats(path: Path) -> tuple[dict, float, float]:
    if not path.exists():
        return {}, 80.0, 100.0
    df = pd.read_csv(path)

    stats = {}
    league_pace = df["Pace"].mean() if "Pace" in df.columns else 80.0
    league_ortg = df["ORtg"].mean() if "ORtg" in df.columns else 100.0
    league_drtg = df["DRtg"].mean() if "DRtg" in df.columns else 100.0

    for row in df.itertuples():
        team = getattr(row, "Team", getattr(row, "team", None))
        if not team:
            continue
        team_key = norm_team(team)
        pace = float(row.Pace) if hasattr(row, "Pace") and pd.notna(row.Pace) else league_pace
        ortg = float(row.ORtg) if hasattr(row, "ORtg") and pd.notna(row.ORtg) else league_ortg
        drtg = float(row.DRtg) if hasattr(row, "DRtg") and pd.notna(row.DRtg) else league_drtg

        stats[team_key] = {
            "pace": pace,
            "ortg": ortg,
            "drtg": drtg,
            "pace_factor": pace / league_pace if league_pace else 1.0
        }
    return stats, league_pace, league_drtg


def project_player(pl_logs: pd.DataFrame, opp: str, game_code: str, is_home: bool,
                   def_factor: float, adv_stats: dict, game_projections: dict,
                   lg_drtg: float, lg_pace: float) -> dict:
    n = len(pl_logs)
    empty = {"proj": 0.0, "notes": "no game data", "games": 0, "cur_games": 0,
             "recent": 0.0, "season_avg": 0.0, "vol": 0.0, "last_game": None,
             "proj_min": 0.0}
    if n == 0:
        return empty

    max_season = pl_logs["season"].max()
    cur = pl_logs[pl_logs["season"] == max_season]
    prev = pl_logs[pl_logs["season"] < max_season]
    n_cur = len(cur)
    last_game = pl_logs["date"].max()

    # Minutes: exponentially weighted over the CURRENT season only (halflife ~3 games)
    # so an old-season workload can never inflate today's role.
    recent = cur.tail(10)
    w = [0.5 ** ((len(recent) - 1 - i) / 3.0) for i in range(len(recent))]
    wmin = sum(wi * m for wi, m in zip(w, recent["minutes"]) if pd.notna(m))
    wtot = sum(wi for wi, m in zip(w, recent["minutes"]) if pd.notna(m))
    proj_minutes = wmin / wtot if wtot > 0 else 15.0
    if pd.isna(proj_minutes) or proj_minutes == 0:
        proj_minutes = 15.0

    minutes_down = False
    if n_cur >= 6:
        last3 = cur.tail(3)["minutes"].mean()
        prior5 = cur.tail(8).head(5)["minutes"].mean()
        if pd.notna(last3) and pd.notna(prior5) and prior5 > 0 and last3 < prior5 * 0.7:
            minutes_down = True

    season_fppm = cur["dkpts"].sum() / cur["minutes"].sum() if cur["minutes"].sum() > 0 else 0

    # Form: current season only, decay-weighted
    form_logs = cur.tail(12)
    weights = [0.5 ** ((len(form_logs) - 1 - i) / 4.0) for i in range(len(form_logs))]
    weighted_dk = sum(w * row.dkpts for w, row in zip(weights, form_logs.itertuples()))
    weighted_min = sum(w * row.minutes for w, row in zip(weights, form_logs.itertuples()))
    recent_fppm = weighted_dk / weighted_min if weighted_min > 0 else season_fppm

    blended_fppm = (0.65 * recent_fppm) + (0.35 * season_fppm)

    # Small-sample players (returning from injury, new signings): use last season as a
    # weak prior only, discounted — players easing back in rarely match old production.
    if n_cur < 5 and len(prev) > 0 and prev["minutes"].sum() > 0:
        prev_fppm = prev["dkpts"].sum() / prev["minutes"].sum()
        w_cur = n_cur / (n_cur + 3.0)
        blended_fppm = w_cur * blended_fppm + (1 - w_cur) * prev_fppm * 0.85

    # Volatility: std of current-season DK scores — this is the GPP ceiling signal
    vol_logs = cur.tail(10)["dkpts"]
    vol = float(vol_logs.std(ddof=0)) if len(vol_logs) >= 3 else 0.0

    opp_stats = adv_stats.get(opp, {"pace_factor": 1.0, "drtg": lg_drtg})

    # Evaluate game-level pace adjustments cleanly
    if game_code in game_projections:
        pace_modifier = game_projections[game_code]["pace"] / lg_pace if lg_pace else 1.0
    else:
        pace_modifier = opp_stats.get("pace_factor", 1.0)

    # Matchup defensive rating factor (higher DRtg = friendlier matchup environment)
    eff_modifier = opp_stats["drtg"] / lg_drtg if lg_drtg else 1.0

    base_proj = proj_minutes * blended_fppm * pace_modifier * eff_modifier

    home_adj = 1.02 if is_home else 0.98
    proj = base_proj * def_factor * home_adj

    notes = []
    if proj_minutes > 20 and blended_fppm > 1.1:
        notes.append(f"High Usage/FPPM ({blended_fppm:.2f})")
    if pace_modifier > 1.03:
        notes.append("Pace Up Matchup")
    if eff_modifier > 1.03:
        notes.append("Weak Defensive Opponent")
    if minutes_down:
        notes.append("MINUTES TRENDING DOWN")
    if n_cur < 5:
        notes.append(f"only {n_cur} games this season")

    return {
        "proj": round(proj, 2),
        "games": n,
        "cur_games": n_cur,
        "last_game": last_game,
        "proj_min": round(proj_minutes, 1),
        "recent": round(recent_fppm * proj_minutes, 2),
        "season_avg": round(season_fppm * proj_minutes, 2),
        "vol": round(vol, 2),
        "def_factor": round(def_factor, 3),
        "notes": "; ".join(notes)
    }


SALARY_CAP = 50_000
PUNT_SALARY = 4_500   # players below this are "punts" — cap how many a lineup can carry


def optimize(players: list[dict], top_n: int = 5, locks: set | None = None,
             max_overlap: int = 3, max_punts: int = 1):
    locks = locks or set()
    guards = sorted([p for p in players if p["slot"] == "G"], key=lambda p: -p["opt"])
    forwards = sorted([p for p in players if p["slot"] == "F"], key=lambda p: -p["opt"])
    all_sorted = sorted(players, key=lambda p: -p["opt"])
    min_util_sal = min(p["salary"] for p in players)

    locked_g = {p["name"] for p in guards if p["name"] in locks}
    locked_f = {p["name"] for p in forwards if p["name"] in locks}

    best = {}
    POOL_SIZE = 1500
    current_cut = -1e18

    def prune():
        nonlocal current_cut
        if len(best) > POOL_SIZE * 2:
            sorted_items = sorted(best.items(), key=lambda kv: -kv[1][0])
            for k, _ in sorted_items[POOL_SIZE:]:
                del best[k]
            current_cut = sorted_items[POOL_SIZE - 1][1][0]
        elif len(best) >= POOL_SIZE and current_cut == -1e18:
            current_cut = min(v[0] for v in best.values())

    for g1, g2 in itertools.combinations(guards, 2):
        gset = {g1["name"], g2["name"]}
        if len(locked_g) >= 2 and not locked_g <= gset:
            continue
        g_sal = g1["salary"] + g2["salary"]
        g_proj = g1["opt"] + g2["opt"]
        if g_sal + min_util_sal > SALARY_CAP:
            continue

        for f1, f2, f3 in itertools.combinations(forwards, 3):
            fset = {f1["name"], f2["name"], f3["name"]}
            if len(locked_f) >= 3 and not locked_f <= fset:
                continue
            sal5 = g_sal + f1["salary"] + f2["salary"] + f3["salary"]
            if sal5 + min_util_sal > SALARY_CAP:
                continue
            proj5 = g_proj + f1["opt"] + f2["opt"] + f3["opt"]
            used = gset | fset
            games5 = {g1["game"], g2["game"], f1["game"], f2["game"], f3["game"]}

            missing_locks = locks - used
            if len(missing_locks) > 1:
                continue

            budget = SALARY_CAP - sal5
            found = 0

            if proj5 + all_sorted[0]["opt"] <= current_cut:
                continue

            pool = all_sorted if not missing_locks else [p for p in all_sorted if p["name"] in missing_locks]

            for u in pool:
                if found >= POOL_SIZE:
                    break
                total = proj5 + u["opt"]
                if total <= current_cut:
                    break
                if u["name"] in used or u["salary"] > budget:
                    continue
                if len(games5) < 2 and u["game"] in games5:
                    continue
                lineup = [g1, g2, f1, f2, f3, u]
                names = tuple(sorted(p["name"] for p in lineup))
                prev = best.get(names)
                if prev is None or total > prev[0]:
                    best[names] = (total, sal5 + u["salary"], lineup)
                    if len(best) == POOL_SIZE:
                        current_cut = min(v[0] for v in best.values())
                found += 1
            prune()

    all_generated = sorted(best.values(), key=lambda v: -v[0])
    final_lineups = []

    for total, salary, lu in all_generated:
        if len(final_lineups) >= top_n:
            break

        game_counts = defaultdict(int)
        for p in lu:
            game_counts[p["game"]] += 1
        is_stacked = any(count >= 3 for count in game_counts.values())
        if not is_stacked:
            continue

        # Winning lineups almost never carry two sub-$4.5k fliers — cap punts
        if sum(1 for p in lu if p["salary"] < PUNT_SALARY) > max_punts:
            continue

        too_similar = False
        current_names = set(p["name"] for p in lu)
        for _, _, existing_lu in final_lineups:
            existing_names = set(p["name"] for p in existing_lu)
            if len(current_names.intersection(existing_names)) > max_overlap:
                too_similar = True
                break

        if not too_similar:
            final_lineups.append((total, salary, lu))

    # Progressive loop relaxation ensures diversification instead of near-duplicates
    if len(final_lineups) < top_n:
        print("\nNote: Strict stacking/overlap constraints met resistance. Progressively scaling constraints to balance slate diversity...")
        for relaxed_overlap in range(max_overlap + 1, 6):
            if len(final_lineups) >= top_n:
                break
            for total, salary, lu in all_generated:
                if len(final_lineups) >= top_n:
                    break
                if any(set(p["name"] for p in lu) == set(p["name"] for p in existing[2]) for existing in final_lineups):
                    continue

                too_similar = False
                current_names = set(p["name"] for p in lu)
                for _, _, existing_lu in final_lineups:
                    existing_names = set(p["name"] for p in existing_lu)
                    if len(current_names.intersection(existing_names)) > relaxed_overlap:
                        too_similar = True
                        break
                if not too_similar:
                    final_lineups.append((total, salary, lu))

    return final_lineups

def main():
    ap = argparse.ArgumentParser(description="WNBA DraftKings lineup optimizer")
    ap.add_argument("--salaries", required=True, help="DraftKings salary CSV for today")
    ap.add_argument("--gamelogs", default="output/gamelogs.csv")
    ap.add_argument("--rosters", default="output/rosters.csv")
    ap.add_argument("--adv_stats", default="output/advanced_stats.csv", help="Advanced stats for pace/efficiency calculations")
    ap.add_argument("--injuries", default="injuries.txt", help="text file of injured players, one name per line")
    ap.add_argument("--exclude", default="", help="comma-separated extra names to exclude")
    ap.add_argument("--lock", default="", help="comma-separated names to force into lineup")
    ap.add_argument("--top", type=int, default=5, help="number of lineups to show")
    ap.add_argument("--max_overlap", type=int, default=3, help="Max players shared between lineups")
    ap.add_argument("--max_idle_days", type=int, default=7,
                    help="drop players who haven't played within this many days of the newest gamelog")
    ap.add_argument("--min_cur_games", type=int, default=2,
                    help="drop players with fewer current-season games than this (unless DK avg fallback)")
    ap.add_argument("--ceiling_weight", type=float, default=0.0,
                    help="optimizer score = proj + w*volatility; 0 (pure projection) "
                         "backtested best over 07/07-07/10")
    ap.add_argument("--max_punts", type=int, default=1,
                    help="max players under $4,500 per lineup")
    ap.add_argument("--no_fetch_injuries", action="store_true",
                    help="skip the live ESPN injury feed (offline mode)")
    ap.add_argument("--out", default="projections_today.csv")
    args = ap.parse_args()

    logs = load_gamelogs(Path(args.gamelogs))
    pos_map = load_positions(Path(args.rosters))
    sal = load_salaries(Path(args.salaries))
    adv_stats, lg_pace, lg_drtg = load_advanced_stats(Path(args.adv_stats))
    injured = load_injuries(Path(args.injuries))
    injured |= {norm_name(x) for x in args.exclude.split(",") if x.strip()}
    locks = {x.strip() for x in args.lock.split(",") if x.strip()}

    gtd_status = {}
    if not args.no_fetch_injuries:
        live_out, gtd_status = fetch_live_injuries()
        if live_out:
            print(f"\nLive injury feed: {len(live_out)} players OUT, "
                  f"{len(gtd_status)} day-to-day/questionable")
        injured |= live_out

    # --- Data freshness check: stale gamelogs were feeding lineups players who
    # hadn't been on the floor in weeks. Refuse to fail silently.
    slate_date = slate_date_from_salaries(sal)
    logs_max = logs["date"].max()
    if slate_date is not None:
        days_behind = (slate_date - logs_max).days
        if days_behind > 1:
            print(f"\n{'!'*78}")
            print(f"!! WARNING: gamelogs end {logs_max.date()} but the slate is {slate_date.date()}")
            print(f"!! You are missing {days_behind - 1} day(s) of games. Re-run wnba_scraper.py")
            print(f"!! before trusting these projections — recent form and injuries are blind.")
            print(f"{'!'*78}")
    # Idle cutoff is measured against the newest gamelog, so a stale scrape doesn't
    # wrongly flag everyone as inactive.
    idle_cutoff = logs_max - pd.Timedelta(days=args.max_idle_days)

    for row in sal.itertuples():
        pos_map[row.name_key] = row.slot
    dvp = defense_vs_position(logs, pos_map)

    matcher = build_name_matcher(set(logs["name_key"].unique()))

    # Calculate and Display Game-Level Context First
    unique_games = set(sal["game"].dropna().unique())
    game_projections = {}

    print(f"\n{'='*78}\nSLATE GAME TOTAL PROJECTIONS\n{'='*78}")
    for g in unique_games:
        if "@" not in g:
            continue
        away, home = g.split("@")
        away, home = norm_team(away), norm_team(home)

        a_stats = adv_stats.get(away, {"pace": lg_pace, "ortg": 100.0, "drtg": lg_drtg})
        h_stats = adv_stats.get(home, {"pace": lg_pace, "ortg": 100.0, "drtg": lg_drtg})

        exp_pace = (a_stats["pace"] * h_stats["pace"]) / lg_pace if lg_pace else lg_pace
        away_rtg = (a_stats["ortg"] * h_stats["drtg"]) / lg_drtg if lg_drtg else 100.0
        home_rtg = (h_stats["ortg"] * a_stats["drtg"]) / lg_drtg if lg_drtg else 100.0

        away_proj = (away_rtg * exp_pace) / 100.0
        home_proj = (home_rtg * exp_pace) / 100.0
        total_proj = away_proj + home_proj

        game_projections[g] = {"pace": exp_pace, "total": total_proj}
        print(f"  {away:<4} @ {home:<4}  |  Projected Score: {away_proj:.1f} - {home_proj:.1f}  |  Total: {total_proj:.1f}")

    players, skipped_injured, unmatched, skipped_stale = [], [], [], []
    for row in sal.itertuples():
        if row.name_key in injured:
            skipped_injured.append(row.Name)
            continue
        gl_key = matcher(row.name_key)
        if gl_key is None:
            unmatched.append(row.Name)
            pl_logs = logs.iloc[0:0]
        else:
            pl_logs = logs[logs["name_key"] == gl_key]

        def_factor = dvp.get((row.opp, row.slot), 1.0)
        pr = project_player(pl_logs, row.opp, row.game, row.is_home, def_factor,
                            adv_stats, game_projections, lg_drtg, lg_pace)

        # --- ACTIVITY FILTER: this is what was putting 0-point players in lineups.
        # A player with gamelogs must have played recently AND have current-season games.
        if pr["games"] > 0:
            if pr["last_game"] < idle_cutoff:
                skipped_stale.append(f"{row.Name} (last played {pr['last_game'].date()})")
                continue
            if pr["cur_games"] < args.min_cur_games:
                skipped_stale.append(f"{row.Name} (only {pr['cur_games']} games this season)")
                continue

        if pr["games"] == 0 and row.AvgPointsPerGame > 0:
            # No scraped history at all (e.g. new signing the scraper missed).
            # Keep with DK's own average, but flag loudly — verify before playing.
            pr["proj"] = round(float(row.AvgPointsPerGame) * 0.95, 2)
            pr["vol"] = round(pr["proj"] * 0.35, 2)
            pr["notes"] = "NO SCRAPED HISTORY — DK avg only, VERIFY ACTIVE"

        if row.name_key in gtd_status:
            tag = f"STATUS: {gtd_status[row.name_key]} — check before lock"
            pr["notes"] = f"{pr['notes']}; {tag}" if pr["notes"] else tag

        players.append({
            "name": row.Name, "team": row.team, "opp": row.opp,
            "slot": row.slot, "salary": int(row.Salary), "game": row.game,
            "dk_avg": row.AvgPointsPerGame, **pr,
            "opt": round(pr["proj"] + args.ceiling_weight * pr.get("vol", 0.0), 2),
            "value": round(pr["proj"] / row.Salary * 1000, 2) if row.Salary else 0,
        })

    proj_df = pd.DataFrame(players).sort_values("proj", ascending=False)
    proj_df.to_csv(args.out, index=False)

    print(f"\n{'='*78}\nPLAYER PROJECTIONS  (saved to {args.out})\n{'='*78}")
    cols = ["name", "team", "opp", "slot", "salary", "proj", "vol", "opt", "proj_min",
            "recent", "season_avg", "def_factor", "value", "notes"]
    print(proj_df[cols].head(30).to_string(index=False))

    if skipped_injured:
        print(f"\nExcluded (injury list): {', '.join(skipped_injured)}")
    if skipped_stale:
        print(f"\nExcluded (inactive — no game within {args.max_idle_days} days / too few 2026 games):")
        for s in skipped_stale:
            print(f"   {s}")
    if unmatched:
        print(f"\nNo gamelog match (using DK avg — VERIFY these are actually playing): {', '.join(unmatched)}")

    bad_locks = locks - {p["name"] for p in players}
    if bad_locks:
        sys.exit(f"Locked player(s) not in the available pool: {', '.join(bad_locks)}")

    lineups = optimize(players, top_n=args.top, locks=locks,
                       max_overlap=args.max_overlap, max_punts=args.max_punts)

    opt_desc = "pure projection" if args.ceiling_weight == 0 else f"proj + {args.ceiling_weight}*volatility"
    print(f"\n{'='*78}\nTOP {len(lineups)} LINEUPS  (optimized on {opt_desc})\n{'='*78}")
    lineup_rows = []
    for rank, (total, salary, lu) in enumerate(lineups, 1):
        slots = ["G", "G", "F", "F", "F", "UTIL"]
        proj_total = sum(p["proj"] for p in lu)
        print(f"\n#{rank}  projected {proj_total:.1f} DK pts (opt score {total:.1f})   salary ${salary:,} "
              f"(${SALARY_CAP - salary:,} left)")
        for slot, p in zip(slots, lu):
            note = f"  [{p['notes']}]" if p.get("notes") else ""
            lg = p["last_game"].date() if p.get("last_game") is not None and pd.notna(p.get("last_game")) else "??"
            print(f"   {slot:<4} {p['name']:<26} {p['team']} vs {p['opp']:<4}"
                  f" ${p['salary']:>6,}  proj {p['proj']:>5.1f}  last gm {lg}{note}")
            lineup_rows.append({"lineup": rank, "slot": slot, **{k: p[k] for k in
                               ("name", "team", "opp", "salary", "proj")}})
    pd.DataFrame(lineup_rows).to_csv("lineups_today.csv", index=False)
    print("\nLineups saved to lineups_today.csv")

    # Pre-submit checklist: the single most common way lineups die is a late scratch.
    uniq = {}
    for _, _, lu in lineups:
        for p in lu:
            uniq[p["name"]] = p
    flagged = [p for p in uniq.values()
               if p.get("last_game") is None or pd.isna(p.get("last_game"))
               or (logs_max - p["last_game"]).days > 3
               or "VERIFY" in str(p.get("notes", "")) or "STATUS" in str(p.get("notes", ""))]
    print(f"\n{'='*78}\nPRE-SUBMIT CHECKLIST — confirm these players are ACTIVE tonight:\n{'='*78}")
    if flagged:
        for p in flagged:
            lg = p["last_game"].date() if p.get("last_game") is not None and pd.notna(p.get("last_game")) else "no data"
            print(f"   !! {p['name']:<26} last game: {lg}")
    else:
        print("   All lineup players have played within the last 3 days of your data.")
    print("   (Also check beat-writer news ~30 min before lock and update injuries.txt.)")

if __name__ == "__main__":
    main()
