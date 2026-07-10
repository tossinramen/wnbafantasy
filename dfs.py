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
                   def_factor: float, adv_stats: dict, game_projections: dict, lg_drtg: float) -> dict:
    n = len(pl_logs)
    if n == 0:
        return {"proj": 0.0, "notes": "no game data", "games": 0, "recent": 0.0, "season_avg": 0.0}

    max_season = pl_logs["season"].max()
    cur = pl_logs[pl_logs["season"] == max_season]
    
    recent_logs = pl_logs.tail(5)
    proj_minutes = recent_logs["minutes"].mean() if not recent_logs.empty else cur["minutes"].mean()
    if pd.isna(proj_minutes) or proj_minutes == 0:
        proj_minutes = 15.0 

    season_fppm = cur["dkpts"].sum() / cur["minutes"].sum() if cur["minutes"].sum() > 0 else 0
    
    form_logs = pl_logs.tail(15)
    weights = [0.5 ** ((len(form_logs) - 1 - i) / 4.0) for i in range(len(form_logs))]
    weighted_dk = sum(w * row.dkpts for w, row in zip(weights, form_logs.itertuples()))
    weighted_min = sum(w * row.minutes for w, row in zip(weights, form_logs.itertuples()))
    recent_fppm = weighted_dk / weighted_min if weighted_min > 0 else season_fppm

    blended_fppm = (0.7 * recent_fppm) + (0.3 * season_fppm)
    
    opp_stats = adv_stats.get(opp, {"pace_factor": 1.0, "drtg": lg_drtg})
    
    # Evaluate game-level pace adjustments cleanly
    if game_code in game_projections:
        pace_modifier = game_projections[game_code]["pace"] / 80.0
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

    return {
        "proj": round(proj, 2), 
        "games": n, 
        "recent": round(recent_fppm * proj_minutes, 2),
        "season_avg": round(season_fppm * proj_minutes, 2),
        "def_factor": round(def_factor, 3),
        "notes": "; ".join(notes)
    }


SALARY_CAP = 50_000


def optimize(players: list[dict], top_n: int = 5, locks: set | None = None, max_overlap: int = 3):
    locks = locks or set()
    guards = sorted([p for p in players if p["slot"] == "G"], key=lambda p: -p["proj"])
    forwards = sorted([p for p in players if p["slot"] == "F"], key=lambda p: -p["proj"])
    all_sorted = sorted(players, key=lambda p: -p["proj"])
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
        g_proj = g1["proj"] + g2["proj"]
        if g_sal + min_util_sal > SALARY_CAP:
            continue
            
        for f1, f2, f3 in itertools.combinations(forwards, 3):
            fset = {f1["name"], f2["name"], f3["name"]}
            if len(locked_f) >= 3 and not locked_f <= fset:
                continue
            sal5 = g_sal + f1["salary"] + f2["salary"] + f3["salary"]
            if sal5 + min_util_sal > SALARY_CAP:
                continue
            proj5 = g_proj + f1["proj"] + f2["proj"] + f3["proj"]
            used = gset | fset
            games5 = {g1["game"], g2["game"], f1["game"], f2["game"], f3["game"]}
           
            missing_locks = locks - used
            if len(missing_locks) > 1:
                continue

            budget = SALARY_CAP - sal5
            found = 0
            
            if proj5 + all_sorted[0]["proj"] <= current_cut:
                continue  
                
            pool = all_sorted if not missing_locks else [p for p in all_sorted if p["name"] in missing_locks]
            
            for u in pool:
                if found >= POOL_SIZE:
                    break
                total = proj5 + u["proj"]
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
    ap.add_argument("--out", default="projections_today.csv")
    args = ap.parse_args()

    logs = load_gamelogs(Path(args.gamelogs))
    pos_map = load_positions(Path(args.rosters))
    sal = load_salaries(Path(args.salaries))
    adv_stats, lg_pace, lg_drtg = load_advanced_stats(Path(args.adv_stats))
    injured = load_injuries(Path(args.injuries))
    injured |= {norm_name(x) for x in args.exclude.split(",") if x.strip()}
    locks = {x.strip() for x in args.lock.split(",") if x.strip()}

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

    players, skipped_injured, unmatched = [], [], []
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
        pr = project_player(pl_logs, row.opp, row.game, row.is_home, def_factor, adv_stats, game_projections, lg_drtg)
        
        if pr["games"] == 0 and row.AvgPointsPerGame > 0:
            pr["proj"] = round(float(row.AvgPointsPerGame) * 0.95, 2)
            pr["notes"] = "no scraped history — using DK season avg"
            
        players.append({
            "name": row.Name, "team": row.team, "opp": row.opp,
            "slot": row.slot, "salary": int(row.Salary), "game": row.game,
            "dk_avg": row.AvgPointsPerGame, **pr,
            "value": round(pr["proj"] / row.Salary * 1000, 2) if row.Salary else 0,
        })

    proj_df = pd.DataFrame(players).sort_values("proj", ascending=False)
    proj_df.to_csv(args.out, index=False)

    print(f"\n{'='*78}\nPLAYER PROJECTIONS  (saved to {args.out})\n{'='*78}")
    cols = ["name", "team", "opp", "slot", "salary", "proj", "recent",
            "season_avg", "def_factor", "value", "notes"]
    print(proj_df[cols].head(30).to_string(index=False))

    if skipped_injured:
        print(f"\nExcluded (injury list): {', '.join(skipped_injured)}")
    if unmatched:
        print(f"\nNo gamelog match (using DK avg): {', '.join(unmatched)}")

    bad_locks = locks - {p["name"] for p in players}
    if bad_locks:
        sys.exit(f"Locked player(s) not in the available pool: {', '.join(bad_locks)}")

    lineups = optimize(players, top_n=args.top, locks=locks, max_overlap=args.max_overlap)
    
    print(f"\n{'='*78}\nTOP {len(lineups)} LINEUPS\n{'='*78}")
    lineup_rows = []
    for rank, (total, salary, lu) in enumerate(lineups, 1):
        slots = ["G", "G", "F", "F", "F", "UTIL"]
        print(f"\n#{rank}  projected {total:.1f} DK pts   salary ${salary:,} "
              f"(${SALARY_CAP - salary:,} left)")
        for slot, p in zip(slots, lu):
            note = f"  [{p['notes']}]" if p.get("notes") else ""
            print(f"   {slot:<4} {p['name']:<26} {p['team']} vs {p['opp']:<4}"
                  f" ${p['salary']:>6,}  proj {p['proj']:>5.1f}{note}")
            lineup_rows.append({"lineup": rank, "slot": slot, **{k: p[k] for k in
                               ("name", "team", "opp", "salary", "proj")}})
    pd.DataFrame(lineup_rows).to_csv("lineups_today.csv", index=False)
    print("\nLineups saved to lineups_today.csv")

if __name__ == "__main__":
    main()