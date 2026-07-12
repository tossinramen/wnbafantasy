import asyncio
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup, Comment

from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions


BASE = "https://www.basketball-reference.com"
YEARS = [2025, 2026]
CURRENT_YEAR = max(YEARS)  
OUTPUT_DIR = Path("output")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
TEAMS_CSV = OUTPUT_DIR / "teams.csv"
ROSTERS_CSV = OUTPUT_DIR / "rosters.csv"
GAMELOGS_CSV = OUTPUT_DIR / "gamelogs.csv"

MIN_DELAY = 3.5   
MAX_DELAY = 6.5  
MAX_RETRIES = 3
HEADLESS = True   




if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        ck = json.loads(CHECKPOINT_FILE.read_text())
        
        
        ck["teams_done"] = False 
        
        
        bad_keys = ["TUL-2025", "TUL-2026", "SAS-2025", "SAS-2026"]
        ck["rosters_done"] = [k for k in ck.get("rosters_done", []) if k not in bad_keys]

        
        ck["rosters_done"] = [k for k in ck["rosters_done"] if not k.endswith(f"-{CURRENT_YEAR}")]
        ck.setdefault("update_date", "")
        ck.setdefault("update_players_done", [])

        return ck

    return {"teams_done": False, "rosters_done": [], "players_done": [],
            "update_date": "", "update_players_done": []}


def save_checkpoint(ck: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(ck, indent=2))


def polite_sleep() -> None:
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def soupify(html: str) -> BeautifulSoup:
    """
    basketball-reference hides many tables inside HTML comments and reveals
    them with JS. pydoll runs real Chrome so they're usually already in the
    DOM, but we also unwrap any remaining commented-out tables to be safe.
    """
    soup = BeautifulSoup(html, "html.parser")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            c.replace_with(BeautifulSoup(c, "html.parser"))
    return BeautifulSoup(str(soup), "html.parser")


async def fetch(tab, url: str) -> BeautifulSoup | None:
    """Load a page in the browser tab with retries + rate-limit backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await tab.go_to(url)
            await asyncio.sleep(1.5)  
            html = await tab.page_source
            if "Rate Limited Request" in html or "429 error" in html:
                wait = 60 * attempt * 2
                log(f"  !! rate limited on {url} — sleeping {wait}s")
                await asyncio.sleep(wait)
                continue
            if "Page Not Found" in html and "404" in html:
                log(f"  -- 404: {url}")
                return None
            return soupify(html)
        except Exception as e:
            log(f"  !! error on {url} (attempt {attempt}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(10 * attempt)
    log(f"  !! giving up on {url}")
    return None


def rows_from_table(table) -> list[dict]:
    """
    Convert a bbref stats table into a list of dicts using the data-stat
    attributes on each cell (much more reliable than positional parsing).
    Skips repeated-header rows and 'totals' summary rows.
    """
    out = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        cls = tr.get("class") or []
        if "thead" in cls:
            continue
        row = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if stat:
                row[stat] = cell.get_text(strip=True)
                a = cell.find("a")
                if a and a.get("href"):
                    row[f"{stat}_href"] = a["href"]
        if row:
            out.append(row)
    return out



async def get_active_teams(tab) -> list[dict]:
    log("Fetching list of active WNBA franchises...")
    soup = await fetch(tab, f"{BASE}/wnba/teams/")
    if soup is None:
        raise RuntimeError("Could not load the teams index page.")

    table = soup.find("table", id="active") or soup.find("table")
    teams = []
    seen = set()
    
   
    code_map = {
        "TUL": "DAL",  
        "SAS": "LVA",  
    }

    for a in table.find_all("a", href=re.compile(r"^/wnba/teams/[A-Z]{2,4}/$")):
        code = a["href"].rstrip("/").split("/")[-1]
        code = code_map.get(code, code) 
        
        if code not in seen:
            seen.add(code)
            teams.append({"team_code": code, "team_name": a.get_text(strip=True)})
            
    log(f"  found {len(teams)} active franchises: {', '.join(t['team_code'] for t in teams)}")
    return teams



async def get_roster(tab, team_code: str, year: int) -> list[dict]:
    url = f"{BASE}/wnba/teams/{team_code}/{year}.html"
    soup = await fetch(tab, url)
    if soup is None:
        return []  

    table = soup.find("table", id="roster")
    if table is None:
        log(f"  -- no roster table on {url}")
        return []

    players = []
    for row in rows_from_table(table):
        href = row.get("player_href", "")
        m = re.search(r"/wnba/players/\w/(\w+)\.html", href)
        if not m:
            continue
        players.append({
            "player_id": m.group(1),
            "player_name": row.get("player", ""),
            "player_url": BASE + href,
            "team_code": team_code,
            "season": year,
            "number": row.get("number", ""),
            "pos": row.get("pos", ""),
            "height": row.get("height", ""),
            "weight": row.get("weight", ""),
            "birth_date": row.get("birth_date", ""),
            "experience": row.get("years_experience", row.get("exp", "")),
            "college": row.get("college", ""),
        })
    return players



def is_gamelog_table(table) -> bool:
    stats = {c.get("data-stat") for c in table.find_all(["th", "td"])}
    return "date" in {s.lower() for s in stats if s} or "date_game" in stats


async def get_player_gamelog(tab, player: dict, year: int) -> list[dict]:
    pid = player["player_id"]
    first_letter = pid[0]
    url = f"{BASE}/wnba/players/{first_letter}/{pid}/gamelog/{year}/"
    soup = await fetch(tab, url)
    if soup is None:
        return []

    games = []
    for table in soup.find_all("table"):
        tid = table.get("id", "")
        
        if "pgl" not in tid and "gamelog" not in tid and not is_gamelog_table(table):
            continue
        if not is_gamelog_table(table):
            continue
        is_playoffs = "playoff" in tid.lower()
        for row in rows_from_table(table):
            
            date = row.get("date_game") or row.get("date") or ""
            if not date:
                continue
            game = {
                "player_id": pid,
                "player_name": player["player_name"],
                "season": year,
                "is_playoffs": is_playoffs,
                "source_table": tid,
            }
            
            for k, v in row.items():
                if not k.endswith("_href"):
                    game[k] = v
            games.append(game)
    return games



def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def game_key(g: dict) -> tuple:

    return (g.get("player_id"), g.get("date_game") or g.get("date"))


def roster_key(p: dict) -> tuple:
    return (p.get("player_id"), p.get("team_code"), str(p.get("season")))


def dedupe_rows(rows: list[dict], key_fn) -> list[dict]:
    out: dict = {}
    for r in rows:
        out[key_fn(r)] = r  
    return list(out.values())


def rows_to_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"  wrote {csv_path} ({len(rows)} rows)")



async def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ck = load_checkpoint()
    rosters_jsonl = OUTPUT_DIR / "rosters.jsonl"
    gamelogs_jsonl = OUTPUT_DIR / "gamelogs.jsonl"


    today = time.strftime("%Y-%m-%d")
    if ck.get("update_date") != today:
        ck["update_date"] = today
        ck["update_players_done"] = []

    options = ChromiumOptions()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")
    options.add_argument("--disable-blink-features=AutomationControlled")

    async with Chrome(options=options) as browser:
        tab = await browser.start()

        
        teams = await get_active_teams(tab)
        if not ck["teams_done"]:
            with TEAMS_CSV.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["team_code", "team_name"])
                w.writeheader()
                w.writerows(teams)
            ck["teams_done"] = True
            save_checkpoint(ck)
        polite_sleep()

        
        all_players: dict[str, dict] = {}
        for team in teams:
            for year in YEARS:
                key = f"{team['team_code']}-{year}"
                if key in ck["rosters_done"]:
                    continue  
                log(f"Roster: {team['team_code']} {year}")
                roster = await get_roster(tab, team["team_code"], year)
                append_jsonl(rosters_jsonl, roster)
                ck["rosters_done"].append(key)
                save_checkpoint(ck)
                polite_sleep()

        
        if rosters_jsonl.exists():
            for line in rosters_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                p = json.loads(line)
                all_players[p["player_id"]] = p
        log(f"Unique players across all rosters: {len(all_players)}")

        
        done = set(ck["players_done"])
        update_done = set(ck["update_players_done"])
        new_players = [p for pid, p in sorted(all_players.items()) if pid not in done]
        update_players = [p for pid, p in sorted(all_players.items())
                          if pid in done and pid not in update_done]
        log(f"New players (full scrape): {len(new_players)} | "
            f"players to refresh for {CURRENT_YEAR}: {len(update_players)}")

        for i, player in enumerate(new_players, 1):
            log(f"(new {i}/{len(new_players)}) Game logs: {player['player_name']} [{player['player_id']}]")
            player_games = []
            for year in YEARS:
                games = await get_player_gamelog(tab, player, year)
                log(f"    {year}: {len(games)} games")
                player_games.extend(games)
                polite_sleep()
            append_jsonl(gamelogs_jsonl, player_games)
            ck["players_done"].append(player["player_id"])
            ck["update_players_done"].append(player["player_id"])  
            save_checkpoint(ck)

        for i, player in enumerate(update_players, 1):
            log(f"(refresh {i}/{len(update_players)}) {CURRENT_YEAR} game log: "
                f"{player['player_name']} [{player['player_id']}]")
            games = await get_player_gamelog(tab, player, CURRENT_YEAR)
            log(f"    {CURRENT_YEAR}: {len(games)} games")
            append_jsonl(gamelogs_jsonl, games)
            ck["update_players_done"].append(player["player_id"])
            save_checkpoint(ck)
            polite_sleep()


    log("Building final CSVs (deduped)...")
    rosters = dedupe_rows(read_jsonl(rosters_jsonl), roster_key)
    rosters.sort(key=lambda p: (str(p.get("season", "")), p.get("team_code", ""), p.get("player_id", "")))
    games = dedupe_rows(read_jsonl(gamelogs_jsonl), game_key)
    games.sort(key=lambda g: (g.get("player_id", ""), str(g.get("season", "")),
                              g.get("date_game") or g.get("date") or ""))

    write_jsonl(rosters_jsonl, rosters)
    write_jsonl(gamelogs_jsonl, games)
    rows_to_csv(rosters, ROSTERS_CSV)
    rows_to_csv(games, GAMELOGS_CSV)
    log("Done. Files are in ./output/")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Re-run to resume.")
        sys.exit(1)