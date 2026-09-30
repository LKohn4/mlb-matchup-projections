# Setting up mlb-matchup-projections in PyCharm (Mac)

This guide assumes no Python experience. It takes about 15 minutes, plus the training run.
If you set up `mlb-hit-probability` before, most steps will feel familiar.

## 1. Put the project in place

1. Unzip `mlb-matchup-projections.zip`.
2. Move the `mlb-matchup-projections` folder into `Desktop/PY/PYPROJ/`, next to
   `mlb-hit-probability`. You should end up with:

   ```
   PYPROJ/
   ├── mlb-hit-probability/
   └── mlb-matchup-projections/     <- this project
   ```

> **Important:** open the `mlb-matchup-projections` folder itself in PyCharm, not `PYPROJ`.
> (Last time, opening the parent folder sent results to the wrong place.)

## 2. Open it in PyCharm

1. In PyCharm: **File → Open…**
2. Select `Desktop/PY/PYPROJ/mlb-matchup-projections` and click **Open**.
3. When asked, choose **New Window** so both projects stay separate.
4. If PyCharm asks whether to trust the project, click **Trust Project**.

## 3. Create the project's Python environment

1. Open settings with **⌘ ,** (Command + comma).
2. Go to **Project: mlb-matchup-projections → Python Interpreter**.
3. Click **Add Interpreter → Add Local Interpreter…**
4. Choose **Virtualenv Environment → New**.
   - **Location:** `.../mlb-matchup-projections/.venv` (inside this project)
   - **Base interpreter:** Python 3.11 or 3.12 (the same one you used for project 1)
5. Click **OK**, then **OK** again.

## 4. Install the packages

1. Open PyCharm's terminal: **fn + ⌥ + F12** (or View → Tool Windows → Terminal).
   The prompt should start with `(.venv)`.
2. Paste this and press Enter:

   ```bash
   pip install -e ".[dev]"
   ```

   This installs pybaseball, pandas, scikit-learn, TensorFlow and the project itself.
   It takes a few minutes. When it's done, check it worked:

   ```bash
   mlb-matchup --version
   ```

## 5. Reuse the 2026 data you already downloaded

Project 1 already downloaded the 2026 season. Copying it saves 10–30 minutes.
In the same terminal:

```bash
cp ../mlb-hit-probability/data/statcast_2026-03-25_2026-09-27.pkl data/
```

Or in Finder: press **⌘ ⇧ G**, go to `~/Desktop/PY/PYPROJ/mlb-hit-probability/data`,
copy `statcast_2026-03-25_2026-09-27.pkl`, and paste it into this project's `data` folder.
(If you skip this, the first training run downloads the season automatically.)

## 6. Run it (top-right dropdown next to the green ▶ button)

The run configurations are already set up. Pick one from the dropdown and press ▶:

| Order | Run configuration | What it does | Time |
| --- | --- | --- | --- |
| 1 | **Tests** | Checks that everything works (uses fake data, no internet) | ~2 min |
| 2 | **Quick train (fast test run)** | A short end-to-end run to confirm your setup | ~3–5 min |
| 3 | **1. Train models (full season)** | The real run: backtest, train, build all 15 figures | ~20–40 min |
| 4 | **2. Make game card (LAD at NYY)** | Writes a filled-in scoresheet to `game_cards/` | ~1 min |
| 5 | **3. Project game (LAD at NYY card)** | Projects that game and writes a report | ~1 min |
| – | **Search players** | Look up names and IDs (edit "smith" in the configuration) | ~1 min |
| – | **Refresh data (re-download 2026)** | Only if Statcast updates the season data | long |

When training finishes, look in:

- `results/figures/` for the 15 dashboard images
- `results/model_scorecard.md` for the model results table
- `results/games/<game>/report.md` for the example game's full report
- `game_cards/example_…yaml` for the auto-generated example card

## 7. Project your own game

1. Run **2. Make game card**. It creates `game_cards/2026-10-02_LAD_at_NYY.yaml`,
   filled from each team's recent lineups, rotation and bullpen.
2. Open that file in PyCharm and edit anything: names, the batting order, the starter,
   relievers, closer, park, date, or `time: day` / `night`.
3. Run **3. Project game**. The results print in the Run window, and the report and figures
   go to `results/games/2026-10-02_LAD_at_NYY/`.

For other teams, edit the **2. Make game card** configuration
(**Run → Edit Configurations…**) and change `LAD NYY` to any two team codes (away first).
Or type it in the terminal:

```bash
mlb-matchup card HOU SEA --date 2026-10-03 --time day
mlb-matchup project game_cards/2026-10-03_HOU_at_SEA.yaml
```

**Name tips:** names must match 2026 players. If a name is shared, the error message lists the
options. Write it as `"Will Smith (LAD)"` or use the ID shown, e.g. `"ID 669257"`. Typos get
a "Did you mean…?" suggestion.

## 8. Publish to GitHub

1. Open `README.md` and fill in the **Results** table from `results/model_scorecard.md`
   (see the comment in that section). The dashboard gallery images appear automatically once
   `results/figures/` exists.
2. **Git → GitHub → Share Project on GitHub** (older PyCharm: VCS → Share Project on GitHub).
   - Repository name: `mlb-matchup-projections`
   - Description: *Project how every MLB player performs in a specific matchup, from 2026 Statcast data*
   - Leave **Private** unticked, then click **Share**.
3. In the commit window, make sure `results/figures/` is ticked and `data/`/`models/` are not
   (the `.gitignore` handles this). Commit message: `Initial commit`.
4. On github.com, open the repository, click the ⚙️ next to **About**, and add topics:
   `baseball`, `mlb`, `statcast`, `machine-learning`, `monte-carlo-simulation`,
   `sports-analytics`, `python`.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named matchup` | Terminal: `pip install -e ".[dev]"` (check the prompt shows `(.venv)`). |
| Run configurations missing | Make sure you opened the `mlb-matchup-projections` folder, not `PYPROJ`. Then File → Reload All from Disk. |
| Results appear in a strange folder | The startup log prints the Code, Data and Results paths. They should all be inside this project. |
| "No trained model" | Run **1. Train models** (or **Quick train**) before projecting. |
| TensorFlow install fails | Use Python 3.11 or 3.12, or add `--skip-nn` to the train configuration's parameters. |
| Day/night shows as "unknown" | The MLB Stats API couldn't be reached. Check your internet, delete `data/schedule_*.csv`, and train again. |
| A name isn't recognized | Run **Search players**, or use `"ID 123456"` in the card. |
