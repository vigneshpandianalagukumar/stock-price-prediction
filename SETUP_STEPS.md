# Pushing this to GitHub

Code-only route: the `.joblib` and `.keras` files stay off GitHub, everything else
goes up. `.gitignore` already excludes them, so you cannot commit them by accident.

What still needs copying in: figures, result CSVs, the notebook and the report.
Those are all small.

---

## 1. Unzip and copy your files in

```bash
cd ~                                  # or wherever you want the repo to live
unzip stock-price-prediction-scaffold.zip
cd stock-price-prediction
REPO=$(pwd)

cd ~/results\(5\)

# Figures -- 4 files
cp diagnostic_plots_h20.png shap_summary_h20.png \
   shap_bar_h20.png todays_picks_h20.png "$REPO/figures/"

# Results -- small CSVs plus the config, which is worth keeping even
# though the models themselves are excluded: it records the feature
# lists and the published run's metrics.
cp summary_h20.csv backtest_h20.csv multi_horizon_results.csv \
   live_predictions_h20.csv pipeline_config_h20.json "$REPO/results/"
cp pipeline_config_h20.json "$REPO/models/"

# holdout_predictions_h20.csv is 118,267 rows. Check before copying:
du -h holdout_predictions_h20.csv
# under ~20 MB -> cp it to "$REPO/results/" ; larger -> leave it out
```

Then the notebook and the report:

```bash
cp your_notebook.ipynb "$REPO/notebook/"
cp /path/to/PBL_Report_Sandhyaa_Vignesh.docx "$REPO/docs/"
```

---

## 2. Check nothing oversized slipped in

```bash
cd "$REPO"
find . -path ./.git -prune -o -type f -size +50M -print
```

Silence means you are clear. Anything listed, remove it before committing — a file
over 100 MB makes GitHub reject the entire push, not just that file, and by then it is
already in your commit history.

```bash
pytest -q        # 12 pass, 2 skip. The skips are the end-to-end tests,
                 # which need the model files -- expected here.
```

---

## 3. Sign in to GitHub

GitHub stopped accepting account passwords over HTTPS, so `git push` needs either a
token or the CLI. The CLI is less hassle — it opens a browser and you click approve:

```bash
# install once, if you don't have it
sudo apt update && sudo apt install -y gh

gh auth login
# choose: GitHub.com -> HTTPS -> Yes (authenticate Git) -> Login with a web browser
# copy the code it shows, press Enter, paste it into the browser page
```

If `gh` will not install, the alternative is a Personal Access Token: github.com →
Settings → Developer settings → Personal access tokens → Tokens (classic) →
Generate new token → tick `repo` → copy it. Use that string as your **password** when
`git push` asks, with your username as the username.

---

## 4. Create the repo and push

```bash
cd "$REPO"
git init
git add .
git commit -m "Regime-adaptive Bayesian stacking for cross-sectional equity forecasting"
git branch -M main

gh repo create stock-price-prediction --public --source=. --remote=origin --push
```

That one command creates the repository and pushes in a single step.

Without `gh`, create the repository yourself at github.com/new — name it
`stock-price-prediction`, public, and do **not** let it add a README or .gitignore,
since this folder already has both. Then:

```bash
git remote add origin https://github.com/vigneshpandianalagukumar/stock-price-prediction.git
git push -u origin main
```

---

## 5. Check the result

Open https://github.com/vigneshpandianalagukumar/stock-price-prediction and confirm:

- the README renders, with the results table and the diagnostic image showing
- `src/`, `tests/`, `predict.py`, `app/app.py` are all there
- `models/` shows only `README.md`

---

## Checklist

- [ ] 4 PNGs in `figures/`
- [ ] result CSVs in `results/`, `pipeline_config_h20.json` in both `results/` and `models/`
- [ ] notebook in `notebook/`
- [ ] report in `docs/`
- [ ] `find . -type f -size +50M` returns nothing
- [ ] `pytest -q` gives 12 passed, 2 skipped
- [ ] pushed, README renders, figures display
