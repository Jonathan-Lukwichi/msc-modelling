# Push to GitHub — workflow

The local git repository is initialised and the initial commit is in place
(hash `345e647`, 88 files, 14,410 lines). Pushing to a remote is two steps:

## 1. Create the remote repository

Go to https://github.com/new and create a **private** repository:

| Field | Value |
|---|---|
| Owner | your GitHub user (jonathanlukwichi or similar) |
| Repository name | `msc-thesis--modelling-and-evaluation` |
| Description | "MSc thesis — Chapter 6 modelling and evaluation pipeline (Steve Biko ED demand forecasting)" |
| Visibility | **Private** (the LICENSE allows academic re-use but the work is unpublished) |
| Initialise with README | **No** (we already have one) |
| .gitignore template | **None** (we already have one) |
| License | **None** (we already have one) |

Click "Create repository". GitHub will show a "set up an existing repository"
page with copy-pasteable commands.

## 2. Add the remote and push

Open a terminal in this repo (`c:\Users\BIBINBUSINESS\OneDrive\Desktop\msc-thesis--modelling-and-evaluation`) and run:

```bash
# Replace JonathanLukwichi with your actual GitHub username
git remote add origin https://github.com/JonathanLukwichi/msc-thesis--modelling-and-evaluation.git

# Verify
git remote -v

# First push — set upstream so future 'git push' works without args
git push -u origin main
```

GitHub will prompt for credentials. Recommended: use a **personal access token**
(not your password) — settings → developer settings → personal access tokens
→ generate new token (scope: `repo`).

## 3. Add Prof. Bean as collaborator

In the new repo: Settings → Collaborators → Add people → search for
Prof. Bean's GitHub username → send invite.

## 4. Future commits

```bash
# After running more scripts and updating artefacts/figures, artefacts/tables,
# or RESULTS.md, stage and commit:
git add artefacts/figures/ artefacts/tables/ artefacts/metrics/ artefacts/RESULTS.md
git commit -m "Update: LSTM k=10 CV results + ablation study + Task 2 leaderboard"
git push
```

## What is NOT in the repo (by design)

These are all in `.gitignore`:

| Path | Why excluded |
|---|---|
| `configs/paths.local.yaml` | Your local absolute paths — different per machine |
| `data/` | Hospital register data is confidential and tracked elsewhere |
| `artefacts/predictions/` | Regenerable per-model CSV outputs (large) |
| `artefacts/models/` | Trained model pickles for cloud deployment (regenerable, large) |
| `.claude/` | IDE state |
| `__pycache__/`, `.pytest_cache/` | Python build caches |

If a collaborator wants the trained `.pkl` files for cloud deployment, the
suggested workflow is:

1. They clone the repo
2. They set up `paths.local.yaml`
3. They run `python scripts/13_save_for_cloud.py` which regenerates every
   `.pkl` under `artefacts/models/deploy/` deterministically (seed = 42)

This keeps the repo small while still being fully reproducible.

## Verification once pushed

After `git push`, verify the public URL works:

```bash
gh repo view JonathanLukwichi/msc-thesis--modelling-and-evaluation --web
# or just open https://github.com/JonathanLukwichi/msc-thesis--modelling-and-evaluation in a browser
```

You should see:
- README rendering with the title "MSc Thesis — Chapter 6: Hospital ED Demand Forecasting"
- The figures previewed inline (GitHub renders PNGs in markdown links)
- A green LICENSE badge if added in repository topics
- 88 files / 14,410 lines at the initial commit

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| "Authentication failed" | Generate a personal access token; use it as password |
| "fatal: refusing to merge unrelated histories" | Don't initialise the remote with README/LICENSE/gitignore; or run `git pull origin main --allow-unrelated-histories` then resolve |
| "Updates were rejected" | `git pull --rebase origin main` then `git push` |
| Want to rename master→main | We already created on main; if GitHub created on master, run `git branch -M main` then push |
