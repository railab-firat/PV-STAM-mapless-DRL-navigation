# How to upload this repository to GitHub

Everything is prepared in `C:\Users\anasm\Desktop\PV-STAM-repo\`. Follow the steps in order.
Total upload: **201 files, 16 MB**. Nothing here exceeds GitHub's limits.

> **Before you start:** change your GitHub password. It was shared in a screenshot, so treat it as exposed. Turn on two-factor authentication while you are in the settings.

---

## Step 0 — Accept the repository invitation

Open the email from GitHub sent **6 Sep at 06:09**, subject:

> railab-firat invited you to railab-firat/PV-STAM-mapless-DRL-navigation

Click **Accept**. There is a second, older invitation from `raifirat-hash` — ignore or decline that one. The paper's Data Availability Statement points to `railab-firat`, so that is the repository that matters.

Confirm you have access by opening:

```
https://github.com/railab-firat/PV-STAM-mapless-DRL-navigation
```

---

## Step 1 — Log in to the GitHub CLI

Open **Git Bash** or **PowerShell** and run:

```bash
gh auth login
```

Answer the prompts:

| Prompt | Answer |
|---|---|
| What account do you want to log into? | **GitHub.com** |
| What is your preferred protocol? | **HTTPS** |
| Authenticate Git with your GitHub credentials? | **Yes** |
| How would you like to authenticate? | **Login with a web browser** |

It prints an eight-character code, then opens your browser. Paste the code, approve, and return to the terminal.

Verify:

```bash
gh auth status
```

You should see `Logged in to github.com as <your-username>`.

---

## Step 2 — Set your git identity (first time only)

```bash
git config --global user.name "Anas Mahyoub Naji Saeed Alqadhi"
git config --global user.email "anas.m.alqadhi1111@gmail.com"
```

Use the email address attached to your GitHub account, otherwise the commits will not be linked to your profile.

---

## Step 3 — Make the first commit

```bash
cd "C:/Users/anasm/Desktop/PV-STAM-repo"
git add -A
git commit -m "Initial release: PV-STAM code, deployed policies and evaluation data"
```

Check that 201 files were staged:

```bash
git status --short | wc -l
```

---

## Step 4 — Connect to the repository and push

```bash
git branch -M main
git remote add origin https://github.com/railab-firat/PV-STAM-mapless-DRL-navigation.git
git push -u origin main
```

If the remote already has a README or LICENSE created by your professor, the push will be rejected. In that case:

```bash
git pull --rebase origin main
git push -u origin main
```

If the rebase reports a conflict on `README.md` or `LICENSE`, keep your version:

```bash
git checkout --ours README.md LICENSE
git add README.md LICENSE
git rebase --continue
git push -u origin main
```

---

## Step 5 — Confirm it is private

The Data Availability Statement says the code is released **upon acceptance**, so the repository must stay private until then.

```bash
gh repo view railab-firat/PV-STAM-mapless-DRL-navigation --json visibility
```

If it says `"visibility": "PUBLIC"`, ask your professor to switch it to private — as a collaborator you may not have permission yourself. On the web: **Settings → General → Danger Zone → Change repository visibility**.

**When the paper is accepted**, make it public and the link in the paper starts working.

---

## Step 6 — Set the description and topics

This is what makes the repository findable, and it costs nothing.

```bash
gh repo edit railab-firat/PV-STAM-mapless-DRL-navigation \
  --description "PV-STAM: velocity-aware spatio-temporal attention for mapless deep reinforcement learning navigation with 2D LiDAR. Code, trained policies and 130 real-robot trials (Applied Sciences, 2026)." \
  --add-topic deep-reinforcement-learning \
  --add-topic mapless-navigation \
  --add-topic lidar \
  --add-topic attention-mechanism \
  --add-topic sim-to-real \
  --add-topic dynamic-obstacle-avoidance \
  --add-topic soft-actor-critic \
  --add-topic turtlebot3 \
  --add-topic ros2 \
  --add-topic gazebo \
  --add-topic mobile-robot
```

You can also do this on the web with the gear icon next to **About**.

---

## Step 7 — Upload the training checkpoints as a Release

The full training checkpoints are **423 MB**, and one file — `sac_v8_s42/ckpt_best_sr.pt` — is **130 MB**, above GitHub's 100 MB hard limit for tracked files. They therefore cannot go in the repository itself.

Releases allow up to **2 GB per file**, which is the standard way research repositories publish weights.

First zip them:

```bash
cd "C:/Users/anasm/Desktop/PV-STAM_Paper(Tubitak2209)/03_PROJECT_ARCHIVE_AND_RAW_DATA/PV_STAM_PROJECT_ARCHIVE/after_reviews_extracted/after_reviews/PVSTAM_PAPER_RESOURCES_ARCHIVE"
"/c/Program Files/WinRAR/Rar.exe" a -r -ep1 -m5 "$HOME/Desktop/PV-STAM_training_checkpoints.rar" 02_model_checkpoints/
```

Then create the release (do this **after** the paper is accepted, along with making the repo public):

```bash
gh release create v1.0.0 \
  "$HOME/Desktop/PV-STAM_training_checkpoints.rar" \
  --repo railab-firat/PV-STAM-mapless-DRL-navigation \
  --title "v1.0.0 — Applied Sciences 2026" \
  --notes "Full training checkpoints (actor, critic and optimiser state) for all seven variants across seeds 42, 777 and 123. Deployed actor weights used in the physical trials are in models/ within the repository."
```

The README already links to the Releases page, so it will resolve once the release exists.

---

## What is in the repository

```
README.md              overview, results, build/train/evaluate, reproduction snippet
LICENSE                MIT  ← confirm with Prof. Ucar before making public
CITATION.cff           gives GitHub a "Cite this repository" button
.gitignore             __pycache__, *.bak, build artefacts, large checkpoints
requirements.txt       torch, numpy, scipy, matplotlib
src/tb3_drl_nav/       47 Python files — agents, environments, 13 launch files
models/                5 deployed actor weights (4.7 MB) used on the Jetson AGX Orin
data/evaluation/       108 per-episode CSVs (45 benchmark + 63 curriculum-phase)
data/hardware/         trials_index.csv, bags.npz, Figure 2 arrays, attention weights
scripts/               20 figure-rendering scripts, pv_data.py is the shared loader
```

Already removed before staging: 13 junk items — `*.bak`, `*.bak_resume`, `*.backup_20260406_133722`, `__pycache__`, and a **nested `.git` directory** inside `tb3_drl_nav/tb3_drl_nav/` that would have broken the push.

Verified: **0 files over 50 MB**, no `.pyc`, no backup files.

---

## Decisions still open

**1. The licence.** MIT is in place as a placeholder — it is the usual choice for research code. Apache-2.0 is the alternative if you want an explicit patent grant. This is your professor's lab repository, so it should be her call. One file to change if she prefers otherwise.

**2. The ORCID field in `CITATION.cff`** is empty. If Prof. Ucar has an ORCID, add it — it links the citation to her researcher profile.

**3. Timing.** Push now (private) so the work is safe and versioned. Make it public and cut the release when the paper is accepted.

---

## If something goes wrong

**`Permission denied` or `403` on push** — the invitation has not been accepted, or you accepted the wrong one (`raifirat-hash` instead of `railab-firat`).

**`remote: error: File ... is 130.00 MB; this exceeds GitHub's file size limit`** — a checkpoint has been staged by mistake. Check `.gitignore` is present and run `git rm --cached <file>`.

**`failed to push some refs`** — the remote has commits you do not. Use the `git pull --rebase` sequence in Step 4.

**Push is slow** — 16 MB over a normal connection takes well under a minute. If it stalls, `Ctrl+C` and retry; git resumes cleanly.

---

## Quick version

```bash
gh auth login                                    # browser, HTTPS, yes to git credentials
cd "C:/Users/anasm/Desktop/PV-STAM-repo"
git add -A
git commit -m "Initial release: PV-STAM code, deployed policies and evaluation data"
git branch -M main
git remote add origin https://github.com/railab-firat/PV-STAM-mapless-DRL-navigation.git
git push -u origin main
```
