# Work log

A running record of every change made to this repository. It covers what was done, why, and
what is still open. Add a new entry at the top of **Log** for each piece of work, and keep the
**Commit history** table in step with `git log`.

---

## Log

### 2026-09-28 — First push to GitHub

- Pushed the full history (24 commits) to `Biswadev-9/fnal_thesis` on `main`.
- HTTP 403 (token lacked repo access), then HTTP 408 (6.8 MB upload timed out on a slow link). Fixed with `http.version=HTTP/1.1` and a larger `http.postBuffer`.
- Renamed the local branch to `main`. `origin` now points to `fnal_thesis`; the old repo is the `thesis-old` remote.
- A token was pasted into a chat session. Revoke it and replace it.

### 2026-09-28 — Repository migrated to `fnal_thesis`, README rewritten

**What was done**
- Moved all code from the `swin-backbone` branch of `Biswadev-9/thesis` to
  `Biswadev-9/fnal_thesis` as `main`. All 22 commits and their history were kept.
- Rewrote the commit messages to drop `Co-Authored-By` trailers left by tooling, so each commit
  lists only the author. Five commits were affected; their code is unchanged.
- Checked that no editor or assistant files were tracked (`.claude/`, `.vscode/`, `.idea/`,
  `.DS_Store`). Added `.claude/`, `CLAUDE.md` and `.DS_Store` to `.gitignore` to keep them out.
- Rewrote `README.md` from the source code. It has eight Mermaid diagrams:
  - the system overview
  - the proposed model's architecture
  - the spatial multiscale gate
  - the staged training sequence
  - the Steps 4–25 dependency graph
  - the A0–A8 + P ablation ladder
  - the EfficientNet-B0 vs Swin-T comparison
  - a class map of the code

  It also has tables for RQ1–RQ10, the fixed protocol, per-step outputs, the quantum circuits,
  fusion variants, repository structure, quick start and limitations.
- Started this work log.

**Notes**
- The original `Biswadev-9/thesis` repository was not modified.
- `docs/Instruction BY asif vai.md` (the supervisor's specification) is included because it was
  already tracked on the branch.

**Open**
- Swin-T arm: nothing has been run yet. Follow `docs/SWIN_EXPERIMENT.md` §3 (runtime checks)
  before any training.
- Full-protocol results have not been produced for either backbone arm.

---

## Commit history

| Commit | Date | Summary |
|---|---|---|
| `5aea36b` | 2026-08-08 | Initial commit (Lightning–Hydra template) |
| `d0a350d` | 2026-08-08 | Implementation plan |
| `201dfae` | 2026-08-12 | Phase 1: data foundation and model layer |
| `2213637` | 2026-08-12 | Phase 2: preprocessing and imbalance studies |
| `3576382` | 2026-08-12 | Phases 3–4: baselines, fixed protocol, branches |
| `ace4c83` | 2026-08-13 | Phases 5–6: fusion, final classifier, Kaggle runner |
| `7f8e61e` | 2026-08-13 | Fail loudly when the dataset is missing |
| `c89ef9c` | 2026-08-13 | Locate the Kaggle dataset by structure |
| `0eaf9d0` | 2026-08-14 | Zero dataloader workers by default; stage timeout and heartbeat |
| `c77b148` | 2026-08-14 | Idempotent Kaggle dataset linking |
| `a67b30e` | 2026-08-14 | Heal a stray file at the dataset link path |
| `883ef39` | 2026-08-14 | Stop streaming stage output into the notebook |
| `350b335` | 2026-08-15 | Phase 7: evaluation, robustness, explainability |
| `39aad3f` | 2026-08-15 | Phase 8: ablation analysis and orchestration |
| `594025c` | 2026-08-16 | Step 6 confirmation; Step 24/25 ablations |
| `c94a68f` | 2026-08-16 | Fixes |
| `bb82a81` | 2026-08-17 | Pipeline resume and interruption handling |
| `d18fdec` | 2026-08-19 | Fixes |
| `6e9f840` | 2026-08-23 | Fix Step 21 instantiation and Step 24/25 recipe resolution |
| `ce52756` | 2026-09-09 | Update README |
| `e6e49d6` | 2026-09-18 | Swin-T backbone arm alongside EfficientNet-B0 |
| `9574344` | 2026-09-19 | Swin-T arm of the Step 21 ablation matrix |
| `d3b1a11` | 2026-09-28 | README rewrite with architecture and protocol diagrams |
| `08f1b7e` | 2026-09-28 | Add this work log |
| _next_ | 2026-09-28 | Log the first push to GitHub |
