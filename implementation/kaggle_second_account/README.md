# Run the EDR study on another Kaggle account

Everything here is independent of the first account. 8 kernels are prepared (hover waiting = main study, landed = extra pass).
Free Kaggle allows 5 concurrent CPU sessions per account, so pick up to 5. Suggested for a second account (the first account is already
running border/geo/paper/loop hover and border-landed): **`edr-loop-landed`, `edr-geo-landed`, `edr-paper-landed`**.
Use CPU, internet OFF, no accelerator (the work is CPU-bound).

## Option A - website only (no CLI, no username needed)
1. Kaggle -> Datasets -> New Dataset -> upload `dataset/edr_code.zip` (name it `edr-code`, keep it private).
2. Kaggle -> Code -> New Notebook -> File -> Import / or paste the contents of `kernels/<name>/entry.py` into one cell.
3. Right side "Add input" -> your `edr-code` dataset. Session options: CPU, Internet off, no accelerator.
4. Save Version -> "Save & Run All (Commit)". When finished, the notebook's Output tab has `results_bundle.zip`; download it.

## Option B - Kaggle CLI with a second account
The CLI uses ONE login at a time. Switch with a separate config folder (nothing of the first account is touched):
```bash
mkdir -p ~/.kaggle_second && chmod 700 ~/.kaggle_second
# put the second account's token file there (Kaggle -> Settings -> API -> Create New Token):
#   new-style token -> save as  ~/.kaggle_second/access_token      (or kaggle.json for the legacy key)
export KAGGLE_CONFIG_DIR=~/.kaggle_second          # use this in the shell where you run the second account
kaggle config view                                  # should show the SECOND username
cd implementation
.venv/bin/python scripts/package_for_kaggle.py --user <SECOND_USERNAME> --wait-mode landed --out kaggle_second_account --maps loop geo paper
kaggle datasets create -p kaggle_second_account/dataset
kaggle kernels push -p kaggle_second_account/kernels/edr-loop-landed
kaggle kernels status <SECOND_USERNAME>/edr-loop-landed
kaggle kernels output <SECOND_USERNAME>/edr-loop-landed -p out/loop_landed
```
Unset with `unset KAGGLE_CONFIG_DIR` to go back to the first account. (I have not tested a second account myself; I can only log in to the first one.)

## What each run produces
`results_bundle.zip` = trained models, logs and CSVs for that map x 3 seeds. Unzip into `implementation/` (it contains `results/...`) so the
report scripts find them: `python scripts/aggregate_seeds.py <tag>`.
