# Kaggle script kernel: 3000 more REINFORCE steps for the 4U-2G base policy on the paper's map (3 seeds in parallel), then greedy evaluation.
import glob, json, os, shutil, subprocess, sys, zipfile
zips = glob.glob("/kaggle/input/**/edr_code.zip", recursive=True)
work = "/kaggle/working/edr"
if zips:
    zipfile.ZipFile(zips[-1]).extractall(work)
else:
    cand = glob.glob("/kaggle/input/**/scripts/train.py", recursive=True)
    shutil.copytree(os.path.dirname(os.path.dirname(cand[0])), work, dirs_exist_ok=True)
os.chdir(work)
sys.path.insert(0, "src")
print("cpus:", os.cpu_count(), flush=True)
procs = []
for seed in (0, 1, 2):
    d = f"results/models/U15G5_4U2G_paper" + (f"_seed{seed}" if seed else "")
    os.makedirs(d, exist_ok=True)
    shutil.copy(glob.glob(f"/kaggle/input/**/p0_seed{seed}.pt", recursive=True)[0], d + "/p0.pt")
    procs.append(subprocess.Popen([sys.executable, "scripts/train.py", "p0more", "--uav", "4", "--ugv", "2", "--roads", "paper", "--rl-steps", "3000",
                                   "--ft-lr", "3e-4", "--threads", "1", "--seed", str(seed), "--out", "results/models"]))
for p in procs:
    p.wait()
import numpy as np, torch
from edr.evaluate import build_cases, load_model
from edr.rl import Setting
out = {}
for seed in (0, 1, 2):
    d = f"results/models/U15G5_4U2G_paper" + (f"_seed{seed}" if seed else "")
    s = Setting(15, 5, 4, 2, "paper")
    for tag in ("p0_short.pt", "p0.pt"):
        cs = build_cases(s, load_model(os.path.join(d, tag)), 100)
        out[f"seed{seed}_{tag}"] = float(np.mean([c.t0 for c in cs])); print(seed, tag, out[f"seed{seed}_{tag}"], flush=True)
json.dump(out, open("/kaggle/working/eval_greedy.json", "w"))
with zipfile.ZipFile("/kaggle/working/results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk("results/models"):
        for f in files:
            if f.endswith((".pt", ".json")):
                z.write(os.path.join(base, f))
