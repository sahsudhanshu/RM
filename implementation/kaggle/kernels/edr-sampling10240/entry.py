# Kaggle script kernel: sampled decoding with N = 10240 routes per instance on the paper's map (4 teams in parallel, CPU).
import glob, os, shutil, subprocess, sys, zipfile
zips = glob.glob("/kaggle/input/**/edr_code.zip", recursive=True)
work = "/kaggle/working/edr"
if zips:
    zipfile.ZipFile(zips[-1]).extractall(work)
else:  # Kaggle auto-extracts single-zip datasets
    cand = glob.glob("/kaggle/input/**/scripts/sampling_eval.py", recursive=True)
    shutil.copytree(os.path.dirname(os.path.dirname(cand[0])), work, dirs_exist_ok=True)
os.chdir(work)
for t in ("1U1G", "2U1G", "2U2G", "4U2G"):
    f = glob.glob(f"/kaggle/input/**/p0_{t}.pt", recursive=True)[0]
    d = f"results/models/U15G5_{t}_paper"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f, d + "/p0.pt")
print("cpus:", os.cpu_count(), flush=True)
procs = [subprocess.Popen([sys.executable, "scripts/sampling_eval.py", "--teams", tm, "--samples", "10240", "--n", "50",
                           "--workers", "1", "--chunk", "256", "--out", f"results/sampling10240_{tm.replace(',', '')}.csv"]) for tm in ("1,1", "2,1", "2,2", "4,2")]
for p in procs:
    p.wait()
with zipfile.ZipFile("/kaggle/working/results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk("results"):
        for f in files:
            if f.endswith(".csv"):
                z.write(os.path.join(base, f))
