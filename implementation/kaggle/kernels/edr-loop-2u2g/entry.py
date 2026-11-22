# Kaggle script kernel: trains and evaluates the EDR study for map(s) loop (CPU; no internet needed).
import glob, os, shutil, subprocess, sys, zipfile
src = None
for z in glob.glob("/kaggle/input/**/edr_code.zip", recursive=True):
    src = z
work = "/kaggle/working/edr"
if src:
    zipfile.ZipFile(src).extractall(work)
else:
    cand = glob.glob("/kaggle/input/**/scripts/kaggle_run.py", recursive=True)
    shutil.copytree(os.path.dirname(os.path.dirname(cand[0])), work, dirs_exist_ok=True)
os.chdir(work)
import torch
print("cpus:", os.cpu_count(), "torch", torch.__version__, flush=True)
sys.exit(subprocess.call([sys.executable, "scripts/kaggle_run.py", "--maps", 'loop', "--seeds", "0", "1", "2", "--wait-mode", 'hover', "--uav", "2", "--ugv", "2", 
                          "--procs", str(max(1, os.cpu_count())), "--zip", "/kaggle/working/results_bundle.zip"]))
