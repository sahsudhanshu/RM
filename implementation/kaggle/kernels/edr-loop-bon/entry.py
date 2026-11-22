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
import shutil
for sd in ['0', '1', '2']:
    src = glob.glob('/kaggle/input/**/p0_seed%s.pt' % sd, recursive=True)
    if src:
        dst = os.path.join(work, 'results', 'models', 'U15G5_2U1G' + ('' if sd == '0' else '_seed' + sd))
        os.makedirs(dst, exist_ok=True); shutil.copy(src[0], os.path.join(dst, 'p0.pt'))
sys.exit(subprocess.call([sys.executable, "scripts/kaggle_run.py", "--maps", 'loop', "--seeds", '0', '1', '2', "--wait-mode", 'hover', "--uav", "2", "--ugv", "1", '--variants', 'p0c', 'p4', 'bon_t', 'bon_nb', 'bon_p4x', 'p3h_b90', 'p3h_b100', '--ft', '400', '--stages', 'A', 'B',
                          "--procs", str(max(1, os.cpu_count())), "--zip", "/kaggle/working/results_bundle.zip"]))
