#!/usr/bin/env python
"""Build kaggle/dataset (code zip + metadata) and one kernel folder per map (entry script + metadata).

  python scripts/package_for_kaggle.py --user vichanshur --maps border geo loop paper

Then (you run these; nothing is uploaded by this script):
  kaggle datasets create -p kaggle/dataset                 # private by default
  kaggle kernels push -p kaggle/kernels/edr-border         # one per map; check with: kaggle kernels status <user>/edr-border
  kaggle kernels output <user>/edr-border -p out/border   # download results_bundle.zip when finished
"""
import argparse, json, os, shutil, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = '''# Kaggle script kernel: trains and evaluates the EDR study for map(s) {maps} (CPU; no internet needed).
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
{prefill}sys.exit(subprocess.call([sys.executable, "scripts/kaggle_run.py", "--maps", {maps_args}, "--seeds", {seeds}, "--wait-mode", {wait!r}, "--uav", "{uav}", "--ugv", "{ugv}", {extra}
                          "--procs", str(max(1, os.cpu_count())), "--zip", "/kaggle/working/results_bundle.zip"]))
'''

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--maps", nargs="+", default=["border", "geo", "loop", "paper"])
    ap.add_argument("--wait-mode", default="hover", choices=["hover", "landed"])
    ap.add_argument("--out", default="kaggle", help="output folder (relative to implementation/)")
    ap.add_argument("--uav", type=int, default=2)
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--name", default="", help="kernel name suffix, e.g. long")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--p0-dataset", default="", help="dataset holding p0_seedN.pt files to start from (skips P0 training)")
    ap.add_argument("--extra", default="", help="extra kaggle_run args, e.g. \"--rl 3000 --variants p0c --stages A\"")
    a = ap.parse_args()
    kd = os.path.join(ROOT, a.out)
    ds = os.path.join(kd, "dataset")
    shutil.rmtree(ds, ignore_errors=True)       # kernels of earlier packagings are kept
    os.makedirs(ds)
    with zipfile.ZipFile(os.path.join(ds, "edr_code.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for top in ("src", "scripts", "tests"):
            for base, dirs, files in os.walk(os.path.join(ROOT, top)):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in files:
                    if not f.endswith(".pyc"):
                        z.write(os.path.join(base, f), os.path.relpath(os.path.join(base, f), ROOT))
        for f in ("pyproject.toml", "README.md"):
            z.write(os.path.join(ROOT, f), f)
    json.dump({"title": "EDR code", "id": f"{a.user}/edr-code", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(ds, "dataset-metadata.json"), "w"), indent=2)
    for m in a.maps:
        name = f"edr-{m}" + ("" if (a.uav, a.ugv) == (2, 1) else f"-{a.uav}u{a.ugv}g") + ("-landed" if a.wait_mode == "landed" else "") + (f"-{a.name}" if a.name else "")
        extra = ", ".join(repr(x) for x in a.extra.split()) + ("," if a.extra.split() else "")
        prefill = ""
        if a.p0_dataset:      # start from existing base policies: copy them where kaggle_run expects them
            tg = f"U15G5_{a.uav}U{a.ugv}G" + ("" if m == "loop" else f"_{m}")
            prefill = ("import shutil\nfor sd in %r:\n    src = glob.glob('/kaggle/input/**/p0_seed%%s.pt' %% sd, recursive=True)\n"
                       "    if src:\n        dst = os.path.join(work, 'results', 'models', %r + ('' if sd == '0' else '_seed' + sd))\n"
                       "        os.makedirs(dst, exist_ok=True); shutil.copy(src[0], os.path.join(dst, 'p0.pt'))\n") % (a.seeds, tg)
        kdir = os.path.join(kd, "kernels", name)
        shutil.rmtree(kdir, ignore_errors=True)
        os.makedirs(kdir)
        open(os.path.join(kdir, "entry.py"), "w").write(ENTRY.format(maps=m, maps_args=repr(m), wait=a.wait_mode, uav=a.uav, ugv=a.ugv, extra=extra, seeds=", ".join(repr(x) for x in a.seeds), prefill=prefill))
        json.dump({"id": f"{a.user}/{name}", "title": name.replace("-", " ").upper(), "code_file": "entry.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": False,
                   "dataset_sources": [f"{a.user}/edr-code"] + ([a.p0_dataset] if a.p0_dataset else []), "competition_sources": [], "kernel_sources": []},
                  open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
    print("built", kd, "->", sorted(os.listdir(os.path.join(kd, "kernels"))), os.path.getsize(os.path.join(ds, "edr_code.zip")) // 1024, "KB")
