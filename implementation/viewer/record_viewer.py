#!/usr/bin/env python3
"""Record the 3D viewer (index_offline.html) to MP4 without a screen: headless Chrome, paused viewer, time set frame by frame
through the scrubber (so the video is smooth and deterministic), screenshots via the DevTools protocol, ffmpeg to stitch.

  python3 record_viewer.py                       # the three demo scenarios -> videos/
  python3 record_viewer.py --case 2 --out x.mp4  # one scenario by its index in missions.json
"""
import argparse, asyncio, base64, json, os, shutil, subprocess, sys, tempfile, time, urllib.request
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = [  # (case index in missions.json, output name)
    (2, "3d_viewer_ringroad_best_case.mp4"),       # ring road, slow edges, best case: replanning saves ~87 min
    (7, "3d_viewer_border_worst_case.mp4"),        # border, blocked road, worst case: replanning is slower (shown honestly)
    (6, "3d_viewer_ringroad_cut_road.mp4"),        # bridge cut: open-loop fails, replanning completes
]
ap = argparse.ArgumentParser()
ap.add_argument("--case", type=int, default=None); ap.add_argument("--out", default=None)
ap.add_argument("--speed", type=float, default=14.0, help="mission minutes per video second")
ap.add_argument("--fps", type=int, default=20); ap.add_argument("--size", default="1500x900"); ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the last frame")
ap.add_argument("--page", default=os.path.join(HERE, "index_offline.html")); ap.add_argument("--outdir", default=os.path.join(HERE, "videos"))
a = ap.parse_args()
W, H = map(int, a.size.split("x"))


class CDP:
    def __init__(self, ws): self.ws, self.n = ws, 0
    async def call(self, method, **params):
        self.n += 1; i = self.n
        await self.ws.send(json.dumps(dict(id=i, method=method, params=params)))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == i:
                if "error" in m: raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})
    async def js(self, expr, await_promise=False):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=await_promise)
        if "exceptionDetails" in r: raise RuntimeError(str(r["exceptionDetails"])[:300])
        return r.get("result", {}).get("value")


async def record(cdp, case, out):
    await cdp.js(f"(()=>{{const s=document.getElementById('case'); s.value='{case}'; s.dispatchEvent(new Event('change')); return s.options[s.selectedIndex].text;}})()")
    await cdp.js("(()=>{const b=document.getElementById('play'); if(b.textContent.trim()==='Pause') b.click();})()")   # pause: we drive time ourselves
    tmax = float(await cdp.js("document.getElementById('scrub').max")); title = await cdp.js("document.getElementById('case').options[document.getElementById('case').selectedIndex].text")
    step = a.speed / a.fps; n = int(tmax / step) + 1; tmp = tempfile.mkdtemp(prefix="viewerframes_")
    print(f"case {case}: {title} | {tmax:.0f} min -> {n} frames", flush=True)
    for k in range(n + int(a.hold * a.fps)):
        t = min(tmax, k * step)
        await cdp.js(f"(()=>{{const e=document.getElementById('scrub'); e.value={t:.3f}; e.dispatchEvent(new Event('input'));}})()")
        await cdp.js("new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))", await_promise=True)
        shot = await cdp.call("Page.captureScreenshot", format="jpeg", quality=88)
        open(os.path.join(tmp, f"f{k:05d}.jpg"), "wb").write(base64.b64decode(shot["data"]))
        if k % 100 == 0: print(f"  frame {k}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", out], check=True)
    shutil.rmtree(tmp); print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB", flush=True)


async def main():
    prof = tempfile.mkdtemp(prefix="chrome_prof_"); port = 9333
    chrome = subprocess.Popen(["google-chrome", "--headless=new", f"--remote-debugging-port={port}", f"--window-size={W},{H}", "--hide-scrollbars", "--no-first-run", "--disable-extensions",
                               "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", f"--user-data-dir={prof}", "--force-device-scale-factor=1", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try: tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")); break
            except Exception: time.sleep(0.5)
        ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            cdp = CDP(ws); await cdp.call("Page.enable"); await cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            await cdp.call("Page.navigate", url="file://" + os.path.abspath(a.page)); await asyncio.sleep(3.0)
            ok = await cdp.js("!!document.getElementById('gl') && !document.getElementById('nogl').hidden === false")
            if not await cdp.js("document.getElementById('nogl').hidden"): sys.exit("WebGL is not available in this headless Chrome")
            jobs = [(a.case, a.out or os.path.join(a.outdir, "3d_viewer_case.mp4"))] if a.case is not None else [(c, os.path.join(a.outdir, n)) for c, n in SCENARIOS]
            for c, out in jobs:
                await cdp.js(f"(()=>{{const s=document.getElementById('case'); s.value='{c}'; s.dispatchEvent(new Event('change'));}})()")
                await asyncio.sleep(0.5)
                full = int(await cdp.js("Math.ceil(Math.max(document.documentElement.scrollHeight, document.body.scrollHeight))"))
                h2 = min(max(H, full), 1500); h2 += h2 % 2
                await cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=h2, deviceScaleFactor=1, mobile=False); await asyncio.sleep(0.8)
                print(f"viewport {W}x{h2} (page height {full})", flush=True)
                await record(cdp, c, out)
    finally:
        chrome.terminate(); shutil.rmtree(prof, ignore_errors=True)

asyncio.run(main())
