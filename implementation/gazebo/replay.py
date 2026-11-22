#!/usr/bin/env python
"""Replay one planned mission in Gazebo (headless). UGVs are wheeled DiffDrive vehicles driven by a pure-pursuit controller
along the planned road edges (real contact physics, acceleration and turn-rate limits). UAVs are velocity-controlled
(kinematic) and fly the planned legs. Dependencies are enforced at run time: a UAV can only be served after the UGV has
*actually* arrived (and the UGV is free), so physical delays propagate into mission time and hover energy.
Lock-step: the world is paused and advanced `--dt` seconds per controller tick."""
import argparse, json, math, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.entity_factory_pb2 import EntityFactory
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from make_world import world_sdf

ap = argparse.ArgumentParser()
ap.add_argument("plan"); ap.add_argument("--out", default="runs"); ap.add_argument("--dt", type=float, default=0.2)
ap.add_argument("--no-video", action="store_true"); ap.add_argument("--max-sim", type=float, default=0.0); ap.add_argument("--terrain", default=""); ap.add_argument("--partition", default="")
a = ap.parse_args()
plan = json.load(open(a.plan)); name = plan["id"].replace(" ", "_").replace("(", "").replace(")", "")
os.makedirs(a.out, exist_ok=True)
part = a.partition or f"edr{os.getpid()}"
os.environ["GZ_PARTITION"] = part
terrain = None
if a.terrain:
    tdir = os.path.abspath(a.terrain); info = json.load(open(os.path.join(tdir, "info.json"))); dem = np.load(os.path.join(tdir, "dem.npz"))["z"]
    terrain = dict(dir=tdir, size=info["size"], relief=info["relief"], dem=dem)
from make_world import terrain_height_fn
gzh = terrain_height_fn(terrain) if terrain else (lambda x, y: 0.0)
world_path = os.path.join(a.out, name + ".sdf"); open(world_path, "w").write(world_sdf(plan, terrain=terrain))
server = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", "-v", "1", world_path], stdout=open(os.path.join(a.out, name + ".server.log"), "w"), stderr=subprocess.STDOUT)
node = Node(); poses = {}; sim = [0.0]; frames = []
def on_pose(m):
    for p in m.pose:
        poses[p.name] = (p.position.x, p.position.y, p.position.z, 2 * math.atan2(p.orientation.z, p.orientation.w))
    sim[0] = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
node.subscribe(Pose_V, "/world/edr/dynamic_pose/info", on_pose)
node.subscribe(Image, "/overview", lambda m: frames.append(bytes(m.data)))
cmd = {}
def publisher(n):
    if n not in cmd: cmd[n] = node.advertise(f"/model/{n}/cmd_vel", Twist)
    return cmd[n]
def drive(n, v, w):
    t = Twist(); t.linear.x = v; t.angular.z = w; publisher(n).publish(t)
def vel(n, vx, vy):
    t = Twist(); t.linear.x = vx; t.linear.y = vy; publisher(n).publish(t)
def set_pose(n, x, y, z):
    p = Pose(); p.name = n; p.position.x, p.position.y, p.position.z = x, y, z; p.orientation.w = 1.0
    return node.request("/world/edr/set_pose", p, Pose, Boolean, 2000)
def step(k):
    w = WorldControl(); w.multi_step = k; w.pause = True
    return node.request("/world/edr/control", w, WorldControl, Boolean, 5000)[0]

t_wait = time.time()
while not all((f"ugv{g['id']}" in poses) for g in plan["ugvs"]) or not all((f"uav{u['id']}" in poses) for u in plan["uavs"]):
    if time.time() - t_wait > 90: sys.exit("world did not come up")
    time.sleep(0.5)
time.sleep(1.0)

# ---- state --------------------------------------------------------------------------------------------------
xy = plan["roads"]["xy"]; dt = a.dt; steps = int(round(dt / 0.01)); HOV = plan["hover_kj_per_s"]
ugvs = {g["id"]: dict(drives=g["drives"], idx=0, mode="idle", arr=[None] * len(g["drives"]), free_at=0.0, err=[]) for g in plan["ugvs"]}
sv_by_leg = {}
for s in plan["services"]:
    for u in plan["uavs"]:
        if u["id"] != s["uav"]: continue
        for k, l in enumerate(u["legs"]):
            if l["land"] and abs(l["t1"] - s["touchdown"]) < 1.0: sv_by_leg[(u["id"], k)] = s
def ugv_dep(s):  # index of the drive after which the UGV is at the service node
    best = None
    for i, d in enumerate(ugvs[s["ugv"]]["drives"]):
        if d["b"] == s["node"] and d["t1"] <= s["start"] + 1.0: best = i
    return best
queue = {g: sorted([s for s in plan["services"] if s["ugv"] == g], key=lambda s: s["start"]) for g in ugvs}
qpos = {g: 0 for g in ugvs}
uavs = {u["id"]: dict(legs=u["legs"], k=0, mode="wait", gate=0.0, t_start=None, arr=None, p=list(u["start"]), flight_s=0.0, wait_s=0.0, snaps=0, done_at=0.0, E_legs=0.0, svc=None) for u in plan["uavs"]}
ALT = (terrain["relief"] + 60.0) if terrain else 60.0
ev = plan["event"]; ev_done = False
def spawn_event():
    for i, e in enumerate(ev["edges"]):
        p, q = xy[plan["roads"]["edges"][e][0]], xy[plan["roads"]["edges"][e][1]]
        L = math.hypot(q[0] - p[0], q[1] - p[1]); yaw = math.atan2(q[1] - p[1], q[0] - p[0]); col = "0.85 0.1 0.1 1" if ev["kind"] == "block" else "0.95 0.65 0.05 1"
        sdf = f"<sdf version='1.9'><model name='ev{i}'><static>true</static><pose>{(p[0]+q[0])/2:.1f} {(p[1]+q[1])/2:.1f} 0.3 0 0 {yaw:.4f}</pose><link name='l'><visual name='v'><geometry><box><size>{L:.1f} 70 0.3</size></box></geometry><material><ambient>{col}</ambient><diffuse>{col}</diffuse></material></visual></link></model></sdf>"
        r = EntityFactory(); r.sdf = sdf; node.request("/world/edr/create", r, EntityFactory, Boolean, 3000)

wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
def ugv_tick(g, st, t):
    px, py, _, yaw = poses[f"ugv{g}"]
    if st["mode"] == "idle":
        drv = st["drives"]
        if st["idx"] < len(drv) and t >= max(drv[st["idx"]]["t0"], st["free_at"]): st["mode"] = "drive"; st["t_begin"] = t
        else: drive(f"ugv{g}", 0.0, 0.0); return
    d = st["drives"][st["idx"]]; A, B = np.array(xy[d["a"]]), np.array(xy[d["b"]]); L = float(np.linalg.norm(B - A))
    v_plan = L / max(d["t1"] - d["t0"], 1.0); u = (B - A) / L; r = np.array([px, py]) - A; s = float(r @ u); remain = L - s
    if remain < 3.0:
        drive(f"ugv{g}", 0.0, 0.0); st["arr"][st["idx"]] = t; st["err"].append(t - d["t1"]); st["free_at"] = t; st["idx"] += 1; st["mode"] = "idle"; return
    tgt = A + u * min(L, max(s, 0.0) + 14.0); e = wrap(math.atan2(tgt[1] - py, tgt[0] - px) - yaw)
    w = max(-1.2, min(1.2, 1.6 * e)); v = 0.0 if abs(e) > 0.9 else v_plan * max(0.0, math.cos(e))
    v = min(v, math.sqrt(2 * 1.2 * max(remain - 2.0, 0.0)) + 0.3); drive(f"ugv{g}", v, w)

def uav_tick(uid, st, t):
    n = f"uav{uid}"; legs = st["legs"]
    if st["mode"] == "wait":
        if st["k"] >= len(legs): st["mode"] = "done"; st["done_at"] = max(st["done_at"], st["arr"] or 0.0); vel(n, 0, 0); return
        l = legs[st["k"]]
        if t < max(l["t0"], st["gate"]): vel(n, 0, 0); return
        cur = poses[n]
        if math.hypot(cur[0] - l["p0"][0], cur[1] - l["p0"][1]) > 15.0: st["snaps"] += 1
        set_pose(n, l["p0"][0], l["p0"][1], ALT); st["mode"] = "fly"; st["t_start"] = t; st["p"] = list(l["p0"])
    if st["mode"] == "fly":
        l = legs[st["k"]]; cur = poses[n]; dx, dy = l["p1"][0] - cur[0], l["p1"][1] - cur[1]; dist = math.hypot(dx, dy)
        v = math.hypot(l["p1"][0] - l["p0"][0], l["p1"][1] - l["p0"][1]) / max(l["t1"] - l["t0"], 1.0)
        if dist <= max(v * dt, 1.0):
            set_pose(n, l["p1"][0], l["p1"][1], (gzh(l["p1"][0], l["p1"][1]) + 4.0) if l["land"] else ALT); vel(n, 0, 0)
            st["arr"] = t; st["flight_s"] += t - st["t_start"]; st["E_legs"] += l["E"]
            if l["land"]: st["mode"] = "landed"
            else: st["k"] += 1; st["mode"] = "wait"
            return
        vel(n, v * dx / dist, v * dy / dist); return
    if st["mode"] == "landed":
        s = sv_by_leg.get((uid, st["k"]))
        if s is None: st["k"] += 1; st["mode"] = "wait"; return
        g = s["ugv"]; q = queue[g]
        if qpos[g] < len(q) and q[qpos[g]] is s:
            dep = ugv_dep(s); ready = True if dep is None else ugvs[g]["arr"][dep] is not None
            if ready and t >= ugvs[g].get("free_at_svc", 0.0):
                st["wait_s"] += t - st["arr"]; st["svc_end"] = t + (s["end"] - s["start"]); st["mode"] = "charge"
                ugvs[g]["free_at_svc"] = st["svc_end"]; qpos[g] += 1
        return
    if st["mode"] == "charge":
        if t >= st["svc_end"]: st["gate"] = st["svc_end"]; st["k"] += 1; st["mode"] = "wait"; st["done_at"] = max(st["done_at"], st["svc_end"])

# ---- video ----------------------------------------------------------------------------------------------------
ff = None
if not a.no_video:
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "20", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", os.path.join(a.out, name + ".mp4")], stdin=subprocess.PIPE)
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", 26); FS = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", 18)
except Exception: FONT = FS = ImageFont.load_default()
def write_frame(buf, t):
    im = PILImage.frombytes("RGB", (1280, 720), buf); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1280, 44], fill=(255, 255, 255)); d.text((14, 8), f"{plan['title']}  |  {plan['mode']}", fill=(20, 40, 70), font=FONT)
    d.text((1060, 8), f"t = {int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}", fill=(20, 40, 70), font=FONT)
    d.rectangle([0, 684, 1280, 720], fill=(255, 255, 255))
    for i, (txt, col) in enumerate([("UGV", (217, 64, 38)), ("UAV", (26, 102, 230)), ("blocked road", (217, 26, 26)), ("slow road", (242, 166, 13)), ("task", (38, 140, 115))]):
        d.ellipse([14 + i * 150, 696, 30 + i * 150, 712], fill=col); d.text((38 + i * 150, 692), txt, fill=(20, 40, 70), font=FS)
    ff.stdin.write(im.tobytes())

# ---- run --------------------------------------------------------------------------------------------------------
T_plan = max(l["t1"] for u in plan["uavs"] for l in u["legs"]); horizon = a.max_sim or (1.4 * max(plan["T"], T_plan) + 1800)
t0w = time.time(); t = 0.0; nfr = 0; last_print = 0.0; trace = []
while t < horizon:
    t_next = t + dt
    if not ev_done and t >= ev["t"]: spawn_event(); ev_done = True
    for g, st in ugvs.items(): ugv_tick(g, st, t)
    for uid, st in uavs.items(): uav_tick(uid, st, t)
    step(steps)
    for _ in range(2000):
        if sim[0] >= t_next - 1e-6: break
        time.sleep(0.001)
    t = t_next
    if int(t / dt) % 25 == 0: trace.append([round(t, 1)] + [round(v, 2) for g in ugvs for v in poses[f"ugv{g}"][:3]])
    while frames:
        b = frames.pop(0)
        if ff: write_frame(b, t)
        nfr += 1
    if t - last_print >= 600:
        last_print = t; print(f"  sim {t/60:6.1f} min  wall {time.time()-t0w:6.0f} s  UAV {[s['mode'] for s in uavs.values()]} UGV {[ (s['mode'], s['idx']) for s in ugvs.values()]}", flush=True)
    if all(s["mode"] == "done" for s in uavs.values()) and all(s["idx"] >= len(s["drives"]) for s in ugvs.values()): break
wall = time.time() - t0w
if ff: ff.stdin.close(); ff.wait()
server.terminate()
T_act = max(s["done_at"] for s in uavs.values()); wait_act = sum(s["wait_s"] for s in uavs.values()); E_legs = sum(s["E_legs"] for s in uavs.values())
E_act = E_legs + HOV * wait_act
plan_wait_services = sum(max(0.0, s["start"] - s["touchdown"]) for s in plan["services"])
errs = [e for st in ugvs.values() for e in st["err"]]
res = dict(id=plan["id"], case=plan["case"], mode=plan["mode"], T_plan_s=plan["T"], T_gz_s=T_act, T_legs_plan_s=T_plan, E_plan=plan["E"], E_gz=E_act, E_legs=E_legs, wait_plan_s=plan["wait"], wait_plan_from_services_s=plan_wait_services,
           wait_gz_s=wait_act, ugv_drives=len(errs), ugv_arrival_err_mean_s=float(np.mean(errs)) if errs else 0.0, ugv_arrival_err_max_s=float(np.max(np.abs(errs))) if errs else 0.0,
           ugv_arrival_err_final_s=float(errs[-1]) if errs else 0.0, uav_snaps=sum(s["snaps"] for s in uavs.values()), terrain=bool(terrain), ugv_z_above_ground=[round(poses[f"ugv{g}"][2] - gzh(poses[f"ugv{g}"][0], poses[f"ugv{g}"][1]), 2) for g in ugvs], sim_s=t, wall_s=wall, unfinished=not (all(s["mode"] == "done" for s in uavs.values())))
res["ugv_trace"] = trace[::4] if len(trace) > 400 else trace
json.dump(res, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
print(json.dumps(res, indent=1))
