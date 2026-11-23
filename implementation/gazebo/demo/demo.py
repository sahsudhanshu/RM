#!/usr/bin/env python3
"""Runnable 3D Gazebo demo of a planned UAV-UGV mission (no experiments).

The plan (exported from the simulator) is scaled to a demo world: space x1/50 and time x1/60, so a ~3.5 min video/real-time
run replays a ~200 min mission, with the same relative speeds. UGVs are wheeled DiffDrive vehicles (real contact physics) driven
along the planned road edges; UAVs are velocity-controlled quadrotors that fly the planned legs, land on the UGV roof pad, charge
while riding on it and take off again. Road events appear as cones/strips at the planned time; visited tasks turn grey.

  python3 demo.py plans/<plan>.json                 # Gazebo window (needs a display), real-time
  python3 demo.py plans/<plan>.json --terrain terrain/thar24    # real-terrain heightmap
  python3 demo.py plans/<plan>.json --headless --record out.mp4 # no window: render an oblique camera to a video
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_world import world_sdf, terrain_fn, event_marker_sdf, UGV_ROOF, UAV_REST
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("plan"); ap.add_argument("--scale", type=float, default=50.0, help="space scale 1:N")
ap.add_argument("--speed", type=float, default=60.0, help="time compression (plan seconds per demo second)")
ap.add_argument("--rtf", type=float, default=None, help="wall-clock pace of the demo (1 = real time; 0 = as fast as possible)")
ap.add_argument("--terrain", default="", help="folder with heightmap.png, diffuse.png, normal.png, info.json, dem.npz")
ap.add_argument("--vex", type=float, default=1.4, help="vertical exaggeration of the terrain")
ap.add_argument("--headless", action="store_true", help="no window (server only)"); ap.add_argument("--record", default="", help="headless: write an MP4 of an oblique camera")
ap.add_argument("--max-sim", type=float, default=0.0, help="stop after this many demo seconds"); ap.add_argument("--follow", default="", help="GUI: follow a vehicle, e.g. ugv0")
ap.add_argument("--cam", default="follow", choices=["follow", "overview"], help="recording camera: follow the first UGV or a fixed overview")
ap.add_argument("--out", default="demo_out"); ap.add_argument("--no-verify", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
K, C = 1.0 / a.scale, a.speed
rtf = a.rtf if a.rtf is not None else (1.0 if not a.headless else 0.0)

# ---- scale the plan --------------------------------------------------------------------------------------------------
P = json.load(open(a.plan)); P["id"] = P["id"]
sc = lambda p: [p[0] * K, p[1] * K]
plan = dict(id=P["id"], title=P["title"], mode=P["mode"], roads=dict(xy=[sc(p) for p in P["roads"]["xy"]], edges=P["roads"]["edges"]),
            tasks=dict(xy=[sc(p) for p in P["tasks"]["xy"]], n_ground=P["tasks"]["n_ground"], depot=sc(P["tasks"]["depot"])),
            ugvs=[dict(id=g["id"], start=sc(g["start"]), drives=[dict(t0=d["t0"] / C, t1=d["t1"] / C, a=d["a"], b=d["b"]) for d in g["drives"]]) for g in P["ugvs"]],
            uavs=[dict(id=u["id"], start=sc(u["start"]), legs=[dict(t0=l["t0"] / C, t1=l["t1"] / C, p0=sc(l["p0"]), p1=sc(l["p1"]), land=l["land"]) for l in u["legs"]]) for u in P["uavs"]],
            services=[dict(uav=s["uav"], ugv=s["ugv"], touchdown=s["touchdown"] / C, start=s["start"] / C, end=s["end"] / C, node=s["node"]) for s in P["services"]],
            event=dict(t=P["event"]["t"] / C, kind=P["event"]["kind"], edges=P["event"]["edges"]), T=P["T"] / C)
terrain = None
if a.terrain:
    td = os.path.abspath(a.terrain); info = json.load(open(os.path.join(td, "info.json")))
    from PIL import Image as _I
    dem = np.asarray(_I.open(os.path.join(td, "heightmap.png")).convert("L"), dtype=np.float64)[::-1] / 255.0 * info["relief"]   # row 0 = south edge (y = 0)
    terrain = dict(dir=td, size=info["size"] * K, relief=info["relief"] * K * a.vex, dem=dem * K * a.vex)
gzh = terrain_fn(terrain) if terrain else (lambda x, y: 0.0)
xy = plan["roads"]["xy"]
allx = [p[0] for p in xy] + [p[0] for p in plan["tasks"]["xy"]]; ally = [p[1] for p in xy] + [p[1] for p in plan["tasks"]["xy"]]
cx, cy = (min(allx) + max(allx)) / 2, (min(ally) + max(ally)) / 2; span = max(max(allx) - min(allx), (max(ally) - min(ally)) * 16 / 9) + 50
cam_d, cam_h = span * 0.80, span * 0.55; cam_pitch = math.atan2(cam_h, cam_d)
cam = (cx, cy - cam_d, cam_h + (terrain["relief"] if terrain else 0.0), cam_pitch, math.pi / 2)
name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in plan["id"])
world_path = os.path.abspath(os.path.join(a.out, name + ".sdf"))
open(world_path, "w").write(world_sdf(plan, terrain, record_cam=cam if a.record else None))
print(f"plan: {plan['title']} [{plan['mode']}]  demo duration ~{plan['T']:.0f} s  (space 1:{a.scale:.0f}, time x{C:.0f})", flush=True)

# ---- start Gazebo --------------------------------------------------------------------------------------------------------
env = dict(os.environ); env.setdefault("GZ_IP", "127.0.0.1"); env["GZ_PARTITION"] = f"edrdemo{os.getpid()}"
os.environ.update(GZ_IP=env["GZ_IP"], GZ_PARTITION=env["GZ_PARTITION"])
cmd = ["gz", "sim", "-s", "--headless-rendering", "-v", "1", world_path] if a.headless else ["gz", "sim", "-v", "2", world_path]
server = subprocess.Popen(cmd, env=env)
node = Node(); poses = {}; sim = [0.0]; frames = []
def on_pose(m):
    for p in m.pose: poses[p.name] = (p.position.x, p.position.y, p.position.z, 2 * math.atan2(p.orientation.z, p.orientation.w))
    sim[0] = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
node.subscribe(Pose_V, "/world/edr/dynamic_pose/info", on_pose)
if a.record: node.subscribe(Image, "/rec_cam", lambda m: frames.append(bytes(m.data)))
pubs = {}
def vel(n, vx, vy, vz=0.0, wz=0.0):
    if n not in pubs: pubs[n] = node.advertise(f"/model/{n}/cmd_vel", Twist)
    t = Twist(); t.linear.x, t.linear.y, t.linear.z, t.angular.z = vx, vy, vz, wz; pubs[n].publish(t)
def set_pose(n, x, y, z, yaw=0.0, pitch=0.0):
    p = Pose(); p.name = n; p.position.x, p.position.y, p.position.z = x, y, z
    cp, sp_, cy_, sy_ = math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = cp * cy_, -sp_ * sy_, sp_ * cy_, cp * sy_
    return node.request("/world/edr/set_pose", p, Pose, Boolean, 2000)
def step(k):
    w = WorldControl(); w.multi_step = k; w.pause = True
    return node.request("/world/edr/control", w, WorldControl, Boolean, 5000)[0]
def create(sdf):
    r = EntityFactory(); r.sdf = sdf; return node.request("/world/edr/create", r, EntityFactory, Boolean, 3000)
def remove(model):
    e = Entity(); e.name = model; e.type = Entity.MODEL; return node.request("/world/edr/remove", e, Entity, Boolean, 3000)

t_wait = time.time()
while not (all(f"ugv{g['id']}" in poses for g in plan["ugvs"]) and all(f"uav{u['id']}" in poses for u in plan["uavs"])):
    if server.poll() is not None or time.time() - t_wait > 120: sys.exit("Gazebo did not start (is gz-harmonic installed and, without --headless, a display available?)")
    time.sleep(0.5)
time.sleep(1.0)

def set_gui_camera():
    from gz.msgs10.gui_camera_pb2 import GUICamera
    x, y, z, pitch, yaw = cam
    g = GUICamera(); g.name = "gzclient_camera"; g.pose.position.x, g.pose.position.y, g.pose.position.z = x, y, z
    cp, sp_, cy_, sy_ = math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    g.pose.orientation.w, g.pose.orientation.x, g.pose.orientation.y, g.pose.orientation.z = cp * cy_, -sp_ * sy_, sp_ * cy_, cp * sy_
    for _ in range(40):
        ok = node.request("/gui/move_to/pose", g, GUICamera, Boolean, 1000)
        if ok[0] and ok[1].data: break
        time.sleep(1.0)
    if a.follow:
        from gz.msgs10.stringmsg_pb2 import StringMsg
        s = StringMsg(); s.data = a.follow; node.request("/gui/follow", s, StringMsg, Boolean, 1000)
if not a.headless: set_gui_camera()

# ---- controllers ------------------------------------------------------------------------------------------------------------
dt = 0.03; nsteps = 3; ALT = 14.0 + (terrain["relief"] if terrain else 0.0)
wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
ugvs = {g["id"]: dict(drives=g["drives"], idx=0, mode="idle", arr=[None] * len(g["drives"]), free_at=0.0) for g in plan["ugvs"]}
sv_by_leg = {}
for s in plan["services"]:
    for u in plan["uavs"]:
        if u["id"] == s["uav"]:
            for k, l in enumerate(u["legs"]):
                if l["land"] and abs(l["t1"] - s["touchdown"]) < 1.0 / C * 60: sv_by_leg[(u["id"], k)] = s
queue = {g: sorted([s for s in plan["services"] if s["ugv"] == g], key=lambda s: s["start"]) for g in ugvs}; qpos = {g: 0 for g in ugvs}
uavs = {u["id"]: dict(legs=u["legs"], k=0, mode="wait", gate=0.0, ride=None, done_at=0.0) for u in plan["uavs"]}
def ugv_dep(s):
    best = None
    for i, d in enumerate(ugvs[s["ugv"]]["drives"]):
        if d["b"] == s["node"] and d["t1"] <= s["start"] + 1e-6: best = i
    return best
visited = set()
def mark_done(p):
    G = plan["tasks"]["n_ground"]
    best = min(range(G, len(plan["tasks"]["xy"])), key=lambda k: math.hypot(plan["tasks"]["xy"][k][0] - p[0], plan["tasks"]["xy"][k][1] - p[1]))
    if best in visited or math.hypot(plan["tasks"]["xy"][best][0] - p[0], plan["tasks"]["xy"][best][1] - p[1]) > 4.0: return
    visited.add(best); x, y = plan["tasks"]["xy"][best]; remove(f"task{best}")
    from demo_world import task_marker
    create(f"<sdf version='1.9'>{task_marker(f'taskdone{best}', x, y, gzh(x, y), True, done=True)}</sdf>")
def ugv_tick(g, st, t):
    n = f"ugv{g}"; px, py, _, yaw = poses[n]
    if st["mode"] == "idle":
        if st["idx"] < len(st["drives"]) and t >= max(st["drives"][st["idx"]]["t0"], st["free_at"]): st["mode"] = "drive"
        else: vel(n, 0, 0, 0, 0); return
    d = st["drives"][st["idx"]]; A, B = np.array(xy[d["a"]]), np.array(xy[d["b"]]); L = float(np.linalg.norm(B - A))
    v_plan = L / max(d["t1"] - d["t0"], 0.05); u = (B - A) / L; s = float((np.array([px, py]) - A) @ u); remain = L - s
    if remain < 1.5:
        vel(n, 0, 0, 0, 0); st["arr"][st["idx"]] = t; st["free_at"] = t; st["idx"] += 1; st["mode"] = "idle"; return
    tgt = A + u * min(L, max(s, 0.0) + 6.0); e = wrap(math.atan2(tgt[1] - py, tgt[0] - px) - yaw)
    w = max(-1.3, min(1.3, 1.8 * e)); v = 0.0 if abs(e) > 0.9 else v_plan * max(0.0, math.cos(e)); v = min(v, math.sqrt(2 * 2.5 * max(remain - 1.0, 0.0)) + 0.4); vel(n, v, 0, 0, w)
def uav_tick(uid, st, t):
    n = f"uav{uid}"; legs = st["legs"]; px, py, pz, _ = poses[n]
    if st["mode"] == "charge":
        g = st["ride"]; ux, uy, uz, uyaw = poses[f"ugv{g}"]
        set_pose(n, ux - 0.9 * math.cos(uyaw), uy - 0.9 * math.sin(uyaw), uz + 0.62 + 0.55, uyaw)
        if t >= st["svc_end"]: st["gate"] = st["svc_end"]; st["k"] += 1; st["mode"] = "wait"
        return
    if st["mode"] == "wait":
        if st["k"] >= len(legs): st["mode"] = "done"; st["done_at"] = t; vel(n, 0, 0, 0); return
        l = legs[st["k"]]
        if st["ride"] is not None:                       # sitting on a UGV: keep following it until take-off
            g = st["ride"]; ux, uy, uz, uyaw = poses[f"ugv{g}"]; set_pose(n, ux - 0.9 * math.cos(uyaw), uy - 0.9 * math.sin(uyaw), uz + 0.62 + 0.55, uyaw)
        if t < max(l["t0"], st["gate"]): vel(n, 0, 0, 0); return
        st["ride"] = None; st["mode"] = "fly"
    if st["mode"] == "fly":
        l = legs[st["k"]]; dx, dy = l["p1"][0] - px, l["p1"][1] - py; dist = math.hypot(dx, dy)
        v = math.hypot(l["p1"][0] - l["p0"][0], l["p1"][1] - l["p0"][1]) / max(l["t1"] - l["t0"], 0.05)
        near_land = l["land"] and dist < 14.0
        zt = (gzh(l["p1"][0], l["p1"][1]) + 5.0) if near_land else ALT
        vz = max(-4.0, min(4.0, 1.4 * (zt - pz)))
        if dist <= max(v * dt, 0.8):
            vel(n, 0, 0, vz)
            if l["land"]: st["mode"] = "landed"; st["arr"] = t
            else: mark_done(l["p1"]); st["k"] += 1; st["mode"] = "wait"
            return
        vel(n, v * dx / dist, v * dy / dist, vz); return
    if st["mode"] == "landed":
        s = sv_by_leg.get((uid, st["k"]))
        if s is None: st["k"] += 1; st["mode"] = "wait"; return
        g = s["ugv"]; q = queue[g]
        vel(n, 0, 0, max(-3.0, min(3.0, 1.5 * ((gzh(px, py) + 3.0) - pz))))
        if qpos[g] < len(q) and q[qpos[g]] is s:
            dep = ugv_dep(s); ready = dep is None or ugvs[g]["arr"][dep] is not None
            if ready and t >= ugvs[g].get("free_at_svc", 0.0):
                st["svc_end"] = t + (s["end"] - s["start"]); st["mode"] = "charge"; st["ride"] = g; ugvs[g]["free_at_svc"] = st["svc_end"]; qpos[g] += 1
        return
ev = plan["event"]; ev_done = False
def spawn_event():
    for i, e in enumerate(ev["edges"]):
        p, q = xy[plan["roads"]["edges"][e][0]], xy[plan["roads"]["edges"][e][1]]
        create(event_marker_sdf(f"event{i}", p[0], p[1], q[0], q[1], (gzh(*p) + gzh(*q)) / 2, ev["kind"]))

# ---- video (headless) ---------------------------------------------------------------------------------------------------------
ff = None
if a.record:
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "15", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", a.record], stdin=subprocess.PIPE)
    from PIL import Image as PILImage, ImageDraw, ImageFont
    try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", 26)
    except Exception: FONT = ImageFont.load_default()

# ---- run ------------------------------------------------------------------------------------------------------------------------
horizon = a.max_sim or (plan["T"] * 1.4 + 30); t = 0.0; t_wall0 = time.time(); last = -1
print("running; close the Gazebo window or press Ctrl+C to stop" if not a.headless else "running headless", flush=True)
try:
    while t < horizon and server.poll() is None:
        t_next = t + dt
        if not ev_done and t >= ev["t"]: spawn_event(); ev_done = True
        for g, st in ugvs.items(): ugv_tick(g, st, t)
        if a.record and a.cam == "follow":
            fx, fy, fz, fyaw = poses[f"ugv{plan['ugvs'][0]['id']}"]; set_pose("rec_cam", fx - 24 * math.cos(fyaw), fy - 24 * math.sin(fyaw), fz + 12, fyaw, 0.42)
        for uid, st in uavs.items(): uav_tick(uid, st, t)
        step(nsteps)
        for _ in range(2000):
            if sim[0] >= t_next - 1e-6 or server.poll() is not None: break
            time.sleep(0.0005)
        t = t_next
        if rtf > 0:
            lag = t_wall0 + t / rtf - time.time()
            if lag > 0: time.sleep(lag)
        while frames:
            b = frames.pop(0)
            if ff:
                im = PILImage.frombytes("RGB", (1280, 720), b); d = ImageDraw.Draw(im); pm = t * C / 60
                d.rectangle([0, 0, 1280, 44], fill=(255, 255, 255)); d.text((14, 8), f"{plan['title']} | {plan['mode']}", fill=(20, 40, 70), font=FONT); d.text((1010, 8), f"plan time {int(pm // 60)}:{int(pm % 60):02d} h", fill=(20, 40, 70), font=FONT)
                ff.stdin.write(im.tobytes())
        if int(t) != last and int(t) % 10 == 0:
            last = int(t); print(f"  demo {t:5.0f} s  (plan {t * C / 60:5.0f} min)  UAV {[s['mode'] for s in uavs.values()]}  UGV {[(s['mode'], s['idx']) for s in ugvs.values()]}", flush=True)
        if all(s["mode"] == "done" for s in uavs.values()) and all(s["idx"] >= len(s["drives"]) for s in ugvs.values()):
            print(f"mission complete after {t:.0f} demo seconds (plan time {t * C / 60:.0f} min)", flush=True); break
    if not a.headless and server.poll() is None and not a.max_sim:
        print("done - the world stays open; close the Gazebo window to exit", flush=True)
        while server.poll() is None: time.sleep(1.0)
except KeyboardInterrupt:
    pass
finally:
    if ff: ff.stdin.close(); ff.wait()
    if server.poll() is None: server.terminate()
