"""3D demo world (SDF 1.9) for a scaled plan: sand ground (flat or real-terrain heightmap), roads, task markers,
wheeled UGVs (DiffDrive physics, roof landing pad) and quadrotor UAVs with spinning rotors. All sizes are in demo metres."""
import math

ROAD_W = 5.0
COL = dict(road="0.22 0.23 0.26 1", node="0.30 0.31 0.34 1", ground="0.86 0.78 0.60 1", aerial="0.12 0.62 0.45 1", done="0.55 0.58 0.58 1",
           pad="0.95 0.75 0.15 1", depot="0.95 0.55 0.10 1", ugv="0.28 0.33 0.22 1", cab="0.20 0.24 0.17 1", uav=["0.10 0.45 0.95 1", "0.85 0.20 0.60 1", "0.10 0.75 0.80 1", "0.95 0.50 0.10 1"])
UGV_Z = 0.75
UGV_ROOF = 1.25          # top of the landing pad above the model origin height (m)
UAV_REST = 0.6           # UAV model height above the surface it rests on


def mat(c): return f"<material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>"
def box(name, size, pose, c, extra=""): return f'<visual name="{name}"><pose>{pose}</pose><geometry><box><size>{size}</size></box></geometry>{mat(c)}{extra}</visual>'
def cyl(name, r, h, pose, c): return f'<visual name="{name}"><pose>{pose}</pose><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>{mat(c)}</visual>'
def sph(name, r, pose, c): return f'<visual name="{name}"><pose>{pose}</pose><geometry><sphere><radius>{r}</radius></sphere></geometry>{mat(c)}</visual>'
def inertia(m, ix, iy, iz): return f"<inertial><mass>{m}</mass><inertia><ixx>{ix}</ixx><iyy>{iy}</iyy><izz>{iz}</izz></inertia></inertial>"


def terrain_fn(t):
    z, size = t["dem"], t["size"]; n = z.shape[0] - 1
    def f(x, y):
        u = min(max(x / size * n, 0.0), n - 1e-6); v = min(max(y / size * n, 0.0), n - 1e-6); i, j = int(v), int(u); fu, fv = u - j, v - i
        return float(z[i, j] * (1 - fu) * (1 - fv) + z[i, j + 1] * fu * (1 - fv) + z[i + 1, j] * (1 - fu) * fv + z[i + 1, j + 1] * fu * fv)
    return f


def ugv_sdf(name, x, y, z, yaw):
    wheel = lambda n, ys: (f'<link name="{n}"><pose relative_to="chassis">0 {ys} -0.05 -1.5707 0 0</pose>{inertia(8, 0.6, 0.6, 0.6)}'
                           f'<visual name="v"><geometry><cylinder><radius>0.7</radius><length>0.5</length></cylinder></geometry>{mat("0.08 0.08 0.08 1")}</visual>'
                           f'<collision name="c"><geometry><cylinder><radius>0.7</radius><length>0.5</length></cylinder></geometry><surface><friction><ode><mu>1.4</mu><mu2>1.4</mu2></ode></friction></surface></collision></link>')
    joint = lambda n, c: f'<joint name="{n}" type="revolute"><parent>chassis</parent><child>{c}</child><axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>'
    caster = lambda n, xs: (f'<link name="{n}"><pose relative_to="chassis">{xs} 0 -0.40 0 0 0</pose>{inertia(2, 0.05, 0.05, 0.05)}'
                            f'<collision name="c"><geometry><sphere><radius>0.35</radius></sphere></geometry><surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision></link>')
    vis = (box("body", "5.2 2.4 0.8", "0 0 0 0 0 0", COL["ugv"]) + box("cab", "1.9 2.2 0.9", "1.5 0 0.85 0 0 0", COL["cab"])
           + box("window", "0.1 1.8 0.5", "2.47 0 0.95 0 0 0", "0.55 0.75 0.85 1") + box("bed", "2.6 2.2 0.15", "-0.9 0 0.48 0 0 0", COL["cab"])
           + box("pad", "2.6 2.2 0.08", "-0.9 0 0.58 0 0 0", COL["pad"]) + box("H1", "0.2 1.2 0.02", "-1.6 0 0.63 0 0 0", "0.1 0.1 0.1 1")
           + box("H2", "0.2 1.2 0.02", "-0.2 0 0.63 0 0 0", "0.1 0.1 0.1 1") + box("H3", "1.4 0.2 0.02", "-0.9 0 0.63 0 0 0", "0.1 0.1 0.1 1")
           + cyl("mast", 0.06, 1.6, "2.0 0.8 1.9 0 0 0", "0.1 0.1 0.1 1") + sph("beacon", 0.18, "2.0 0.8 2.75 0 0 0", "0.95 0.2 0.1 1")
           + "".join(cyl(f"w{i}", 0.7, 0.5, f"{xs} {ys} -0.05 -1.5707 0 0", "0.08 0.08 0.08 1") for i, (xs, ys) in enumerate([(1.9, 1.3), (1.9, -1.3), (-1.9, 1.3), (-1.9, -1.3)])))
    return (f'<model name="{name}"><pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.4f}</pose>'
            f'<link name="chassis">{inertia(600, 320, 1382, 1638)}{vis}<collision name="body"><geometry><box><size>5.2 2.4 0.8</size></box></geometry></collision></link>'
            f'{wheel("left_wheel", 1.3)}{wheel("right_wheel", -1.3)}{caster("caster_f", 2.0)}{caster("caster_r", -2.0)}'
            f'{joint("left_wheel_joint", "left_wheel")}{joint("right_wheel_joint", "right_wheel")}'
            f'<joint name="cf" type="fixed"><parent>chassis</parent><child>caster_f</child></joint><joint name="cr" type="fixed"><parent>chassis</parent><child>caster_r</child></joint>'
            f'<plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive"><left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>'
            f'<wheel_separation>2.6</wheel_separation><wheel_radius>0.7</wheel_radius><max_linear_velocity>9</max_linear_velocity><min_linear_velocity>-9</min_linear_velocity>'
            f'<max_linear_acceleration>3</max_linear_acceleration><min_linear_acceleration>-5</min_linear_acceleration><max_angular_velocity>1.3</max_angular_velocity><min_angular_velocity>-1.3</min_angular_velocity>'
            f'<topic>/model/{name}/cmd_vel</topic></plugin></model>')


def uav_sdf(name, x, y, z, color):
    arm = lambda i, yaw: box(f"arm{i}", "3.4 0.14 0.09", f"0 0 0.05 0 0 {yaw}", "0.12 0.12 0.14 1")
    rotor = lambda i, rx, ry: (f'<link name="rotor{i}"><gravity>false</gravity><pose relative_to="body">{rx} {ry} 0.2 0 0 0</pose>{inertia(0.05, 0.001, 0.001, 0.002)}'
                               f'{box("b1", "1.5 0.13 0.02", "0 0 0 0 0 0", "0.05 0.05 0.06 0.9")}{box("b2", "0.13 1.5 0.02", "0 0 0 0 0 0", "0.05 0.05 0.06 0.9")}{cyl("hub", 0.1, 0.1, "0 0 -0.03 0 0 0", "0.3 0.3 0.33 1")}</link>')
    pos = [(1.2, 1.2), (-1.2, 1.2), (-1.2, -1.2), (1.2, -1.2)]
    joints = "".join(f'<joint name="rj{i}" type="revolute"><parent>body</parent><child>rotor{i}</child><axis><xyz>0 0 1</xyz></axis></joint>' for i in range(4))
    ctrl = "".join(f'<plugin filename="gz-sim-joint-controller-system" name="gz::sim::systems::JointController"><joint_name>rj{i}</joint_name><initial_velocity>{55 * (1 if i % 2 == 0 else -1)}</initial_velocity></plugin>' for i in range(4))
    vis = (box("hull", "1.1 1.1 0.3", "0 0 0.1 0 0 0", "0.16 0.16 0.19 1") + box("plate", "0.95 0.95 0.04", "0 0 0.29 0 0 0", color) + arm(0, math.pi / 4) + arm(1, -math.pi / 4)
           + sph("cam", 0.2, "0.45 0 -0.12 0 0 0", "0.05 0.05 0.06 1")
           + "".join(cyl(f"leg{i}", 0.04, 0.5, f"{lx} {ly} -0.25 0 0 0", "0.12 0.12 0.14 1") for i, (lx, ly) in enumerate([(0.4, 0.4), (-0.4, 0.4), (-0.4, -0.4), (0.4, -0.4)]))
           + "".join(cyl(f"mt{i}", 0.12, 0.2, f"{rx} {ry} 0.1 0 0 0", "0.2 0.2 0.23 1") for i, (rx, ry) in enumerate(pos)))
    return (f'<model name="{name}"><pose>{x:.2f} {y:.2f} {z:.2f} 0 0 0</pose><link name="body"><gravity>false</gravity>{inertia(2, 0.1, 0.1, 0.15)}{vis}</link>'
            + "".join(rotor(i, rx, ry) for i, (rx, ry) in enumerate(pos)) + joints + ctrl
            + f'<plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl"><topic>/model/{name}/cmd_vel</topic></plugin></model>')


def task_marker(name, x, y, z, aerial, done=False):
    if aerial:
        c = COL["done"] if done else COL["aerial"]
        return (f'<model name="{name}"><static>true</static><pose>{x:.2f} {y:.2f} {z:.2f} 0 0 0</pose><link name="l">'
                + cyl("pole", 0.12, 7.0, "0 0 3.5 0 0 0", "0.25 0.25 0.28 1") + sph("top", 1.3, "0 0 7.4 0 0 0", c) + cyl("base", 1.6, 0.1, "0 0 0.05 0 0 0", c) + "</link></model>")
    c = COL["pad"]
    return (f'<model name="{name}"><static>true</static><pose>{x:.2f} {y:.2f} {z:.2f} 0 0 0</pose><link name="l">'
            + box("pad", "9 9 0.12", "0 0 0.06 0 0 0", c) + box("h1", "0.6 4.4 0.02", "-1.4 0 0.13 0 0 0", "0.1 0.1 0.1 1") + box("h2", "0.6 4.4 0.02", "1.4 0 0.13 0 0 0", "0.1 0.1 0.1 1")
            + box("h3", "3.4 0.6 0.02", "0 0 0.13 0 0 0", "0.1 0.1 0.1 1") + "</link></model>")


def event_marker_sdf(name, x0, y0, x1, y1, z, kind):
    L = math.hypot(x1 - x0, y1 - y0); yaw = math.atan2(y1 - y0, x1 - x0); c = "0.85 0.10 0.10 1" if kind == "block" else "0.95 0.65 0.05 1"
    cones = "".join(f'<visual name="k{i}"><pose>{(-L / 2 + (i + 0.5) * L / max(2, int(L // 6))):.2f} 0 0.9 0 0 0</pose><geometry><cylinder><radius>0.45</radius><length>1.8</length></cylinder></geometry>{mat(c)}</visual>' for i in range(max(2, int(L // 6))))
    return (f"<sdf version='1.9'><model name='{name}'><static>true</static><pose>{(x0 + x1) / 2:.2f} {(y0 + y1) / 2:.2f} {z:.2f} 0 0 {yaw:.4f}</pose><link name='l'>"
            f"{box('strip', f'{L:.1f} 7.0 0.25', '0 0 0.12 0 0 0', c)}{cones}</link></model></sdf>")


def world_sdf(plan, terrain=None, record_cam=None, cam_res=(1280, 720)):
    """plan: scaled plan (demo metres). terrain: dict(dir, size, relief, dem). record_cam: (x,y,z,pitch,yaw) -> adds a camera sensor."""
    gz = terrain_fn(terrain) if terrain else (lambda x, y: 0.0)
    xy, edges = plan["roads"]["xy"], plan["roads"]["edges"]
    body = []
    for k, (a, b) in enumerate(edges):
        (x0, y0), (x1, y1) = xy[a], xy[b]; L = math.hypot(x1 - x0, y1 - y0); yaw = math.atan2(y1 - y0, x1 - x0); n = max(1, int(L // 8)) if terrain else 1
        for q in range(n):
            f0, f1 = q / n, (q + 1) / n; ax, ay, bx, by = x0 + (x1 - x0) * f0, y0 + (y1 - y0) * f0, x0 + (x1 - x0) * f1, y0 + (y1 - y0) * f1
            mx, my = (ax + bx) / 2, (ay + by) / 2; pitch = -math.atan2(gz(bx, by) - gz(ax, ay), math.hypot(bx - ax, by - ay)) if terrain else 0.0
            body.append(box(f"r{k}_{q}", f"{L / n + 0.8:.2f} {ROAD_W} 0.12", f"{mx:.2f} {my:.2f} {gz(mx, my) + (0.4 if terrain else 0.1):.2f} 0 {pitch:.4f} {yaw:.4f}", COL["road"]))
    for k, (x, y) in enumerate(xy): body.append(cyl(f"n{k}", ROAD_W / 2, 0.12, f"{x:.2f} {y:.2f} {gz(x, y) + 0.12:.2f} 0 0 0", COL["node"]))
    dx, dy = plan["tasks"]["depot"]
    depot = (f'<model name="depot"><static>true</static><pose>{dx:.2f} {dy:.2f} {gz(dx, dy):.2f} 0 0 0</pose><link name="l">' + cyl("pad", 9.5, 0.14, "0 0 0.07 0 0 0", COL["depot"]) + box("h1", "0.7 5 0.02", "-1.5 0 0.15 0 0 0", "0.1 0.1 0.1 1")
             + box("h2", "0.7 5 0.02", "1.5 0 0.15 0 0 0", "0.1 0.1 0.1 1") + box("h3", "3.7 0.7 0.02", "0 0 0.15 0 0 0", "0.1 0.1 0.1 1") + cyl("pole", 0.1, 9, "6.5 0 4.5 0 0 0", "0.3 0.3 0.3 1") + box("flag", "0.05 3 1.6", "6.5 1.5 8.2 0 0 0", "0.85 0.15 0.15 1") + "</link></model>")
    tasks = []
    G = plan["tasks"]["n_ground"]
    for k, (x, y) in enumerate(plan["tasks"]["xy"]):
        if k < G and [x, y] == [dx, dy]: continue
        tasks.append(task_marker(f"task{k}", x, y, gz(x, y), aerial=k >= G))
    vehicles = []
    for g in plan["ugvs"]:
        d = g["drives"]; yaw = 0.0
        if d:
            a, b = xy[d[0]["a"]], xy[d[0]["b"]]; yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        vehicles.append(ugv_sdf(f"ugv{g['id']}", g["start"][0], g["start"][1], gz(*g["start"]) + UGV_Z + 0.25, yaw))
    for u in plan["uavs"]:
        vehicles.append(uav_sdf(f"uav{u['id']}", u["start"][0], u["start"][1], gz(*u["start"]) + UAV_REST + 0.2, COL["uav"][u["id"] % 4]))
    xs = [p[0] for p in xy] + [p[0] for p in plan["tasks"]["xy"]]; ys = [p[1] for p in xy] + [p[1] for p in plan["tasks"]["xy"]]
    if terrain:
        h = terrain["size"] / 2; u = "file://" + terrain["dir"]
        hm = lambda extra: f'<heightmap><uri>{u}/heightmap.png</uri><size>{terrain["size"]:.1f} {terrain["size"]:.1f} {terrain["relief"]:.3f}</size>{extra}</heightmap>'
        tex = f'<texture><size>{terrain["size"]:.1f}</size><diffuse>{u}/diffuse.png</diffuse><normal>{u}/normal.png</normal></texture>'
        ground = (f'<model name="ground"><static>true</static><pose>{h:.1f} {h:.1f} 0 0 0 0</pose><link name="g"><collision name="c"><geometry>{hm("")}</geometry></collision></link></model>'
                  f'<model name="ground_vis"><static>true</static><link name="g"><visual name="v"><geometry>{hm(f"<pos>{h:.1f} {h:.1f} 0</pos>" + tex)}</geometry></visual></link></model>')
    else:
        ground = (f'<model name="ground"><static>true</static><link name="g"><collision name="c"><pose>0 0 -0.5 0 0 0</pose><geometry><box><size>3000 3000 1</size></box></geometry>'
                  f'<surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision><visual name="v"><pose>0 0 -0.5 0 0 0</pose><geometry><box><size>3000 3000 1</size></box></geometry>{mat(COL["ground"])}</visual></link></model>')
    cam = ""
    if record_cam:
        cx, cy, cz, cp, cyaw = record_cam
        cam = (f'<model name="rec_cam"><pose>{cx:.1f} {cy:.1f} {cz:.1f} 0 {cp:.4f} {cyaw:.4f}</pose><link name="l"><gravity>false</gravity><inertial><mass>1</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial><sensor name="cam" type="camera"><camera><horizontal_fov>1.05</horizontal_fov>'
               f'<image><width>{cam_res[0]}</width><height>{cam_res[1]}</height></image><clip><near>0.5</near><far>3000</far></clip></camera><always_on>1</always_on><update_rate>15</update_rate><topic>rec_cam</topic></sensor></link></model>')
    return f'''<?xml version="1.0"?>
<sdf version="1.9"><world name="edr">
  <physics name="p" type="ode"><max_step_size>0.01</max_step_size><real_time_factor>1</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <scene><ambient>0.55 0.55 0.58 1</ambient><background>0.60 0.76 0.92 1</background><sky/></scene>
  <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 200 0 0 0</pose><diffuse>1 0.97 0.9 1</diffuse><specular>0.25 0.25 0.25 1</specular><direction>-0.5 0.3 -1</direction></light>
  {ground}
  <model name="map"><static>true</static><link name="l">{''.join(body)}</link></model>
  {depot}{''.join(tasks)}{''.join(vehicles)}{cam}
</world></sdf>'''
