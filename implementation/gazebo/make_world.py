#!/usr/bin/env python
"""Generate a Gazebo (SDF 1.9) world for one exported plan: road strips, task markers, wheeled UGVs (DiffDrive, real
physics), velocity-controlled UAVs, and a top-down camera. Vehicles have real sizes; large visual-only beacons make
them visible at the 10 km map scale."""
import json, math, os, sys

COLORS = dict(road="0.25 0.27 0.30 1", ground="0.80 0.84 0.76 1", task="0.15 0.55 0.45 1", depot="0.9 0.7 0.1 1", ugv="0.85 0.25 0.15 0.9", uav="0.1 0.4 0.9 0.9")
ROAD_W = 45.0
UGV_Z = 0.45


def box_visual(name, size, color, z=0.0, extra=""):
    return f'<visual name="{name}"><pose>0 0 {z} 0 0 0</pose><geometry><box><size>{size}</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>{extra}</visual>'


def cyl_visual(name, r, h, color, z=0.0):
    return f'<visual name="{name}"><pose>0 0 {z} 0 0 0</pose><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>'


def terrain_height_fn(t):
    import numpy as np
    z, size = t["dem"], t["size"]; n = z.shape[0] - 1
    def f(x, y):
        u = min(max(x / size * n, 0.0), n - 1e-6); v = min(max(y / size * n, 0.0), n - 1e-6); i, j = int(v), int(u); fu, fv = u - j, v - i
        return float(z[i, j] * (1 - fu) * (1 - fv) + z[i, j + 1] * fu * (1 - fv) + z[i + 1, j] * (1 - fu) * fv + z[i + 1, j + 1] * fu * fv)
    return f


def ugv_sdf(name, x, y, yaw, z0=UGV_Z):
    wheel = lambda n, ys: f'''<link name="{n}"><pose relative_to="chassis">0 {ys} -0.15 -1.5707 0 0</pose>
      <inertial><mass>2</mass><inertia><ixx>0.05</ixx><iyy>0.05</iyy><izz>0.05</izz></inertia></inertial>
      <visual name="v"><geometry><cylinder><radius>0.3</radius><length>0.15</length></cylinder></geometry><material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
      <collision name="c"><geometry><cylinder><radius>0.3</radius><length>0.15</length></cylinder></geometry><surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision></link>'''
    joint = lambda n, c: f'<joint name="{n}" type="revolute"><parent>chassis</parent><child>{c}</child><axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>'
    return f'''<model name="{name}"><pose>{x:.2f} {y:.2f} {z0:.2f} 0 0 {yaw:.4f}</pose>
    <link name="chassis"><inertial><mass>400</mass><inertia><ixx>60</ixx><iyy>120</iyy><izz>150</izz></inertia></inertial>
      <visual name="body"><geometry><box><size>2.2 1.2 0.5</size></box></geometry><material><ambient>0.3 0.35 0.3 1</ambient><diffuse>0.3 0.35 0.3 1</diffuse></material></visual>
      <collision name="body"><geometry><box><size>2.2 1.2 0.5</size></box></geometry></collision>
      {cyl_visual("beacon", 110, 12, COLORS["ugv"], 40)}
    </link>
    {wheel("left_wheel", 0.7)}{wheel("right_wheel", -0.7)}
    <link name="caster"><pose relative_to="chassis">-0.9 0 -0.30 0 0 0</pose><inertial><mass>1</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
      <collision name="c"><geometry><sphere><radius>0.15</radius></sphere></geometry><surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision></link>
    {joint("left_wheel_joint", "left_wheel")}{joint("right_wheel_joint", "right_wheel")}
    <joint name="caster_joint" type="fixed"><parent>chassis</parent><child>caster</child></joint>
    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
      <wheel_separation>1.4</wheel_separation><wheel_radius>0.3</wheel_radius>
      <max_linear_velocity>8</max_linear_velocity><min_linear_velocity>-8</min_linear_velocity>
      <max_linear_acceleration>1.5</max_linear_acceleration><min_linear_acceleration>-3</min_linear_acceleration>
      <max_angular_velocity>1.2</max_angular_velocity><min_angular_velocity>-1.2</min_angular_velocity>
      <topic>/model/{name}/cmd_vel</topic>
    </plugin></model>'''


def uav_sdf(name, x, y, z):
    return f'''<model name="{name}"><pose>{x:.2f} {y:.2f} {z} 0 0 0</pose>
    <link name="body"><gravity>false</gravity><inertial><mass>2</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial>
      {box_visual("body", "0.8 0.8 0.15", "0.1 0.1 0.1 1")}
      <visual name="beacon"><geometry><sphere><radius>85</radius></sphere></geometry><material><ambient>{COLORS["uav"]}</ambient><diffuse>{COLORS["uav"]}</diffuse></material></visual>
    </link>
    <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl"><topic>/model/{name}/cmd_vel</topic></plugin></model>'''


def world_sdf(plan, res=(1280, 720), terrain=None):
    """terrain: dict(dir=<abs dir with heightmap.png, diffuse.png, normal.png>, size=, relief=, dem=<2-D array of z>)"""
    gz = terrain_height_fn(terrain) if terrain else (lambda x, y: 0.0)
    xy, edges = plan["roads"]["xy"], plan["roads"]["edges"]
    xs = [p[0] for p in xy] + [p[0] for p in plan["tasks"]["xy"]]; ys = [p[1] for p in xy] + [p[1] for p in plan["tasks"]["xy"]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs) * 1.0, (max(ys) - min(ys)) * res[0] / res[1]) + 1600
    hfov = math.radians(60); height = span / 2 / math.tan(hfov / 2)
    road_links = []
    for k, (a, b) in enumerate(edges):
        (x0, y0), (x1, y1) = xy[a], xy[b]; L = math.hypot(x1 - x0, y1 - y0); yaw = math.atan2(y1 - y0, x1 - x0)
        nseg = max(1, int(L // 150)) if terrain else 1
        for q in range(nseg):
            f0, f1 = q / nseg, (q + 1) / nseg; ax, ay, bx, by = x0 + (x1 - x0) * f0, y0 + (y1 - y0) * f0, x0 + (x1 - x0) * f1, y0 + (y1 - y0) * f1
            mx, my = (ax + bx) / 2, (ay + by) / 2; zz = gz(mx, my) + 4.0; pitch = -math.atan2(gz(bx, by) - gz(ax, ay), math.hypot(bx - ax, by - ay)) if terrain else 0.0
            road_links.append(f'<visual name="r{k}_{q}"><pose>{mx:.1f} {my:.1f} {zz + 0.05:.2f} 0 {pitch:.4f} {yaw:.4f}</pose><geometry><box><size>{L / nseg + 4:.1f} {ROAD_W} 0.1</size></box></geometry><material><ambient>{COLORS["road"]}</ambient><diffuse>{COLORS["road"]}</diffuse></material></visual>')
    for k, (x, y) in enumerate(xy):
        road_links.append(f'<visual name="n{k}"><pose>{x:.1f} {y:.1f} {gz(x, y) + 4.2:.2f} 0 0 0</pose><geometry><cylinder><radius>{ROAD_W/2}</radius><length>0.1</length></cylinder></geometry><material><ambient>{COLORS["road"]}</ambient><diffuse>{COLORS["road"]}</diffuse></material></visual>')
    for k, (x, y) in enumerate(plan["tasks"]["xy"]):
        col = COLORS["task"]
        road_links.append(f'<visual name="t{k}"><pose>{x:.1f} {y:.1f} {gz(x, y) + 4.6:.2f} 0 0 0</pose><geometry><cylinder><radius>75</radius><length>0.2</length></cylinder></geometry><material><ambient>{col}</ambient><diffuse>{col}</diffuse></material></visual>')
    dx, dy = plan["tasks"]["depot"]
    road_links.append(f'<visual name="depot"><pose>{dx:.1f} {dy:.1f} {gz(dx, dy) + 5.0:.2f} 0 0 0</pose><geometry><cylinder><radius>110</radius><length>0.2</length></cylinder></geometry><material><ambient>{COLORS["depot"]}</ambient><diffuse>{COLORS["depot"]}</diffuse></material></visual>')
    vehicles = []
    for g in plan["ugvs"]:
        d = g["drives"]
        yaw = 0.0
        if d:
            a, b = plan["roads"]["xy"][d[0]["a"]], plan["roads"]["xy"][d[0]["b"]]; yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        vehicles.append(ugv_sdf(f"ugv{g['id']}", g["start"][0], g["start"][1], yaw, gz(g["start"][0], g["start"][1]) + UGV_Z + 0.2))
    for u in plan["uavs"]:
        vehicles.append(uav_sdf(f"uav{u['id']}", u["start"][0], u["start"][1], (terrain["relief"] + 60) if terrain else 60))
    W, H = res
    if terrain:
        u = "file://" + terrain["dir"]; hm = lambda extra: f'<heightmap><uri>{u}/heightmap.png</uri><size>{terrain["size"]:.0f} {terrain["size"]:.0f} {terrain["relief"]:.2f}</size>{extra}</heightmap>'
        tex = f'<texture><size>{terrain["size"]:.0f}</size><diffuse>{u}/diffuse.png</diffuse><normal>{u}/normal.png</normal></texture>'
        h = terrain["size"] / 2
        ground_model = (f'<model name="ground"><static>true</static><pose>{h:.0f} {h:.0f} 0 0 0 0</pose><link name="g"><collision name="c"><geometry>{hm("")}</geometry></collision></link></model>'
                        f'<model name="ground_vis"><static>true</static><link name="g"><visual name="v"><geometry>{hm(f"<pos>{h:.0f} {h:.0f} 0</pos>" + tex)}</geometry></visual></link></model>')
    else:
        ground_model = f'<model name="ground"><static>true</static><link name="g"><collision name="c"><pose>0 0 -0.5 0 0 0</pose><geometry><box><size>40000 40000 1</size></box></geometry><surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision><visual name="v"><geometry><plane><normal>0 0 1</normal><size>40000 40000</size></plane></geometry><material><ambient>{COLORS["ground"]}</ambient><diffuse>{COLORS["ground"]}</diffuse></material></visual></link></model>'
    return f'''<?xml version="1.0"?>
<sdf version="1.9"><world name="edr">
  <physics name="p" type="ode"><max_step_size>0.01</max_step_size><real_time_factor>1000</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <scene><ambient>0.9 0.9 0.9 1</ambient><background>0.55 0.7 0.85 1</background></scene>
  <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 100 0 0 0</pose><diffuse>1 1 1 1</diffuse><specular>0.1 0.1 0.1 1</specular><direction>-0.2 0.2 -1</direction></light>
  {ground_model}
  <model name="map"><static>true</static><link name="l">{''.join(road_links)}</link></model>
  {''.join(vehicles)}
  <model name="overview"><static>true</static><pose>{cx:.1f} {cy:.1f} {height + (terrain["relief"] if terrain else 0):.1f} 0 1.5707963 1.5707963</pose><link name="l">
    <sensor name="cam" type="camera"><camera><horizontal_fov>{hfov:.5f}</horizontal_fov><image><width>{W}</width><height>{H}</height></image><clip><near>10</near><far>{height*3:.0f}</far></clip></camera>
      <always_on>1</always_on><update_rate>0.05</update_rate><visualize>false</visualize><topic>overview</topic></sensor></link></model>
</world></sdf>'''


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    open(sys.argv[2], "w").write(world_sdf(plan))
    print("wrote", sys.argv[2])
