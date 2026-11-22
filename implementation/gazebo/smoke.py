import sys, time, threading
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
n = Node(); poses = {}; sim = [0.0]; imgs = []
def on_pose(m):
    for p in m.pose: poses[p.name] = (p.position.x, p.position.y, p.position.z)
    sim[0] = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
def on_img(m): imgs.append(m)
print("sub pose", n.subscribe(Pose_V, "/world/edr/dynamic_pose/info", on_pose)); print("sub img", n.subscribe(Image, "/overview", on_img))
pub = n.advertise("/model/ugv0/cmd_vel", Twist); pubu = n.advertise("/model/uav0/cmd_vel", Twist)
time.sleep(2)
def step(k):
    w = WorldControl(); w.multi_step = k; w.pause = True
    return n.request("/world/edr/control", w, WorldControl, Boolean, 3000)
print("names", sorted(poses)[:12])
t = Twist(); t.linear.x = 4.0; u = Twist(); u.linear.x = 10.0
t0 = time.time()
for i in range(200):
    pub.publish(t); pubu.publish(u)
    ok = step(20)
    for _ in range(200):
        if sim[0] >= (i + 1) * 0.2 - 1e-6: break
        time.sleep(0.002)
print("steps ok", ok, "sim", sim[0], "wall", round(time.time() - t0, 2), "rate", round(sim[0] / (time.time() - t0), 1), "x")
print({k: tuple(round(v, 1) for v in poses[k]) for k in ("ugv0", "uav0") if k in poses}); print("images", len(imgs))
if imgs:
    m = imgs[-1]; print(m.width, m.height, m.pixel_format_type, len(m.data)); open("/tmp/frame.rgb", "wb").write(m.data)
