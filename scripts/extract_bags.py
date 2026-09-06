#!/usr/bin/env python3
"""Extract odom trajectories + cmd_vel time series from all 130 trials -> npz cache."""
import glob, os, numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

OUT = os.path.expanduser("~/Desktop/PVSTAM_FIGURES_FINAL/data/bags.npz")
store = {}
bags = sorted(glob.glob(os.path.expanduser(
    "~/Desktop/PVSTAM_hardware_trials/keep/*_2026*")))
print(f"{len(bags)} bags")
for n, bag in enumerate(bags):
    tid = os.path.basename(bag)
    sr = rosbag2_py.SequentialReader()
    try:
        sr.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    except Exception as e:
        print(f"  SKIP {tid}: {e}"); continue
    ot, ox, oy, ct, cl, ca = [], [], [], [], [], []
    while sr.has_next():
        topic, data, t = sr.read_next()
        if topic == "/odom":
            m = deserialize_message(data, Odometry)
            ot.append(t/1e9); ox.append(m.pose.pose.position.x); oy.append(m.pose.pose.position.y)
        elif topic == "/cmd_vel":
            m = deserialize_message(data, Twist)
            ct.append(t/1e9); cl.append(m.linear.x); ca.append(m.angular.z)
    if not ot: continue
    t0 = min(ot[0], ct[0] if ct else ot[0])
    store[f"{tid}|ot"] = np.array(ot) - t0
    store[f"{tid}|ox"] = np.array(ox); store[f"{tid}|oy"] = np.array(oy)
    store[f"{tid}|ct"] = (np.array(ct) - t0) if ct else np.array([])
    store[f"{tid}|cl"] = np.array(cl); store[f"{tid}|ca"] = np.array(ca)
    if (n+1) % 25 == 0: print(f"  {n+1}/{len(bags)}")
np.savez_compressed(OUT, **store)
print(f"saved {OUT}  ({len(store)//5} trials)")
