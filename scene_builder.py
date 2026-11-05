"""Procedural room generation from a seed (CLAUDE.md 6.1).

build_scene(cfg, seed) returns a Scene with
  * `xml`            — the full MJCF (room + obstacles + charging station + robot),
  * `spec`           — MissionSpec: what the ROBOT is allowed to know a priori,
  * `sim_only`       — simulator/evaluation-only facts (true start pose, obstacle list, true map).
The generator alone may use the true map (to guarantee a collision-free path).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.grid import GridSpec, inflate, rasterize_shapes
from core.interfaces import MissionSpec
from core.planner import astar
from core.se2 import make_T

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "sim" / "models" / "turtlebot3_burger.xml"
# ArUco DICT_4X4_50 id 0 with a 1-module white quiet zone, made with cv2.aruco.generateImageMarker and
# mirrored left-right: MuJoCo maps a 2-D texture onto the -x face of a box mirrored as seen from outside
# (verified empirically at CP1 in the main repository).
MARKER_TEXTURE = ROOT / "sim" / "assets" / "aruco_dict_4x4_50_id0_q1_mirrored.png"

# Marker frame M on the east wall (normal -x_W): x_M = -y_W, y_M = +z_W, z_M = -x_W (CLAUDE.md 4.1)
R_W_M_EAST = np.array([[0.0, 0.0, -1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])


@dataclass
class Scene:
    seed: int
    xml: str
    spec: MissionSpec
    sim_only: dict = field(default_factory=dict)


def _dist_point_rect(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    d = np.maximum(np.maximum(lo - p, 0.0), p - hi)
    return float(np.linalg.norm(d))


def _dist_point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _sample_obstacles(rng: np.random.Generator, sc: dict, keepouts: dict) -> list[dict] | None:
    o = sc["obstacles"]
    hx, hy = sc["room"]["size_x_m"] / 2, sc["room"]["size_y_m"] / 2
    obs: list[dict] = []
    tries = 0
    while len(obs) < o["n"] and tries < o["max_tries"]:
        tries += 1
        kind = o["types"][int(rng.integers(len(o["types"])))]
        if kind == "box":
            half = rng.uniform(*o["box_half_extent_m"], size=2)
            yaw = float(rng.uniform(-np.pi, np.pi))
            rb = float(np.hypot(*half))
            shape = {"type": "box", "half": half.tolist(), "yaw": yaw}
        else:
            r = float(rng.uniform(*o["cylinder_radius_m"]))
            rb = r
            shape = {"type": "cylinder", "radius": r}
        lim_x, lim_y = hx - rb - o["wall_margin_m"], hy - rb - o["wall_margin_m"]
        p = np.array([rng.uniform(-lim_x, lim_x), rng.uniform(-lim_y, lim_y)])
        ok = _dist_point_rect(p, *keepouts["start_rect"]) > rb + sc["start"]["clear_radius_m"]
        ok &= all(np.linalg.norm(p - w) > rb + sc["waypoint_clear_radius_m"] for w in keepouts["waypoints"])
        ok &= _dist_point_rect(p, *keepouts["dock_rect"]) > rb
        ok &= _dist_point_segment(p, *keepouts["los"]) > rb + sc["line_of_sight"]["half_width_m"]
        ok &= all(np.linalg.norm(p - np.array(q["pos"])) > rb + q["rbound"] + o["min_gap_m"] for q in obs)
        if ok:
            shape.update({"pos": p.tolist(), "rbound": rb})
            obs.append(shape)
    return obs if len(obs) == o["n"] else None


def _true_map_path_ok(sc: dict, obstacles: list[dict], station: dict, route: list[np.ndarray]) -> bool:
    pc = sc["path_check"]
    spec = GridSpec.for_room(sc["room"]["size_x_m"], sc["room"]["size_y_m"], pc["resolution_m"])
    occ = rasterize_shapes(spec, obstacles + [station])
    X, Y = spec.centres()
    hx, hy = sc["room"]["size_x_m"] / 2, sc["room"]["size_y_m"] / 2
    occ |= (np.abs(X) > hx) | (np.abs(Y) > hy)  # walls
    blocked = inflate(occ, pc["inflation_m"], spec.res_m)
    cells = [tuple(spec.world_to_cell(p)[0]) for p in route]
    for a, b in zip(cells[:-1], cells[1:]):
        if astar(blocked, a, b) is None:
            return False
    return True


def build_scene(cfg: dict, seed: int) -> Scene:
    """Generate the scene for `seed`. cfg is the composed config (core.config.load_config)."""
    sc = cfg["scene"]
    rng = np.random.default_rng(seed)
    hx, hy = sc["room"]["size_x_m"] / 2, sc["room"]["size_y_m"] / 2

    # --- charging station and marker (known infrastructure) ---
    mk = sc["marker"]
    st = sc["station"]
    y_m = float(rng.uniform(*mk["y_range_m"]))
    face_x = hx - st["depth_m"]
    panel_t = 0.002
    p_W_M = np.array([face_x - 2 * panel_t, y_m, mk["centre_height_m"]])
    T_W_M = make_T(R_W_M_EAST, p_W_M)
    n_W = R_W_M_EAST[:, 2][:2]  # marker normal z_M projected on the floor (= -x_W)
    theta_D = float(np.arctan2(-n_W[1], -n_W[0]))  # B faces the marker: x_B = -z_M
    dock = np.array([*(p_W_M[:2] + sc["dock"]["standoff_m"] * n_W), theta_D])
    predock = np.array([*(p_W_M[:2] + sc["dock"]["predock_distance_m"] * n_W), theta_D])

    # --- start pose: nominal known to robot, true pose sampled (sim only) ---
    nominal = np.array(sc["start"]["nominal"], float)
    half = np.array(sc["start"]["half_range"], float)
    true_start = nominal + rng.uniform(-half, half)
    waypoints = np.array(sc["waypoints"], float).reshape(-1, 2)

    station_shape = {"type": "box", "pos": [hx - st["depth_m"] / 2, y_m],
                     "half": [st["depth_m"] / 2, st["width_m"] / 2], "yaw": 0.0}
    cw = sc["dock"]["corridor_half_width_m"]
    keepouts = {
        "start_rect": (nominal[:2] - half[:2], nominal[:2] + half[:2]),
        "waypoints": list(waypoints),
        "dock_rect": (np.array([predock[0] - 0.3, y_m - cw]), np.array([hx, y_m + cw])),
        "los": (nominal[:2], p_W_M[:2]),
    }
    route = [nominal[:2], *waypoints, predock[:2], dock[:2]]
    obstacles = None
    for _ in range(50):
        obstacles = _sample_obstacles(rng, sc, keepouts)
        if obstacles is not None and _true_map_path_ok(sc, obstacles, station_shape, route):
            break
        obstacles = None
    if obstacles is None:
        raise RuntimeError(f"seed {seed}: could not generate a valid obstacle layout")

    # --- MJCF assembly ---
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="aruco_tex", type="2d", file=MARKER_TEXTURE.as_posix())
    ET.SubElement(asset, "material", name="aruco_mat", texture="aruco_tex", texrepeat="1 1",
                  texuniform="false", specular="0", shininess="0", reflectance="0")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker",
                  rgb1="0.62 0.62 0.60", rgb2="0.55 0.55 0.53", width="512", height="512")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="10 10",
                  texuniform="true", reflectance="0")
    ET.SubElement(asset, "material", name="wall_mat", rgba="0.85 0.83 0.78 1")
    ET.SubElement(asset, "material", name="obst_mat", rgba="0.55 0.35 0.2 1")
    ET.SubElement(asset, "material", name="station_mat", rgba="0.2 0.3 0.6 1")

    wb = root.find("worldbody")
    robot_body = wb.find("body")
    wb.remove(robot_body)
    ET.SubElement(wb, "light", name="sun", pos="0 0 4", dir="0 0 -1", diffuse="0.6 0.6 0.6",
                  castshadow="false")
    ET.SubElement(wb, "light", name="fill", pos="-2 0 2", dir="1 0 -0.5", diffuse="0.3 0.3 0.3",
                  castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size=f"{hx + 1} {hy + 1} 0.1",
                  material="floor_mat", group="0")
    wt, wh = sc["room"]["wall_thickness_m"], sc["room"]["wall_height_m"]
    walls = {"wall_e": (hx + wt / 2, 0, wt / 2, hy + wt), "wall_w": (-hx - wt / 2, 0, wt / 2, hy + wt),
             "wall_n": (0, hy + wt / 2, hx + wt, wt / 2), "wall_s": (0, -hy - wt / 2, hx + wt, wt / 2)}
    for n, (x, y, sx, sy) in walls.items():
        ET.SubElement(wb, "geom", name=n, type="box", pos=f"{x} {y} {wh / 2}", size=f"{sx} {sy} {wh / 2}",
                      material="wall_mat", group="0")
    for i, o in enumerate(obstacles):
        h = sc["obstacles"]["height_m"]
        if o["type"] == "box":
            ET.SubElement(wb, "geom", name=f"obst_{i}", type="box",
                          pos=f"{o['pos'][0]} {o['pos'][1]} {h / 2}",
                          size=f"{o['half'][0]} {o['half'][1]} {h / 2}",
                          euler=f"0 0 {o['yaw']}", material="obst_mat", group="0")
        else:
            ET.SubElement(wb, "geom", name=f"obst_{i}", type="cylinder",
                          pos=f"{o['pos'][0]} {o['pos'][1]} {h / 2}", size=f"{o['radius']} {h / 2}",
                          material="obst_mat", group="0")
    station = ET.SubElement(wb, "body", name="charging_station", pos=f"{station_shape['pos'][0]} {y_m} 0")
    ET.SubElement(station, "geom", name="station_box", type="box", pos=f"0 0 {st['height_m'] / 2}",
                  size=f"{st['depth_m'] / 2} {st['width_m'] / 2} {st['height_m'] / 2}",
                  material="station_mat", group="0")
    panel_half = mk["side_m"] * (1 + 2 * mk["quiet_zone_modules"] / 6.0) / 2  # 6 modules for 4x4
    ET.SubElement(station, "geom", name="marker_panel", type="box",
                  pos=f"{-st['depth_m'] / 2 - panel_t} 0 {mk['centre_height_m']}",
                  size=f"{panel_t} {panel_half} {panel_half}", material="aruco_mat",
                  contype="0", conaffinity="0", group="0")

    # observer camera for the video (not a robot sensor)
    ET.SubElement(wb, "camera", name="overview", pos=f"0 {-hy - 2.2} 4.2", xyaxes="1 0 0 0 0.75 0.66",
                  fovy="60")

    # robot at the TRUE start pose (simulator initial condition only)
    robot_body.set("pos", f"{true_start[0]} {true_start[1]} 0.0")
    robot_body.set("euler", f"0 0 {true_start[2]}")
    wb.append(robot_body)
    root.set("model", f"tb3_room_seed{seed}")

    spec = MissionSpec(T_W_M=T_W_M, marker_side_m=mk["side_m"], marker_id=mk["id"],
                       aruco_dictionary=mk["dictionary"], dock_pose=dock, predock_pose=predock,
                       waypoints=waypoints, nominal_start=nominal,
                       room_size_m=(sc["room"]["size_x_m"], sc["room"]["size_y_m"]))
    sim_only = {"true_start_pose": true_start, "obstacles": obstacles, "station_shape": station_shape}
    return Scene(seed=seed, xml=ET.tostring(root, encoding="unicode"), spec=spec, sim_only=sim_only)
