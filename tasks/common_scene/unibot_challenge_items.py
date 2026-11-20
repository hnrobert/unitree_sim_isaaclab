# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Challenge-task tabletop items, driven by unibot_task_layout.yaml.

The YAML lists, per table, the items to place (type / at / color / up /
yaw). This module maps each item TYPE to a spawn spec:

* generated primitives  -> RigidObjectCfg (dynamic) or AssetBaseCfg
  (static slabs/racks), sized/masses after the challenge photos
  (T04 fruits+plates, T13 tubes+rack, ...)
* USD props             -> prepared single-body USDs under UNIBOT_ASSETS
  (T20's simready kraft paper cup, flattened offline - raw multi-prim/
  instanced SimReady spawns fail the env prim regex)
"""

import math
import os

import isaaclab.sim as sim_utils
import yaml
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

UNIBOT_ASSETS = os.environ.get("UNIBOT_ASSETS", "/workspaces/unibot_demo_assets")
LAYOUT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unibot_task_layout.yaml")
TABLE_TOP = 0.78
GAP = 0.002  # spawn clearance above the tabletop


def _sr(*p):
    return os.path.join(UNIBOT_ASSETS, "simready", *p)


def _av(*p):
    return os.path.join(UNIBOT_ASSETS, "artvip", *p)


def _kit(name):
    return os.path.join(UNIBOT_ASSETS, "warehouse", "Props", name + ".usd")


def _qz(deg):
    r = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(r), math.cos(r))


def _qmul(a, b):
    """Hamilton product, xyzw."""
    return (
        a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
        a[1] * b[3] + a[3] * b[1] + a[2] * b[0] - a[0] * b[2],
        a[2] * b[3] + a[3] * b[2] + a[0] * b[1] - a[1] * b[0],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )


def _dyn(spawn, pos, rot=(0, 0, 0, 1)):
    return RigidObjectCfg(
        prim_path="{PLACEHOLDER}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(pos), rot=list(rot)),
    )


def _stat(spawn, pos, rot=(0, 0, 0, 1)):
    return AssetBaseCfg(
        prim_path="{PLACEHOLDER}",
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(pos), rot=list(rot)),
    )


def _vis(color, metallic=0.1, roughness=0.6):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic, roughness=roughness)


def _rigid_body():
    return sim_utils.RigidBodyPropertiesCfg()


def _col():
    return sim_utils.CollisionPropertiesCfg()


def _mass(m):
    return sim_utils.MassPropertiesCfg(mass=m)


def _usd_spawn(path, scale=None):
    cfg = dict(usd_path=path, rigid_props=_rigid_body(), collision_props=_col())
    if scale is not None:
        cfg["scale"] = tuple(scale)
    return sim_utils.UsdFileCfg(**cfg)


# ---- item registry -----------------------------------------------------
# type -> (kind, half_z above tabletop, factory(color) -> spawn cfg)
# kind: "dyn" RigidObjectCfg | "stat" AssetBaseCfg | "statics" multi-part
REGISTRY = {
    # T20 paper cups: the SimReady kraft prop, offline-flattened by the
    # repo's tools/prepare_paper_cup_usd.py to ONE root rigid body (0.05
    # kg) + SDF mesh collider (concave cup - stacks nest). Asset origin
    # sits at the BASE, hence half_z = 0; scale 0.504 (= 0.72 x 0.70,
    # user-requested 70% of the previous size) -> dia 0.052 m,
    # height 0.056 m (= FLIP_H below: a flipped cup's origin must be
    # raised by the height, the body then hangs down onto its rim)
    "cup": ("dyn", 0.0, lambda c: _usd_spawn(
        _sr("paper_cup", "paper_cup.usda"), scale=(0.504,) * 3)),
    "apple": ("dyn", 0.040, lambda c: sim_utils.SphereCfg(
        radius=0.040, rigid_props=_rigid_body(), mass_props=_mass(0.10),
        collision_props=_col(), visual_material=_vis(c or (0.85, 0.15, 0.12), 0.3, 0.5))),
    "orange": ("dyn", 0.0375, lambda c: sim_utils.SphereCfg(
        radius=0.0375, rigid_props=_rigid_body(), mass_props=_mass(0.09),
        collision_props=_col(), visual_material=_vis(c or (0.95, 0.55, 0.1), 0.3, 0.5))),
    "lime": ("dyn", 0.0325, lambda c: sim_utils.SphereCfg(
        radius=0.0325, rigid_props=_rigid_body(), mass_props=_mass(0.07),
        collision_props=_col(), visual_material=_vis(c or (0.2, 0.75, 0.25), 0.3, 0.5))),
    "book": ("dyn", 0.0175, lambda c: sim_utils.CuboidCfg(
        size=(0.19, 0.13, 0.035), rigid_props=_rigid_body(), mass_props=_mass(0.30),
        collision_props=_col(), visual_material=_vis(c or (0.85, 0.3, 0.25), 0.0, 0.8))),
    "towel": ("dyn", 0.0125, lambda c: sim_utils.CuboidCfg(
        size=(0.28, 0.28, 0.025), rigid_props=_rigid_body(), mass_props=_mass(0.15),
        collision_props=_col(), visual_material=_vis(c or (0.92, 0.9, 0.86), 0.0, 0.95))),
    "battery": ("dyn", 0.0225, lambda c: sim_utils.CylinderCfg(
        radius=0.007, height=0.045, rigid_props=_rigid_body(), mass_props=_mass(0.03),
        collision_props=_col(), visual_material=_vis(c or (0.3, 0.3, 0.32), 0.75, 0.35))),
    "block": ("dyn", 0.0225, lambda c: sim_utils.CuboidCfg(
        size=(0.045, 0.045, 0.045), rigid_props=_rigid_body(), mass_props=_mass(0.05),
        collision_props=_col(), visual_material=_vis(c or (0.9, 0.2, 0.15), 0.2, 0.5))),
    "plate": ("dyn", 0.006, lambda c: sim_utils.CylinderCfg(
        radius=0.09, height=0.012, rigid_props=_rigid_body(), mass_props=_mass(0.2),
        collision_props=_col(), visual_material=_vis(c or (0.92, 0.92, 0.9), 0.1, 0.4))),
    "tube": ("dyn", 0.050, lambda c: sim_utils.CylinderCfg(
        radius=0.008, height=0.10, rigid_props=_rigid_body(), mass_props=_mass(0.02),
        collision_props=_col(), visual_material=_vis(c or (0.9, 0.75, 0.2), 0.4, 0.3))),
    "pencil": ("dyn", 0.004, lambda c: sim_utils.CylinderCfg(
        radius=0.004, height=0.15, rigid_props=_rigid_body(), mass_props=_mass(0.01),
        collision_props=_col(), visual_material=_vis(c or (0.95, 0.75, 0.15), 0.3, 0.5))),
    # flat holders (static slabs with collision)
    "tray": ("stat", 0.004, lambda c: sim_utils.CuboidCfg(
        size=(0.32, 0.24, 0.008), collision_props=_col(),
        visual_material=_vis(c or (0.25, 0.7, 0.35), 0.1, 0.7))),
    "box": ("stat", 0.025, lambda c: sim_utils.CuboidCfg(
        size=(0.12, 0.09, 0.05), collision_props=_col(),
        visual_material=_vis(c or (0.3, 0.7, 0.4), 0.2, 0.6))),
    # generated stand-ins for the artvip articulated twins (artvip
    # models are MULTI-PRIM hierarchies - spawning them raw makes the
    # env prim regex match 3 prims where 1 is expected; the demo
    # needed a whole rigidify pass to use them, not worth it here)
    "stapler_g": ("dyn", 0.0225, lambda c: sim_utils.CuboidCfg(
        size=(0.16, 0.05, 0.045), rigid_props=_rigid_body(), mass_props=_mass(0.10),
        collision_props=_col(), visual_material=_vis(c or (0.25, 0.27, 0.3), 0.5, 0.4))),
    # USD-prop stand-ins as GENERATED shapes: the demo's simready/
    # artvip props only ever worked through its rigidify USD surgery;
    # raw UsdFileCfg spawns here produced "found 0 prims" / multi-
    # prim regex failures. Single rigid bodies, fully deterministic.
    "mug_g": ("dyn", 0.0475, lambda c: sim_utils.CylinderCfg(
        radius=0.04, height=0.095, rigid_props=_rigid_body(), mass_props=_mass(0.20),
        collision_props=_col(), visual_material=_vis(c or (0.25, 0.68, 0.68), 0.3, 0.4))),
    "banana_g": ("dyn", 0.0175, lambda c: sim_utils.CuboidCfg(
        size=(0.17, 0.035, 0.033), rigid_props=_rigid_body(), mass_props=_mass(0.05),
        collision_props=_col(), visual_material=_vis(c or (0.95, 0.85, 0.25), 0.1, 0.6))),
    "box_g": ("dyn", 0.06, lambda c: sim_utils.CuboidCfg(
        size=(0.25, 0.18, 0.12), rigid_props=_rigid_body(), mass_props=_mass(0.30),
        collision_props=_col(), visual_material=_vis(c or (0.72, 0.55, 0.32), 0.0, 0.85))),
}

# compound wooden holders: base + uprights, all static cuboids.
# factory(world_x, world_y, yaw_deg, color) -> list[(local_xyz, size, color)]
_WOOD = (0.62, 0.45, 0.28)


def _cup_rack_parts(_wx, _wy, _yaw, color=None):
    c = color or _WOOD
    return [
        (((0.0, 0.0, 0.008)), (0.12, 0.12, 0.016), c),      # round-ish base
        (((0.0, 0.0, 0.15)), (0.028, 0.028, 0.30), c),      # trunk
        (((0.0, 0.0, 0.30)), (0.26, 0.026, 0.026), c),      # hanging bar
    ]


def _bookstand_parts(_wx, _wy, _yaw, color=None):
    c = color or _WOOD
    return [
        (((0.0, 0.0, 0.006)), (0.32, 0.14, 0.012), c),      # base slab
        (((-0.15, 0.0, 0.10)), (0.012, 0.14, 0.20), c),     # left bookend
        (((0.15, 0.0, 0.10)), (0.012, 0.14, 0.20), c),      # right bookend
    ]


def _tube_rack_parts(_wx, _wy, _yaw, color=None):
    blue = color or (0.2, 0.4, 0.85)
    parts = [
        (((0.0, 0.0, 0.012)), (0.18, 0.08, 0.024), blue),   # base
        (((0.0, -0.034, 0.06)), (0.18, 0.012, 0.10), blue),  # back panel
    ]
    # three pre-inserted tubes standing on the base
    for dx, cap in ((-0.055, (0.95, 0.8, 0.2)), (0.0, (0.95, 0.55, 0.15)), (0.055, (0.2, 0.75, 0.3))):
        parts.append(((dx, 0.0, 0.024 + 0.05), (0.014, 0.014, 0.10), (0.85, 0.88, 0.92)))
        parts.append(((dx, 0.0, 0.024 + 0.103), (0.018, 0.018, 0.012), cap))
    return parts


COMPOUNDS = {
    "cup_rack": _cup_rack_parts,
    "bookstand": _bookstand_parts,
    "tube_rack": _tube_rack_parts,
}


# flipped (upside-down) base-origin USD props: the spawn z must be
# raised by the item height - keep in sync with the REGISTRY scales
FLIP_H = {"cup": 0.111 * 0.504}


def _yaw_rot(entry, lying=False):
    yaw = float(entry.get("yaw", 0.0))
    q = _qz(yaw)
    if lying:  # cylinder lying along local x: yaw(z) * pitch(y, 90deg)
        q = _qmul(q, (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)))
    return q


def _color(entry):
    """YAML colors arrive as LISTS - PreviewSurfaceCfg chokes on them
    ('expected GfVec3f, got vector<VtValue>'); coerce to float tuple."""
    c = entry.get("color")
    return tuple(float(v) for v in c) if c else None


def make_item_cfgs(entry, table_pos):
    """One YAML item entry -> (cfgs, manifests, pick_spec).

    cfgs carry a PLACEHOLDER prim_path (filled by build_table_items);
    manifests are the per-prim records for the material dressing
    pass (type + tint color); pick_spec is non-None for dynamic
    single-part items (usable as the env's task object).
    """
    typ = entry["type"]
    color = _color(entry)
    lx, ly = entry["at"]
    up = float(entry.get("up", 0.0))
    wx, wy = table_pos[0] + lx, table_pos[1] + ly

    if typ in COMPOUNDS:
        yaw = float(entry.get("yaw", 0.0))
        q = _qz(yaw)
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        cfgs, manifests = [], []
        for (ox, oy, oz), size, color in COMPOUNDS[typ](wx, wy, yaw, color):
            px = wx + ox * cy - oy * sy
            py = wy + ox * sy + oy * cy
            cfgs.append(_stat(
                sim_utils.CuboidCfg(size=size, collision_props=_col(), visual_material=_vis(color, 0.05, 0.7)),
                (px, py, TABLE_TOP + oz + GAP), q))
            manifests.append({"type": typ, "color": color})
        return cfgs, manifests, None

    kind, half_z, factory = REGISTRY[typ]
    z = TABLE_TOP + GAP + (half_z if half_z > 0 else 0.0) + up
    rot = _yaw_rot(entry, lying=(typ == "pencil"))
    if entry.get("flip"):
        # upside-down spawn: 180 deg about the local x after the yaw -
        # for base-origin USD props (cup) the body then hangs DOWN from
        # the origin, so raise z by the item's height (rim rests on the
        # table through the spawn gap, gravity settles it)
        rot = _qmul(rot, (1.0, 0.0, 0.0, 0.0))
        z += FLIP_H.get(typ, 0.0)
    cfg = _dyn(factory(color), (wx, wy, z), rot) if kind == "dyn" \
        else _stat(factory(color), (wx, wy, z), rot)
    pick = {"type": typ, "pos": (wx, wy, z), "rot": rot,
            "spawn": factory(color)} if kind == "dyn" else None
    return [cfg], [{"type": typ, "color": color}], pick


def load_layout():
    with open(LAYOUT_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


# per-prim records for the material dressing pass (unibot_dress_
# materials.dress_items): {"prim": leaf name, "type", "color"}
DRESS_MANIFEST = []


def build_table_items(table_world_pos):
    """All layout items across tables.

    table_world_pos: {"table_00": (x, y), ...} from the scene module.
    Returns (cfgs, table_color, pick) - cfgs have UNIQUE prim paths.
    """
    layout = load_layout()
    cfgs = []
    n = 0
    pick = None
    pick_sel = layout.get("pick") or {}
    for tname, tdef in layout.get("tables", {}).items():
        if tname not in table_world_pos:
            raise KeyError(f"{tname} in unibot_task_layout.yaml is not a scene table")
        for i, entry in enumerate(tdef.get("items", [])):
            item_cfgs, manifests, item_pick = make_item_cfgs(entry, table_world_pos[tname])
            is_pick = item_pick is not None and tname == pick_sel.get("table") and i == pick_sel.get("item")
            if is_pick:
                # the env's "object" entity IS this item - do NOT also
                # attach it as a scene field (a duplicate would spawn
                # two overlapping bodies)
                pick = item_pick
                continue
            for c, m in zip(item_cfgs, manifests):
                leaf = f"task_item_{n:03d}"
                c.prim_path = f"{{ENV_REGEX_NS}}/{leaf}"
                cfgs.append(c)
                DRESS_MANIFEST.append({"prim": leaf, "type": m["type"], "color": m["color"]})
                n += 1
    if pick is not None:
        # the env object prim (env_.*/Object) needs dressing too
        DRESS_MANIFEST.append({"prim": "Object", "type": pick["type"], "color": (0.45, 0.8, 0.8)})
    return cfgs, layout.get("table_color", (0.88, 0.87, 0.84)), pick
