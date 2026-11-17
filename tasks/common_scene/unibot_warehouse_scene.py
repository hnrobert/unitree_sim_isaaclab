# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""The unibot demo's warehouse hall, rebuilt as an IsaacLab scene.

Ported from the sibling repo demo/g1d_scene.py + g1d_config.py
(2026-11-14): 48x30 m hall, 3+3 rack rows, the 12-table divider
grid (DIVIDER_X x +-1.7), 4 east dressing tables, floor tiles /
lane stripes / perimeter walls from the Simple_Warehouse kit.

Kit USDs come from the demo's assets (bind-mounted read-only at
UNIBOT_ASSETS, default /workspaces/unibot_demo_assets) - the MDL
materials live in the same tree, so relative refs resolve.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg  # isort: skip

UNIBOT_ASSETS = os.environ.get("UNIBOT_ASSETS", "/workspaces/unibot_demo_assets")
WAREHOUSE = os.path.join(UNIBOT_ASSETS, "warehouse")

# ---- layout constants (verbatim from demo/g1d_config.py) --------------
HALL_SCALE = 5.0
HALL_X = 4.8 * HALL_SCALE  # +-24
HALL_Y = 3.0 * HALL_SCALE  # +-15
RACK_ROWS = (5.0, 8.0, 12.0)
DIVIDER_X = (-8.5, -11.0, -13.75, -15.25, -18.0, -19.5)
ROW_DY = 1.7
TABLE_POS = tuple((float(x), float(dy)) for x in DIVIDER_X for dy in (-ROW_DY, ROW_DY))
TABLE_TOP = 0.78
TABLE_SIZE = (1.8, 0.9, 0.06)  # long axis X - the corridor-facing
# face is the full 1.8 m edge and the DEPTH toward the far side is
# only 0.9 m, so far-edge items (racks/plates rows) stay within the
# arm's reach from the corridor side (was 1.8 m deep - unreachable)
TABLE_COLOR = (0.56, 0.42, 0.27)
LEG_COLOR = (0.32, 0.32, 0.34)
DRESSING_TABLES = ((6.0, 3.0), (10.0, 3.2), (6.0, -3.0), (12.0, -3.4))


def _prop(name):
    return os.path.join(WAREHOUSE, "Props", name + ".usd")


def _quat_z(deg):
    """Yaw quat (x, y, z, w) for InitialStateCfg.rot."""
    r = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(r), math.cos(r))


def _centers(half, step):
    n = 2 * half / step
    assert abs(n - round(n)) < 1e-6, f"{half} m span not a multiple of the {step} m panel"
    return [-half + step * (i + 0.5) for i in range(int(round(n)))]


class _PropFactory:
    """Numbered prop cfgs - every prim path must be UNIQUE (a shared
    path makes later spawns overwrite earlier ones)."""

    def __init__(self):
        self.n = 0

    def cfg(self, name, pos, rot_z=0.0):
        rot = _quat_z(rot_z) if rot_z else (0.0, 0.0, 0.0, 1.0)
        c = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/wh_prop_{self.n:04d}",
            spawn=UsdFileCfg(usd_path=_prop(name)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=list(pos), rot=list(rot)),
        )
        self.n += 1
        return c


def build_tables(table_color=TABLE_COLOR):
    """12 work tables + the dressing tables: top slab + four legs each."""
    cfgs = []
    tx_len, ty_len, t_thick = TABLE_SIZE
    leg_h = TABLE_TOP - t_thick
    positions = [(ti, cx, cy) for ti, (cx, cy) in enumerate(TABLE_POS)]
    positions += [(len(TABLE_POS) + i, cx, cy) for i, (cx, cy) in enumerate(DRESSING_TABLES)]
    for ti, cx, cy in positions:
        cfgs.append(
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Table{ti:02d}",
                spawn=sim_utils.CuboidCfg(
                    size=(tx_len, ty_len, t_thick),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=table_color, roughness=0.55, metallic=0.05),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, TABLE_TOP - t_thick / 2)),
            )
        )
        for li, (lx, ly) in enumerate((
            (-tx_len / 2 + 0.09, -ty_len / 2 + 0.09),
            (tx_len / 2 - 0.09, -ty_len / 2 + 0.09),
            (-tx_len / 2 + 0.09, ty_len / 2 - 0.09),
            (tx_len / 2 - 0.09, ty_len / 2 - 0.09),
        )):
            cfgs.append(
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Table{ti:02d}_L{li}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.07, 0.07, leg_h),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=LEG_COLOR),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(cx + lx, cy + ly, leg_h / 2)),
                )
            )
    return cfgs


def build_warehouse_props():
    """All static Simple_Warehouse dressing (mirrors the demo)."""
    p = _PropFactory()
    props = []
    # concrete-look 6 m floor tiles over the whole hall (visual; the
    # ground plane below provides collision)
    for x in _centers(HALL_X, 6.0):
        for y in _centers(HALL_Y, 6.0):
            props.append(p.cfg("SM_floor02", (x, y, 0.002)))
    # center-corridor lane stripes
    for x in _centers(HALL_X - 4.0, 4.0):
        for y in (-0.62, 0.62):
            props.append(p.cfg("SM_FloorDecal_StripeFull_4m", (x, y, 0.004)))
    # perimeter: 3 m panel runs, corners cross-overlapping
    for y in (HALL_Y, -HALL_Y):
        for x in _centers(HALL_X, 3.0):
            props.append(p.cfg("SM_WallA_3M", (x, y, 0.0), rot_z=90.0))
    for x in (HALL_X, -HALL_X):
        for y in _centers(HALL_Y, 3.0):
            props.append(p.cfg("SM_WallA_3M", (x, y, 0.0)))
    # THREE parallel rack rows per side (y +-5/+-8/+-12), 5 m bays
    _bay_goods = ("SM_CardBoxA_01", "SM_CardBoxB_01", "SM_CardBoxC_01",
                  "SM_CardBoxA_02", "SM_CardBoxD_01")
    for y_r in [s * r for r in RACK_ROWS for s in (1.0, -1.0)]:
        for bi, x_c in enumerate(_centers(HALL_X - 4.0, 5.0)):
            for z_lvl in (0.35, 1.45):
                props.append(p.cfg("SM_RackShelf_01", (x_c, y_r, z_lvl)))
            for x_e in (x_c - 2.0, x_c + 2.0):
                props.append(p.cfg("SM_RackFrame_03", (x_e, y_r, 0.0)))
            props.append(p.cfg("SM_RackPile_03", (x_c - 1.2, y_r, 0.40)))
            props.append(p.cfg("SM_RackPile_04", (x_c + 0.9, y_r, 0.42)))
            g0 = _bay_goods[bi % len(_bay_goods)]
            g1 = _bay_goods[(bi + 2) % len(_bay_goods)]
            for gx, gn in ((x_c - 0.35, g0), (x_c + 0.35, g1)):
                props.append(p.cfg(gn, (gx, y_r, 0.38)))
            for gx, gn in ((x_c - 1.0, g1), (x_c, g0), (x_c + 1.0, g1)):
                props.append(p.cfg(gn, (gx, y_r, 1.48)))
    # traffic furniture beside the table rows + aisle sign
    props.append(p.cfg("S_TrafficCone", (-2.5, 2.8, 0.0)))
    props.append(p.cfg("S_TrafficCone", (-6.8, -2.8, 0.0)))
    # S_AisleSign origin is CENTERED (local z -0.39..0.39) - lift it
    props.append(p.cfg("S_AisleSign", (-4.5, 3.2, 0.39), rot_z=180.0))
    # palletized stacks in both far corners
    for px, py in ((20.0, 4.0), (-21.5, 5.5), (-21.6, -6.2), (20.4, -4.5)):
        props.append(p.cfg("SM_PaletteA_02", (px, py, 0.0)))
        props.append(p.cfg("SM_CardBoxD_01", (px, py, 0.145)))
    return props


@configclass
class UnibotWarehouseSceneCfg(InteractiveSceneCfg):
    """Our demo warehouse hall; robot/cameras/object are attached by
    the task env cfg (see pick_place_cylinder_g1d_dex1_unibot)."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # AGV "mast" camera: rides above the base, pitched ~55 deg down
    # toward the manipulation workspace in front - the record-overhead
    # view. Parents under the AGV link so it follows the robot.
    from isaaclab.sensors import CameraCfg
    overhead_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/AGV_link/overhead_cam",
        update_period=0.05, height=960, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 1.70), rot=(0.0, 0.4037, 0.0, 0.9149),
            convention="opengl"),
    )

    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-8.6, 3.6, 1.6),
        rot_offset=(0.00617, 0.70708, -0.70708, -0.00617),
        focal_length=16.5)


# ---- challenge-task tabletop dressing (unibot_task_layout.yaml) --------
# table_i -> world (x, y); work tables 00-11, dressing tables 12-15
TABLE_WORLD = {f"table_{_i:02d}": _pos for _i, _pos in enumerate(TABLE_POS)}
TABLE_WORLD.update({f"table_{len(TABLE_POS) + _i:02d}": _pos
                    for _i, _pos in enumerate(DRESSING_TABLES)})

from tasks.common_scene.unibot_challenge_items import build_table_items  # noqa: E402

_TASK_ITEM_CFGS, TABLE_COLOR_CFG, PICK_OBJECT = build_table_items(TABLE_WORLD)


# tables and warehouse dressing attach as numbered fields
# (configclass turns every class attribute into an entity)
for _i, _cfg in enumerate(build_tables(tuple(TABLE_COLOR_CFG))):
    setattr(UnibotWarehouseSceneCfg, f"table_{_i:03d}", _cfg)

for _i, _cfg in enumerate(build_warehouse_props()):
    setattr(UnibotWarehouseSceneCfg, f"wh_prop_{_i:04d}", _cfg)

for _i, _cfg in enumerate(_TASK_ITEM_CFGS):
    setattr(UnibotWarehouseSceneCfg, f"task_item_{_i:03d}", _cfg)
