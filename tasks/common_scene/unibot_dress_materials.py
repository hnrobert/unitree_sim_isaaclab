# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Real materials on the generated challenge items.

Port of the demo's proven pipeline (g1d_usd.bind_object_materials +
_write_box_uv): IsaacLab shape spawners create meshes with NO uv
primvar and bind a FLAT diffuse color - a texture would collapse to
one texel. Here we author triplanar faceVarying `st` primvars and
rebind UsdUVTexture-driven PreviewSurface networks (stronger than
the spawner's material) on every item gprim.

Tinting: UsdUVTexture has a float4 `scale` input - the YAML item
color multiplies the grayscale-ish procedural texture, so one set
of PNGs covers every color variant.

Called from sim_main right after env.reset() (prims exist by then);
no-ops when the layout manifest is empty (non-unibot tasks).
"""

import os

TEXTURE_DIR = os.path.join(
    os.environ.get("UNIBOT_ASSETS", "/workspaces/unibot_demo_assets"), "materials")

# item type -> (png, roughness, metallic); the tint color comes from
# the layout entry (YAML color or the type default). USD-prop types
# (e.g. "cup" - SimReady kraft paper cup) ship their own materials
# and must stay OUT of this map.
ITEM_TEXTURES = {
    "mug_g": ("metal.png", 0.35, 0.75),
    "apple": ("apple_red.png", 0.45, 0.05),
    "orange": ("orange.png", 0.45, 0.05),
    "lime": ("lime_green.png", 0.45, 0.05),
    "banana_g": ("plastic_yellow.png", 0.5, 0.05),
    "book": ("paper_red.png", 0.8, 0.0),
    "towel": ("fabric.png", 0.95, 0.0),
    "battery": ("metal.png", 0.3, 0.8),
    "block": ("plastic_blue.png", 0.45, 0.1),
    "plate": ("paper.png", 0.35, 0.05),
    "tube": ("plastic_blue.png", 0.25, 0.1),
    "pencil": ("wood.png", 0.6, 0.0),
    "tray": ("plastic_blue.png", 0.4, 0.1),
    "box": ("plastic_blue.png", 0.4, 0.1),
    "stapler_g": ("metal.png", 0.35, 0.7),
    "box_g": ("paper.png", 0.85, 0.0),
    # compound-holder parts
    "cup_rack": ("wood.png", 0.65, 0.0),
    "bookstand": ("wood.png", 0.65, 0.0),
    "tube_rack": ("plastic_blue.png", 0.4, 0.1),
}

_GPRIM_TYPES = ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane")


def _write_triplanar_uv(prim) -> None:
    """Author an `st` primvar via triplanar box projection (mesh
    points -> dominant-axis uv); authored UVs are kept."""
    import numpy as np
    from pxr import Sdf, UsdGeom

    mesh = UsdGeom.Mesh(prim)
    if UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").IsDefined():
        return
    pts = mesh.GetPointsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not fvi:
        return
    pts = np.array([[c for c in p] for p in pts])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    mid = (lo + hi) / 2.0
    uvs, idx = [], []
    for v in fvi:
        q = (pts[v] - lo) / span
        ax = int(np.argmax(np.abs(pts[v] - mid)))
        uvs.append((q[1], q[2]) if ax == 0 else (q[0], q[2]) if ax == 1 else (q[0], q[1]))
        idx.append(len(uvs) - 1)
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    pv.Set(uvs)
    pv.SetIndicesAttr(idx)


def _textured_material(stage, path, png, tint, roughness, metallic):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    tex = UsdShade.Shader.Define(stage, path + "/DiffuseTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    # float4 scale tints the texture with the layout color
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        (float(tint[0]), float(tint[1]), float(tint[2]), 1.0))
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    reader = UsdShade.Shader.Define(stage, path + "/UVReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def dress_items(env_ns: str = "/World/envs/env_0") -> int:
    """Bind textured materials to every manifest item prim."""
    import omni.usd
    from pxr import UsdShade

    from tasks.common_scene.unibot_challenge_items import DRESS_MANIFEST

    if not DRESS_MANIFEST:
        return 0
    stage = omni.usd.get_context().get_stage()
    cache = {}
    bound = 0
    for entry in DRESS_MANIFEST:
        spec = ITEM_TEXTURES.get(entry["type"])
        if spec is None:
            continue
        png, rough, metal = spec
        png_path = os.path.join(TEXTURE_DIR, png)
        if not os.path.isfile(png_path):
            continue
        tint = entry.get("color") or (1.0, 1.0, 1.0)
        key = (png, tuple(tint), rough, metal)
        if key not in cache:
            cache[key] = _textured_material(
                stage, f"{env_ns}/ItemMaterials/M{len(cache):03d}", png_path, tint, rough, metal)
        root = stage.GetPrimAtPath(f"{env_ns}/{entry['prim']}")
        if not root or not root.IsValid():
            continue
        for p in stage.Traverse():
            path = str(p.GetPath())
            if path != f"{env_ns}/{entry['prim']}" and not path.startswith(f"{env_ns}/{entry['prim']}/"):
                continue
            if p.GetTypeName() not in _GPRIM_TYPES:
                continue
            rel = p.GetRelationship("material:binding")
            if rel:
                rel.ClearTargets(True)  # drop the spawner's flat color
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                cache[key], UsdShade.Tokens.strongerThanDescendants)
            _write_triplanar_uv(p)
            bound += 1
    print(f"[unibot] textured materials bound to {bound} item prims", flush=True)
    return bound
