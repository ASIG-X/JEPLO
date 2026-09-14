import math
import threading

import mujoco
import numpy as np

try:
    MJT_OBJ = mujoco.mjtObj
except AttributeError:
    MJT_OBJ = mujoco._enums.mjtObj


class TerrainSwitcher:
    """Runtime terrain switcher for named static terrain bodies."""

    TERRAIN_KEYS = {
        "1": "stairs",
        "2": "boxes",
        "3": "gap",
        "4": "boxes_overhang",
        "5": "pyramid_stairs",
        "6": "bumped_stairs",
        "7": "short_box_stairs",
        "8": "discrete_grid",
    }
    TERRAIN_LABELS = {
        "stairs": "stairs",
        "boxes": "boxes",
        "gap": "gap",
        "boxes_overhang": "overhanging boxes",
        "pyramid_stairs": "pyramid and inverted-pyramid stairs",
        "bumped_stairs": "stairs with riser bumps",
        "short_box_stairs": "five-step stairs with boxes",
        "discrete_grid": "discrete grid terrain",
    }
    PRE_STAIR_GEOMS = ("pre1", "pre2", "pre3")
    TERRAIN_FRONT_OFFSETS = {
        "pyramid_stairs": 1.5,
    }
    HIDDEN_POS = np.array([1000.0, 1000.0, 1000.0])
    HIDDEN_GEOM_POS = np.array([1000.0, 1000.0, 1000.0])

    def __init__(self, model, data, front_distance=3.0):
        self.model = model
        self.data = data
        self.front_distance = front_distance
        self.base_body_id = self._find_first_body(("base_link", "torso_link", "pelvis"))
        self._pending = None
        self._pending_lock = threading.Lock()
        self.enabled = False
        self.active_terrain = "stairs"
        self.pre_stair_boxes_enabled = False

        self.terrain = {}
        for name in self.TERRAIN_LABELS:
            body_id = self._body_id(name)
            if body_id == -1:
                continue
            geom_ids = self._body_geom_ids(body_id)
            self.terrain[name] = {
                "body_id": body_id,
                "z": float(self.model.body_pos[body_id][2]),
                "geom_ids": geom_ids,
            }

        self.geom_defaults = {}
        all_geom_ids = set()
        for info in self.terrain.values():
            all_geom_ids.update(info["geom_ids"])
        for geom_name in self.PRE_STAIR_GEOMS:
            geom_id = self._geom_id(geom_name)
            if geom_id != -1:
                all_geom_ids.add(geom_id)

        for geom_id in all_geom_ids:
            self.geom_defaults[geom_id] = {
                "rgba": self.model.geom_rgba[geom_id].copy(),
                "contype": int(self.model.geom_contype[geom_id]),
                "conaffinity": int(self.model.geom_conaffinity[geom_id]),
                "pos": self.model.geom_pos[geom_id].copy(),
            }

        if not self.terrain:
            print("Terrain switcher: no named terrain bodies found.")
            return
        if self.base_body_id == -1:
            print("Terrain switcher: no robot base body found.")
            return

        self.enabled = True
        if self.active_terrain not in self.terrain:
            self.active_terrain = next(iter(self.terrain))
        self._apply_active_terrain()
        self._print_help()

    def key_callback(self, key):
        if not self.enabled:
            return
        if not self._ctrl_is_pressed():
            return
        char = chr(key).upper() if 0 <= key < 256 else ""
        if char in self.TERRAIN_KEYS:
            self._queue(("switch", self.TERRAIN_KEYS[char]))
        elif char == "B":
            self._queue(("toggle_pre_stairs", None))
        elif char == "R":
            self._queue(("reposition", None))

    def apply_pending(self):
        if not self.enabled:
            return
        with self._pending_lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return

        command, value = pending
        if command == "switch":
            self._switch(value)
        elif command == "toggle_pre_stairs":
            self.pre_stair_boxes_enabled = not self.pre_stair_boxes_enabled
            self._apply_pre_stair_boxes()
            mujoco.mj_forward(self.model, self.data)
            state = "enabled" if self.pre_stair_boxes_enabled else "disabled"
            print(f"Terrain switcher: pre-stair boxes {state}.")
        elif command == "reposition":
            self._apply_active_terrain()
            print(f"Terrain switcher: repositioned {self._active_label()} in front of robot.")

    def _queue(self, pending):
        with self._pending_lock:
            self._pending = pending

    def _switch(self, terrain_name):
        if terrain_name not in self.terrain:
            print(f"Terrain switcher: '{terrain_name}' is not available in this model.")
            return
        self.active_terrain = terrain_name
        self._apply_active_terrain()
        print(f"Terrain switcher: active terrain is {self._active_label()}.")

    def _apply_active_terrain(self):
        for name, info in self.terrain.items():
            if name == self.active_terrain:
                self._show_body(info)
            else:
                self._hide_body(info)
        self._apply_pre_stair_boxes()
        mujoco.mj_forward(self.model, self.data)

    def _show_body(self, info):
        yaw = self._base_yaw()
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        base_pos = self.data.xpos[self.base_body_id].copy()
        front_distance = self.front_distance + self.TERRAIN_FRONT_OFFSETS.get(
            self.active_terrain, 0.0
        )
        pos = base_pos + forward * front_distance
        pos[2] = info["z"]

        body_id = info["body_id"]
        self.model.body_pos[body_id] = pos
        self.model.body_quat[body_id] = self._yaw_quat(yaw)
        self._restore_geoms(info["geom_ids"])

    def _hide_body(self, info):
        self.model.body_pos[info["body_id"]] = self.HIDDEN_POS
        self._disable_geoms(info["geom_ids"])

    def _apply_pre_stair_boxes(self):
        geom_ids = [self._geom_id(name) for name in self.PRE_STAIR_GEOMS]
        geom_ids = [geom_id for geom_id in geom_ids if geom_id != -1]
        if self.pre_stair_boxes_enabled and self.active_terrain == "stairs":
            self._restore_geoms(geom_ids)
        else:
            self._disable_geoms(geom_ids)

    def _restore_geoms(self, geom_ids):
        for geom_id in geom_ids:
            defaults = self.geom_defaults.get(geom_id)
            if defaults is None:
                continue
            self.model.geom_rgba[geom_id] = defaults["rgba"]
            self.model.geom_contype[geom_id] = defaults["contype"]
            self.model.geom_conaffinity[geom_id] = defaults["conaffinity"]
            self.model.geom_pos[geom_id] = defaults["pos"]

    def _disable_geoms(self, geom_ids):
        for geom_id in geom_ids:
            defaults = self.geom_defaults.get(geom_id)
            if defaults is None:
                continue
            rgba = defaults["rgba"].copy()
            rgba[3] = 0.0
            self.model.geom_rgba[geom_id] = rgba
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            self.model.geom_pos[geom_id] = self.HIDDEN_GEOM_POS

    def _body_geom_ids(self, body_id):
        geom_start = int(self.model.body_geomadr[body_id])
        geom_count = int(self.model.body_geomnum[body_id])
        if geom_start < 0 or geom_count <= 0:
            return []
        return list(range(geom_start, geom_start + geom_count))

    def _base_yaw(self):
        quat = self.data.xquat[self.base_body_id]
        w, x, y, z = quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _active_label(self):
        return self.TERRAIN_LABELS.get(self.active_terrain, self.active_terrain)

    @staticmethod
    def _yaw_quat(yaw):
        half_yaw = 0.5 * yaw
        return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])

    def _body_id(self, name):
        return mujoco.mj_name2id(self.model, MJT_OBJ.mjOBJ_BODY, name)

    def _geom_id(self, name):
        return mujoco.mj_name2id(self.model, MJT_OBJ.mjOBJ_GEOM, name)

    def _find_first_body(self, names):
        for name in names:
            body_id = self._body_id(name)
            if body_id != -1:
                return body_id
        return -1

    @staticmethod
    def _ctrl_is_pressed():
        glfw = mujoco.glfw.glfw
        window = glfw.get_current_context()
        if window is None:
            return False
        return (
            glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )

    def _print_help(self):
        print(
            "Terrain switcher keys: Ctrl+1 stairs, Ctrl+2 boxes, Ctrl+3 gap, "
            "Ctrl+4 overhanging boxes, Ctrl+5 pyramid stairs, "
            "Ctrl+6 bumped stairs, Ctrl+7 five-step box stairs, "
            "Ctrl+8 discrete grid, Ctrl+B toggle pre-stair boxes, "
            "Ctrl+R reposition current terrain in front of robot."
        )
