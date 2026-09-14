/*
This file is part of JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion

Copyright (c) 2026 Qihao Yuan

Developer: Qihao Yuan <qihao.yuan@rug.nl>

For commercial use, please contact me at <qihao.yuan@rug.nl> or Kailai Li at <kailai.li@liu.se>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#pragma once

#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

// Mirrors simulate_python/terrain_switcher.py. Keyboard callbacks only queue
// requests; update() applies model changes while the physics thread owns the
// simulation mutex.
class TerrainSwitcher {
  public:
    bool queueKey(int key) {
        Pending pending;
        if (key >= '1' && key <= '8') {
            pending.type = Command::kSwitch;
            pending.terrain = kTerrainDefinitions[key - '1'].name;
        } else if (key == 'B') {
            pending.type = Command::kTogglePreStairs;
        } else if (key == 'R') {
            pending.type = Command::kReposition;
        } else {
            return false;
        }

        const std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_ = pending;
        return true;
    }

    void update(mjModel *model, mjData *data) {
        if (!model || !data) {
            return;
        }
        if (model != model_) {
            initialize(model, data);
        }

        Pending pending;
        {
            const std::lock_guard<std::mutex> lock(pending_mutex_);
            pending = pending_;
            pending_ = {};
        }
        if (!enabled_ || pending.type == Command::kNone) {
            return;
        }

        switch (pending.type) {
        case Command::kSwitch:
            switchTerrain(model, data, pending.terrain);
            break;
        case Command::kTogglePreStairs:
            pre_stair_boxes_enabled_ = !pre_stair_boxes_enabled_;
            applyPreStairBoxes(model);
            mj_forward(model, data);
            std::cout << "Terrain switcher: pre-stair boxes "
                      << (pre_stair_boxes_enabled_ ? "enabled" : "disabled") << ".\n";
            break;
        case Command::kReposition:
            applyActiveTerrain(model, data);
            std::cout << "Terrain switcher: repositioned " << activeLabel()
                      << " in front of robot.\n";
            break;
        case Command::kNone:
            break;
        }
    }

  private:
    struct TerrainDefinition {
        const char *name;
        const char *label;
        mjtNum front_offset;
    };

    inline static constexpr std::array<TerrainDefinition, 8> kTerrainDefinitions = {{
        {"stairs", "stairs", 0.0},
        {"boxes", "boxes", 0.0},
        {"gap", "gap", 0.0},
        {"boxes_overhang", "overhanging boxes", 0.0},
        {"pyramid_stairs", "pyramid and inverted-pyramid stairs", 1.5},
        {"bumped_stairs", "stairs with riser bumps", 0.0},
        {"short_box_stairs", "five-step stairs with boxes", 0.0},
        {"discrete_grid", "discrete grid terrain", 0.0},
    }};
    inline static constexpr std::array<const char *, 3> kPreStairGeomNames = {
        "pre1", "pre2", "pre3"};

    enum class Command { kNone, kSwitch, kTogglePreStairs, kReposition };

    struct Pending {
        Command type = Command::kNone;
        std::string terrain;
    };

    struct GeomDefaults {
        bool captured = false;
        std::array<float, 4> rgba{};
        int contype = 0;
        int conaffinity = 0;
        std::array<mjtNum, 3> pos{};
    };

    struct Terrain {
        std::string name;
        std::string label;
        int body_id = -1;
        mjtNum z = 0.0;
        mjtNum front_offset = 0.0;
        std::vector<int> geom_ids;
    };

    void initialize(mjModel *model, mjData *data) {
        model_ = model;
        enabled_ = false;
        base_body_id_ = findFirstBody(model, {"base_link", "torso_link", "pelvis"});
        active_terrain_ = "stairs";
        pre_stair_boxes_enabled_ = false;
        terrains_.clear();
        pre_stair_geom_ids_.clear();
        geom_defaults_.assign(model->ngeom, {});

        for (const TerrainDefinition &definition : kTerrainDefinitions) {
            const int body_id = mj_name2id(model, mjOBJ_BODY, definition.name);
            if (body_id < 0) {
                continue;
            }

            Terrain terrain;
            terrain.name = definition.name;
            terrain.label = definition.label;
            terrain.body_id = body_id;
            terrain.z = model->body_pos[3 * body_id + 2];
            terrain.front_offset = definition.front_offset;

            const int geom_start = model->body_geomadr[body_id];
            const int geom_count = model->body_geomnum[body_id];
            if (geom_start >= 0) {
                for (int i = 0; i < geom_count; ++i) {
                    const int geom_id = geom_start + i;
                    terrain.geom_ids.push_back(geom_id);
                    captureGeomDefaults(model, geom_id);
                }
            }
            terrains_.push_back(std::move(terrain));
        }

        for (const char *name : kPreStairGeomNames) {
            const int geom_id = mj_name2id(model, mjOBJ_GEOM, name);
            if (geom_id >= 0) {
                pre_stair_geom_ids_.push_back(geom_id);
                captureGeomDefaults(model, geom_id);
            }
        }

        if (terrains_.empty()) {
            std::cout << "Terrain switcher: no named terrain bodies found.\n";
            return;
        }
        if (base_body_id_ < 0) {
            std::cout << "Terrain switcher: no robot base body found.\n";
            return;
        }
        if (!findTerrain(active_terrain_)) {
            active_terrain_ = terrains_.front().name;
        }

        enabled_ = true;
        applyActiveTerrain(model, data);
        printHelp();
    }

    static int findFirstBody(mjModel *model, std::initializer_list<const char *> names) {
        for (const char *name : names) {
            const int body_id = mj_name2id(model, mjOBJ_BODY, name);
            if (body_id >= 0) {
                return body_id;
            }
        }
        return -1;
    }

    void captureGeomDefaults(const mjModel *model, int geom_id) {
        if (geom_id < 0 || geom_id >= static_cast<int>(geom_defaults_.size()) ||
            geom_defaults_[geom_id].captured) {
            return;
        }

        GeomDefaults &defaults = geom_defaults_[geom_id];
        defaults.captured = true;
        for (int i = 0; i < 4; ++i) {
            defaults.rgba[i] = model->geom_rgba[4 * geom_id + i];
        }
        defaults.contype = model->geom_contype[geom_id];
        defaults.conaffinity = model->geom_conaffinity[geom_id];
        for (int i = 0; i < 3; ++i) {
            defaults.pos[i] = model->geom_pos[3 * geom_id + i];
        }
    }

    Terrain *findTerrain(const std::string &name) {
        for (Terrain &terrain : terrains_) {
            if (terrain.name == name) {
                return &terrain;
            }
        }
        return nullptr;
    }

    const Terrain *findTerrain(const std::string &name) const {
        for (const Terrain &terrain : terrains_) {
            if (terrain.name == name) {
                return &terrain;
            }
        }
        return nullptr;
    }

    void switchTerrain(mjModel *model, mjData *data, const std::string &name) {
        if (!findTerrain(name)) {
            std::cout << "Terrain switcher: '" << name
                      << "' is not available in this model.\n";
            return;
        }
        active_terrain_ = name;
        applyActiveTerrain(model, data);
        std::cout << "Terrain switcher: active terrain is " << activeLabel() << ".\n";
    }

    void applyActiveTerrain(mjModel *model, mjData *data) {
        for (Terrain &terrain : terrains_) {
            if (terrain.name == active_terrain_) {
                showBody(model, data, terrain);
            } else {
                hideBody(model, terrain);
            }
        }
        applyPreStairBoxes(model);
        mj_forward(model, data);
    }

    void showBody(mjModel *model, const mjData *data, const Terrain &terrain) {
        const mjtNum *quat = data->xquat + 4 * base_body_id_;
        const mjtNum yaw = std::atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]));
        const mjtNum distance = front_distance_ + terrain.front_offset;
        const mjtNum *base_pos = data->xpos + 3 * base_body_id_;
        mjtNum *body_pos = model->body_pos + 3 * terrain.body_id;
        body_pos[0] = base_pos[0] + std::cos(yaw) * distance;
        body_pos[1] = base_pos[1] + std::sin(yaw) * distance;
        body_pos[2] = terrain.z;

        mjtNum *body_quat = model->body_quat + 4 * terrain.body_id;
        body_quat[0] = std::cos(0.5 * yaw);
        body_quat[1] = 0.0;
        body_quat[2] = 0.0;
        body_quat[3] = std::sin(0.5 * yaw);
        restoreGeoms(model, terrain.geom_ids);
    }

    void hideBody(mjModel *model, const Terrain &terrain) {
        mjtNum *body_pos = model->body_pos + 3 * terrain.body_id;
        body_pos[0] = kHiddenPosition;
        body_pos[1] = kHiddenPosition;
        body_pos[2] = kHiddenPosition;
        disableGeoms(model, terrain.geom_ids);
    }

    void applyPreStairBoxes(mjModel *model) {
        if (pre_stair_boxes_enabled_ && active_terrain_ == "stairs") {
            restoreGeoms(model, pre_stair_geom_ids_);
        } else {
            disableGeoms(model, pre_stair_geom_ids_);
        }
    }

    void restoreGeoms(mjModel *model, const std::vector<int> &geom_ids) const {
        for (int geom_id : geom_ids) {
            if (geom_id < 0 || geom_id >= static_cast<int>(geom_defaults_.size())) {
                continue;
            }
            const GeomDefaults &defaults = geom_defaults_[geom_id];
            if (!defaults.captured) {
                continue;
            }
            for (int i = 0; i < 4; ++i) {
                model->geom_rgba[4 * geom_id + i] = defaults.rgba[i];
            }
            model->geom_contype[geom_id] = defaults.contype;
            model->geom_conaffinity[geom_id] = defaults.conaffinity;
            for (int i = 0; i < 3; ++i) {
                model->geom_pos[3 * geom_id + i] = defaults.pos[i];
            }
        }
    }

    void disableGeoms(mjModel *model, const std::vector<int> &geom_ids) const {
        for (int geom_id : geom_ids) {
            if (geom_id < 0 || geom_id >= static_cast<int>(geom_defaults_.size())) {
                continue;
            }
            const GeomDefaults &defaults = geom_defaults_[geom_id];
            if (!defaults.captured) {
                continue;
            }
            for (int i = 0; i < 4; ++i) {
                model->geom_rgba[4 * geom_id + i] = defaults.rgba[i];
            }
            model->geom_rgba[4 * geom_id + 3] = 0.0f;
            model->geom_contype[geom_id] = 0;
            model->geom_conaffinity[geom_id] = 0;
            for (int i = 0; i < 3; ++i) {
                model->geom_pos[3 * geom_id + i] = kHiddenPosition;
            }
        }
    }

    std::string activeLabel() const {
        const Terrain *terrain = findTerrain(active_terrain_);
        return terrain ? terrain->label : active_terrain_;
    }

    static void printHelp() {
        std::cout
            << "Terrain switcher keys: Ctrl+1 stairs, Ctrl+2 boxes, Ctrl+3 gap, "
               "Ctrl+4 overhanging boxes, Ctrl+5 pyramid stairs, "
               "Ctrl+6 bumped stairs, Ctrl+7 five-step box stairs, "
               "Ctrl+8 discrete grid, Ctrl+B toggle pre-stair boxes, "
               "Ctrl+R reposition current terrain in front of robot.\n";
    }

    inline static constexpr mjtNum kHiddenPosition = 1000.0;
    mjtNum front_distance_ = 3.0;
    mjModel *model_ = nullptr;
    int base_body_id_ = -1;
    bool enabled_ = false;
    bool pre_stair_boxes_enabled_ = false;
    std::string active_terrain_ = "stairs";
    std::vector<Terrain> terrains_;
    std::vector<int> pre_stair_geom_ids_;
    std::vector<GeomDefaults> geom_defaults_;
    std::mutex pending_mutex_;
    Pending pending_;
};
