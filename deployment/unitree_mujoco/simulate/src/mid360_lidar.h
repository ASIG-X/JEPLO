/*
This file is part of JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion

Copyright (c) 2026 Qihao Yuan

Developer: Qihao Yuan <qihao.yuan@rug.nl>

For commercial use, please contact me at <qihao.yuan@rug.nl> or Kailai Li at <kailai.li@liu.se>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#pragma once

#include <mujoco/mujoco.h>
#include <zmq.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

// Publishes FOV-filtered Livox Mid360 point clouds in the wire format consumed
// by simulate_python/lidar_depth_pub/lidar_depth_pub.cpp --sim:
//   [3 float sensor position][9 float sensor rotation][N * 3 float local points]
class Mid360LidarPublisher {
  public:
    Mid360LidarPublisher(int port, double frequency_hz, bool legacy_fov)
        : frequency_hz_(frequency_hz), legacy_fov_(legacy_fov) {
        if (frequency_hz_ <= 0.0) {
            error_ = "LiDAR frequency must be positive";
            return;
        }
        if (port <= 0 || port > 65535) {
            error_ = "LiDAR port must be between 1 and 65535";
            return;
        }

        zmq_context_ = zmq_ctx_new();
        if (!zmq_context_) {
            error_ = "failed to create ZMQ context";
            return;
        }
        zmq_publisher_ = zmq_socket(zmq_context_, ZMQ_PUB);
        if (!zmq_publisher_) {
            error_ = "failed to create ZMQ publisher socket";
            return;
        }

        const int high_water_mark = 2;
        const int linger_ms = 0;
        zmq_setsockopt(
            zmq_publisher_, ZMQ_SNDHWM, &high_water_mark, sizeof(high_water_mark));
        zmq_setsockopt(zmq_publisher_, ZMQ_LINGER, &linger_ms, sizeof(linger_ms));

        const std::string endpoint = "tcp://*:" + std::to_string(port);
        if (zmq_bind(zmq_publisher_, endpoint.c_str()) != 0) {
            error_ = "ZMQ bind failed on " + endpoint + ": " + zmq_strerror(zmq_errno());
            return;
        }
        transport_ready_ = true;
        std::cout << "[Mid360] ZMQ PUB bound to " << endpoint << std::endl;
    }

    ~Mid360LidarPublisher() {
        if (zmq_publisher_)
            zmq_close(zmq_publisher_);
        if (zmq_context_)
            zmq_ctx_destroy(zmq_context_);
    }

    Mid360LidarPublisher(const Mid360LidarPublisher &) = delete;
    Mid360LidarPublisher &operator=(const Mid360LidarPublisher &) = delete;

    bool loadPattern(const std::filesystem::path &path) {
        if (!transport_ready_)
            return false;
        ready_ = false;

        std::vector<float> angles;
        size_t angle_count = 0;
        if (!loadNpyFloat32x2(path, angles, angle_count))
            return false;

        const size_t frame_count = angle_count / std::gcd(angle_count, kSamplesPerFrame);
        frames_.clear();
        frames_.reserve(frame_count);

        size_t cursor = 0;
        size_t min_rays = kSamplesPerFrame;
        size_t max_rays = 0;
        size_t total_rays = 0;
        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            std::vector<mjtNum> local_rays;
            local_rays.reserve(kSamplesPerFrame * 3);

            for (size_t i = 0; i < kSamplesPerFrame; ++i) {
                const size_t pattern_index = (cursor + i) % angle_count;
                const double azimuth = std::remainder(
                    static_cast<double>(angles[2 * pattern_index]), 2.0 * kPi);
                const double elevation = static_cast<double>(angles[2 * pattern_index + 1]);
                if (!insideOutputGrid(azimuth, elevation))
                    continue;

                const double cos_elevation = std::cos(elevation);
                local_rays.push_back(cos_elevation * std::cos(azimuth));
                local_rays.push_back(cos_elevation * std::sin(azimuth));
                local_rays.push_back(std::sin(elevation));
            }

            const size_t ray_count = local_rays.size() / 3;
            min_rays = std::min(min_rays, ray_count);
            max_rays = std::max(max_rays, ray_count);
            total_rays += ray_count;
            frames_.push_back(std::move(local_rays));
            cursor = (cursor + kSamplesPerFrame) % angle_count;
        }

        world_rays_.resize(max_rays * 3);
        distances_.resize(max_rays);
        geom_ids_.resize(max_rays);
        message_.resize(kPoseFloatCount + max_rays * 3);
        frame_index_ = 0;
        if (max_rays == 0) {
            error_ = "Mid360 pattern has no rays inside the selected FOV";
            return false;
        }

        const double average_rays = static_cast<double>(total_rays) / frames_.size();
        std::cout << "[Mid360] Loaded " << angle_count << " scan angles from " << path
                  << " | precomputed " << frames_.size() << " frames | FOV "
                  << (legacy_fov_ ? "legacy 25x60" : "wide 25x120") << " | rays/frame "
                  << min_rays << ".." << max_rays << " (mean " << average_rays << ")"
                  << std::endl;
        ready_ = true;
        return true;
    }

    const std::string &error() const { return error_; }

    // Called with exclusive access to model/data after stepping the simulation.
    void update(const mjModel *model, mjData *data) {
        if (!ready_ || !model || !data)
            return;
        if (model != model_ && !initializeModel(model))
            return;

        if (!schedule_initialized_ || data->time < last_simulation_time_) {
            next_scan_time_ = data->time;
            schedule_initialized_ = true;
        }
        last_simulation_time_ = data->time;
        if (data->time + 1e-9 < next_scan_time_)
            return;

        const double period = 1.0 / frequency_hz_;
        do {
            next_scan_time_ += period;
        } while (next_scan_time_ <= data->time);

        publishFrame(model, data);
    }

  private:
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr size_t kSamplesPerFrame = 24000;
    static constexpr size_t kPoseFloatCount = 12;
    // Keep the tracing cutoff independent of lidar_depth_pub's 2 m image range.
    // A short mj_multiRay cutoff can cull the static floor based on its geom origin.
    static constexpr double kRayTraceCutoff = 100.0;

    // Must match lidar_depth_pub's spherical projection exactly.
    static constexpr int kGridRows = 25;
    static constexpr double kElevationMin = -4.5 * kPi / 180.0;
    static constexpr double kElevationResolution = 2.0 * kPi / 180.0;

    bool insideOutputGrid(double azimuth, double elevation) const {
        const int columns = legacy_fov_ ? 60 : 120;
        const double azimuth_min = (legacy_fov_ ? -60.0 : -120.0) * kPi / 180.0;
        const double azimuth_resolution = 2.0 * kPi / 180.0;
        const int row = static_cast<int>(std::round(
            (elevation - kElevationMin) / kElevationResolution));
        const int column = static_cast<int>(std::round(
            (azimuth - azimuth_min) / azimuth_resolution));
        return row >= 0 && row < kGridRows && column >= 0 && column < columns;
    }

    bool initializeModel(const mjModel *model) {
        model_ = model;
        lidar_site_id_ = mj_name2id(model, mjOBJ_SITE, "lidar");
        body_exclude_id_ = mj_name2id(model, mjOBJ_BODY, "base_link");
        if (lidar_site_id_ < 0 || body_exclude_id_ < 0) {
            std::cerr << "[Mid360] Model must contain site 'lidar' and body 'base_link'"
                      << std::endl;
            model_valid_ = false;
            return false;
        }

        geom_group_.fill(1);
        for (int group = 3; group < mjNGROUP; ++group)
            geom_group_[group] = 0;
        model_valid_ = true;
        schedule_initialized_ = false;
        std::cout << "[Mid360] Using site 'lidar' and excluding body 'base_link'" << std::endl;
        return true;
    }

    void publishFrame(const mjModel *model, mjData *data) {
        if (!model_valid_ || frames_.empty())
            return;

        const std::vector<mjtNum> &local_rays = frames_[frame_index_];
        frame_index_ = (frame_index_ + 1) % frames_.size();
        const int ray_count = static_cast<int>(local_rays.size() / 3);

        const auto start = std::chrono::steady_clock::now();
        const mjtNum *sensor_rotation = data->site_xmat + 9 * lidar_site_id_;
        for (int i = 0; i < ray_count; ++i) {
            mju_mulMatVec3(
                world_rays_.data() + 3 * i, sensor_rotation, local_rays.data() + 3 * i);
        }

        mj_multiRay(
            model, data, data->site_xpos + 3 * lidar_site_id_, world_rays_.data(),
            geom_group_.data(), 1, body_exclude_id_, geom_ids_.data(), distances_.data(),
            ray_count, kRayTraceCutoff);

        const mjtNum *sensor_position = data->site_xpos + 3 * lidar_site_id_;
        for (int i = 0; i < 3; ++i)
            message_[i] = static_cast<float>(sensor_position[i]);
        for (int i = 0; i < 9; ++i)
            message_[3 + i] = static_cast<float>(sensor_rotation[i]);

        float *points = message_.data() + kPoseFloatCount;
        for (int i = 0; i < ray_count; ++i) {
            const mjtNum distance = distances_[i];
            if (distance < 0) {
                points[3 * i] = 0.0f;
                points[3 * i + 1] = 0.0f;
                points[3 * i + 2] = 0.0f;
            } else {
                points[3 * i] = static_cast<float>(distance * local_rays[3 * i]);
                points[3 * i + 1] = static_cast<float>(distance * local_rays[3 * i + 1]);
                points[3 * i + 2] = static_cast<float>(distance * local_rays[3 * i + 2]);
            }
        }

        const size_t message_bytes =
            (kPoseFloatCount + static_cast<size_t>(ray_count) * 3) * sizeof(float);
        zmq_send(zmq_publisher_, message_.data(), message_bytes, ZMQ_DONTWAIT);

        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
        accumulated_time_ms_ += elapsed_ms;
        max_time_ms_ = std::max(max_time_ms_, elapsed_ms);
        ++published_frames_;
        if (published_frames_ % 100 == 0) {
            std::cout << "[Mid360] " << published_frames_ << " frames | last " << ray_count
                      << " rays | mean " << accumulated_time_ms_ / 100.0 << " ms | max "
                      << max_time_ms_ << " ms" << std::endl;
            accumulated_time_ms_ = 0.0;
            max_time_ms_ = 0.0;
        }
    }

    bool loadNpyFloat32x2(
        const std::filesystem::path &path, std::vector<float> &values, size_t &rows) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            error_ = "cannot open Mid360 pattern: " + path.string();
            return false;
        }

        char magic[6];
        file.read(magic, sizeof(magic));
        if (!file || std::memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0) {
            error_ = "invalid NumPy file: " + path.string();
            return false;
        }

        uint8_t version[2];
        file.read(reinterpret_cast<char *>(version), sizeof(version));
        uint32_t header_length = 0;
        if (version[0] == 1) {
            uint8_t bytes[2];
            file.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
            header_length = static_cast<uint32_t>(bytes[0]) |
                            (static_cast<uint32_t>(bytes[1]) << 8);
        } else if (version[0] == 2 || version[0] == 3) {
            uint8_t bytes[4];
            file.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
            header_length = static_cast<uint32_t>(bytes[0]) |
                            (static_cast<uint32_t>(bytes[1]) << 8) |
                            (static_cast<uint32_t>(bytes[2]) << 16) |
                            (static_cast<uint32_t>(bytes[3]) << 24);
        } else {
            error_ = "unsupported NumPy version in " + path.string();
            return false;
        }
        if (!file || header_length == 0 || header_length > 1024 * 1024) {
            error_ = "invalid NumPy header in " + path.string();
            return false;
        }

        std::string header(header_length, '\0');
        file.read(header.data(), header.size());
        if (!file || header.find("'<f4'") == std::string::npos ||
            header.find("False") == std::string::npos) {
            error_ = "Mid360 pattern must be a C-order little-endian float32 array";
            return false;
        }

        const size_t shape = header.find("shape");
        const size_t open = header.find('(', shape);
        const size_t comma = header.find(',', open);
        const size_t close = header.find(')', comma);
        if (shape == std::string::npos || open == std::string::npos ||
            comma == std::string::npos || close == std::string::npos) {
            error_ = "Mid360 pattern must have shape (N, 2)";
            return false;
        }

        try {
            rows = std::stoull(header.substr(open + 1, comma - open - 1));
            const size_t columns = std::stoull(header.substr(comma + 1, close - comma - 1));
            if (columns != 2) {
                error_ = "Mid360 pattern must have shape (N, 2)";
                return false;
            }
        } catch (const std::exception &) {
            error_ = "cannot parse Mid360 pattern shape";
            return false;
        }
        if (rows == 0) {
            error_ = "Mid360 pattern is empty";
            return false;
        }

        values.resize(rows * 2);
        file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float));
        if (!file) {
            error_ = "Mid360 pattern data is truncated";
            return false;
        }
        return true;
    }

    double frequency_hz_;
    bool legacy_fov_;
    bool ready_ = false;
    bool transport_ready_ = false;
    std::string error_;

    void *zmq_context_ = nullptr;
    void *zmq_publisher_ = nullptr;

    std::vector<std::vector<mjtNum>> frames_;
    std::vector<mjtNum> world_rays_;
    std::vector<mjtNum> distances_;
    std::vector<int> geom_ids_;
    std::vector<float> message_;
    size_t frame_index_ = 0;

    const mjModel *model_ = nullptr;
    bool model_valid_ = false;
    int lidar_site_id_ = -1;
    int body_exclude_id_ = -1;
    std::array<mjtByte, mjNGROUP> geom_group_{};

    bool schedule_initialized_ = false;
    double next_scan_time_ = 0.0;
    double last_simulation_time_ = 0.0;

    uint64_t published_frames_ = 0;
    double accumulated_time_ms_ = 0.0;
    double max_time_ms_ = 0.0;
};
