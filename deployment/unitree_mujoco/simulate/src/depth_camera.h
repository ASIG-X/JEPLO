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

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "simulate.h"

// ──────────────────────────────────────────────────────────────────────────────
// DepthCamera — offscreen depth rendering, post-processing, ZMQ publishing,
//               and overlay visualization for the Unitree MuJoCo simulator.
//
// Camera parameters match the Intel RealSense D435i reference from the
// go2_parkour_deploy Python code:
//   - Camera name in XML:  "d435i_camera"
//   - Mounted on base_link: pos="0.33 0 0.08" euler="0 -20.2 48.7"
//
// Pipeline:
//   1. Offscreen render at 640×480 → read float depth buffer
//   2. Convert OpenGL depth to metric depth (meters)
//   3. Clamp to [0, max_distance] (default 2.0 m)
//   4. Center-crop to 480×480
//   5. Downscale to 64×64 (area-average / box filter)
//   6. Normalize: depth / max_distance → [0, 1]
//   7. Gaussian blur (5×5 kernel, σ ≈ 1.0)
//   8. Publish 64×64 float32 over ZMQ PUB (tcp://*:5560)
//   9. Upscale to 320×320 orange→blue colormap RGB for overlay visualization
// ──────────────────────────────────────────────────────────────────────────────

class DepthCamera {
  public:
    // Render resolution
    static constexpr int kRenderW = 640;
    static constexpr int kRenderH = 480;

    // After center-crop (take middle 480 cols from 640)
    static constexpr int kCropSize = 480;
    static constexpr int kCropOffsetX = (kRenderW - kCropSize) / 2; // 80

    // Final output
    static constexpr int kOutputSize = 64;

    // Visualization (upscaled for overlay)
    static constexpr int kVizSize = 320;

    // Depth parameters
    static constexpr float kMaxDistance = 2.0f;

    // ZMQ port
    static constexpr int kZmqPort = 5560;

    DepthCamera() {
        // Allocate buffers
        raw_depth_.resize(kRenderW * kRenderH, 0.0f);
        metric_depth_.resize(kRenderW * kRenderH, 0.0f);
        cropped_depth_.resize(kCropSize * kCropSize, 0.0f);
        output_depth_.resize(kOutputSize * kOutputSize, 0.0f);
        blur_tmp_.resize(kOutputSize * kOutputSize, 0.0f);
        viz_rgb_.resize(kVizSize * kVizSize * 3, 0);

        // Initialize MuJoCo visualization objects
        mjv_defaultScene(&scn_);
        mjv_makeScene(nullptr, &scn_, 10000);
        mjv_defaultCamera(&cam_);
        mjv_defaultOption(&opt_);

        // Initialize ZMQ
        zmq_ctx_ = zmq_ctx_new();
        zmq_pub_ = zmq_socket(zmq_ctx_, ZMQ_PUB);
        int sndhwm = 2; // only keep latest 2 messages
        zmq_setsockopt(zmq_pub_, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
        char endpoint[64];
        snprintf(endpoint, sizeof(endpoint), "tcp://*:%d", kZmqPort);
        int rc = zmq_bind(zmq_pub_, endpoint);
        if (rc != 0) {
            std::cerr << "[DepthCamera] ZMQ bind failed on " << endpoint << ": "
                      << zmq_strerror(zmq_errno()) << std::endl;
        } else {
            std::cout << "[DepthCamera] ZMQ PUB bound to " << endpoint << std::endl;
        }

        // Pre-build ZMQ message buffer: header (w, h) + float32 data
        zmq_buf_.resize(sizeof(uint32_t) * 2 + kOutputSize * kOutputSize * sizeof(float));
        uint32_t w = kOutputSize, h = kOutputSize;
        std::memcpy(zmq_buf_.data(), &w, sizeof(uint32_t));
        std::memcpy(zmq_buf_.data() + sizeof(uint32_t), &h, sizeof(uint32_t));
    }

    ~DepthCamera() {
        mjv_freeScene(&scn_);
        if (zmq_pub_)
            zmq_close(zmq_pub_);
        if (zmq_ctx_)
            zmq_ctx_destroy(zmq_ctx_);
    }

    // Non-copyable
    DepthCamera(const DepthCamera &) = delete;
    DepthCamera &operator=(const DepthCamera &) = delete;

    // Check if the camera has been initialized with a model
    bool isInitialized() const { return initialized_; }

    // Return the model pointer used during last init (for detecting model reloads)
    const mjModel *lastModel() const { return last_model_; }

    // ────────────────────────────────────────────────────────────────────────────
    // Call once after model is loaded to look up the camera ID.
    // Returns true if the camera was found.
    // ────────────────────────────────────────────────────────────────────────────
    bool init(mjModel *m) {
        cam_id_ = mj_name2id(m, mjOBJ_CAMERA, "d435i_camera");
        if (cam_id_ < 0) {
            std::cerr << "[DepthCamera] Camera 'd435i_camera' not found in model!\n";
            initialized_ = false;
            last_model_ = nullptr;
            return false;
        }
        std::cout << "[DepthCamera] Found camera 'd435i_camera' (id=" << cam_id_
                  << "), rendering at " << kRenderW << "x" << kRenderH << std::endl;
        initialized_ = true;
        last_model_ = m;
        return true;
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Main per-frame entry point.  Must be called from the render thread
    // (which owns the OpenGL context).
    //
    // sim  — the Simulate object (for mutex and overlay image submission)
    // m, d — model/data pointers (caller ensures they are valid)
    // con  — the mjrContext that holds the OpenGL resources
    // ────────────────────────────────────────────────────────────────────────────
    void renderFrame(mujoco::Simulate &sim, mjModel *m, mjData *d, mjrContext &con) {
        if (!initialized_ || cam_id_ < 0)
            return;

        // ── 1. Offscreen render ──────────────────────────────────────────────────
        // We need to lock the sim mutex while accessing m/d for scene update.
        {
            const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

            // Ensure offscreen buffer is large enough
            if (con.offWidth < kRenderW || con.offHeight < kRenderH) {
                mjr_resizeOffscreen(kRenderW, kRenderH, &con);
            }

            // Set up camera
            cam_.type = mjCAMERA_FIXED;
            cam_.fixedcamid = cam_id_;

            // Update scene from current state
            mjv_updateScene(m, d, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);

            // Switch to offscreen, render, read depth
            mjr_setBuffer(mjFB_OFFSCREEN, &con);

            mjrRect viewport = {0, 0, kRenderW, kRenderH};
            mjr_render(viewport, &scn_, &con);
            mjr_readPixels(nullptr, raw_depth_.data(), viewport, &con);

            // Switch back to window buffer
            mjr_setBuffer(mjFB_WINDOW, &con);

            // Cache znear/zfar for depth conversion
            znear_ = static_cast<float>(m->vis.map.znear * m->stat.extent);
            zfar_ = static_cast<float>(m->vis.map.zfar * m->stat.extent);
        }
        // Mutex released — post-processing can proceed without blocking physics

        // ── 2. Convert OpenGL depth to metric depth ─────────────────────────────
        convertDepth();

        // ── 3. Center-crop 640×480 → 480×480 ────────────────────────────────────
        centerCrop();

        // ── 4. Downscale 480×480 → 64×64 (area average) ────────────────────────
        downscale();

        // ── 5. Normalize by max distance ────────────────────────────────────────
        normalize();

        // ── 6. Gaussian blur (5×5, σ≈1.0) ──────────────────────────────────────
        gaussianBlur();

        // ── 7. Publish over ZMQ ─────────────────────────────────────────────────
        publishZmq();

        // ── 8. Prepare overlay visualization ────────────────────────────────────
        prepareVizOverlay(sim);
    }

  private:
    // ──────────────────────────────────────────────────────────────────────────
    // Convert raw OpenGL depth buffer to metric depth (meters).
    // MuJoCo default depth map: 0 = znear, 1 = zfar (mjDEPTH_ZERONEAR).
    // Metric z = znear * zfar / (zfar - raw * (zfar - znear))
    // ──────────────────────────────────────────────────────────────────────────
    void convertDepth() {
        const float range = zfar_ - znear_;
        const int n = kRenderW * kRenderH;
        for (int i = 0; i < n; ++i) {
            float d = raw_depth_[i];
            if (d >= 1.0f) {
                // At or beyond far plane — clamp to max distance
                metric_depth_[i] = kMaxDistance;
            } else {
                float z = znear_ * zfar_ / (zfar_ - d * range);
                metric_depth_[i] = std::min(z, kMaxDistance);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Center-crop: extract the middle 480 columns from 640-wide image.
    // Also flip vertically (OpenGL has origin at bottom-left).
    // ──────────────────────────────────────────────────────────────────────────
    void centerCrop() {
        for (int row = 0; row < kCropSize; ++row) {
            // Flip: output row 0 = input row (kRenderH - 1)
            int src_row = kRenderH - 1 - row;
            const float *src = metric_depth_.data() + src_row * kRenderW + kCropOffsetX;
            float *dst = cropped_depth_.data() + row * kCropSize;
            std::memcpy(dst, src, kCropSize * sizeof(float));
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Downscale 480×480 → 64×64 using area-average (box filter).
    // Each output pixel covers a (480/64 = 7.5) × 7.5 region.
    // We use a simple area-average with fractional edges.
    // ──────────────────────────────────────────────────────────────────────────
    void downscale() {
        const float scale = static_cast<float>(kCropSize) / kOutputSize; // 7.5
        for (int oy = 0; oy < kOutputSize; ++oy) {
            float src_y0 = oy * scale;
            float src_y1 = (oy + 1) * scale;
            for (int ox = 0; ox < kOutputSize; ++ox) {
                float src_x0 = ox * scale;
                float src_x1 = (ox + 1) * scale;

                float sum = 0.0f;
                float weight = 0.0f;

                int iy0 = static_cast<int>(src_y0);
                int iy1 = std::min(static_cast<int>(std::ceil(src_y1)), kCropSize);
                int ix0 = static_cast<int>(src_x0);
                int ix1 = std::min(static_cast<int>(std::ceil(src_x1)), kCropSize);

                for (int iy = iy0; iy < iy1; ++iy) {
                    float wy = 1.0f;
                    if (iy == iy0)
                        wy = 1.0f - (src_y0 - iy0);
                    if (iy == iy1 - 1 && src_y1 < iy1)
                        wy = src_y1 - (iy1 - 1);

                    for (int ix = ix0; ix < ix1; ++ix) {
                        float wx = 1.0f;
                        if (ix == ix0)
                            wx = 1.0f - (src_x0 - ix0);
                        if (ix == ix1 - 1 && src_x1 < ix1)
                            wx = src_x1 - (ix1 - 1);

                        float w = wx * wy;
                        sum += w * cropped_depth_[iy * kCropSize + ix];
                        weight += w;
                    }
                }

                output_depth_[oy * kOutputSize + ox] = (weight > 0.0f) ? (sum / weight) : 0.0f;
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Normalize depth values to [0, 1] by dividing by max distance.
    // ──────────────────────────────────────────────────────────────────────────
    void normalize() {
        const float inv_max = 1.0f / kMaxDistance;
        for (int i = 0; i < kOutputSize * kOutputSize; ++i) {
            output_depth_[i] = std::min(output_depth_[i] * inv_max, 1.0f);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Gaussian blur with a 5×5 kernel (σ ≈ 1.0) on the 64×64 output buffer.
    // Uses blur_tmp_ as scratch space. Edge pixels use clamped (replicate)
    // boundary conditions.
    // ──────────────────────────────────────────────────────────────────────────
    void gaussianBlur() {
        // 5×5 Gaussian kernel, σ = 1.0, unnormalized weights:
        //   1  4  7  4  1
        //   4 16 26 16  4
        //   7 26 41 26  7
        //   4 16 26 16  4
        //   1  4  7  4  1
        // Sum = 273
        static constexpr int kKernel[5][5] = {
            {1, 4, 7, 4, 1},
            {4, 16, 26, 16, 4},
            {7, 26, 41, 26, 7},
            {4, 16, 26, 16, 4},
            {1, 4, 7, 4, 1}};
        static constexpr float kKernelSum = 273.0f;
        constexpr int N = kOutputSize;
        constexpr int R = 2; // kernel radius

        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                float sum = 0.0f;
                for (int ky = -R; ky <= R; ++ky) {
                    int sy = std::max(0, std::min(y + ky, N - 1));
                    for (int kx = -R; kx <= R; ++kx) {
                        int sx = std::max(0, std::min(x + kx, N - 1));
                        sum += kKernel[ky + R][kx + R] * output_depth_[sy * N + sx];
                    }
                }
                blur_tmp_[y * N + x] = sum / kKernelSum;
            }
        }
        std::swap(output_depth_, blur_tmp_);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Publish the 64×64 float32 normalized depth image over ZMQ.
    // Message format: [uint32 width][uint32 height][float32 × width × height]
    // ──────────────────────────────────────────────────────────────────────────
    void publishZmq() {
        constexpr size_t header_size = sizeof(uint32_t) * 2;
        constexpr size_t data_size = kOutputSize * kOutputSize * sizeof(float);
        std::memcpy(zmq_buf_.data() + header_size, output_depth_.data(), data_size);

        // Non-blocking send (ZMQ_DONTWAIT) — drops if no subscriber
        zmq_send(zmq_pub_, zmq_buf_.data(), zmq_buf_.size(), ZMQ_DONTWAIT);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Orange → Blue colormap.  t=0 (near) → orange, t=1 (far) → blue.
    // ──────────────────────────────────────────────────────────────────────────
    static void depthColormap(float t, unsigned char &r, unsigned char &g, unsigned char &b) {
        t = std::max(0.0f, std::min(t, 1.0f));
        // Orange: (255, 140, 0)   Blue: (30, 60, 255)
        r = static_cast<unsigned char>(255.0f + t * (30.0f - 255.0f));
        g = static_cast<unsigned char>(140.0f + t * (60.0f - 140.0f));
        b = static_cast<unsigned char>(0.0f + t * (255.0f - 0.0f));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Upscale 64×64 → 320×320 (bilinear interpolation) and apply orange→blue
    // colormap, then submit as overlay for the MuJoCo viewer with a
    // color bar legend.
    // ──────────────────────────────────────────────────────────────────────────
    void prepareVizOverlay(mujoco::Simulate &sim) {
        // Layout: [depth image 320×320] [color bar 20×320]  = 340×320 total
        constexpr int kBarWidth = 20;
        constexpr int kTotalW = kVizSize + kBarWidth;
        constexpr int kTotalH = kVizSize;

        // Resize viz buffer if needed (first call or size changed)
        const size_t total_pixels = kTotalW * kTotalH * 3;
        if (viz_rgb_.size() != total_pixels) {
            viz_rgb_.resize(total_pixels, 0);
        }

        const float scale = static_cast<float>(kOutputSize) / kVizSize; // 0.2

        // Build depth image with orange→blue colormap (bottom-up for OpenGL).
        // Flip vertically: OpenGL row 0 is bottom, but output_depth_ row 0
        // is the top of the camera view (already flipped in centerCrop).
        for (int vy = 0; vy < kVizSize; ++vy) {
            // Flip: vy=0 (screen bottom) → last source row (image bottom)
            int vy_flipped = kVizSize - 1 - vy;

            for (int vx = 0; vx < kVizSize; ++vx) {
                // Bilinear interpolation from 64×64 source
                float sx = (vx + 0.5f) * scale - 0.5f;
                float sy = (vy_flipped + 0.5f) * scale - 0.5f;
                sx = std::max(0.0f, std::min(sx, static_cast<float>(kOutputSize - 1)));
                sy = std::max(0.0f, std::min(sy, static_cast<float>(kOutputSize - 1)));

                int x0 = static_cast<int>(sx);
                int y0 = static_cast<int>(sy);
                int x1 = std::min(x0 + 1, kOutputSize - 1);
                int y1 = std::min(y0 + 1, kOutputSize - 1);
                float fx = sx - x0;
                float fy = sy - y0;

                float val = (1 - fx) * (1 - fy) * output_depth_[y0 * kOutputSize + x0] +
                            fx * (1 - fy) * output_depth_[y0 * kOutputSize + x1] +
                            (1 - fx) * fy * output_depth_[y1 * kOutputSize + x0] +
                            fx * fy * output_depth_[y1 * kOutputSize + x1];

                unsigned char r, g, b;
                depthColormap(val, r, g, b);

                int idx = (vy * kTotalW + vx) * 3;
                viz_rgb_[idx + 0] = r;
                viz_rgb_[idx + 1] = g;
                viz_rgb_[idx + 2] = b;
            }

            // Color bar: vertical gradient (bottom = near/0, top = far/1)
            // OpenGL row 0 is bottom, so vy=0 is bottom → near (t=0)
            float t = static_cast<float>(vy) / (kVizSize - 1);
            unsigned char br, bg, bb;
            depthColormap(t, br, bg, bb);
            for (int bx = 0; bx < kBarWidth; ++bx) {
                int idx = (vy * kTotalW + kVizSize + bx) * 3;
                viz_rgb_[idx + 0] = br;
                viz_rgb_[idx + 1] = bg;
                viz_rgb_[idx + 2] = bb;
            }
        }

        // Submit the image as an overlay in the bottom-left corner of the viewer.
        mjrRect viz_rect = {10, 10, kTotalW, kTotalH};

        auto rgb_copy = std::make_unique<unsigned char[]>(total_pixels);
        std::memcpy(rgb_copy.get(), viz_rgb_.data(), total_pixels);

        sim.user_images_new_.clear();
        sim.user_images_new_.emplace_back(viz_rect, std::move(rgb_copy));
        sim.newimagerequest.store(1);
    }

    // MuJoCo visualization objects (dedicated to depth camera)
    mjvScene scn_;
    mjvCamera cam_;
    mjvOption opt_;

    // Camera ID in the model
    int cam_id_ = -1;
    bool initialized_ = false;
    const mjModel *last_model_ = nullptr;

    // Depth conversion parameters
    float znear_ = 0.0f;
    float zfar_ = 0.0f;

    // Buffers
    std::vector<float> raw_depth_;       // kRenderW × kRenderH
    std::vector<float> metric_depth_;    // kRenderW × kRenderH
    std::vector<float> cropped_depth_;   // kCropSize × kCropSize (480×480)
    std::vector<float> output_depth_;    // kOutputSize × kOutputSize (64×64)
    std::vector<float> blur_tmp_;        // kOutputSize × kOutputSize (scratch for blur)
    std::vector<unsigned char> viz_rgb_; // kVizSize × kVizSize × 3 (320×320)

    // ZMQ
    void *zmq_ctx_ = nullptr;
    void *zmq_pub_ = nullptr;
    std::vector<uint8_t> zmq_buf_;
};
