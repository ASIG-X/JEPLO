/*
This file is part of JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion

Copyright (c) 2026 Qihao Yuan

Developer: Qihao Yuan <qihao.yuan@rug.nl>

For commercial use, please contact me at <qihao.yuan@rug.nl> or Kailai Li at <kailai.li@liu.se>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

// ─────────────────────────────────────────────────────────────────────────────
// lidar_depth_pub.cpp — Livox Mid360 LiDAR depth image publisher.
//
// Subscribes to /livox/lidar (livox_ros_driver2 CustomMsg), builds a spherical
// depth image matching the simulation preprocessing pipeline, and
// publishes the result over ZMQ PUB on tcp://*:5560.
//
// Pipeline (matches simulation LiDAR preprocessing):
//   1. Receive raw 3D points from Livox Mid360 via ROS2 CustomMsg
//   2. Downsample raw points (default: keep every point)
//   3. Filter invalid / out-of-range points
//   4. Project to the selected 25×60, 25×90, or 25×120 spherical depth grid
//   5. Normalize: clamp [0, 2.0], divide by 2.0 → [0, 1]
//   6. Push into a configurable-length frame ring buffer (default: 5)
//   7. Aggregate via element-wise min across the buffer
//   8. Publish float32 grid over ZMQ PUB (tcp://*:5560)
//      Message: [uint32 width][uint32 height=25][float32 × width × height]
//
// Build (colcon):
//   colcon build --packages-select lidar_depth_pub
//
// Run:
//   ros2 run lidar_depth_pub lidar_depth_pub --fov 25x90
//   ros2 run lidar_depth_pub lidar_depth_pub --fov 25x120 --port 5560 --downsample-rate 2
// ─────────────────────────────────────────────────────────────────────────────

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <rclcpp/rclcpp.hpp>

#include <zmq.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ── Constants ────────────────────────────────────────────────────────────────
// Spherical grid dimensions. Buffers reserve space for the widest layout;
// the active width/FOV is selected with the required --fov argument.
static constexpr int kGridRows = 25;
static constexpr int kMaxGridCols = 120;
static constexpr int kMaxGridSize = kGridRows * kMaxGridCols;
static int gGridCols = 0;
static int gGridSize = kGridRows * gGridCols;

// Real Mid360 expanded-FOV cleanup: mask image-bottom far-left/far-right
// triangular corners where the 25x120 projection can report spurious near-zero
// returns.
static constexpr int kWideBottomCornerMaskRows = 6;
static constexpr int kWideBottomCornerMaskMaxCols = 12;

// Vertical FOV: -4.5° to 45.5° (total 50°)
static constexpr float kThetaMinDeg = -4.5f;
static constexpr float kThetaMaxDeg = 45.5f;
static constexpr float kThetaTotalDeg = kThetaMaxDeg - kThetaMinDeg; // 50°

// Horizontal FOV is configured at startup.
static float gPhiMinDeg = 0.0f;
static float gPhiMaxDeg = 0.0f;
static float gPhiTotalDeg = gPhiMaxDeg - gPhiMinDeg;

// Precomputed radian values
static constexpr float kDeg2Rad = static_cast<float>(M_PI) / 180.0f;
static constexpr float kThetaMinRad = kThetaMinDeg * kDeg2Rad;
static constexpr float kThetaResRad = (kThetaTotalDeg * kDeg2Rad) / kGridRows;
static float gPhiMinRad = gPhiMinDeg * kDeg2Rad;
static float gPhiResRad = 0.0f;

// Sensor limits
static constexpr float kMaxDistance = 2.0f;
static constexpr float kMinDistance = 0.1f;

// Temporal accumulation
static constexpr int kDefaultNumStacked = 5; // default number of frames in sliding window
static constexpr double kPublishRate = 10.0; // publish rate in Hz

// Cage pillar occlusion mask parameters (matches Python SphericalDepthGenerator)
static float gPhiResDeg = 0.0f;
static constexpr float kCagePillarAnglesDeg[2] = {-45.0f, 45.0f};
static constexpr float kCagePillarJitterDeg = 2.0f;
// Tier definitions: (row_start, row_end, hw_top_deg, hw_bot_deg)
struct CageTier {
    int row_start, row_end;
    float hw_top_deg, hw_bot_deg;
};
static const CageTier kCageTiers[3] = {
    {0, kGridRows / 3, 3.0f, 3.5f},                 // top tier
    {kGridRows / 3, 2 * kGridRows / 3, 5.0f, 7.0f}, // middle tier
    {2 * kGridRows / 3, kGridRows, 7.0f, 11.0f},    // bottom tier
};

// Optional synthetic occlusion bands. Their thickness is based on the image
// height so it remains consistent across all supported horizontal FOVs.
static constexpr int kMaxExtraOcclusionBands = 2;
static constexpr float kBandWidthHeightFraction = 0.4f;

enum class BandPattern {
    Top,
    Bottom,
    Left,
    Right,
    DiagonalDown,
    DiagonalUp,
};

struct GridPoint {
    float row;
    float col;
};

// ── Signal handling ──────────────────────────────────────────────────────────
static volatile sig_atomic_t g_running = 1;
static void signalHandler(int) { g_running = 0; }

static size_t gridDataSize() { return static_cast<size_t>(gGridSize) * sizeof(float); }

static bool configureFov(const std::string &layout) {
    if (layout == "25x60") {
        gGridCols = 60;
    } else if (layout == "25x90") {
        gGridCols = 90;
    } else if (layout == "25x120") {
        gGridCols = 120;
    } else {
        return false;
    }

    gGridSize = kGridRows * gGridCols;
    gPhiMinDeg = -static_cast<float>(gGridCols);
    gPhiMaxDeg = static_cast<float>(gGridCols);
    gPhiTotalDeg = gPhiMaxDeg - gPhiMinDeg;
    gPhiMinRad = gPhiMinDeg * kDeg2Rad;
    gPhiResRad = (gPhiTotalDeg * kDeg2Rad) / gGridCols;
    gPhiResDeg = gPhiTotalDeg / gGridCols;
    return true;
}

static bool parsePositiveInt(const char *text, int *value) {
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 1 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

static bool parseBandPattern(const std::string &text, BandPattern *pattern) {
    if (text == "top") {
        *pattern = BandPattern::Top;
    } else if (text == "bottom") {
        *pattern = BandPattern::Bottom;
    } else if (text == "left") {
        *pattern = BandPattern::Left;
    } else if (text == "right") {
        *pattern = BandPattern::Right;
    } else if (text == "diagonal-down" || text == "diag-down") {
        *pattern = BandPattern::DiagonalDown;
    } else if (text == "diagonal-up" || text == "diag-up") {
        *pattern = BandPattern::DiagonalUp;
    } else {
        return false;
    }
    return true;
}

static const char *bandPatternName(BandPattern pattern) {
    switch (pattern) {
    case BandPattern::Top:
        return "top";
    case BandPattern::Bottom:
        return "bottom";
    case BandPattern::Left:
        return "left";
    case BandPattern::Right:
        return "right";
    case BandPattern::DiagonalDown:
        return "diagonal-down";
    case BandPattern::DiagonalUp:
        return "diagonal-up";
    }
    return "unknown";
}

// ── Cage pillar mask ─────────────────────────────────────────────────────────

// Build a boolean cage pillar occlusion mask. Two vertical stripes at ±45°
// azimuth, widest at the bottom rows, narrowing toward the top. Per-pillar
// jitter and a global scale factor are randomized once.
static void buildCageMask(bool *mask) {
    std::fill(mask, mask + gGridSize, false);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> jitter_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.8f, 1.2f);

    const float jitter_cols = kCagePillarJitterDeg / gPhiResDeg;
    const float scale = scale_dist(rng);

    int masked_count = 0;
    for (int p = 0; p < 2; ++p) {
        const float center =
            (kCagePillarAnglesDeg[p] - gPhiMinDeg) / gPhiResDeg + jitter_dist(rng) * jitter_cols;

        for (int t = 0; t < 3; ++t) {
            const auto &tier = kCageTiers[t];
            const int num_rows = tier.row_end - tier.row_start;
            for (int i = 0; i < num_rows; ++i) {
                const int r = tier.row_start + i;
                const float frac = static_cast<float>(i) / std::max(num_rows - 1, 1);
                const float hw_deg = tier.hw_top_deg + frac * (tier.hw_bot_deg - tier.hw_top_deg);
                const float hw_cols = (hw_deg * scale) / gPhiResDeg;
                for (int c = 0; c < gGridCols; ++c) {
                    if (std::abs(static_cast<float>(c) - center) <= hw_cols) {
                        mask[r * gGridCols + c] = true;
                        ++masked_count;
                    }
                }
            }
        }
    }
    std::cout << "[LidarDepth] Cage mask: " << masked_count << "/" << gGridSize
              << " pixels masked (" << 100.0f * masked_count / gGridSize << "%)" << std::endl;
}

// Add one or two thick, slightly curved quadratic Bezier bands to an existing
// mask. Each requested placement selects a region of the FOV; its exact
// position, width, endpoint tilt, and curvature are randomized once at startup.
static void addBezierBandMasks(bool *mask, const std::vector<BandPattern> &patterns) {
    if (patterns.empty())
        return;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> width_scale_dist(0.9f, 1.1f);
    std::uniform_real_distribution<float> unit_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> curve_magnitude_dist(0.04f, 0.10f);

    const float row_max = static_cast<float>(kGridRows - 1);
    const float col_max = static_cast<float>(gGridCols - 1);
    const float border_margin = std::max(1.5f, 0.06f * static_cast<float>(kGridRows));

    for (BandPattern pattern : patterns) {
        const float band_width =
            kBandWidthHeightFraction * static_cast<float>(kGridRows) * width_scale_dist(rng);
        const float half_width = 0.5f * band_width;
        const float safe_row_min = border_margin + half_width;
        const float safe_row_max = row_max - border_margin - half_width;
        const float row_jitter = unit_dist(rng) * 0.035f * static_cast<float>(kGridRows);
        const float col_jitter = unit_dist(rng) * 0.025f * static_cast<float>(gGridCols);
        const float endpoint_tilt = unit_dist(rng) * 0.04f * static_cast<float>(kGridRows);
        const float curve_sign = unit_dist(rng) < 0.0f ? -1.0f : 1.0f;

        GridPoint p0{};
        GridPoint p1{};
        GridPoint p2{};

        if (pattern == BandPattern::Top || pattern == BandPattern::Bottom) {
            const float top_center = std::clamp(
                0.32f * row_max + row_jitter, safe_row_min, 0.5f * row_max);
            const float center = pattern == BandPattern::Top ? top_center : row_max - top_center;
            const float along_margin =
                std::max(border_margin + half_width, 0.08f * col_max);
            p0 = {std::clamp(center - endpoint_tilt, safe_row_min, safe_row_max), along_margin};
            p2 = {std::clamp(center + endpoint_tilt, safe_row_min, safe_row_max),
                  col_max - along_margin};
            const float curve = curve_sign * curve_magnitude_dist(rng) * kGridRows;
            p1 = {std::clamp(center + curve, safe_row_min, safe_row_max),
                  0.5f * col_max + col_jitter};
        } else if (pattern == BandPattern::Left || pattern == BandPattern::Right) {
            const float safe_col_min = std::max(border_margin + half_width, 0.08f * col_max);
            const float safe_col_max = col_max - safe_col_min;
            const float left_center = std::clamp(
                0.20f * col_max + col_jitter, safe_col_min, 0.5f * col_max);
            const float center = pattern == BandPattern::Left ? left_center : col_max - left_center;
            const float along_margin = safe_row_min;
            const float vertical_tilt = unit_dist(rng) * 0.025f * static_cast<float>(gGridCols);
            p0 = {along_margin, std::clamp(center - vertical_tilt, safe_col_min, safe_col_max)};
            p2 = {row_max - along_margin,
                  std::clamp(center + vertical_tilt, safe_col_min, safe_col_max)};
            const float curve = curve_sign * curve_magnitude_dist(rng) * kGridRows;
            p1 = {0.5f * row_max + row_jitter,
                  std::clamp(center + curve, safe_col_min, safe_col_max)};
        } else {
            const float col_margin = std::max(border_margin + half_width, 0.08f * col_max);
            const float upper_row = safe_row_min;
            const float lower_row = safe_row_max;
            p0 = pattern == BandPattern::DiagonalDown
                     ? GridPoint{upper_row, col_margin}
                     : GridPoint{lower_row, col_margin};
            p2 = pattern == BandPattern::DiagonalDown
                     ? GridPoint{lower_row, col_max - col_margin}
                     : GridPoint{upper_row, col_max - col_margin};

            const float dr = p2.row - p0.row;
            const float dc = p2.col - p0.col;
            const float length = std::max(std::sqrt(dr * dr + dc * dc), 1.0f);
            const float curve = curve_sign * curve_magnitude_dist(rng) * kGridRows;
            p1 = {std::clamp(0.5f * (p0.row + p2.row) - dc / length * curve,
                             safe_row_min, safe_row_max),
                  std::clamp(0.5f * (p0.col + p2.col) + dr / length * curve,
                             col_margin, col_max - col_margin)};
        }

        // Sampling at four points per pixel gives a smooth enough centerline
        // for this small (at most 25x120) raster.
        const int samples = 4 * std::max(kGridRows, gGridCols);
        const float radius_sq = half_width * half_width;
        for (int i = 0; i <= samples; ++i) {
            const float t = static_cast<float>(i) / samples;
            const float one_minus_t = 1.0f - t;
            const float curve_row = one_minus_t * one_minus_t * p0.row +
                                    2.0f * one_minus_t * t * p1.row + t * t * p2.row;
            const float curve_col = one_minus_t * one_minus_t * p0.col +
                                    2.0f * one_minus_t * t * p1.col + t * t * p2.col;
            const int row_begin = std::max(0, static_cast<int>(std::floor(curve_row - half_width)));
            const int row_end =
                std::min(kGridRows - 1, static_cast<int>(std::ceil(curve_row + half_width)));
            const int col_begin = std::max(0, static_cast<int>(std::floor(curve_col - half_width)));
            const int col_end =
                std::min(gGridCols - 1, static_cast<int>(std::ceil(curve_col + half_width)));
            for (int r = row_begin; r <= row_end; ++r) {
                for (int c = col_begin; c <= col_end; ++c) {
                    const float dr = static_cast<float>(r) - curve_row;
                    const float dc = static_cast<float>(c) - curve_col;
                    if (dr * dr + dc * dc <= radius_sq)
                        mask[r * gGridCols + c] = true;
                }
            }
        }

        std::cout << "[LidarDepth] Extra occlusion band: " << bandPatternName(pattern)
                  << ", width=" << band_width << " px" << std::endl;
    }

    const int masked_count = static_cast<int>(std::count(mask, mask + gGridSize, true));
    std::cout << "[LidarDepth] Combined occlusion mask: " << masked_count << "/" << gGridSize
              << " pixels masked (" << 100.0f * masked_count / gGridSize << "%)" << std::endl;
}

// Apply the cage and optional band masks: set masked pixels to 1.0 (max range).
static void applyOcclusionMask(float *grid, const bool *mask) {
    for (int i = 0; i < gGridSize; ++i) {
        if (mask[i])
            grid[i] = 1.0f;
    }
}

// Apply only to the expanded 25x120 layout. The 25x60 and 25x90 layouts are
// intentionally unchanged.
static void applyWideBottomCornerMask(float *grid) {
    if (gGridCols != kMaxGridCols)
        return;

    for (int i = 0; i < kWideBottomCornerMaskRows; ++i) {
        const int r = kGridRows - kWideBottomCornerMaskRows + i;
        const int cols =
            static_cast<int>(std::ceil((i + 1) * static_cast<float>(kWideBottomCornerMaskMaxCols) /
                                       kWideBottomCornerMaskRows));
        for (int c = 0; c < cols; ++c) {
            grid[r * gGridCols + c] = 1.0f;
            grid[r * gGridCols + (gGridCols - 1 - c)] = 1.0f;
        }
    }
}

// ── Processing functions ─────────────────────────────────────────────────────

// Project filtered 3D points onto the configured spherical depth grid.
// Each bin keeps the minimum (closest) distance. Unfilled bins = max_distance.
// Grid is row-major: grid[theta_bin * gGridCols + phi_bin].
static void
projectToSphericalGrid(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg, float *grid, int downsample_rate) {
    // Initialize grid to max distance
    std::fill(grid, grid + gGridSize, kMaxDistance);

    const uint32_t step = static_cast<uint32_t>(std::max(downsample_rate, 1));
    for (uint32_t i = 0; i < msg->point_num; i += step) {
        const auto &pt = msg->points[i];
        const float x = pt.x;
        const float y = pt.y;
        const float z = pt.z;

        // Reject near-zero noise points
        if (std::abs(x) < 1e-3f && std::abs(y) < 1e-3f && std::abs(z) < 1e-3f)
            continue;

        const float r = std::sqrt(x * x + y * y + z * z);

        // Reject invalid / too close / too far
        if (!std::isfinite(r) || r <= kMinDistance || r > kMaxDistance)
            continue;

        const float theta = std::asin(z / r); // elevation
        const float phi = std::atan2(y, x);   // azimuth

        // Compute bin indices (round to nearest bin)
        const int theta_bin = static_cast<int>(std::round((theta - kThetaMinRad) / kThetaResRad));
        const int phi_bin = static_cast<int>(std::round((phi - gPhiMinRad) / gPhiResRad));

        // Reject out-of-bounds
        if (theta_bin < 0 || theta_bin >= kGridRows || phi_bin < 0 || phi_bin >= gGridCols)
            continue;

        // Scatter with min-reduce: keep closest hit per bin
        const int idx = theta_bin * gGridCols + phi_bin;
        grid[idx] = std::min(grid[idx], r);
    }
}

// Normalize grid: clamp to [0, max_distance], then divide by max_distance → [0, 1].
// 0.0 = at sensor (0m), 1.0 = at/beyond max range (2.0m).
static void normalizeGrid(float *grid) {
    const float inv_max = 1.0f / kMaxDistance;
    for (int i = 0; i < gGridSize; ++i) {
        grid[i] = std::min(std::max(grid[i], 0.0f) * inv_max, 1.0f);
    }
}

// Aggregate ring buffer by taking element-wise minimum across all frames.
// This densifies the sparse Livox temporal scan pattern.
static void accumulateMin(const float *hist, int num_frames, float *output) {
    std::copy(hist, hist + gGridSize, output);
    for (int f = 1; f < num_frames; ++f) {
        const float *frame = hist + static_cast<size_t>(f) * gGridSize;
        for (int i = 0; i < gGridSize; ++i) {
            output[i] = std::min(output[i], frame[i]);
        }
    }
}

// ── ROS2 Node ────────────────────────────────────────────────────────────────
class LidarDepthPub : public rclcpp::Node {
  public:
    LidarDepthPub(void *zmq_pub, const bool *occlusion_mask, int downsample_rate, int num_stacked)
        : Node("lidar_depth_pub")
        , zmq_pub_(zmq_pub)
        , downsample_rate_(std::max(downsample_rate, 1))
        , num_stacked_(std::max(num_stacked, 1))
        , hist_scans_(static_cast<size_t>(num_stacked_) * gGridSize, 1.0f) {
        // Copy the fixed mask (cage, optional bands, or both).
        std::copy(occlusion_mask, occlusion_mask + gGridSize, occlusion_mask_);

        // Pre-build ZMQ message buffer: [uint32 w][uint32 h][float32 × w × h]
        zmq_buf_.resize(kHeaderSize + gridDataSize());
        {
            uint32_t w = gGridCols, h = kGridRows;
            std::memcpy(zmq_buf_.data(), &w, sizeof(uint32_t));
            std::memcpy(zmq_buf_.data() + sizeof(uint32_t), &h, sizeof(uint32_t));
        }

        // Subscribe to Livox CustomMsg
        sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/livox/lidar", 10,
            std::bind(&LidarDepthPub::lidarCallback, this, std::placeholders::_1));

        // Create publish timer at configured rate
        auto period = std::chrono::duration<double>(1.0 / kPublishRate);
        pub_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&LidarDepthPub::publishTimerCallback, this));

        t0_ = std::chrono::steady_clock::now();
        RCLCPP_INFO(
            this->get_logger(),
            "Subscribed to /livox/lidar | Grid: %dx%d | FOV: [%.1f, %.1f]° × [%.1f, %.1f]° "
            "| Max dist: %.1fm | Window: %d frames | Downsample: every %d point(s) | "
            "Publish: %.0f Hz | ZMQ PUB ready",
            kGridRows, gGridCols, kThetaMinDeg, kThetaMaxDeg, gPhiMinDeg, gPhiMaxDeg, kMaxDistance,
            num_stacked_, downsample_rate_, kPublishRate);
    }

  private:
    // Called by ROS2 subscription — just buffer the frame
    void lidarCallback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
        // ── Step 1: Spherical projection ─────────────────────────────────────
        float frame[kMaxGridSize];
        projectToSphericalGrid(msg, frame, downsample_rate_);

        // ── Step 2: Normalize → [0, 1] ──────────────────────────────────────
        normalizeGrid(frame);

        // ── Step 2a: Mask expanded-FOV bottom corner artifacts ──────────────
        applyWideBottomCornerMask(frame);

        // ── Step 2b: Apply fixed cage/extra occlusion mask ──────────────────
        applyOcclusionMask(frame, occlusion_mask_);

        // ── Step 3: Push into ring buffer (sliding window) ───────────────────
        // Shift left: discard oldest (index 0), append new at end
        std::lock_guard<std::mutex> lock(buf_mutex_);
        if (num_stacked_ > 1) {
            std::memmove(hist_scans_.data(), hist_scans_.data() + gGridSize,
                         static_cast<size_t>(num_stacked_ - 1) * gridDataSize());
        }
        std::copy(frame, frame + gGridSize,
                  hist_scans_.data() + static_cast<size_t>(num_stacked_ - 1) * gGridSize);

        ++frame_count_;
        last_point_num_ = msg->point_num;
    }

    // Called by wall timer at 10 Hz — aggregate and publish
    void publishTimerCallback() {
        float aggregated[kMaxGridSize];
        {
            std::lock_guard<std::mutex> lock(buf_mutex_);
            accumulateMin(hist_scans_.data(), num_stacked_, aggregated);
        }

        std::memcpy(zmq_buf_.data() + kHeaderSize, aggregated, gridDataSize());
        zmq_send(zmq_pub_, zmq_buf_.data(), zmq_buf_.size(), ZMQ_DONTWAIT);

        ++publish_count_;

        // Print stats every 5 seconds
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0_).count();
        if (elapsed >= 5.0) {
            RCLCPP_INFO(
                this->get_logger(),
                "%lu frames in %.1fs (%.1f fps), %lu published (%.1f pub/s), "
                "last msg: %u points",
                frame_count_, elapsed, frame_count_ / elapsed, publish_count_,
                publish_count_ / elapsed, last_point_num_);
            frame_count_ = 0;
            publish_count_ = 0;
            t0_ = now;
        }
    }

    // ZMQ
    void *zmq_pub_;
    int downsample_rate_;
    int num_stacked_;
    static constexpr size_t kHeaderSize = sizeof(uint32_t) * 2;
    std::vector<uint8_t> zmq_buf_;

    // Fixed cage and optional Bezier-band occlusion mask
    bool occlusion_mask_[kMaxGridSize]{};

    // Ring buffer for temporal accumulation (sliding window of latest frames)
    std::mutex buf_mutex_;
    std::vector<float> hist_scans_;

    // ROS2
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_;
    rclcpp::TimerBase::SharedPtr pub_timer_;
    uint32_t last_point_num_ = 0;

    // Stats
    uint64_t frame_count_ = 0;
    uint64_t publish_count_ = 0;
    std::chrono::steady_clock::time_point t0_;
};

// ── ZMQ source mode (receives points from unitree_mujoco simulation) ────────
// Message format from unitree_mujoco:
//   [3 floats sensor_pos][9 floats sensor_rot][N*3 floats local_points]
// Local points are in sensor frame — ready for spherical projection.

static void projectRawPointsToGrid(
    const float *pts, int num_points, float *grid, int downsample_rate) {
    std::fill(grid, grid + gGridSize, kMaxDistance);

    const int step = std::max(downsample_rate, 1);
    for (int i = 0; i < num_points; i += step) {
        const float x = pts[i * 3 + 0];
        const float y = pts[i * 3 + 1];
        const float z = pts[i * 3 + 2];

        if (std::abs(x) < 1e-3f && std::abs(y) < 1e-3f && std::abs(z) < 1e-3f)
            continue;

        const float r = std::sqrt(x * x + y * y + z * z);

        if (!std::isfinite(r) || r <= kMinDistance || r > kMaxDistance)
            continue;

        const float theta = std::asin(z / r);
        const float phi = std::atan2(y, x);

        const int theta_bin = static_cast<int>(std::round((theta - kThetaMinRad) / kThetaResRad));
        const int phi_bin = static_cast<int>(std::round((phi - gPhiMinRad) / gPhiResRad));

        if (theta_bin < 0 || theta_bin >= kGridRows || phi_bin < 0 || phi_bin >= gGridCols)
            continue;

        const int idx = theta_bin * gGridCols + phi_bin;
        grid[idx] = std::min(grid[idx], r);
    }
}

static void runZmqSource(
    int sub_port, int pub_port, const bool *occlusion_mask, int downsample_rate, int num_stacked) {
    downsample_rate = std::max(downsample_rate, 1);
    num_stacked = std::max(num_stacked, 1);

    // ── ZMQ PUB (depth output) ──────────────────────────────────────────────
    void *zmq_ctx = zmq_ctx_new();
    void *zmq_pub = zmq_socket(zmq_ctx, ZMQ_PUB);
    int sndhwm = 2;
    zmq_setsockopt(zmq_pub, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));

    char pub_endpoint[64];
    snprintf(pub_endpoint, sizeof(pub_endpoint), "tcp://*:%d", pub_port);
    if (zmq_bind(zmq_pub, pub_endpoint) != 0) {
        std::cerr << "[LidarDepth] ZMQ PUB bind failed on " << pub_endpoint << ": "
                  << zmq_strerror(zmq_errno()) << std::endl;
        zmq_close(zmq_pub);
        zmq_ctx_destroy(zmq_ctx);
        return;
    }
    std::cout << "[LidarDepth] ZMQ PUB bound to " << pub_endpoint << std::endl;

    // ── ZMQ SUB (point cloud input from unitree_mujoco) ─────────────────────
    void *zmq_sub = zmq_socket(zmq_ctx, ZMQ_SUB);
    zmq_setsockopt(zmq_sub, ZMQ_SUBSCRIBE, "", 0);

    char sub_endpoint[64];
    snprintf(sub_endpoint, sizeof(sub_endpoint), "tcp://localhost:%d", sub_port);
    if (zmq_connect(zmq_sub, sub_endpoint) != 0) {
        std::cerr << "[LidarDepth] ZMQ SUB connect failed on " << sub_endpoint << ": "
                  << zmq_strerror(zmq_errno()) << std::endl;
        zmq_close(zmq_sub);
        zmq_close(zmq_pub);
        zmq_ctx_destroy(zmq_ctx);
        return;
    }
    std::cout << "[LidarDepth] ZMQ SUB connected to " << sub_endpoint << std::endl;

    // ── Ring buffer (same as ROS2 path: sliding window, shift-left) ─────────
    std::mutex buf_mutex;
    std::vector<float> hist_scans(static_cast<size_t>(num_stacked) * gGridSize, 1.0f);

    static constexpr size_t kHeaderSize = sizeof(uint32_t) * 2;
    // Header in incoming message: 3 floats pos + 9 floats rot = 12 floats
    static constexpr size_t kPoseFloats = 12;
    static constexpr size_t kPoseBytes = kPoseFloats * sizeof(float);

    uint64_t frame_count = 0;
    uint64_t publish_count = 0;
    uint32_t last_point_num = 0;
    auto t0 = std::chrono::steady_clock::now();

    std::cout << "[LidarDepth] Pipeline: ZMQ points → downsample every " << downsample_rate
              << " point(s) → filter → spherical " << kGridRows << "×" << gGridCols << " "
              << "→ normalize ÷ " << kMaxDistance << " → accumulate ×" << num_stacked
              << " min → ZMQ (" << kPublishRate << " Hz)" << std::endl;
    std::cout << "[LidarDepth] Running ZMQ source... (Ctrl+C to stop)" << std::endl;

    // ── Receiver thread: buffer incoming frames (like lidarCallback) ────────
    std::vector<uint8_t> recv_buf(1024 * 1024); // 1 MB max

    std::thread recv_thread([&]() {
        while (g_running) {
            int nbytes = zmq_recv(zmq_sub, recv_buf.data(), recv_buf.size(), 0);
            if (nbytes < 0) {
                if (zmq_errno() == EINTR)
                    continue;
                break;
            }

            size_t msg_size = static_cast<size_t>(nbytes);
            if (msg_size < kPoseBytes + 3 * sizeof(float))
                continue;

            // Skip pose header (12 floats), use local-frame points directly
            const float *pts = reinterpret_cast<const float *>(recv_buf.data() + kPoseBytes);
            int num_points = static_cast<int>((msg_size - kPoseBytes) / (3 * sizeof(float)));

            // ── Spherical projection ────────────────────────────────────────
            float frame[kMaxGridSize];
            projectRawPointsToGrid(pts, num_points, frame, downsample_rate);

            // ── Normalize ───────────────────────────────────────────────────
            normalizeGrid(frame);

            // ── Mask expanded-FOV bottom corner artifacts ───────────────────
            applyWideBottomCornerMask(frame);

            // ── Apply fixed cage/extra occlusion mask ──────────────────────
            applyOcclusionMask(frame, occlusion_mask);

            // ── Push into ring buffer (shift-left, same as ROS2 path) ───────
            {
                std::lock_guard<std::mutex> lock(buf_mutex);
                if (num_stacked > 1) {
                    std::memmove(hist_scans.data(), hist_scans.data() + gGridSize,
                                 static_cast<size_t>(num_stacked - 1) * gridDataSize());
                }
                std::copy(frame, frame + gGridSize,
                          hist_scans.data() + static_cast<size_t>(num_stacked - 1) * gGridSize);
                frame_count++;
                last_point_num = static_cast<uint32_t>(num_points);
            }
        }
    });

    // ── Publish timer: aggregate and publish at kPublishRate Hz ─────────────
    //    (mirrors publishTimerCallback in the ROS2 path)
    std::vector<uint8_t> zmq_buf(kHeaderSize + gridDataSize());
    {
        uint32_t w = gGridCols, h = kGridRows;
        std::memcpy(zmq_buf.data(), &w, sizeof(uint32_t));
        std::memcpy(zmq_buf.data() + sizeof(uint32_t), &h, sizeof(uint32_t));
    }

    const auto pub_period = std::chrono::duration<double>(1.0 / kPublishRate);
    auto next_pub = std::chrono::steady_clock::now() + pub_period;

    while (g_running) {
        std::this_thread::sleep_until(next_pub);
        next_pub += std::chrono::duration_cast<std::chrono::steady_clock::duration>(pub_period);

        // Aggregate ring buffer (same as publishTimerCallback)
        float aggregated[kMaxGridSize];
        {
            std::lock_guard<std::mutex> lock(buf_mutex);
            accumulateMin(hist_scans.data(), num_stacked, aggregated);
        }

        std::memcpy(zmq_buf.data() + kHeaderSize, aggregated, gridDataSize());
        zmq_send(zmq_pub, zmq_buf.data(), zmq_buf.size(), ZMQ_DONTWAIT);
        publish_count++;

        // ── Stats (every 5 seconds) ─────────────────────────────────────────
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (elapsed >= 5.0) {
            std::cout << "[LidarDepth] " << frame_count << " frames in " << elapsed << "s ("
                      << frame_count / elapsed << " fps), " << publish_count << " published ("
                      << publish_count / elapsed << " pub/s), last: " << last_point_num << " points"
                      << std::endl;
            frame_count = 0;
            publish_count = 0;
            t0 = now;
        }
    }

    // ── Cleanup ─────────────────────────────────────────────────────────────
    std::cout << "\n[LidarDepth] Shutting down ZMQ source..." << std::endl;
    recv_thread.join();
    zmq_close(zmq_sub);
    zmq_close(zmq_pub);
    zmq_ctx_destroy(zmq_ctx);
    std::cout << "[LidarDepth] Done." << std::endl;
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    // Set up signal handling for clean shutdown
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // ── Parse CLI args ──────────────────────────────────────────────────────
    int port = 5560;
    bool use_sim = false;
    int sim_port = 5590;
    bool enable_cage_mask = true;
    bool fov_specified = false;
    int downsample_rate = 1;
    int num_stacked = kDefaultNumStacked;
    std::vector<BandPattern> band_patterns;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--sim") {
            use_sim = true;
        } else if (arg == "--sim-port" && i + 1 < argc) {
            sim_port = std::atoi(argv[++i]);
        } else if (arg == "--downsample-rate" && i + 1 < argc) {
            const char *value = argv[++i];
            if (!parsePositiveInt(value, &downsample_rate)) {
                std::cerr << "[LidarDepth] Invalid --downsample-rate '" << value
                          << "'; using 1" << std::endl;
                downsample_rate = 1;
            }
        } else if (arg == "--stacked-frames") {
            if (i + 1 >= argc || !parsePositiveInt(argv[i + 1], &num_stacked)) {
                std::cerr << "[LidarDepth] --stacked-frames requires a positive integer"
                          << std::endl;
                return 1;
            }
            ++i;
        } else if (arg == "--fov") {
            if (i + 1 >= argc) {
                std::cerr << "[LidarDepth] --fov requires one of: 25x60, 25x90, 25x120"
                          << std::endl;
                return 1;
            }
            const std::string layout(argv[++i]);
            if (!configureFov(layout)) {
                std::cerr << "[LidarDepth] Invalid --fov '" << layout
                          << "'; expected 25x60, 25x90, or 25x120" << std::endl;
                return 1;
            }
            fov_specified = true;
        } else if (arg == "--no-cage-mask") {
            enable_cage_mask = false;
        } else if (arg == "--occlusion-pattern") {
            if (i + 1 >= argc) {
                std::cerr << "[LidarDepth] --occlusion-pattern requires one of: "
                          << "top, bottom, left, right, diagonal-down, diagonal-up" << std::endl;
                return 1;
            }
            if (band_patterns.size() >= kMaxExtraOcclusionBands) {
                std::cerr << "[LidarDepth] At most " << kMaxExtraOcclusionBands
                          << " --occlusion-pattern options are supported" << std::endl;
                return 1;
            }
            BandPattern pattern;
            const std::string value(argv[++i]);
            if (!parseBandPattern(value, &pattern)) {
                std::cerr << "[LidarDepth] Invalid --occlusion-pattern '" << value
                          << "'; expected top, bottom, left, right, diagonal-down, or diagonal-up"
                          << std::endl;
                return 1;
            }
            band_patterns.push_back(pattern);
        }
    }
    if (!fov_specified) {
        std::cerr << "[LidarDepth] Missing required --fov; expected 25x60, 25x90, or 25x120"
                  << std::endl;
        return 1;
    }
    std::cout << "[LidarDepth] Horizontal FOV: [" << gPhiMinDeg << ", " << gPhiMaxDeg
              << "] deg, " << gGridCols << " columns" << std::endl;
    std::cout << "[LidarDepth] Downsample rate: every " << downsample_rate << " point(s)"
              << std::endl;
    std::cout << "[LidarDepth] Stacked frames: " << num_stacked << std::endl;

    // Build cage pillar occlusion mask
    bool cage_mask[kMaxGridSize];
    std::fill(cage_mask, cage_mask + gGridSize, false);
    if (enable_cage_mask) {
        buildCageMask(cage_mask);
    } else {
        std::cout << "[LidarDepth] Cage mask disabled" << std::endl;
    }
    addBezierBandMasks(cage_mask, band_patterns);

    if (use_sim) {
        // ── ZMQ-to-ZMQ mode: receive from unitree_mujoco, publish depth ─────
        std::cout << "[LidarDepth] Using simulation source (ZMQ SUB port " << sim_port << ")"
                  << std::endl;
        runZmqSource(sim_port, port, cage_mask, downsample_rate, num_stacked);
        return 0;
    }

    // ── Default: ROS2 Livox source ──────────────────────────────────────────

    // ── Initialize ZMQ ───────────────────────────────────────────────────────
    void *zmq_ctx = zmq_ctx_new();
    void *zmq_pub = zmq_socket(zmq_ctx, ZMQ_PUB);
    int sndhwm = 2;
    zmq_setsockopt(zmq_pub, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));

    char endpoint[64];
    snprintf(endpoint, sizeof(endpoint), "tcp://*:%d", port);
    if (zmq_bind(zmq_pub, endpoint) != 0) {
        std::cerr << "[LidarDepth] ZMQ bind failed on " << endpoint << ": "
                  << zmq_strerror(zmq_errno()) << std::endl;
        return 1;
    }
    std::cout << "[LidarDepth] ZMQ PUB bound to " << endpoint << std::endl;

    // ── Initialize ROS2 ──────────────────────────────────────────────────────
    rclcpp::init(argc, argv);
    auto node =
        std::make_shared<LidarDepthPub>(zmq_pub, cage_mask, downsample_rate, num_stacked);

    std::cout << "[LidarDepth] Pipeline: CustomMsg → downsample every " << downsample_rate
              << " point(s) → filter → spherical " << kGridRows << "×" << gGridCols << " "
              << "→ normalize ÷ " << kMaxDistance << " → accumulate ×" << num_stacked
              << " min → ZMQ" << std::endl;
    std::cout << "[LidarDepth] Running... (Ctrl+C to stop)" << std::endl;

    rclcpp::spin(node);

    // ── Cleanup ──────────────────────────────────────────────────────────────
    std::cout << "\n[LidarDepth] Shutting down..." << std::endl;
    rclcpp::shutdown();
    zmq_close(zmq_pub);
    zmq_ctx_destroy(zmq_ctx);
    std::cout << "[LidarDepth] Done." << std::endl;

    return 0;
}
