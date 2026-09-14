/*
This file is part of JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion

Copyright (c) 2026 Qihao Yuan

Developer: Qihao Yuan <qihao.yuan@rug.nl>

For commercial use, please contact me at <qihao.yuan@rug.nl> or Kailai Li at <kailai.li@liu.se>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <zlib.h>

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

// Argument parser
#include "cxxopts.hpp"

// JSON config
#include "json.hpp"

// ONNX Runtime headers
#include "onnxruntime_cxx_api.h"

// ZMQ
#include "zmq.hpp"

#include "advanced_gamepad.hpp"

// Unitree SDK headers
#include "unitree/common/json/json.hpp"
#include "unitree/common/time/time_tool.hpp"
#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/idl/go2/WirelessController_.hpp"
#include "unitree/robot/b2/motion_switcher/motion_switcher_api.hpp"
#include "unitree/robot/b2/motion_switcher/motion_switcher_client.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"

using namespace std::chrono_literals;
using namespace unitree::common;
using namespace unitree::robot::b2;
using namespace unitree::robot;
using namespace unitree;
using namespace Eigen;

#define LOG_INFO(...)                                                                            \
    do {                                                                                         \
        std::fprintf(stdout, "[INFO] ");                                                         \
        std::fprintf(stdout, __VA_ARGS__);                                                       \
        std::fputc('\n', stdout);                                                               \
    } while (false)
#define LOG_WARN(...)                                                                            \
    do {                                                                                         \
        std::fprintf(stderr, "[WARN] ");                                                         \
        std::fprintf(stderr, __VA_ARGS__);                                                       \
        std::fputc('\n', stderr);                                                               \
    } while (false)
#define LOG_ERROR(...)                                                                           \
    do {                                                                                         \
        std::fprintf(stderr, "[ERROR] ");                                                        \
        std::fprintf(stderr, __VA_ARGS__);                                                       \
        std::fputc('\n', stderr);                                                               \
    } while (false)

#define LOG_THROTTLED(log_macro, interval_ms, ...)                                               \
    do {                                                                                         \
        static auto last_log_time = std::chrono::steady_clock::time_point{};                     \
        const auto log_now = std::chrono::steady_clock::now();                                  \
        if (log_now - last_log_time >= std::chrono::milliseconds(interval_ms)) {                 \
            log_macro(__VA_ARGS__);                                                              \
            last_log_time = log_now;                                                            \
        }                                                                                        \
    } while (false)
#define LOG_WARN_THROTTLED(interval_ms, ...) LOG_THROTTLED(LOG_WARN, interval_ms, __VA_ARGS__)

std::atomic<bool> g_stop_control_signal(false);
std::atomic<bool> g_emergency_stop_mode(false);

namespace {

struct OnnxTestArray {
    std::vector<int64_t> shape;
    std::vector<float> values;
};

struct OnnxTestStats {
    bool ok = true;
    size_t records = 0;
    double worst_max_abs = 0.0;
    double worst_mean_abs = 0.0;
    std::string worst_sample = "-";
    std::string worst_tensor = "-";
};

struct OnnxValidationResult {
    OnnxTestStats policy;
    OnnxTestStats estimator;
};

uint16_t read_le16(const std::vector<uint8_t> &buffer, size_t offset) {
    if (offset + 2 > buffer.size()) {
        throw std::runtime_error("truncated uint16");
    }
    return static_cast<uint16_t>(buffer[offset]) |
           (static_cast<uint16_t>(buffer[offset + 1]) << 8);
}

uint32_t read_le32(const std::vector<uint8_t> &buffer, size_t offset) {
    if (offset + 4 > buffer.size()) {
        throw std::runtime_error("truncated uint32");
    }
    return static_cast<uint32_t>(buffer[offset]) |
           (static_cast<uint32_t>(buffer[offset + 1]) << 8) |
           (static_cast<uint32_t>(buffer[offset + 2]) << 16) |
           (static_cast<uint32_t>(buffer[offset + 3]) << 24);
}

std::vector<uint8_t> read_binary_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(fmt::format("failed to open {}", path));
    }
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0) {
        throw std::runtime_error(fmt::format("failed to size {}", path));
    }
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty()) {
        in.read(reinterpret_cast<char *>(data.data()), size);
    }
    return data;
}

std::string read_text_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(fmt::format("failed to open {}", path));
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::vector<uint8_t>
inflate_raw_deflate(const uint8_t *src, size_t src_size, size_t expected_size) {
    std::vector<uint8_t> out(expected_size);
    z_stream stream{};
    stream.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(src));
    stream.avail_in = static_cast<uInt>(src_size);
    stream.next_out = reinterpret_cast<Bytef *>(out.data());
    stream.avail_out = static_cast<uInt>(out.size());
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
        throw std::runtime_error("inflateInit2 failed");
    }
    int ret = inflate(&stream, Z_FINISH);
    inflateEnd(&stream);
    if (ret != Z_STREAM_END || stream.total_out != expected_size) {
        throw std::runtime_error("deflate stream did not inflate to expected size");
    }
    return out;
}

std::vector<int64_t> parse_npy_shape(const std::string &header) {
    size_t key = header.find("shape");
    size_t left = header.find('(', key);
    size_t right = header.find(')', left);
    if (key == std::string::npos || left == std::string::npos || right == std::string::npos) {
        throw std::runtime_error("npy header missing shape");
    }
    std::vector<int64_t> shape;
    size_t i = left + 1;
    while (i < right) {
        while (i < right && (header[i] == ' ' || header[i] == ',')) {
            ++i;
        }
        size_t j = i;
        while (j < right && header[j] >= '0' && header[j] <= '9') {
            ++j;
        }
        if (j > i) {
            shape.push_back(std::stoll(header.substr(i, j - i)));
        }
        i = j + 1;
    }
    return shape;
}

OnnxTestArray parse_npy_float32(const std::vector<uint8_t> &npy, const std::string &name) {
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (npy.size() < 10 || std::memcmp(npy.data(), magic, sizeof(magic)) != 0) {
        throw std::runtime_error(fmt::format("{} is not a npy array", name));
    }
    uint8_t major = npy[6];
    size_t header_len = 0;
    size_t data_offset = 0;
    if (major == 1) {
        header_len = read_le16(npy, 8);
        data_offset = 10;
    } else if (major == 2 || major == 3) {
        header_len = read_le32(npy, 8);
        data_offset = 12;
    } else {
        throw std::runtime_error(fmt::format("{} has unsupported npy version {}", name, major));
    }
    if (data_offset + header_len > npy.size()) {
        throw std::runtime_error(fmt::format("{} has truncated npy header", name));
    }
    std::string header(
        reinterpret_cast<const char *>(npy.data() + data_offset), header_len);
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        throw std::runtime_error(fmt::format("{} is not float32", name));
    }
    if (header.find("True") != std::string::npos) {
        throw std::runtime_error(fmt::format("{} is Fortran-order; unsupported", name));
    }

    OnnxTestArray array;
    array.shape = parse_npy_shape(header);
    int64_t count = 1;
    for (int64_t dim : array.shape) {
        count *= dim;
    }
    size_t byte_count = static_cast<size_t>(count) * sizeof(float);
    size_t payload_offset = data_offset + header_len;
    if (payload_offset + byte_count > npy.size()) {
        throw std::runtime_error(fmt::format("{} has truncated npy payload", name));
    }
    array.values.resize(static_cast<size_t>(count));
    std::memcpy(array.values.data(), npy.data() + payload_offset, byte_count);
    return array;
}

std::map<std::string, OnnxTestArray> load_npz_float_arrays(const std::string &path) {
    std::vector<uint8_t> zip = read_binary_file(path);
    if (zip.size() < 22) {
        throw std::runtime_error(fmt::format("{} is too small to be npz", path));
    }
    size_t min_eocd = zip.size() > 65557 ? zip.size() - 65557 : 0;
    size_t eocd = std::string::npos;
    for (size_t pos = zip.size() - 22; pos + 1 > min_eocd; --pos) {
        if (read_le32(zip, pos) == 0x06054b50U) {
            eocd = pos;
            break;
        }
        if (pos == 0) {
            break;
        }
    }
    if (eocd == std::string::npos) {
        throw std::runtime_error(fmt::format("{} missing zip central directory", path));
    }

    uint16_t entries = read_le16(zip, eocd + 10);
    uint32_t central_dir_size = read_le32(zip, eocd + 12);
    uint32_t central_dir_offset = read_le32(zip, eocd + 16);
    if (static_cast<size_t>(central_dir_offset) + central_dir_size > zip.size()) {
        throw std::runtime_error(fmt::format("{} has invalid zip central directory", path));
    }

    std::map<std::string, OnnxTestArray> arrays;
    size_t pos = central_dir_offset;
    for (uint16_t i = 0; i < entries; ++i) {
        if (read_le32(zip, pos) != 0x02014b50U) {
            throw std::runtime_error(fmt::format("{} has invalid central directory entry", path));
        }
        uint16_t method = read_le16(zip, pos + 10);
        uint32_t compressed_size = read_le32(zip, pos + 20);
        uint32_t uncompressed_size = read_le32(zip, pos + 24);
        uint16_t name_len = read_le16(zip, pos + 28);
        uint16_t extra_len = read_le16(zip, pos + 30);
        uint16_t comment_len = read_le16(zip, pos + 32);
        uint32_t local_offset = read_le32(zip, pos + 42);
        std::string file_name(
            reinterpret_cast<const char *>(zip.data() + pos + 46), name_len);
        pos += 46 + name_len + extra_len + comment_len;

        if (file_name.size() < 5 || file_name.substr(file_name.size() - 4) != ".npy") {
            continue;
        }
        if (read_le32(zip, local_offset) != 0x04034b50U) {
            throw std::runtime_error(fmt::format("{} has invalid local header", path));
        }
        uint16_t local_name_len = read_le16(zip, local_offset + 26);
        uint16_t local_extra_len = read_le16(zip, local_offset + 28);
        size_t data_offset = local_offset + 30 + local_name_len + local_extra_len;
        if (data_offset + compressed_size > zip.size()) {
            throw std::runtime_error(fmt::format("{} has truncated zip data", path));
        }
        std::vector<uint8_t> npy;
        if (method == 0) {
            npy.assign(zip.begin() + data_offset, zip.begin() + data_offset + compressed_size);
        } else if (method == 8) {
            npy = inflate_raw_deflate(zip.data() + data_offset, compressed_size, uncompressed_size);
        } else {
            throw std::runtime_error(fmt::format("{} uses unsupported zip method {}", path, method));
        }
        std::string array_name = file_name.substr(0, file_name.size() - 4);
        try {
            arrays[array_name] = parse_npy_float32(npy, array_name);
        } catch (const std::exception &) {
            // Ignore integer metadata arrays such as env_id and play_step.
        }
    }
    return arrays;
}

std::vector<std::string> list_npz_samples(const std::filesystem::path &dir) {
    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error(fmt::format("sample directory not found: {}", dir.string()));
    }
    std::vector<std::string> samples;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string name = entry.path().filename().string();
            if (name.rfind("sample_", 0) == 0 && entry.path().extension() == ".npz") {
                samples.push_back(entry.path().string());
            }
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples;
}

std::vector<std::string> json_string_list(const nlohmann::json &items) {
    std::vector<std::string> names;
    for (const auto &item : items) {
        names.push_back(item.get<std::string>());
    }
    return names;
}

std::vector<const char *> cstr_names(const std::vector<std::string> &names) {
    std::vector<const char *> out;
    for (const auto &name : names) {
        out.push_back(name.c_str());
    }
    return out;
}

std::vector<int64_t>
get_session_io_shape(Ort::Session &session, const std::string &name, bool is_input) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t count = is_input ? session.GetInputCount() : session.GetOutputCount();
    for (size_t i = 0; i < count; ++i) {
        auto io_name = is_input ? session.GetInputNameAllocated(i, allocator)
                                : session.GetOutputNameAllocated(i, allocator);
        if (name == io_name.get()) {
            auto type_info = is_input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
            return type_info.GetTensorTypeAndShapeInfo().GetShape();
        }
    }
    throw std::runtime_error(
        fmt::format("ONNX session does not have {} named '{}'", is_input ? "input" : "output", name));
}

int64_t shape_count(const std::vector<int64_t> &shape) {
    int64_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return -1;
        }
        count *= dim;
    }
    return count;
}

std::vector<int64_t> tensor_shape_for_values(
    const std::vector<int64_t> &sample_shape, const std::vector<int64_t> &onnx_shape,
    size_t value_count) {
    if (shape_count(onnx_shape) == static_cast<int64_t>(value_count)) {
        return onnx_shape;
    }

    if (!onnx_shape.empty() && onnx_shape.size() <= sample_shape.size()) {
        std::vector<int64_t> resolved = onnx_shape;
        const size_t sample_offset = sample_shape.size() - onnx_shape.size();
        for (size_t i = 0; i < resolved.size(); ++i) {
            if (resolved[i] <= 0) {
                resolved[i] = sample_shape[sample_offset + i];
            }
        }
        if (shape_count(resolved) == static_cast<int64_t>(value_count)) {
            return resolved;
        }
    }

    return sample_shape;
}

std::vector<float> tensor_to_vector(Ort::Value &value, size_t count) {
    float *ptr = value.GetTensorMutableData<float>();
    return std::vector<float>(ptr, ptr + count);
}

void update_test_stats(
    OnnxTestStats &stats, const std::string &sample_path, const std::string &tensor_name,
    const std::vector<float> &actual, const std::vector<float> &expected, double max_abs_tol,
    double mean_abs_tol) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            fmt::format("{} shape mismatch for {}", sample_path, tensor_name));
    }
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double diff = std::abs(static_cast<double>(actual[i]) - expected[i]);
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
    }
    double mean_abs = actual.empty() ? 0.0 : sum_abs / static_cast<double>(actual.size());
    if (max_abs > stats.worst_max_abs || mean_abs > stats.worst_mean_abs) {
        stats.worst_max_abs = std::max(stats.worst_max_abs, max_abs);
        stats.worst_mean_abs = std::max(stats.worst_mean_abs, mean_abs);
        stats.worst_sample = sample_path;
        stats.worst_tensor = tensor_name;
    }
    if (max_abs > max_abs_tol || mean_abs > mean_abs_tol) {
        stats.ok = false;
    }
}

OnnxTestStats validate_model_test_cases(
    Ort::Session &session, const std::filesystem::path &sample_dir,
    const std::vector<std::string> &input_names, const std::vector<std::string> &output_names,
    double max_abs_tol, double mean_abs_tol) {
    std::vector<std::string> samples = list_npz_samples(sample_dir);
    OnnxTestStats stats;
    if (samples.empty()) {
        stats.ok = false;
        return stats;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<const char *> input_name_ptrs = cstr_names(input_names);
    std::vector<const char *> output_name_ptrs = cstr_names(output_names);

    for (const std::string &sample : samples) {
        auto arrays = load_npz_float_arrays(sample);
        std::vector<Ort::Value> input_tensors;
        for (const auto &name : input_names) {
            auto it = arrays.find(name);
            if (it == arrays.end()) {
                throw std::runtime_error(fmt::format("{} missing input {}", sample, name));
            }
            std::vector<int64_t> tensor_shape = tensor_shape_for_values(
                it->second.shape, get_session_io_shape(session, name, true),
                it->second.values.size());
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, it->second.values.data(), it->second.values.size(),
                tensor_shape.data(), tensor_shape.size()));
        }

        auto outputs = session.Run(
            Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_tensors.data(),
            input_tensors.size(), output_name_ptrs.data(), output_name_ptrs.size());
        ++stats.records;
        for (size_t i = 0; i < output_names.size(); ++i) {
            const auto &name = output_names[i];
            auto it = arrays.find(name);
            if (it == arrays.end()) {
                throw std::runtime_error(fmt::format("{} missing output {}", sample, name));
            }
            update_test_stats(
                stats, sample, name, tensor_to_vector(outputs[i], it->second.values.size()),
                it->second.values, max_abs_tol, mean_abs_tol);
        }
    }
    return stats;
}

} // namespace

void interrupt_handler(int /*signal_num*/) {
    static bool first_interrupt = true;
    if (first_interrupt) {
        g_emergency_stop_mode.store(true, std::memory_order_relaxed);
        first_interrupt = false;
        fmt::print("\n=== EMERGENCY STOP ACTIVATED ===\n");
        fmt::print("Switching to locomotion policy with zero velocity commands\n");
        fmt::print("Robot will attempt to stand in place\n");
        fmt::print("Press Ctrl+C again to fully stop the program\n");
    } else {
        g_stop_control_signal.store(true, std::memory_order_relaxed);
        fmt::print("Second Ctrl+C detected - stopping control...\n");
    }
}

/**
 * @brief Get gravity orientation from quaternion (equivalent to Python's get_gravity_orientation)
 * @param quat Quaternion representing orientation
 * @return Vector3f representing gravity orientation
 */
Vector3f get_gravity_orientation_eigen(const Quaternionf &quat) {
    Vector3f gravity_orientation;
    gravity_orientation.x() = 2.0f * (-quat.z() * quat.x() + quat.w() * quat.y());
    gravity_orientation.y() = -2.0f * (quat.z() * quat.y() + quat.w() * quat.x());
    gravity_orientation.z() = 1.0f - 2.0f * (quat.w() * quat.w() + quat.z() * quat.z());

    return gravity_orientation;
}

uint32_t crc32_core(uint32_t *ptr, uint32_t len) {
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}

template <typename Iterable> auto enumerate(Iterable &&iterable) {
    using Iterator = decltype(std::begin(std::declval<Iterable>()));
    using T = decltype(*std::declval<Iterator>());

    struct Enumerated {
        std::size_t index;
        T element;
    };

    struct Enumerator {
        Iterator iterator;
        std::size_t index;

        auto operator!=(const Enumerator &other) const { return iterator != other.iterator; }

        auto &operator++() {
            ++iterator;
            ++index;
            return *this;
        }

        auto operator*() const { return Enumerated{index, *iterator}; }
    };

    struct Wrapper {
        Iterable &iterable;

        [[nodiscard]] auto begin() const { return Enumerator{std::begin(iterable), 0U}; }

        [[nodiscard]] auto end() const { return Enumerator{std::end(iterable), 0U}; }
    };

    return Wrapper{std::forward<Iterable>(iterable)};
}

// Configuration for Go2 robot (world-model locomotion)
struct LocomotionConfig {
    int control_freq = 50;
    float control_dt = 0.02f;

    std::string lowcmd_topic = "rt/lowcmd";
    std::string lowstate_topic = "rt/lowstate";

    // Model paths and names
    // ONNX files are looked up under <exe_dir>/model/<model_subdir>/ using the
    // fixed default file names below. Set model_subdir to "" to use <exe_dir>/model/ directly.
    std::string model_subdir = "";
    std::string policy_filename = "policy.onnx";
    std::string sensor_estimator_filename = "sensor_estimator.onnx";
    std::string policy_path;
    std::string sensor_estimator_path;
    bool trt_fp16_enabled = false;
    bool trt_cache_enabled = false;

    float kp = 32.0f;
    float kd = 1.0f;

    // Joint index mapping
    std::array<int, 12> joint2motor_idx = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8};

    // Default joint angles
    VectorXf default_angles;
    VectorXf crouch_angles;

    // ordered joint names in simulation (model input)
    std::vector<std::string> sim_joint_names = {
        "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
    };

    std::vector<std::string> motor_joint_names = {
        "FR_hip_joint",   "FR_thigh_joint", "FR_calf_joint",  "FL_hip_joint",
        "FL_thigh_joint", "FL_calf_joint",  "RR_hip_joint",   "RR_thigh_joint",
        "RR_calf_joint",  "RL_hip_joint",   "RL_thigh_joint", "RL_calf_joint",
    };

    std::unordered_map<std::string, float> default_joint_angles = {
        {"FL_hip_joint", 0.1},   {"RL_hip_joint", 0.1},   {"FR_hip_joint", -0.1},
        {"RR_hip_joint", -0.1},  {"FL_thigh_joint", 0.8}, {"RL_thigh_joint", 1.0},
        {"FR_thigh_joint", 0.8}, {"RR_thigh_joint", 1.0}, {"FL_calf_joint", -1.5},
        {"RL_calf_joint", -1.5}, {"FR_calf_joint", -1.5}, {"RR_calf_joint", -1.5},
    };

    std::unordered_map<std::string, float> crouch_joint_angles = {
        {"FL_hip_joint", 0.125},  {"FL_thigh_joint", 1.23}, {"FL_calf_joint", -2.70},
        {"FR_hip_joint", -0.125}, {"FR_thigh_joint", 1.23}, {"FR_calf_joint", -2.70},
        {"RL_hip_joint", 0.47},   {"RL_thigh_joint", 1.25}, {"RL_calf_joint", -2.72},
        {"RR_hip_joint", -0.47},  {"RR_thigh_joint", 1.25}, {"RR_calf_joint", -2.72},
    };

    // Joint limits
    std::vector<std::pair<float, float>> joint_limits = {
        {-1.0472, 1.0472},  {-1.0472, 1.0472},  {-1.0472, 1.0472},  {-1.0472, 1.0472},
        {-1.5708, 3.4907},  {-1.5708, 3.4907},  {-0.5236, 4.5379},  {-0.5236, 4.5379},
        {-2.7227, -0.8378}, {-2.7227, -0.8378}, {-2.7227, -0.8378}, {-2.7227, -0.8378},
    };

    // Observation scales
    float lin_vel_scale = 2.0f;
    float ang_vel_scale = 0.25f;
    float dof_pos_scale = 1.0f;
    float dof_vel_scale = 0.05f;
    float action_obs_scale = 0.1f;
    float obs_clip = 100.0f;

    float forward_clip = 2.5;
    float lateral_clip = 1.5;
    float angular_clip = 3.0;

    // Safety/OOD detection thresholds
    float safety_max_tau_threshold = 30.0f;
    float safety_max_joint_vel_threshold = 25.0f;
    int safety_ood_count_threshold = 20;

    // Action scales
    float action_scale = 0.25f;
    bool use_raw_actions = false;

    // Network dimensions
    int num_actions = 12;
    int num_proprio = 45;

    // Proprioception input dimensions
    int proprio_dim = 45;
    int prop_hist_dim = 450;

    // Sensor estimator dimensions are detected from the ONNX models at load time.
    // The recurrent state and policy feature can have different sizes.
    int wm_update_interval = 5;
    int gru_hidden_dim = 768;                     // GRU hidden / hidden_out size
    int sensor_latent_dim = 768;                  // "feature" / policy `high_feat` size
    int policy_window = 5;                        // policy high_feat_window length
    std::array<int, 3> sensor_size = {2, 25, 60}; // C, H, W (channels-first)

    // Policy GRU dimensions (auto-detected from policy ONNX at load time)
    int rnn_num_layers = 1;
    int rnn_hidden_dim = 512;

    // ZMQ ports
    int depth_sub_port = 5560;
    std::string depth_sub_host = "localhost";
    int cmd_sub_port = 5562;
    std::string cmd_sub_host = "localhost";

    bool use_joystick_gamepad_control = true;
    float gamepad_linear_scale = 1.0f;
    float gamepad_linear_scale_2 = 1.0f;
    float gamepad_angular_scale = 1.0f;
    bool use_sim = false;

    void init() {
        fmt::print("joint2motor_idx: ");
        for (auto &&[i, name] : enumerate(sim_joint_names)) {
            auto index = std::find(motor_joint_names.begin(), motor_joint_names.end(), name) -
                         motor_joint_names.begin();
            joint2motor_idx.at(i) = index;
            fmt::print("{} ", index);
        }
        fmt::print("\nshould be:       3 0 9 6 4 1 10 7 5 2 11 8\n");

        default_angles.resize(num_actions);
        crouch_angles.resize(num_actions);
        for (int i = 0; i < num_actions; ++i) {
            default_angles(i) = default_joint_angles.at(sim_joint_names.at(i));
            crouch_angles(i) = crouch_joint_angles.at(sim_joint_names.at(i));
        }

        // Get the executable directory
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        std::string exe_path = std::string(result, (count > 0) ? count : 0);
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();

        // Set model paths
        std::filesystem::path model_dir = exe_dir / "model";
        if (!model_subdir.empty()) {
            model_dir /= model_subdir;
        }
        policy_path = (model_dir / policy_filename).string();
        sensor_estimator_path = (model_dir / sensor_estimator_filename).string();
        fmt::print("Model directory: {}\n", model_dir.string());
        fmt::print(
            "Policy config: proprio_dim={}, prop_hist_dim={}, sensor_latent_dim={}, "
            "rnn_layers={}, rnn_hidden={}\n",
            proprio_dim, prop_hist_dim, sensor_latent_dim, rnn_num_layers, rnn_hidden_dim);
        fmt::print(
            "Estimator config: wm_update_interval={}, gru_hidden_dim={}, "
            "sensor_size=[{},{},{}]\n",
            wm_update_interval, gru_hidden_dim, sensor_size[0], sensor_size[1], sensor_size[2]);

        // Modify joint limits to reduce range
        for (auto &limit : joint_limits) {
            float range = limit.second - limit.first;
            limit.first += 0.05f * range;
            limit.second -= 0.05f * range;
            if (limit.second <= limit.first) {
                throw std::runtime_error("Joint limits are invalid after modification.");
            }
        }
    }
};

/**
 * @brief World-model locomotion control node.
 *
 * Uses two ONNX models:
 *   - Policy: runs every control step (50 Hz by default)
 *   - Sensor estimator (GRU): runs every wm_update_interval steps (10 Hz by default)
 *
 * Receives depth images via ZMQ SUB.
 * Receives velocity commands via ZMQ SUB.
 */
class LocomotionNode {
  public:
    LocomotionNode(LocomotionConfig config)
        : config_(config)
        , running_(false)
        , crouching_(false)
        , initialized_(false)
        , depth_recv_running_(false) {
        // Initialize ONNX environment once
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LocomotionNodeWM");

        // Load configuration
        load_config();

        // Initialize controller
        initialize_controller();

        // Initialize ZMQ after ONNX shapes have sized the depth buffers.
        initialize_zmq();

        LOG_INFO("LocomotionNode (WM) initialized");
    }

    ~LocomotionNode() {
        // Stop depth receiver thread
        depth_recv_running_.store(false, std::memory_order_relaxed);
        if (depth_recv_thread_.joinable()) {
            depth_recv_thread_.join();
        }

        // Ensure we send zero commands when shutting down
        fmt::print("shutting down locomotion node (WM)...\n");
        if (initialized_) {
            create_zero_cmd();
            send_cmd();
            LOG_INFO("Locomotion control shut down");
        }
    }

    void run_control_loop() {
        fmt::print("attempting to start locomotion control loop...\n");

        if (!initialized_) {
            fmt::print("quit locomotion control loop.\n");
            return;
        }

        start_control_sequence();

        if (config_.control_freq <= 0) {
            throw std::runtime_error("control frequency must be positive");
        }
        const auto control_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / static_cast<double>(config_.control_freq)));
        auto next_cycle = std::chrono::steady_clock::now() + control_period;
        auto wait_for_next_cycle = [&]() {
            const auto now = std::chrono::steady_clock::now();
            if (now < next_cycle) {
                std::this_thread::sleep_until(next_cycle);
            } else if (now - next_cycle >= control_period) {
                // Do not run burst iterations after missing more than one complete cycle.
                next_cycle = now;
            }
            next_cycle += control_period;
        };

        while (true) {
            if (g_stop_control_signal.load(std::memory_order_relaxed)) {
                fmt::print("stopping locomotion control...\n");
                break;
            }

            if (depth_size_mismatch_.load(std::memory_order_relaxed)) {
                LOG_ERROR("Stopping control: incoming depth image size does not match model-required "
                    "size %dx%d",
                    config_.sensor_size[2], config_.sensor_size[1]);
                break;
            }

            // Wait for at least one depth frame before sending any commands
            if (!depth_frame_received_) {
                static int wait_log = 0;
                if (wait_log++ % 50 == 0)
                    fmt::print("Waiting for depth images before sending commands...\n");
                wait_for_next_cycle();
                continue;
            }

            run_controller();

            static int log_counter = 0;
            if (log_counter++ % 50 == 0) {
                fmt::print(
                    "Current cmd: x: {:.2f}, y: {:.2f}, z: {:.2f}\n", cmd_(0), cmd_(1), cmd_(2));
            }

            wait_for_next_cycle();
        }

        stop_control_sequence();
    }

  private:
    // ==================== Configuration ====================

    void load_config() {
        // Initialize controller state variables
        qj_.resize(config_.num_actions);
        qj_.setZero();
        dqj_.resize(config_.num_actions);
        dqj_.setZero();
        raw_action_.resize(config_.num_actions, 0.0f);
        target_dof_pos_ = config_.default_angles;
        previous_target_dof_pos_ = config_.default_angles;
        cmd_ = {0.0f, 0.0f, 0.0f};

        proprio_obs_.resize(config_.num_proprio, 0.0f);

        // Initialize proprioception history buffer (newest first)
        prop_hist_.resize(config_.prop_hist_dim, 0.0f);
        policy_proprio_dims_ = {1, config_.proprio_dim};
        policy_prop_hist_dims_ = {1, config_.prop_hist_dim};
        policy_depth_dims_ = {
            1, config_.sensor_size[0], config_.sensor_size[1], config_.sensor_size[2]};
        policy_high_feat_dims_ = {1, config_.sensor_latent_dim};
        policy_high_feat_window_dims_ = {
            1, config_.policy_window, config_.sensor_latent_dim};
        policy_gru_dims_ = {config_.rnn_num_layers, 1, config_.rnn_hidden_dim};
        estimator_prop_hist_dims_ = {1, config_.prop_hist_dim};
        estimator_depth_dims_ = {
            1, config_.sensor_size[0], config_.sensor_size[1], config_.sensor_size[2]};
        estimator_hidden_dims_ = {1, config_.gru_hidden_dim};

        // Initialize estimator state. Buffers are (re)sized in initialize_onnx_runtime()
        // after the actual ONNX dims are known, but allocate defaults here so they exist.
        high_feat_.assign(config_.sensor_latent_dim, 0.0f);
        high_feat_window_.assign(
            static_cast<size_t>(config_.policy_window) * config_.sensor_latent_dim, 0.0f);
        wm_hidden_.assign(config_.gru_hidden_dim, 0.0f);
        policy_gru_hidden_.assign(
            static_cast<size_t>(config_.rnn_num_layers) * config_.rnn_hidden_dim, 0.0f);
        wm_step_counter_ = 0;

        // Initialize depth buffer (C x H x W, channels-first)
        int depth_pixels = config_.sensor_size[1] * config_.sensor_size[2]; // H * W
        depth_buffer_.resize(config_.sensor_size[0] * depth_pixels, 0.0f);  // C * H * W
        latest_depth_frame_.resize(depth_pixels, 0.0f);
        prev_depth_frame_.resize(depth_pixels, 0.0f);

        LOG_INFO("Locomotion configuration loaded (WM)");
    }

    // ==================== ZMQ ====================

    void initialize_zmq() {
        fmt::print("Initializing ZMQ...\n");

        zmq_ctx_ = std::make_unique<zmq::context_t>(1);

        // Depth subscriber
        zmq_depth_sub_ = std::make_unique<zmq::socket_t>(*zmq_ctx_, zmq::socket_type::sub);
        zmq_depth_sub_->set(zmq::sockopt::subscribe, "");
        zmq_depth_sub_->set(zmq::sockopt::rcvtimeo, 100); // 100ms timeout
        std::string depth_endpoint =
            fmt::format("tcp://{}:{}", config_.depth_sub_host, config_.depth_sub_port);
        zmq_depth_sub_->connect(depth_endpoint);
        fmt::print("ZMQ depth subscriber connected to {}\n", depth_endpoint);

        // Command subscriber
        zmq_cmd_sub_ = std::make_unique<zmq::socket_t>(*zmq_ctx_, zmq::socket_type::sub);
        zmq_cmd_sub_->set(zmq::sockopt::subscribe, "");
        zmq_cmd_sub_->set(zmq::sockopt::conflate, 1); // keep only latest message
        zmq_cmd_sub_->set(zmq::sockopt::rcvtimeo, 0); // non-blocking
        std::string cmd_endpoint =
            fmt::format("tcp://{}:{}", config_.cmd_sub_host, config_.cmd_sub_port);
        zmq_cmd_sub_->connect(cmd_endpoint);
        fmt::print("ZMQ command subscriber connected to {}\n", cmd_endpoint);

        // Start depth receiver thread
        depth_recv_running_.store(true, std::memory_order_relaxed);
        depth_recv_thread_ = std::thread(&LocomotionNode::depth_recv_loop, this);
        fmt::print("ZMQ depth receiver thread started\n");
    }

    void depth_recv_loop() {
        const uint32_t expected_w = static_cast<uint32_t>(config_.sensor_size[2]);
        const uint32_t expected_h = static_cast<uint32_t>(config_.sensor_size[1]);
        const int depth_pixels = config_.sensor_size[1] * config_.sensor_size[2]; // H * W

        while (depth_recv_running_.load(std::memory_order_relaxed)) {
            zmq::message_t msg;
            auto result = zmq_depth_sub_->recv(msg, zmq::recv_flags::none);

            if (!result.has_value()) {
                continue; // timeout, try again
            }

            if (msg.size() < 8) {
                continue; // too short
            }

            // Parse header: [uint32 width][uint32 height]
            uint32_t w, h;
            std::memcpy(&w, msg.data(), sizeof(uint32_t));
            std::memcpy(&h, static_cast<const uint8_t *>(msg.data()) + 4, sizeof(uint32_t));

            if (msg.size() != 8 + w * h * sizeof(float)) {
                continue; // unexpected size
            }

            if (w != expected_w || h != expected_h) {
                if (!depth_size_mismatch_.exchange(true, std::memory_order_relaxed)) {
                    LOG_ERROR("Incoming depth image is %ux%u, but model requires %ux%u",
                        w, h, expected_w, expected_h);
                }
                continue;
            }

            // Store the depth frame
            std::lock_guard<std::mutex> lock(depth_mutex_);
            const float *data_ptr =
                reinterpret_cast<const float *>(static_cast<const uint8_t *>(msg.data()) + 8);
            std::copy_n(data_ptr, depth_pixels, latest_depth_frame_.begin());
            depth_frame_received_ = true;
        }
    }

    // ==================== Controller Init ====================

    void initialize_controller() {
        fmt::print("initializing locomotion controller (WM)...\n");
        try {
            // Initialize command publisher and state subscribers
            lowcmd_publisher_ = std::make_unique<ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(
                config_.lowcmd_topic);
            lowcmd_publisher_->InitChannel();

            lowstate_subscriber_ =
                std::make_unique<ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(
                    config_.lowstate_topic);
            lowstate_subscriber_->InitChannel(
                std::bind(&LocomotionNode::low_state_callback, this, std::placeholders::_1), 10);

            // subscribe to wireless controller
            wireless_controller_.reset(
                new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(
                    "rt/wirelesscontroller"));
            wireless_controller_->InitChannel(
                std::bind(
                    &LocomotionNode::wireless_controller_callback, this, std::placeholders::_1),
                10);

            // Initialize ONNX models
            initialize_onnx_runtime();

            // Initialize the command
            init_cmd_go();

            // Wait for initial state data
            wait_for_low_state();

            // Initialize robot clients
            if (!config_.use_sim) {
                sport_client_ = std::make_unique<unitree::robot::go2::SportClient>();
                motion_switcher_client_ = std::make_unique<MotionSwitcherClient>();

                sport_client_->SetTimeout(5.0f);
                sport_client_->Init();
                sport_client_->StandDown();
                std::this_thread::sleep_for(2s);

                motion_switcher_client_->SetTimeout(5.0f);
                motion_switcher_client_->Init();

                fmt::print("switching to release mode.\n");
                if (motion_switcher_client_->ReleaseMode() == 0) {
                    fmt::print("switch to release mode success.\n");
                } else {
                    throw std::runtime_error("switch to release mode failed.");
                }
                std::this_thread::sleep_for(5s);
            }

            initialized_ = true;
            LOG_INFO("Locomotion controller (WM) initialized");
        } catch (const std::exception &e) {
            LOG_ERROR("Failed to initialize locomotion controller: %s", e.what());
        }
        fmt::print("locomotion controller (WM) initialized.\n");
    }

    void initialize_onnx_runtime() {
        fmt::print("initializing ONNX Runtime for world-model locomotion...\n");

        // Per-model TRT engine cache: TRT keys cached engines by ONNX file path/name,
        // so two different policies named `policy.onnx` would otherwise collide. We give
        // each model_subdir its own cache directory.
        std::string trt_cache_dir =
            config_.model_subdir.empty()
                ? std::string("./trt_engine_cache")
                : fmt::format("./trt_engine_cache/{}", config_.model_subdir);
        std::filesystem::create_directories(trt_cache_dir);
        fmt::print("TensorRT engine cache directory: {}\n", trt_cache_dir);

        // Create session options
        auto create_session_options = [&]() {
            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Enable TensorRT
            OrtTensorRTProviderOptionsV2 *tensorrt_options = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&tensorrt_options));

            std::vector<const char *> option_keys = {
                "trt_fp16_enable",
                "trt_engine_cache_enable",
                "trt_engine_cache_path",
            };
            std::vector<const char *> option_values = {
                config_.trt_fp16_enabled ? "1" : "0",
                config_.trt_cache_enabled ? "1" : "0",
                trt_cache_dir.c_str(), // Per-model engine cache directory
            };
            Ort::ThrowOnError(
                Ort::GetApi().UpdateTensorRTProviderOptions(
                    tensorrt_options, option_keys.data(), option_values.data(),
                    option_keys.size()));

            session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
            Ort::GetApi().ReleaseTensorRTProviderOptions(tensorrt_options);

            return session_options;
        };

        try {
            const std::filesystem::path model_dir =
                std::filesystem::path(config_.policy_path).parent_path();
            const std::filesystem::path test_case_dir = model_dir / "test_cases";
            const bool has_test_cases =
                std::filesystem::exists(test_case_dir / "metadata.json");
            const int max_build_attempts = has_test_cases ? 2 : 1;
            int build_attempt = 1;
            while (true) {
                LOG_WARN("TensorRT build/test attempt %d starting. Cache directory: %s",
                    build_attempt, trt_cache_dir.c_str());

                try {
                    LOG_INFO("Loading policy from %s",
                        config_.policy_path.c_str());
                    auto policy_opts = create_session_options();
                    policy_session_ = std::make_unique<Ort::Session>(
                        env_, config_.policy_path.c_str(), policy_opts);
                    LOG_INFO("Policy ONNX model loaded successfully");

                    LOG_INFO("Loading sensor estimator from %s",
                        config_.sensor_estimator_path.c_str());
                    auto se_opts = create_session_options();
                    sensor_estimator_session_ = std::make_unique<Ort::Session>(
                        env_, config_.sensor_estimator_path.c_str(), se_opts);
                    LOG_INFO("Sensor estimator ONNX model loaded successfully");

                    if (!has_test_cases) {
                        LOG_WARN("No ONNX test cases found at %s; skipping TensorRT validation",
                            test_case_dir.string().c_str());
                        break;
                    }

                    auto validation = run_onnx_test_case_validation(test_case_dir);
                    log_onnx_test_stats("policy", validation.policy);
                    log_onnx_test_stats("sensor_estimator", validation.estimator);

                    if (validation.policy.ok && validation.estimator.ok) {
                        LOG_WARN("TensorRT cached engines passed ONNX test cases on attempt %d",
                            build_attempt);
                        break;
                    }

                    throw std::runtime_error(
                        fmt::format(
                            "TensorRT cached engines failed ONNX test cases on attempt {}",
                            build_attempt));
                } catch (const std::exception &e) {
                    if (build_attempt >= max_build_attempts) {
                        LOG_ERROR("TensorRT build/test attempt %d failed: %s",
                            build_attempt, e.what());
                        throw;
                    }
                    LOG_ERROR("TensorRT build/test attempt %d failed: %s. Clearing cache and "
                        "rebuilding.",
                        build_attempt, e.what());
                }

                policy_session_.reset();
                sensor_estimator_session_.reset();
                std::filesystem::remove_all(trt_cache_dir);
                std::filesystem::create_directories(trt_cache_dir);
                ++build_attempt;
            }

            log_session_io(*policy_session_, "policy");
            log_session_io(*sensor_estimator_session_, "sensor_estimator");

            // ---- Auto-detect dims and verify policy <-> estimator compatibility ----
            policy_uses_proprio_ = has_io(*policy_session_, "proprio", true);
            policy_uses_high_feat_window_ = has_io(*policy_session_, "high_feat_window", true);
            policy_uses_high_feat_ = has_io(*policy_session_, "high_feat", true);
            policy_uses_prop_hist_ = has_io(*policy_session_, "proprio_hist", true);
            policy_uses_depth_stack_ = has_io(*policy_session_, "depth_stack", true);
            policy_uses_gru_state_ = has_io(*policy_session_, "gru_hidden_in", true) &&
                                     has_io(*policy_session_, "gru_hidden_out", false);
            policy_uses_stateless_high_feat_ =
                !policy_uses_high_feat_window_ && !policy_uses_proprio_ &&
                !policy_uses_depth_stack_ && policy_uses_prop_hist_ &&
                policy_uses_high_feat_ && !policy_uses_gru_state_;
            if (policy_uses_high_feat_window_) {
                if (!policy_uses_prop_hist_) {
                    throw std::runtime_error(
                        "Window policy ONNX format requires proprio_hist input");
                }
            } else if (policy_uses_stateless_high_feat_) {
                // Fixed-window transformer export:
                //   inputs: proprio_hist, high_feat
                //   outputs: actions
            } else {
                if (!policy_uses_proprio_ || !policy_uses_high_feat_ || !policy_uses_gru_state_) {
                    throw std::runtime_error(
                        "Unsupported policy ONNX format. Supported formats are: "
                        "(1) proprio + high_feat + gru_hidden_in -> actions + gru_hidden_out, "
                        "(2) proprio_hist + high_feat_window -> actions, "
                        "(3) proprio_hist + high_feat -> actions");
                }
            }

            if (policy_uses_proprio_) {
                auto policy_proprio_shape = get_io_shape(*policy_session_, "proprio", true);
                int detected_proprio_dim = static_cast<int>(policy_proprio_shape.back());
                if (detected_proprio_dim > 0 && detected_proprio_dim != config_.proprio_dim) {
                    config_.proprio_dim = detected_proprio_dim;
                    proprio_obs_.assign(config_.proprio_dim, 0.0f);
                    LOG_INFO("Auto-detected policy proprio dim = %d",
                        config_.proprio_dim);
                }
                policy_proprio_dims_ =
                    concrete_vector_shape(policy_proprio_shape, {1, config_.proprio_dim});
            }

            if (policy_uses_prop_hist_) {
                auto policy_prop_hist_shape = get_io_shape(*policy_session_, "proprio_hist", true);
                int detected_prop_hist_dim = static_cast<int>(policy_prop_hist_shape.back());
                if (detected_prop_hist_dim > 0 && detected_prop_hist_dim != config_.prop_hist_dim) {
                    config_.prop_hist_dim = detected_prop_hist_dim;
                    prop_hist_.assign(config_.prop_hist_dim, 0.0f);
                    LOG_INFO("Auto-detected policy proprio_hist dim = %d",
                        config_.prop_hist_dim);
                }
                policy_prop_hist_dims_ =
                    concrete_vector_shape(policy_prop_hist_shape, {1, config_.prop_hist_dim});
            }

            // Estimator output `feature` -> sensor_latent_dim.
            auto est_feature_shape = get_io_shape(*sensor_estimator_session_, "feature", false);
            int detected_feature_dim = static_cast<int>(est_feature_shape.back());
            int detected_high_feat_dim = 0;
            if (policy_uses_high_feat_window_) {
                auto policy_high_feat_shape =
                    get_io_shape(*policy_session_, "high_feat_window", true);
                if (policy_high_feat_shape.size() != 2 && policy_high_feat_shape.size() != 3) {
                    throw std::runtime_error(
                        fmt::format(
                            "policy high_feat_window expected rank 2 or 3, got rank {}",
                            policy_high_feat_shape.size()));
                }
                const size_t window_dim_index = policy_high_feat_shape.size() - 2;
                if (policy_high_feat_shape[window_dim_index] > 0) {
                    config_.policy_window =
                        static_cast<int>(policy_high_feat_shape[window_dim_index]);
                }
                detected_high_feat_dim = static_cast<int>(policy_high_feat_shape.back());
                policy_high_feat_window_dims_ = concrete_vector_shape(
                    policy_high_feat_shape, {1, config_.policy_window, config_.sensor_latent_dim});
            } else if (policy_uses_high_feat_) {
                auto policy_high_feat_shape = get_io_shape(*policy_session_, "high_feat", true);
                detected_high_feat_dim = static_cast<int>(policy_high_feat_shape.back());
                policy_high_feat_dims_ = concrete_vector_shape(
                    policy_high_feat_shape, {1, config_.sensor_latent_dim});
            } else {
                throw std::runtime_error(
                    "Policy ONNX must have either high_feat_window or high_feat input");
            }
            if (detected_feature_dim != detected_high_feat_dim) {
                throw std::runtime_error(
                    fmt::format(
                        "Policy/estimator mismatch: estimator feature dim={} but policy latent "
                        "input dim={}. Are the policy.onnx and sensor_estimator.onnx from the same "
                        "checkpoint?",
                        detected_feature_dim, detected_high_feat_dim));
            }
            config_.sensor_latent_dim = detected_feature_dim;
            high_feat_.assign(config_.sensor_latent_dim, 0.0f);
            high_feat_window_.assign(
                static_cast<size_t>(config_.policy_window) * config_.sensor_latent_dim, 0.0f);
            if (policy_uses_high_feat_window_) {
                auto policy_high_feat_shape =
                    get_io_shape(*policy_session_, "high_feat_window", true);
                policy_high_feat_window_dims_ = concrete_vector_shape(
                    policy_high_feat_shape, {1, config_.policy_window, config_.sensor_latent_dim});
            } else {
                auto policy_high_feat_shape = get_io_shape(*policy_session_, "high_feat", true);
                policy_high_feat_dims_ = concrete_vector_shape(
                    policy_high_feat_shape, {1, config_.sensor_latent_dim});
            }
            if (policy_uses_depth_stack_) {
                policy_depth_dims_ = concrete_vector_shape(
                    get_io_shape(*policy_session_, "depth_stack", true),
                    {1, config_.sensor_size[0], config_.sensor_size[1], config_.sensor_size[2]});
            }
            LOG_INFO("Auto-detected sensor_latent_dim=%d, policy_window=%d (%s policy format)",
                config_.sensor_latent_dim, config_.policy_window,
                policy_uses_high_feat_window_ ? "high_feat_window" : "single high_feat");

            if (policy_uses_gru_state_) {
                // Older policy exports carry their own GRU state.
                auto policy_gru_shape = get_io_shape(*policy_session_, "gru_hidden_in", true);
                if (policy_gru_shape.size() != 3) {
                    throw std::runtime_error(
                        fmt::format(
                            "policy gru_hidden_in expected to be rank 3, got rank {}",
                            policy_gru_shape.size()));
                }
                config_.rnn_num_layers = static_cast<int>(policy_gru_shape[0]);
                config_.rnn_hidden_dim = static_cast<int>(policy_gru_shape[2]);
                size_t policy_gru_size =
                    static_cast<size_t>(config_.rnn_num_layers) * config_.rnn_hidden_dim;
                policy_gru_hidden_.assign(policy_gru_size, 0.0f);
                policy_gru_dims_ = concrete_vector_shape(
                    policy_gru_shape,
                    {config_.rnn_num_layers, 1, config_.rnn_hidden_dim});
                LOG_INFO("Auto-detected policy GRU: num_layers=%d, hidden_dim=%d",
                    config_.rnn_num_layers, config_.rnn_hidden_dim);
            } else {
                policy_gru_hidden_.clear();
                LOG_INFO("Policy has no recurrent ONNX state");
            }

            // Estimator GRU hidden state.
            auto hidden_out_shape = get_io_shape(*sensor_estimator_session_, "hidden_out", false);
            config_.gru_hidden_dim = static_cast<int>(hidden_out_shape.back());
            wm_hidden_.assign(config_.gru_hidden_dim, 0.0f);
            estimator_prop_hist_dims_ = concrete_vector_shape(
                get_io_shape(*sensor_estimator_session_, "prop_hist", true),
                {1, config_.prop_hist_dim});
            estimator_depth_dims_ = concrete_vector_shape(
                get_io_shape(*sensor_estimator_session_, "depth_stack", true),
                {1, config_.sensor_size[0], config_.sensor_size[1], config_.sensor_size[2]});
            if (policy_uses_depth_stack_ && policy_depth_dims_ != estimator_depth_dims_) {
                throw std::runtime_error(
                    fmt::format(
                        "Policy/estimator depth_stack mismatch: policy={} estimator={}",
                        shape_to_string(policy_depth_dims_),
                        shape_to_string(estimator_depth_dims_)));
            }
            apply_model_depth_shape(estimator_depth_dims_);
            estimator_hidden_dims_ = concrete_vector_shape(
                get_io_shape(*sensor_estimator_session_, "hidden_in", true),
                {1, config_.gru_hidden_dim});
            LOG_INFO("Auto-detected estimator gru_hidden_dim = %d",
                config_.gru_hidden_dim);

            LOG_INFO("Policy inputs: proprio=%s, proprio_hist=%s, depth_stack=%s, high_feat=%s, "
                "high_feat_window=%s, gru_state=%s, stateless_high_feat=%s",
                policy_uses_proprio_ ? "yes" : "no", policy_uses_prop_hist_ ? "yes" : "no",
                policy_uses_depth_stack_ ? "yes" : "no",
                policy_uses_high_feat_ ? "yes" : "no",
                policy_uses_high_feat_window_ ? "yes" : "no",
                policy_uses_gru_state_ ? "yes" : "no",
                policy_uses_stateless_high_feat_ ? "yes" : "no");
            log_runtime_tensor_shapes();

            fmt::print(
                "\nONNX sessions are loaded and validated. Press Enter to continue to robot "
                "control initialization...");
            std::string enter_line;
            std::getline(std::cin, enter_line);

        } catch (const Ort::Exception &e) {
            LOG_ERROR("ONNX initialization error: %s", e.what());
            throw;
        }
    }

    // Find an input or output by name and return its shape (-1 entries are dynamic).
    static std::vector<int64_t>
    get_io_shape(Ort::Session &session, const std::string &name, bool is_input) {
        Ort::AllocatorWithDefaultOptions allocator;
        size_t count = is_input ? session.GetInputCount() : session.GetOutputCount();
        for (size_t i = 0; i < count; ++i) {
            auto io_name = is_input ? session.GetInputNameAllocated(i, allocator)
                                    : session.GetOutputNameAllocated(i, allocator);
            if (name == io_name.get()) {
                auto type_info =
                    is_input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
                return type_info.GetTensorTypeAndShapeInfo().GetShape();
            }
        }
        throw std::runtime_error(
            fmt::format(
                "ONNX session does not have {} named '{}'", is_input ? "input" : "output", name));
    }

    static bool has_io(Ort::Session &session, const std::string &name, bool is_input) {
        Ort::AllocatorWithDefaultOptions allocator;
        size_t count = is_input ? session.GetInputCount() : session.GetOutputCount();
        for (size_t i = 0; i < count; ++i) {
            auto io_name = is_input ? session.GetInputNameAllocated(i, allocator)
                                    : session.GetOutputNameAllocated(i, allocator);
            if (name == io_name.get()) {
                return true;
            }
        }
        return false;
    }

    static std::string shape_to_string(const std::vector<int64_t> &shape) {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i + 1 < shape.size()) {
                s += ",";
            }
        }
        s += "]";
        return s;
    }

    static std::vector<int64_t> concrete_vector_shape(
        const std::vector<int64_t> &onnx_shape, const std::vector<int> &fallback_shape) {
        if (onnx_shape.empty()) {
            return std::vector<int64_t>(fallback_shape.begin(), fallback_shape.end());
        }
        std::vector<int64_t> concrete;
        concrete.reserve(onnx_shape.size());
        const size_t fallback_offset =
            fallback_shape.size() > onnx_shape.size() ? fallback_shape.size() - onnx_shape.size() : 0;
        for (size_t i = 0; i < onnx_shape.size(); ++i) {
            const size_t fallback_index = std::min(fallback_offset + i, fallback_shape.size() - 1);
            concrete.push_back(
                onnx_shape[i] > 0 ? onnx_shape[i] : static_cast<int64_t>(fallback_shape[fallback_index]));
        }
        return concrete;
    }

    void apply_model_depth_shape(const std::vector<int64_t> &depth_dims) {
        if (depth_dims.size() != 4) {
            throw std::runtime_error(
                fmt::format(
                    "depth_stack input must be rank 4 [N,C,H,W], got rank {}",
                    depth_dims.size()));
        }
        if (depth_dims[0] != 1 || depth_dims[1] <= 0 || depth_dims[2] <= 0 ||
            depth_dims[3] <= 0) {
            throw std::runtime_error(
                fmt::format("Unsupported depth_stack shape {}", shape_to_string(depth_dims)));
        }

        const int channels = static_cast<int>(depth_dims[1]);
        const int height = static_cast<int>(depth_dims[2]);
        const int width = static_cast<int>(depth_dims[3]);
        if (channels != 2) {
            throw std::runtime_error(
                fmt::format("Unsupported depth_stack channel count {}; expected 2", channels));
        }

        config_.sensor_size = {channels, height, width};

        const size_t depth_pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
        depth_buffer_.assign(static_cast<size_t>(channels) * depth_pixels, 0.0f);
        latest_depth_frame_.assign(depth_pixels, 0.0f);
        prev_depth_frame_.assign(depth_pixels, 0.0f);
        depth_prev_initialized_ = false;
        depth_frame_received_ = false;

        LOG_INFO("Model depth_stack shape detected: C=%d H=%d W=%d",
            channels, height, width);
    }

    void log_runtime_tensor_shapes() {
        if (policy_uses_high_feat_window_) {
            const std::string prop_hist_shape = shape_to_string(policy_prop_hist_dims_);
            const std::string high_feat_window_shape =
                shape_to_string(policy_high_feat_window_dims_);
            LOG_INFO("Runtime policy tensors: proprio_hist%s, high_feat_window%s",
                prop_hist_shape.c_str(), high_feat_window_shape.c_str());
        } else if (policy_uses_stateless_high_feat_) {
            const std::string prop_hist_shape = shape_to_string(policy_prop_hist_dims_);
            const std::string high_feat_shape = shape_to_string(policy_high_feat_dims_);
            LOG_INFO("Runtime policy tensors: proprio_hist%s, high_feat%s",
                prop_hist_shape.c_str(), high_feat_shape.c_str());
        } else {
            const std::string proprio_shape = shape_to_string(policy_proprio_dims_);
            const std::string prop_hist_shape =
                policy_uses_prop_hist_ ? shape_to_string(policy_prop_hist_dims_) : "unused";
            const std::string depth_shape =
                policy_uses_depth_stack_ ? shape_to_string(policy_depth_dims_) : "unused";
            const std::string high_feat_shape = shape_to_string(policy_high_feat_dims_);
            const std::string gru_shape = shape_to_string(policy_gru_dims_);
            LOG_INFO("Runtime policy tensors: proprio%s, proprio_hist%s, depth_stack%s, high_feat%s, "
                "gru_hidden%s",
                proprio_shape.c_str(), prop_hist_shape.c_str(), depth_shape.c_str(),
                high_feat_shape.c_str(), gru_shape.c_str());
        }
        const std::string estimator_prop_hist_shape = shape_to_string(estimator_prop_hist_dims_);
        const std::string estimator_depth_shape = shape_to_string(estimator_depth_dims_);
        const std::string estimator_hidden_shape = shape_to_string(estimator_hidden_dims_);
        LOG_INFO("Runtime estimator tensors: prop_hist%s, depth_stack%s, hidden_in%s",
            estimator_prop_hist_shape.c_str(), estimator_depth_shape.c_str(),
            estimator_hidden_shape.c_str());
    }

    // Print all input/output names + shapes for an ONNX session.
    void log_session_io(Ort::Session &session, const std::string &label) {
        Ort::AllocatorWithDefaultOptions allocator;
        auto shape_to_str = [](const std::vector<int64_t> &shape) {
            std::string s = "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                s += (shape[i] < 0) ? std::string("?") : std::to_string(shape[i]);
                if (i + 1 < shape.size())
                    s += ",";
            }
            s += "]";
            return s;
        };
        size_t n_in = session.GetInputCount();
        for (size_t i = 0; i < n_in; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            LOG_INFO("  %s input[%zu]: %s %s", label.c_str(), i, name.get(),
                shape_to_str(shape).c_str());
        }
        size_t n_out = session.GetOutputCount();
        for (size_t i = 0; i < n_out; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            LOG_INFO("  %s output[%zu]: %s %s", label.c_str(), i, name.get(),
                shape_to_str(shape).c_str());
        }
    }

    OnnxValidationResult run_onnx_test_case_validation(const std::filesystem::path &test_case_dir) {
        const double policy_max_abs_tol = 0.1;
        const double estimator_max_abs_tol = 0.5;
        const double mean_abs_tol = 0.1;

        std::filesystem::path metadata_path = test_case_dir / "metadata.json";
        LOG_WARN("Validating TensorRT engines against ONNX test cases in %s",
            test_case_dir.string().c_str());
        nlohmann::json metadata = nlohmann::json::parse(read_text_file(metadata_path.string()));

        auto policy_meta = metadata.at("policy");
        auto estimator_meta = metadata.at("sensor_estimator");

        OnnxValidationResult result;
        result.policy = validate_model_test_cases(
            *policy_session_,
            test_case_dir / policy_meta.value("sample_dir", std::string("policy")),
            json_string_list(policy_meta.at("inputs")), json_string_list(policy_meta.at("outputs")),
            policy_max_abs_tol, mean_abs_tol);
        result.estimator = validate_model_test_cases(
            *sensor_estimator_session_,
            test_case_dir /
                estimator_meta.value("sample_dir", std::string("sensor_estimator")),
            json_string_list(estimator_meta.at("inputs")),
            json_string_list(estimator_meta.at("outputs")), estimator_max_abs_tol, mean_abs_tol);
        return result;
    }

    void log_onnx_test_stats(const std::string &label, const OnnxTestStats &stats) {
        LOG_WARN("ONNX TEST %s %s records=%zu worst_tensor=%s worst_sample=%s "
            "worst_max_abs=%.9g worst_mean_abs=%.9g",
            label.c_str(), stats.ok ? "PASS" : "FAIL", stats.records,
            stats.worst_tensor.c_str(), stats.worst_sample.c_str(), stats.worst_max_abs,
            stats.worst_mean_abs);
    }

    // ==================== Callbacks ====================

    /**
     * Poll ZMQ command subscriber (non-blocking).
     * CONFLATE is set so the socket only holds the latest message.
     * Message format: 12 bytes = 3x float32 (vx, vy, wz)
     */
    void poll_zmq_commands() {
        zmq::message_t msg;
        auto result = zmq_cmd_sub_->recv(msg, zmq::recv_flags::dontwait);

        if (!result.has_value() || msg.size() != 3 * sizeof(float)) {
            return;
        }

        float vx, vy, wz;
        std::memcpy(&vx, static_cast<const uint8_t *>(msg.data()) + 0, sizeof(float));
        std::memcpy(&vy, static_cast<const uint8_t *>(msg.data()) + sizeof(float), sizeof(float));
        std::memcpy(
            &wz, static_cast<const uint8_t *>(msg.data()) + 2 * sizeof(float), sizeof(float));

        cmd_(0) = std::clamp(vx, -config_.forward_clip, config_.forward_clip);
        cmd_(1) = std::clamp(vy, -config_.lateral_clip, config_.lateral_clip);
        cmd_(2) = std::clamp(wz, -config_.angular_clip, config_.angular_clip);

        // In emergency stop mode, override with zero commands
        if (g_emergency_stop_mode.load(std::memory_order_relaxed)) {
            cmd_(0) = 0.0f;
            cmd_(1) = 0.0f;
            cmd_(2) = 0.0f;
        }

        // Log commands periodically
        static int log_counter = 0;
        if (log_counter++ % 100 == 0) {
            LOG_INFO("ZMQ commands: [%.2f, %.2f, %.2f]", cmd_[0], cmd_[1], cmd_[2]);
        }
    }

    void low_state_callback(const void *msg) {
        low_state_ = *(unitree_go::msg::dds_::LowState_ *)msg;
    }

    void wireless_controller_callback(const void *msg) {
        wireless_controller_msg_ = *(unitree_go::msg::dds_::WirelessController_ *)msg;
    }

    void wait_for_low_state() {
        if (config_.use_sim) {
            fmt::print("Simulation mode, skipping wait for low state.\n");
            return;
        }
        LOG_INFO("Waiting for robot state...");
        auto start = std::chrono::steady_clock::now();
        while (low_state_.tick() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            if (elapsed > 5) {
                LOG_WARN("Timeout waiting for robot state");
                break;
            }
        }
        LOG_INFO("Successfully connected to the robot");
    }

    // ==================== Motor Commands ====================

    void init_cmd_go() {
        low_cmd_.head()[0] = 0xFE;
        low_cmd_.head()[1] = 0xEF;
        low_cmd_.level_flag() = 0xFF;
        low_cmd_.gpio() = 0;

        float PosStopF = 2.146e9f;
        float VelStopF = 16000.0f;

        for (size_t i = 0; i < low_cmd_.motor_cmd().size(); i++) {
            low_cmd_.motor_cmd()[i].mode() = 0x0A;
            low_cmd_.motor_cmd()[i].q() = PosStopF;
            low_cmd_.motor_cmd()[i].dq() = VelStopF;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 0;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void create_zero_cmd() {
        for (int i = 0; i < 12; i++) {
            low_cmd_.motor_cmd()[i].q() = 0;
            low_cmd_.motor_cmd()[i].dq() = 0;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 0;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void create_damping_cmd() {
        for (int i = 0; i < 12; i++) {
            low_cmd_.motor_cmd()[i].q() = 0;
            low_cmd_.motor_cmd()[i].dq() = 0;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 1;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void send_cmd() {
        if (initialized_) {
            low_cmd_.crc() = crc32_core(
                (uint32_t *)&low_cmd_, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
            lowcmd_publisher_->Write(low_cmd_);
        }
    }

    // ==================== Control Sequences ====================

    void zero_torque_state() {
        LOG_INFO("Entering zero torque state");
        create_zero_cmd();
        send_cmd();
        std::this_thread::sleep_for(1s);
    }

    void move_to_default_pos() {
        LOG_INFO("Moving to default position");

        float total_time = 2.0f;
        int num_steps = static_cast<int>(total_time / config_.control_dt);

        std::vector<float> init_dof_pos(12, 0.0f);
        for (int i = 0; i < 12; i++) {
            init_dof_pos[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
        }

        for (int step = 0; step < num_steps; step++) {
            float alpha = static_cast<float>(step) / num_steps;
            for (int j = 0; j < 12; j++) {
                int motor_idx = config_.joint2motor_idx[j];
                float target_pos = config_.default_angles[j];
                low_cmd_.motor_cmd()[motor_idx].q() =
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha;
                low_cmd_.motor_cmd()[motor_idx].dq() = 0;
                low_cmd_.motor_cmd()[motor_idx].kp() = 35.0f;
                low_cmd_.motor_cmd()[motor_idx].kd() = 0.5f;
                low_cmd_.motor_cmd()[motor_idx].tau() = 0;
            }
            send_cmd();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(config_.control_dt * 1000)));
        }
    }

    void default_pos_state() {
        LOG_INFO("Entering default position state");

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 12; j++) {
                int motor_idx = config_.joint2motor_idx[j];
                low_cmd_.motor_cmd()[motor_idx].q() = config_.default_angles[j];
                low_cmd_.motor_cmd()[motor_idx].dq() = 0;
                low_cmd_.motor_cmd()[motor_idx].kp() = 45.0f;
                low_cmd_.motor_cmd()[motor_idx].kd() = 0.5f;
                low_cmd_.motor_cmd()[motor_idx].tau() = 0;
            }
            send_cmd();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(config_.control_dt * 1000)));
        }
    }

    void move_to_crouch_pose() {
        LOG_INFO("Moving to crouching position");

        float total_time = 2.0f;
        int num_steps = static_cast<int>(total_time / config_.control_dt);

        std::vector<float> init_dof_pos(12, 0.0f);
        for (int i = 0; i < 12; i++) {
            init_dof_pos[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
        }

        for (int step = 0; step < num_steps; step++) {
            float alpha = static_cast<float>(step) / num_steps;
            for (int j = 0; j < 12; j++) {
                int motor_idx = config_.joint2motor_idx[j];
                float target_pos = config_.crouch_angles[j];
                low_cmd_.motor_cmd()[motor_idx].q() =
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha;
                low_cmd_.motor_cmd()[motor_idx].dq() = 0;
                low_cmd_.motor_cmd()[motor_idx].kp() = 35.0f;
                low_cmd_.motor_cmd()[motor_idx].kd() = 0.5f;
                low_cmd_.motor_cmd()[motor_idx].tau() = 0;
            }
            send_cmd();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(config_.control_dt * 1000)));
        }
    }

    void start_control_sequence() {
        if (!initialized_ || running_) {
            return;
        }

        LOG_INFO("Starting locomotion control sequence");

        zero_torque_state();
        move_to_default_pos();
        default_pos_state();

        // Reset all recurrent state before policy takes over.
        reset_recurrent_state();

        running_ = true;
        LOG_INFO("Locomotion control sequence started, robot ready");
    }

    void reset_recurrent_state() {
        std::fill(policy_gru_hidden_.begin(), policy_gru_hidden_.end(), 0.0f);
        std::fill(wm_hidden_.begin(), wm_hidden_.end(), 0.0f);
        std::fill(high_feat_.begin(), high_feat_.end(), 0.0f);
        std::fill(high_feat_window_.begin(), high_feat_window_.end(), 0.0f);
        previous_target_dof_pos_ = target_dof_pos_;
        wm_step_counter_ = 0;
    }

    void stop_control_sequence() {
        if (!initialized_ || !running_) {
            return;
        }

        LOG_INFO("Stopping locomotion control sequence");

        if (!crouching_) {
            if (safety_fault_) {
                LOG_WARN("Safety fault active; skipping stand/crouch transition and keeping damping");
            } else {
                move_to_default_pos();
                move_to_crouch_pose();
            }
        }

        if (!safety_fault_) {
            create_damping_cmd();
        }
        send_cmd();

        running_ = false;
        LOG_INFO("Locomotion stopped");
    }

    // ==================== Main Control Logic ====================

    void run_controller() {
        // Emergency stop with controller
        gamepad_.Update(wireless_controller_msg_);
        if (gamepad_.B.on_press) {
            if (!g_emergency_stop_mode.load(std::memory_order_relaxed)) {
                fmt::print("emergency stop mode activated by wireless controller.\n");
                g_emergency_stop_mode.store(true, std::memory_order_relaxed);
            } else {
                fmt::print("exiting emergency stop mode by wireless controller.\n");
                g_stop_control_signal.store(true, std::memory_order_relaxed);
            }
        }

        // L1 button toggle for crouch/stand
        if (gamepad_.L1.on_press && !g_emergency_stop_mode.load(std::memory_order_relaxed)) {
            if (!crouching_) {
                fmt::print("Entering crouch mode...\n");
                move_to_crouch_pose();
                crouching_ = true;
                fmt::print("Robot is now crouching. Press L1 to stand up.\n");
            } else {
                fmt::print("Standing up...\n");
                move_to_default_pos();
                default_pos_state();
                crouching_ = false;
                // Clear recurrent state so the policy doesn't carry over crouch-time activations.
                reset_recurrent_state();
                fmt::print("Robot is now standing. Policy active.\n");
            }
        }

        // R1 button toggle for gamepad direct control mode
        if (gamepad_.R1.on_press && !g_emergency_stop_mode.load(std::memory_order_relaxed)) {
            gamepad_control_mode_ = !gamepad_control_mode_;
            if (gamepad_control_mode_) {
                fmt::print("=== SWITCHED TO GAMEPAD CONTROL MODE ===\n");
                if (config_.use_joystick_gamepad_control) {
                    fmt::print("Left stick=omnidirectional movement, right stick X=rotation\n");
                } else {
                    fmt::print("Y=rotate left, A=rotate right, X=forward\n");
                }
            } else {
                fmt::print("=== SWITCHED TO ZMQ COMMAND MODE ===\n");
                cmd_(0) = 0.0f;
                cmd_(1) = 0.0f;
                cmd_(2) = 0.0f;
            }
        }

        // Skip policy execution if crouching
        if (crouching_) {
            for (int i = 0; i < 12; i++) {
                int motor_idx = config_.joint2motor_idx[i];
                low_cmd_.motor_cmd()[motor_idx].q() = config_.crouch_angles[i];
                low_cmd_.motor_cmd()[motor_idx].dq() = 0;
                low_cmd_.motor_cmd()[motor_idx].kp() = 35.0f;
                low_cmd_.motor_cmd()[motor_idx].kd() = 0.5f;
                low_cmd_.motor_cmd()[motor_idx].tau() = 0;
            }
            send_cmd();
            return;
        }

        if (safety_fault_) {
            cmd_(0) = 0.0f;
            cmd_(1) = 0.0f;
            cmd_(2) = 0.0f;
            create_damping_cmd();
            send_cmd();
            return;
        }

        // In buttons mode, R2 toggles between the two configured linear speed scales.
        // Joystick mode deliberately ignores R2.
        if (gamepad_control_mode_ && !config_.use_joystick_gamepad_control &&
            gamepad_.R2.on_press) {
            secondary_button_linear_scale_active_ = !secondary_button_linear_scale_active_;
            const float active_scale = secondary_button_linear_scale_active_
                                           ? config_.gamepad_linear_scale_2
                                           : config_.gamepad_linear_scale;
            fmt::print(
                "Button linear scale switched to mode {}: scale={:.3f}, forward speed={:.3f}\n",
                secondary_button_linear_scale_active_ ? 2 : 1, active_scale,
                0.75f * active_scale);
        }

        // Gamepad direct control or ZMQ commands
        if (gamepad_control_mode_) {
            const float active_linear_scale =
                (!config_.use_joystick_gamepad_control && secondary_button_linear_scale_active_)
                    ? config_.gamepad_linear_scale_2
                    : config_.gamepad_linear_scale;
            const float linear_speed = 0.75f * active_linear_scale;
            const float angular_speed = 1.0f * config_.gamepad_angular_scale;
            float vx = 0.0f, vy = 0.0f, wz = 0.0f;
            if (config_.use_joystick_gamepad_control) {
                // Match the button speeds while adding analog omnidirectional control.
                vx = gamepad_.ly * linear_speed;
                vy = -gamepad_.lx * linear_speed;
                wz = -gamepad_.rx * angular_speed;
            } else {
                if (gamepad_.X.pressed)
                    vx = linear_speed;
                if (gamepad_.Y.pressed)
                    wz = angular_speed;
                if (gamepad_.A.pressed)
                    wz -= angular_speed;
            }
            cmd_(0) = std::clamp(vx, -config_.forward_clip, config_.forward_clip);
            cmd_(1) = std::clamp(vy, -config_.lateral_clip, config_.lateral_clip);
            cmd_(2) = std::clamp(wz, -config_.angular_clip, config_.angular_clip);
            if (g_emergency_stop_mode.load(std::memory_order_relaxed)) {
                cmd_(0) = 0.0f;
                cmd_(1) = 0.0f;
                cmd_(2) = 0.0f;
            }
        } else {
            // Poll ZMQ for velocity commands
            poll_zmq_commands();
        }

        // Update joint positions and velocities
        for (int i = 0; i < 12; i++) {
            qj_[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
            dqj_[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].dq();
        }

        // Get IMU state
        auto quat = Quaternionf(
            low_state_.imu_state().quaternion()[0], low_state_.imu_state().quaternion()[1],
            low_state_.imu_state().quaternion()[2], low_state_.imu_state().quaternion()[3]);
        auto ang_vel = std::array<float, 3>{
            low_state_.imu_state().gyroscope()[0], low_state_.imu_state().gyroscope()[1],
            low_state_.imu_state().gyroscope()[2]};
        auto gravity_orientation = get_gravity_orientation_eigen(quat);

        // Build proprioception observation (45 dims)
        // [ang_vel*0.25(3), gravity(3), cmd_lin*2.0(2), cmd_ang*0.25(1),
        //  joint_pos-default(12), joint_vel*0.05(12), actions*0.1(12)]
        proprio_obs_[0] = ang_vel[0] * config_.ang_vel_scale;
        proprio_obs_[1] = ang_vel[1] * config_.ang_vel_scale;
        proprio_obs_[2] = ang_vel[2] * config_.ang_vel_scale;
        proprio_obs_[3] = gravity_orientation[0];
        proprio_obs_[4] = gravity_orientation[1];
        proprio_obs_[5] = gravity_orientation[2];
        proprio_obs_[6] = cmd_[0] * config_.lin_vel_scale;
        proprio_obs_[7] = cmd_[1] * config_.lin_vel_scale;
        proprio_obs_[8] = cmd_[2] * config_.ang_vel_scale;

        for (int i = 0; i < 12; i++) {
            proprio_obs_[9 + i] = (qj_[i] - config_.default_angles[i]) * config_.dof_pos_scale;
            proprio_obs_[21 + i] = dqj_[i] * config_.dof_vel_scale;
            proprio_obs_[33 + i] = raw_action_[i] * config_.action_obs_scale;
        }

        // Clip proprio observation
        for (auto &val : proprio_obs_) {
            val = std::clamp(val, -config_.obs_clip, config_.obs_clip);
        }

        // Update proprioception history: shift right 45 slots, prepend current (newest first)
        std::copy_backward(
            prop_hist_.begin(), prop_hist_.begin() + config_.prop_hist_dim - config_.proprio_dim,
            prop_hist_.end());
        std::copy_n(proprio_obs_.begin(), config_.proprio_dim, prop_hist_.begin());

        // --- Refresh depth buffer every step (policy consumes depth at 50 Hz) ---
        int depth_pixels = config_.sensor_size[1] * config_.sensor_size[2];
        std::vector<float> live_depth_stack(config_.sensor_size[0] * depth_pixels);
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            // Duplicate the latest frame when initializing the depth history.
            if (!depth_prev_initialized_) {
                prev_depth_frame_ = latest_depth_frame_;
                depth_prev_initialized_ = true;
            }
            std::copy_n(prev_depth_frame_.begin(), depth_pixels, live_depth_stack.begin());
            std::copy_n(
                latest_depth_frame_.begin(), depth_pixels, live_depth_stack.begin() + depth_pixels);
            prev_depth_frame_ = latest_depth_frame_;
        }

        std::copy(live_depth_stack.begin(), live_depth_stack.end(), depth_buffer_.begin());

        // --- Sensor estimator step (every wm_update_interval steps) ---
        if (wm_step_counter_ % config_.wm_update_interval == 0) {
            bool wm_updated = run_sensor_estimator();
            if (wm_updated) {
                update_high_feat_window();
            }
        }

        // --- Policy inference (every step) ---
        float max_abs_action = 0.0f;
        float max_abs_action_delta = 0.0f;
        try {
            auto actions = run_policy();

            // NaN/Inf check
            for (auto &val : actions) {
                if (std::isnan(val) || std::isinf(val)) {
                    val = 0.0f;
                }
            }

            // Unless raw action mode is enabled, smooth: action = 0.8 * new + 0.2 * previous
            std::vector<float> previous_action = raw_action_;
            raw_action_ = actions;

            for (size_t i = 0; i < actions.size(); i++) {
                max_abs_action =
                    std::max(max_abs_action, std::abs(raw_action_[i]));
                max_abs_action_delta =
                    std::max(max_abs_action_delta, std::abs(raw_action_[i] - previous_action[i]));
                if (!config_.use_raw_actions) {
                    actions[i] = 0.8f * actions[i] + 0.2f * previous_action[i];
                }
            }

            // Scale actions; the default mode also reduces hip joints (indices 0-3) by 0.5x
            for (int i = 0; i < config_.num_actions; i++) {
                float action_scaled = actions[i] * config_.action_scale;
                if (!config_.use_raw_actions && i < 4) {
                    action_scaled *= 0.5f;
                }
                target_dof_pos_[i] = config_.default_angles[i] + action_scaled;
            }
        } catch (const Ort::Exception &e) {
            LOG_ERROR("Policy ONNX inference error: %s", e.what());
        }

        // Near a joint limit, prevent the target from moving further toward it
        // by holding the target at the current joint position.
        for (int i = 0; i < config_.num_actions; i++) {
            float lo = config_.joint_limits[i].first;
            float hi = config_.joint_limits[i].second;
            float range = hi - lo;
            float lower_threshold = lo + 0.1f * range; // 90% toward lower limit
            float upper_threshold = hi - 0.1f * range; // 90% toward upper limit
            float q = qj_[i];

            if (q <= lower_threshold && target_dof_pos_[i] < q) {
                LOG_WARN_THROTTLED(500, "Joint %d (%s) near LOWER limit: q=%.3f lo=%.3f thresh=%.3f target=%.3f -> "
                    "clamped",
                    i, config_.sim_joint_names[i].c_str(), q, lo, lower_threshold,
                    target_dof_pos_[i]);
                target_dof_pos_[i] = q;
            } else if (q >= upper_threshold && target_dof_pos_[i] > q) {
                LOG_WARN_THROTTLED(500, "Joint %d (%s) near UPPER limit: q=%.3f hi=%.3f thresh=%.3f target=%.3f -> "
                    "clamped",
                    i, config_.sim_joint_names[i].c_str(), q, hi, upper_threshold,
                    target_dof_pos_[i]);
                target_dof_pos_[i] = q;
            }
        }

        if (log_safety_metrics(max_abs_action, max_abs_action_delta)) {
            safety_fault_ = true;
            g_emergency_stop_mode.store(true, std::memory_order_relaxed);
            cmd_(0) = 0.0f;
            cmd_(1) = 0.0f;
            cmd_(2) = 0.0f;
            create_damping_cmd();
            send_cmd();
            return;
        }

        // Build low command
        for (int i = 0; i < 12; i++) {
            int motor_idx = config_.joint2motor_idx[i];
            low_cmd_.motor_cmd()[motor_idx].q() = target_dof_pos_[i];
            low_cmd_.motor_cmd()[motor_idx].dq() = 0;
            low_cmd_.motor_cmd()[motor_idx].kp() = config_.kp;
            low_cmd_.motor_cmd()[motor_idx].kd() = config_.kd;
            low_cmd_.motor_cmd()[motor_idx].tau() = 0;
        }

        // Send the command
        send_cmd();

        // Increment world model step counter
        wm_step_counter_++;
    }

    // ==================== ONNX Inference ====================

    bool log_safety_metrics(float max_abs_action, float max_abs_action_delta) {
        float max_abs_tau_proxy = 0.0f;
        float sum_abs_tau_proxy = 0.0f;
        float max_limit_push = 0.0f;
        float max_abs_target_delta = 0.0f;
        float max_abs_joint_velocity = 0.0f;
        float max_limit_pressure = 0.0f;

        for (int i = 0; i < config_.num_actions; ++i) {
            const float tau_proxy = config_.kp * (target_dof_pos_[i] - qj_[i]) -
                                    config_.kd * dqj_[i];
            const float abs_tau = std::abs(tau_proxy);
            sum_abs_tau_proxy += abs_tau;
            max_abs_tau_proxy = std::max(max_abs_tau_proxy, abs_tau);

            const float lo = config_.joint_limits[i].first;
            const float hi = config_.joint_limits[i].second;
            const float center = 0.5f * (lo + hi);
            const float half_range = 0.5f * (hi - lo);
            const float limit_pressure =
                half_range > 0.0f ? std::clamp(std::abs((qj_[i] - center) / half_range), 0.0f, 1.0f)
                                   : 1.0f;
            max_limit_pressure = std::max(max_limit_pressure, limit_pressure);

            const bool pushing_lower = qj_[i] < center && tau_proxy < 0.0f;
            const bool pushing_upper = qj_[i] > center && tau_proxy > 0.0f;
            const float limit_push =
                (pushing_lower || pushing_upper) ? limit_pressure * abs_tau : 0.0f;
            max_limit_push = std::max(max_limit_push, limit_push);

            max_abs_target_delta =
                std::max(max_abs_target_delta, std::abs(target_dof_pos_[i] - previous_target_dof_pos_[i]));
            max_abs_joint_velocity = std::max(max_abs_joint_velocity, std::abs(dqj_[i]));
        }

        previous_target_dof_pos_ = target_dof_pos_;

        const float mean_abs_tau_proxy =
            sum_abs_tau_proxy / static_cast<float>(config_.num_actions);
        const float high_feat_norm =
            std::sqrt(std::inner_product(high_feat_.begin(), high_feat_.end(), high_feat_.begin(), 0.0f));
        const float wm_hidden_norm =
            std::sqrt(std::inner_product(wm_hidden_.begin(), wm_hidden_.end(), wm_hidden_.begin(), 0.0f));

        const bool ood_sample =
            max_abs_tau_proxy > config_.safety_max_tau_threshold ||
            max_abs_joint_velocity > config_.safety_max_joint_vel_threshold;
        if (ood_sample) {
            safety_ood_counter_++;
        } else {
            safety_ood_counter_ = 0;
        }
        const bool ood_detected = safety_ood_counter_ >= config_.safety_ood_count_threshold;

        static int safety_log_counter = 0;
        const bool should_log = safety_log_counter++ % 10 == 0 || ood_sample;
        if (should_log) {
            fmt::print(
                "SAFETY ood={} ood_count={:2d} max_tau={:8.3f} mean_tau={:8.3f} max_limit_push={:8.3f} max_action={:8.3f} max_action_delta={:8.3f} max_target_delta={:8.3f} max_joint_vel={:8.3f} max_limit_pressure={:8.3f} high_feat_norm={:9.3f} wm_hidden_norm={:9.3f}\n",
                ood_detected ? 1 : 0, safety_ood_counter_, max_abs_tau_proxy, mean_abs_tau_proxy,
                max_limit_push, max_abs_action, max_abs_action_delta, max_abs_target_delta,
                max_abs_joint_velocity, max_limit_pressure, high_feat_norm, wm_hidden_norm);
        }

        if (ood_detected && !safety_fault_logged_) {
            safety_fault_logged_ = true;
            LOG_ERROR("OOD safety fault: max_tau=%.3f threshold=%.3f max_joint_vel=%.3f threshold=%.3f. "
                "Switching to damping commands.",
                max_abs_tau_proxy, config_.safety_max_tau_threshold, max_abs_joint_velocity,
                config_.safety_max_joint_vel_threshold);
        }

        return ood_detected;
    }

    std::vector<float> run_policy() {
        if (policy_uses_high_feat_window_) {
            return run_window_policy();
        }
        if (policy_uses_stateless_high_feat_) {
            return run_stateless_high_feat_policy();
        }

        return run_recurrent_policy();
    }

    void update_high_feat_window() {
        if (!policy_uses_high_feat_window_ || high_feat_window_.empty()) {
            return;
        }
        const size_t latent_dim = static_cast<size_t>(config_.sensor_latent_dim);
        const size_t window = static_cast<size_t>(config_.policy_window);
        if (window == 0 || high_feat_window_.size() != window * latent_dim) {
            return;
        }
        if (window > 1) {
            std::copy_backward(
                high_feat_window_.begin(), high_feat_window_.begin() + (window - 1) * latent_dim,
                high_feat_window_.end());
        }
        std::copy_n(high_feat_.begin(), latent_dim, high_feat_window_.begin());
    }

    std::vector<float> run_window_policy() {
        // Policy export with a feature window:
        //   inputs:  proprio_hist ([1,] prop_hist_dim),
        //            high_feat_window ([1,] window, sensor_latent_dim)
        //   outputs: actions (1, 12)
        const std::array<const char *, 2> input_names = {"proprio_hist", "high_feat_window"};
        const std::array<const char *, 1> output_names = {"actions"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::array<Ort::Value, 2> input_tensors = {
            Ort::Value::CreateTensor<float>(
                memory_info, prop_hist_.data(), prop_hist_.size(), policy_prop_hist_dims_.data(),
                policy_prop_hist_dims_.size()),
            Ort::Value::CreateTensor<float>(
                memory_info, high_feat_window_.data(), high_feat_window_.size(),
                policy_high_feat_window_dims_.data(), policy_high_feat_window_dims_.size())};

        auto output_tensors = policy_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        float *actions_ptr = output_tensors[0].GetTensorMutableData<float>();
        return std::vector<float>(actions_ptr, actions_ptr + config_.num_actions);
    }

    std::vector<float> run_stateless_high_feat_policy() {
        // Fixed-window transformer export:
        //   inputs:  proprio_hist ([1,] prop_hist_dim), high_feat ([1,] sensor_latent_dim)
        //   outputs: actions (1, 12)
        const std::array<const char *, 2> input_names = {"proprio_hist", "high_feat"};
        const std::array<const char *, 1> output_names = {"actions"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::array<Ort::Value, 2> input_tensors = {
            Ort::Value::CreateTensor<float>(
                memory_info, prop_hist_.data(), prop_hist_.size(), policy_prop_hist_dims_.data(),
                policy_prop_hist_dims_.size()),
            Ort::Value::CreateTensor<float>(
                memory_info, high_feat_.data(), high_feat_.size(), policy_high_feat_dims_.data(),
                policy_high_feat_dims_.size())};

        auto output_tensors = policy_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        float *actions_ptr = output_tensors[0].GetTensorMutableData<float>();
        return std::vector<float>(actions_ptr, actions_ptr + config_.num_actions);
    }

    std::vector<float> run_recurrent_policy() {
        // Recurrent policy exports:
        //   inputs:  proprio (1, 45), optional proprio_hist/depth_stack,
        //            high_feat (1, sensor_latent_dim), gru_hidden_in (L,1,H)
        //   outputs used: actions (1, 12), gru_hidden_out (L,1,H)
        std::vector<const char *> input_names = {"proprio"};
        if (policy_uses_prop_hist_) {
            input_names.push_back("proprio_hist");
        }
        if (policy_uses_depth_stack_) {
            input_names.push_back("depth_stack");
        }
        input_names.push_back("high_feat");
        input_names.push_back("gru_hidden_in");

        std::vector<const char *> output_names = {"actions", "gru_hidden_out"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, proprio_obs_.data(), proprio_obs_.size(), policy_proprio_dims_.data(),
                policy_proprio_dims_.size()));
        if (policy_uses_prop_hist_) {
            input_tensors.push_back(
                Ort::Value::CreateTensor<float>(
                    memory_info, prop_hist_.data(), prop_hist_.size(), policy_prop_hist_dims_.data(),
                    policy_prop_hist_dims_.size()));
        }
        if (policy_uses_depth_stack_) {
            input_tensors.push_back(
                Ort::Value::CreateTensor<float>(
                    memory_info, depth_buffer_.data(), depth_buffer_.size(), policy_depth_dims_.data(),
                    policy_depth_dims_.size()));
        }
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, high_feat_.data(), high_feat_.size(), policy_high_feat_dims_.data(),
                policy_high_feat_dims_.size()));
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, policy_gru_hidden_.data(), policy_gru_hidden_.size(),
                policy_gru_dims_.data(), policy_gru_dims_.size()));

        auto output_tensors = policy_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        // Persist updated policy GRU state for the next step.
        float *gru_hidden_out_ptr = output_tensors[1].GetTensorMutableData<float>();
        std::copy_n(
            gru_hidden_out_ptr, policy_gru_hidden_.size(), policy_gru_hidden_.begin());

        float *actions_ptr = output_tensors[0].GetTensorMutableData<float>();
        return std::vector<float>(actions_ptr, actions_ptr + config_.num_actions);
    }

    bool run_sensor_estimator() {
        // Estimator inputs: prop_hist ([1,]450), depth_stack ([1,]C,H,W),
        // hidden_in ([1,]gru_hidden_dim)
        // Estimator outputs used: feature ([1,]sensor_latent_dim),
        // hidden_out ([1,]gru_hidden_dim).
        const std::array<const char *, 3> input_names = {"prop_hist", "depth_stack", "hidden_in"};
        std::vector<const char *> output_names = {"feature", "hidden_out"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, prop_hist_.data(), prop_hist_.size(), estimator_prop_hist_dims_.data(),
                estimator_prop_hist_dims_.size()));
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, depth_buffer_.data(), depth_buffer_.size(), estimator_depth_dims_.data(),
                estimator_depth_dims_.size()));
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, wm_hidden_.data(), wm_hidden_.size(), estimator_hidden_dims_.data(),
                estimator_hidden_dims_.size()));

        try {
            auto output_tensors = sensor_estimator_session_->Run(
                Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                input_tensors.size(), output_names.data(), output_names.size());

            // Cache the policy feature separately from the estimator recurrent state.
            float *feat_ptr = output_tensors[0].GetTensorMutableData<float>();
            std::copy_n(feat_ptr, config_.sensor_latent_dim, high_feat_.begin());

            // Update GRU hidden state.
            float *hidden_ptr = output_tensors[1].GetTensorMutableData<float>();
            std::copy_n(hidden_ptr, config_.gru_hidden_dim, wm_hidden_.begin());

            return true;

        } catch (const Ort::Exception &e) {
            LOG_ERROR("Estimator ONNX inference error: %s", e.what());
            return false;
        }
    }

    // ==================== Member Variables ====================
  private:
    // Go2 Configuration
    LocomotionConfig config_;

    // ONNX Runtime
    Ort::Env env_;
    std::unique_ptr<Ort::Session> policy_session_;
    std::unique_ptr<Ort::Session> sensor_estimator_session_;
    bool policy_uses_proprio_ = false;
    bool policy_uses_prop_hist_ = false;
    bool policy_uses_depth_stack_ = false;
    bool policy_uses_high_feat_ = false;
    bool policy_uses_high_feat_window_ = false;
    bool policy_uses_gru_state_ = false;
    bool policy_uses_stateless_high_feat_ = false;
    std::vector<int64_t> policy_proprio_dims_;
    std::vector<int64_t> policy_prop_hist_dims_;
    std::vector<int64_t> policy_depth_dims_;
    std::vector<int64_t> policy_high_feat_dims_;
    std::vector<int64_t> policy_high_feat_window_dims_;
    std::vector<int64_t> policy_gru_dims_;
    std::vector<int64_t> estimator_prop_hist_dims_;
    std::vector<int64_t> estimator_depth_dims_;
    std::vector<int64_t> estimator_hidden_dims_;

    // Controller state
    VectorXf qj_;
    VectorXf dqj_;
    std::vector<float> raw_action_; // Raw policy output (unscaled)
    VectorXf target_dof_pos_;
    VectorXf previous_target_dof_pos_;
    Vector3f cmd_;
    std::vector<float> proprio_obs_; // Current proprioception (45)
    std::vector<float> prop_hist_;   // Proprioception history (450), newest first

    // Estimator state
    std::vector<float> high_feat_; // Cached high-level sensor latent (sensor_latent_dim)
    std::vector<float> high_feat_window_; // Latest-first high-level latent window
    std::vector<float> wm_hidden_; // GRU hidden state (gru_hidden_dim)
    std::vector<float> policy_gru_hidden_; // Policy GRU state (num_layers * rnn_hidden_dim)
    int wm_step_counter_;          // Counts policy steps for estimator scheduling

    // Depth buffer
    std::vector<float> depth_buffer_;       // (C*H*W) channels-first
    std::vector<float> latest_depth_frame_; // Latest frame from ZMQ
    std::vector<float> prev_depth_frame_;   // Previous depth frame
    bool depth_prev_initialized_ = false;   // True after first depth stack is built
    bool depth_frame_received_ = false;
    std::atomic<bool> depth_size_mismatch_{false};

    // ZMQ
    std::unique_ptr<zmq::context_t> zmq_ctx_;
    std::unique_ptr<zmq::socket_t> zmq_depth_sub_;
    std::unique_ptr<zmq::socket_t> zmq_cmd_sub_;
    std::thread depth_recv_thread_;
    std::atomic<bool> depth_recv_running_;
    std::mutex depth_mutex_;

    // Unitree SDK
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> wireless_controller_;
    std::unique_ptr<unitree::robot::go2::SportClient> sport_client_;
    std::unique_ptr<MotionSwitcherClient> motion_switcher_client_;
    unitree_go::msg::dds_::LowCmd_ low_cmd_{};
    unitree_go::msg::dds_::LowState_ low_state_{};
    unitree_go::msg::dds_::WirelessController_ wireless_controller_msg_{};

    Gamepad gamepad_;

    // Controller status
    bool running_;
    bool crouching_;
    bool initialized_;
    bool gamepad_control_mode_ = false; // R1 toggle: false=ZMQ, true=gamepad direct
    bool secondary_button_linear_scale_active_ = false; // R2 toggle in buttons mode
    bool safety_fault_ = false;
    bool safety_fault_logged_ = false;
    int safety_ood_counter_ = 0;
};

int main(int argc, char *argv[]) {
    signal(SIGINT, interrupt_handler);

    // Parse command line arguments
    cxxopts::Options options("LocomotionNodeWM", "Unitree Go2 World-Model Locomotion Control");
    options.add_options()(
        "n,net", "network interface", cxxopts::value<std::string>()->default_value("lo"))(
        "m,model-dir",
        "subfolder under model/ containing policy.onnx and sensor_estimator.onnx "
        "(empty = use model/ directly)",
        cxxopts::value<std::string>()->default_value(""))(
        "trt-fp16", "Enable TensorRT FP16 inference (disabled by default)")(
        "trt-cache", "Enable the per-model TensorRT engine cache (disabled by default)")(
        "depth-port", "ZMQ depth subscriber port", cxxopts::value<int>()->default_value("5560"))(
        "depth-host", "ZMQ depth subscriber host",
        cxxopts::value<std::string>()->default_value("localhost"))(
        "cmd-port", "ZMQ command subscriber port", cxxopts::value<int>()->default_value("5562"))(
        "cmd-host", "ZMQ command subscriber host (controller IP)",
        cxxopts::value<std::string>()->default_value("localhost"))(
        "gamepad-mode", "Wireless gamepad control mode: joystick or buttons",
        cxxopts::value<std::string>()->default_value("joystick"))(
        "gamepad-linear-scale", "Primary wireless gamepad linear velocity scale",
        cxxopts::value<float>()->default_value("1.0"))(
        "gamepad-linear-scale-2", "Secondary linear velocity scale (buttons mode, toggled by R2)",
        cxxopts::value<float>()->default_value("1.0"))(
        "gamepad-angular-scale", "Wireless gamepad angular velocity scale",
        cxxopts::value<float>()->default_value("1.0"))(
        "ood-count-threshold",
        "Override consecutive OOD sample count required before safety fault",
        cxxopts::value<int>())(
        "raw-actions", "Disable action smoothing and hip scaling; apply action scale only")(
        "h,help", "Print usage");
    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        fmt::print("{}\n", options.help());
        return 0;
    }

    auto net_interface = args["net"].as<std::string>();
    auto gamepad_mode = args["gamepad-mode"].as<std::string>();
    float gamepad_linear_scale = args["gamepad-linear-scale"].as<float>();
    float gamepad_linear_scale_2 = args["gamepad-linear-scale-2"].as<float>();
    float gamepad_angular_scale = args["gamepad-angular-scale"].as<float>();
    if (gamepad_mode != "joystick" && gamepad_mode != "buttons") {
        LOG_ERROR("Invalid --gamepad-mode '%s'; expected 'joystick' or 'buttons'",
            gamepad_mode.c_str());
        return 1;
    }
    if (!std::isfinite(gamepad_linear_scale) || gamepad_linear_scale < 0.0f) {
        LOG_ERROR("Invalid --gamepad-linear-scale; expected a finite non-negative value");
        return 1;
    }
    if (!std::isfinite(gamepad_linear_scale_2) || gamepad_linear_scale_2 < 0.0f) {
        LOG_ERROR("Invalid --gamepad-linear-scale-2; expected a finite non-negative value");
        return 1;
    }
    if (!std::isfinite(gamepad_angular_scale) || gamepad_angular_scale < 0.0f) {
        LOG_ERROR("Invalid --gamepad-angular-scale; expected a finite non-negative value");
        return 1;
    }
    int net_idx = net_interface == "lo" ? 1 : 0;

    // Initialize unitree sdk
    ChannelFactory::Instance()->Init(net_idx, net_interface);

    LocomotionConfig config;
    config.use_joystick_gamepad_control = gamepad_mode == "joystick";
    config.gamepad_linear_scale = gamepad_linear_scale;
    config.gamepad_linear_scale_2 = gamepad_linear_scale_2;
    config.gamepad_angular_scale = gamepad_angular_scale;
    if (config.use_joystick_gamepad_control) {
        fmt::print("Wireless gamepad mode: {}, linear scale: {}, angular scale: {}\n",
            gamepad_mode, gamepad_linear_scale, gamepad_angular_scale);
    } else {
        fmt::print(
            "Wireless gamepad mode: buttons, linear scales: {} / {} (R2 toggle), angular scale: {}\n",
            gamepad_linear_scale, gamepad_linear_scale_2, gamepad_angular_scale);
    }

    std::ifstream conf_file("config.json");
    nlohmann::json yaml_conf = nlohmann::json::parse(conf_file);

    config.use_sim = net_interface == "lo";
    config.kp = yaml_conf["kp"].get<float>();
    config.kd = yaml_conf["kd"].get<float>();
    config.forward_clip = yaml_conf["linear_velocity_clip"].get<float>();
    config.lateral_clip = yaml_conf["linear_velocity_lateral_clip"].get<float>();
    config.angular_clip = yaml_conf["angular_velocity_clip"].get<float>();

    fmt::print("Using kp: {}, kd: {}\n", config.kp, config.kd);

    // Override model directory from CLI if provided
    if (!args["model-dir"].as<std::string>().empty()) {
        config.model_subdir = args["model-dir"].as<std::string>();
        fmt::print("Using model subfolder: model/{}\n", config.model_subdir);
    }
    config.trt_fp16_enabled = args.count("trt-fp16") > 0;
    config.trt_cache_enabled = args.count("trt-cache") > 0;
    fmt::print(
        "TensorRT options: fp16={}, engine_cache={}\n",
        config.trt_fp16_enabled ? "enabled" : "disabled",
        config.trt_cache_enabled ? "enabled" : "disabled");

    // ZMQ ports from CLI
    config.depth_sub_port = args["depth-port"].as<int>();
    config.depth_sub_host = args["depth-host"].as<std::string>();
    config.cmd_sub_port = args["cmd-port"].as<int>();
    config.cmd_sub_host = args["cmd-host"].as<std::string>();
    config.use_raw_actions = args.count("raw-actions") > 0;
    if (args.count("ood-count-threshold") > 0) {
        config.safety_ood_count_threshold = args["ood-count-threshold"].as<int>();
        fmt::print(
            "OOD safety count threshold override: {}\n", config.safety_ood_count_threshold);
    }
    fmt::print(
        "ZMQ: depth={}://{}, cmd={}://{}\n", config.depth_sub_port, config.depth_sub_host,
        config.cmd_sub_port, config.cmd_sub_host);

    config.init();
    auto node = std::make_shared<LocomotionNode>(config);

    // Run control loop
    node->run_control_loop();

    return 0;
}
