#pragma once

#include <array>
#include <memory>
#include <system_error>
#include <vector>
#include <cstdint>

// Error handling macro
#define LEANRL_ASSERT_OK(x) do { \
    if (!(x)) { \
        throw std::system_error(std::error_code(-1, std::system_category()), \
                               "LeanRL operation failed"); \
    } \
} while(0)

namespace leanrl {

// Forward declarations
class Obs4;
class Action2;
class Env4x2;

// Observation class for 4-dimensional observations
class Obs4 {
public:
    Obs4(std::array<float, 4> data);
    Obs4(const Obs4& other);
    Obs4& operator=(const Obs4& other);
    ~Obs4();
    
    std::array<float, 4> get_data() const;
    void set_data(std::array<float, 4> data);
    
    // Array-like access
    float operator[](size_t index) const;
    float& operator[](size_t index);
    
    // Iterator support
    const float* begin() const;
    const float* end() const;
    float* begin();
    float* end();
    
    // Size
    static constexpr size_t size() { return 4; }
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Action class for 2-dimensional actions
class Action2 {
public:
    Action2(std::array<float, 2> data);
    Action2(const Action2& other);
    Action2& operator=(const Action2& other);
    ~Action2();
    
    std::array<float, 2> get_data() const;
    void set_data(std::array<float, 2> data);
    
    // Array-like access
    float operator[](size_t index) const;
    float& operator[](size_t index);
    
    // Iterator support
    const float* begin() const;
    const float* end() const;
    float* begin();
    float* end();
    
    // Size
    static constexpr size_t size() { return 2; }
    
    // Vector operations
    Action2 operator+(const Action2& other) const;
    Action2 operator-(const Action2& other) const;
    Action2 operator*(const Action2& other) const;
    Action2 operator*(float scale) const;
    
    // Clamping
    Action2 clamp(float min, float max) const;
    
    // Statistics
    float max() const;
    float min() const;
    bool is_within_bounds(float min, float max) const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Environment class for 4x2 RL environment
class Env4x2 {
public:
    Env4x2(const std::vector<uint8_t>& weights);
    Env4x2(const Env4x2& other);
    Env4x2& operator=(const Env4x2& other);
    ~Env4x2();
    
    // Core RL interface
    std::unique_ptr<Action2> reset(const Obs4& obs);
    std::unique_ptr<Action2> step(const Obs4& obs);
    
    // State management
    std::pair<uint64_t, uint64_t> get_state() const; // (step_count, episode_count)
    
    // Safety and verification
    bool check_invariant(const Obs4& obs, const Action2& action) const;
    
    // Weight management
    bool update_weights(const std::vector<uint8_t>& weights);
    std::vector<uint8_t> get_weights() const;
    
    // Utility methods
    bool is_valid() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Factory functions
std::unique_ptr<Env4x2> create_env4x2(const std::vector<uint8_t>& weights);

// Utility functions
template<typename T, size_t N>
std::array<T, N> to_array(const std::vector<T>& vec) {
    std::array<T, N> arr{};
    std::copy_n(vec.begin(), std::min(vec.size(), N), arr.begin());
    return arr;
}

template<typename T, size_t N>
std::vector<T> to_vector(const std::array<T, N>& arr) {
    return std::vector<T>(arr.begin(), arr.end());
}

// Eigen integration (optional)
#ifdef LEANRL_USE_EIGEN
#include <Eigen/Dense>

namespace eigen {
    // Convert Obs4 to Eigen::Vector4f
    Eigen::Vector4f to_eigen(const Obs4& obs);
    
    // Convert Action2 to Eigen::Vector2f
    Eigen::Vector2f to_eigen(const Action2& action);
    
    // Convert Eigen::Vector4f to Obs4
    Obs4 from_eigen(const Eigen::Vector4f& vec);
    
    // Convert Eigen::Vector2f to Action2
    Action2 from_eigen(const Eigen::Vector2f& vec);
}
#endif

// TensorRT integration (optional)
#ifdef LEANRL_USE_TENSORRT
#include <NvInfer.h>

namespace tensorrt {
    // Convert Obs4 to TensorRT tensor
    void to_tensorrt(const Obs4& obs, void* tensor_data, size_t tensor_size);
    
    // Convert Action2 from TensorRT tensor
    Action2 from_tensorrt(const void* tensor_data, size_t tensor_size);
}
#endif

} // namespace leanrl

// Global operators
leanrl::Action2 operator*(float scale, const leanrl::Action2& action); 