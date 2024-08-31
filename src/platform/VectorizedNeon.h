#ifndef VECTORIZED_H
#define VECTORIZED_H

#include <arm_neon.h>
#include <array>
#include <type_traits>

#ifdef USE_NEON
namespace Breeze {

    template <typename scalar_t>
    class Vectorized;

    // Specialization for float
    template <>
    class Vectorized<float> {
    private:
        float32x4_t values;

    public:
        static constexpr int size() { return 4; }  // NEON holds 4 floats in a 128-bit register

        Vectorized() : values(vdupq_n_f32(0.0f)) {}

        explicit Vectorized(const float v) : values(vdupq_n_f32(v)) {}

        explicit Vectorized(const float32x4_t v) : values(v) {}

        Vectorized operator+(const Vectorized& other) const {
            return Vectorized(vaddq_f32(values, other.values));
        }

        Vectorized operator-(const Vectorized& other) const {
            return Vectorized(vsubq_f32(values, other.values));
        }

        Vectorized operator*(const Vectorized& other) const {
            return Vectorized(vmulq_f32(values, other.values));
        }

        Vectorized operator/(const Vectorized& other) const {
            return Vectorized(vdivq_f32(values, other.values)); // Use NEON support for division
        }

        static Vectorized loadu(const void* ptr) {
            return Vectorized(vld1q_f32(static_cast<const float*>(ptr)));
        }

        void store(void* ptr) const {
            vst1q_f32(static_cast<float*>(ptr), values);
        }
    };

    // Specialization for double
    // NEON does not natively support double in SIMD, so we'll simulate using float64x2_t
    template <>
    class Vectorized<double> {
    private:
        float64x2_t values;

    public:
        static constexpr int size() { return 2; }  // NEON holds 2 doubles in a 128-bit register

        Vectorized() : values(vdupq_n_f64(0.0)) {}

        explicit Vectorized(const double v) : values(vdupq_n_f64(v)) {}

        explicit Vectorized(const float64x2_t v) : values(v) {}

        Vectorized operator+(const Vectorized& other) const {
            return Vectorized(vaddq_f64(values, other.values));
        }

        Vectorized operator-(const Vectorized& other) const {
            return Vectorized(vsubq_f64(values, other.values));
        }

        Vectorized operator*(const Vectorized& other) const {
            return Vectorized(vmulq_f64(values, other.values));
        }

        Vectorized operator/(const Vectorized& other) const {
            return Vectorized(vdivq_f64(values, other.values)); // Use NEON support for double division
        }

        static Vectorized loadu(const void* ptr) {
            return Vectorized(vld1q_f64(static_cast<const double*>(ptr)));
        }

        void store(void* ptr) const {
            vst1q_f64(static_cast<double*>(ptr), values);
        }
    };

}  // namespace Breeze
#endif // USE_NEON

#endif // VECTORIZED_H
