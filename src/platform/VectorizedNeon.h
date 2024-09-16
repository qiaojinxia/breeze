#ifndef VECTORIZED_H
#define VECTORIZED_H

#include <arm_neon.h>
#include <array>
#include <type_traits>
#include <sleef.h>

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
        static constexpr int size() { return 4; }

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
            return Vectorized(vdivq_f32(values, other.values));
        }

        [[nodiscard]] Vectorized sin() const {
            return Vectorized(Sleef_sinf4_u10(values));
        }

        [[nodiscard]] Vectorized cos() const {
            return Vectorized(Sleef_cosf4_u10(values));
        }

        [[nodiscard]] Vectorized pow(const Vectorized& exponents) const {
            return Vectorized(Sleef_powf4_u10(values, exponents.values));
        }

        [[nodiscard]] Vectorized tan() const {
            return Vectorized(Sleef_tanf4_u10(values));
        }

        [[nodiscard]] Vectorized atan() const {
            return Vectorized(Sleef_atanf4_u10(values));
        }

        static Vectorized loadu(const char* ptr) {
            return Vectorized(vld1q_f32(reinterpret_cast<const float*>(ptr)));
        }

        void store(float* ptr) const {
            vst1q_f32(ptr, values);
        }
    };

    // Specialization for double
    template <>
    class Vectorized<double> {
    private:
        float64x2_t values;

    public:
        static constexpr int size() { return 2; }

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
            return Vectorized(vdivq_f64(values, other.values));
        }

        [[nodiscard]] Vectorized sin() const {
            return Vectorized(Sleef_sind2_u10(values));
        }

        [[nodiscard]] Vectorized cos() const {
            return Vectorized(Sleef_cosd2_u10(values));
        }

        [[nodiscard]] Vectorized pow(const Vectorized& exponents) const {
            return Vectorized(Sleef_powd2_u10(values, exponents.values));
        }

        [[nodiscard]] Vectorized tan() const {
            return Vectorized(Sleef_tand2_u10(values));
        }

        [[nodiscard]] Vectorized atan() const {
            return Vectorized(Sleef_atand2_u10(values));
        }

        static Vectorized loadu(const char* ptr) {
            return Vectorized(vld1q_f64(reinterpret_cast<const double*>(ptr)));
        }

        void store(double* ptr) const {
            vst1q_f64(ptr, values);
        }
    };

}  // namespace Breeze
#endif // USE_NEON

#endif // VECTORIZED_H