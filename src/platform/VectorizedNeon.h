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

        [[nodiscard]] static Vectorized randn(const Vectorized& v1, const Vectorized& v2) {
            const float32x4_t vec_two_pi = vdupq_n_f32(2.0f * M_PI);
            const float32x4_t vec_minus_two = vdupq_n_f32(-2.0f);
            const float32x4_t vec_log_u1 = Sleef_logf4_u10(v1.values);
            const float32x4_t vec_sqrt = Sleef_sqrtf4_u05(vmulq_f32(vec_minus_two, vec_log_u1));
            const float32x4_t vec_cos = Sleef_cosf4_u10(vmulq_f32(vec_two_pi, v2.values));
            const float32x4_t vec_result = vmulq_f32(vec_sqrt, vec_cos);
            return Vectorized(vec_result);
        }

        static Vectorized loadu(const char* ptr) {
            return Vectorized(vld1q_f32(reinterpret_cast<const float*>(ptr)));
        }

        static Vectorized loadu(const float* ptr) {
            return Vectorized(vld1q_f32(ptr));
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

        [[nodiscard]] static Vectorized randn(const Vectorized& v1, const Vectorized& v2) {
            const float64x2_t vec_two_pi = vdupq_n_f64(2.0 * M_PI);
            const float64x2_t vec_minus_two = vdupq_n_f64(-2.0);
            const float64x2_t vec_log_u1 = Sleef_logd2_u10(v1.values);
            const float64x2_t vec_sqrt = Sleef_sqrtd2_u05(vmulq_f64(vec_minus_two, vec_log_u1));
            const float64x2_t vec_cos = Sleef_cosd2_u10(vmulq_f64(vec_two_pi, v2.values));
            const float64x2_t vec_result = vmulq_f64(vec_sqrt, vec_cos);
            return Vectorized(vec_result);
        }

        static Vectorized loadu(const char* ptr) {
            return Vectorized(vld1q_f64(reinterpret_cast<const double*>(ptr)));
        }

        static Vectorized loadu(const double* ptr) {
            return Vectorized(vld1q_f64(ptr));
        }

        void store(double* ptr) const {
            vst1q_f64(ptr, values);
        }
    };

}  // namespace Breeze
#endif // USE_NEON

#endif // VECTORIZED_H