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

        static void prefetch(const float* ptr) {
            asm volatile("prfm pldl1keep, [%0]" : : "r" (ptr));
        }

        [[nodiscard]] float horizontal_sum() const {

            // 先将四个float两两相加，得到两个float
            const float32x2_t sum64 = vadd_f32(vget_low_f32(values), vget_high_f32(values));

            // 再将两个float相加，得到最终结果
            // vpaddq_f32将相邻的两个float相加
            return vget_lane_f32(vpadd_f32(sum64, sum64), 0);
        }

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
            return Vectorized(Sleef_sinf4_u10advsimd(values));
        }

        [[nodiscard]] Vectorized cos() const {
            return Vectorized(Sleef_cosf4_u10advsimd(values));
        }

        [[nodiscard]] Vectorized pow(const Vectorized& exponents) const {
            return Vectorized(Sleef_powf4_u10advsimd(values, exponents.values));
        }

        [[nodiscard]] Vectorized tan() const {
            return Vectorized(Sleef_tanf4_u10advsimd(values));
        }

        [[nodiscard]] Vectorized atan() const {
            return Vectorized(Sleef_atanf4_u10advsimd(values));
        }

        [[nodiscard]] Vectorized log() const {
            return Vectorized(Sleef_logf4_u10advsimd(values));
        }

        [[nodiscard]] Vectorized log2() const {
            return Vectorized(Sleef_log2f4_u10advsimd(values));
        }

        [[nodiscard]] Vectorized log10() const {
            return Vectorized(Sleef_log10f4_u10advsimd(values));
        }

        [[nodiscard]] Vectorized exp() const {
            return Vectorized(Sleef_expf4_u10advsimd(values));
        }

        [[nodiscard]] Vectorized abs() const {
            return Vectorized(Sleef_fabsf4(values));
        }

        [[nodiscard]] Vectorized sqrt() const {
            return Vectorized(Sleef_sqrtf4_u05advsimd(values));
        }

        [[nodiscard]] static Vectorized randn(const Vectorized& v1, const Vectorized& v2) {
            const float32x4_t vec_two_pi = vdupq_n_f32(2.0f * M_PI);
            const float32x4_t vec_minus_two = vdupq_n_f32(-2.0f);
            const float32x4_t vec_log_u1 = Sleef_logf4_u10advsimd(v1.values);
            const float32x4_t vec_sqrt = Sleef_sqrtf4_u05advsimd(vmulq_f32(vec_minus_two, vec_log_u1));
            const float32x4_t vec_cos = Sleef_cosf4_u10advsimd(vmulq_f32(vec_two_pi, v2.values));
            const float32x4_t vec_result = vmulq_f32(vec_sqrt, vec_cos);
            return Vectorized(vec_result);
        }


        static Vectorized loadu(const char* ptr) {
            return Vectorized(vld1q_f32(reinterpret_cast<const float*>(ptr)));
        }

        [[nodiscard]] Vectorized max(const Vectorized& other) const {
            return Vectorized(vmaxq_f32(values, other.values));
        }

        static Vectorized max(const Vectorized& a, const Vectorized& b) {
            return Vectorized(vmaxq_f32(a.values, b.values));
        }

        [[nodiscard]] Vectorized min(const Vectorized& other) const {
            return Vectorized(vminq_f32(values, other.values));
        }

        static Vectorized min(const Vectorized& a, const Vectorized& b) {
            return Vectorized(vminq_f32(a.values, b.values));
        }


        static Vectorized loadu(const float* ptr) {
            return Vectorized(vld1q_f32(ptr));
        }

        template<typename T0, typename T1, typename T2, typename T3>
        static Vectorized loadu_unaligned(const T0* ptr0, const T1* ptr1, const T2* ptr2, const T3* ptr3) {
            float32x4_t result = vdupq_n_f32(0.0f);
            result = vsetq_lane_f32(static_cast<float>(*ptr0), result, 0);
            result = vsetq_lane_f32(static_cast<float>(*ptr1), result, 1);
            result = vsetq_lane_f32(static_cast<float>(*ptr2), result, 2);
            result = vsetq_lane_f32(static_cast<float>(*ptr3), result, 3);
            return Vectorized(result);
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

        // 水平求和
        [[nodiscard]] double horizontal_sum() const {
            return vaddvq_f64(values);
        }

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
            return Vectorized(Sleef_sind2_u10advsimd(values));
        }

        [[nodiscard]] Vectorized cos() const {
            return Vectorized(Sleef_cosd2_u10advsimd(values));
        }

        [[nodiscard]] Vectorized pow(const Vectorized& exponents) const {
            return Vectorized(Sleef_powd2_u10advsimd(values, exponents.values));
        }

        [[nodiscard]] Vectorized tan() const {
            return Vectorized(Sleef_tand2_u10advsimd(values));
        }

        [[nodiscard]] Vectorized atan() const {
            return Vectorized(Sleef_atand2_u10advsimd(values));
        }

        [[nodiscard]] Vectorized log() const {
            return Vectorized(Sleef_logd2_u10advsimd(values));
        }

        [[nodiscard]] Vectorized log2() const {
            return Vectorized(Sleef_log2d2_u10advsimd(values));
        }

        [[nodiscard]] Vectorized log10() const {
            return Vectorized(Sleef_log10d2_u10advsimd(values));
        }

        [[nodiscard]] Vectorized exp() const {
            return Vectorized(Sleef_expd2_u10advsimd(values));
        }

        [[nodiscard]] Vectorized abs() const {
            return Vectorized(Sleef_fabsd2(values));
        }

        [[nodiscard]] Vectorized sqrt() const {
            return Vectorized(Sleef_sqrtd2_u05advsimd(values));
        }

        [[nodiscard]] static Vectorized randn(const Vectorized& v1, const Vectorized& v2) {
            const float64x2_t vec_two_pi = vdupq_n_f64(2.0 * M_PI);
            const float64x2_t vec_minus_two = vdupq_n_f64(-2.0);
            const float64x2_t vec_log_u1 = Sleef_logd2_u10advsimd(v1.values);
            const float64x2_t vec_sqrt = Sleef_sqrtd2_u05advsimd(vmulq_f64(vec_minus_two, vec_log_u1));
            const float64x2_t vec_cos = Sleef_cosd2_u10advsimd(vmulq_f64(vec_two_pi, v2.values));
            const float64x2_t vec_result = vmulq_f64(vec_sqrt, vec_cos);
            return Vectorized(vec_result);
        }

        [[nodiscard]] Vectorized max(const Vectorized& other) const {
            return Vectorized(vmaxq_f64(values, other.values));
        }

        static Vectorized max(const Vectorized& a, const Vectorized& b) {
            return Vectorized(vmaxq_f64(a.values, b.values));
        }

        [[nodiscard]] Vectorized min(const Vectorized& other) const {
            return Vectorized(vminq_f64(values, other.values));
        }

        static Vectorized min(const Vectorized& a, const Vectorized& b) {
            return Vectorized(vminq_f64(a.values, b.values));
        }


        static Vectorized loadu(const char* ptr) {
            return Vectorized(vld1q_f64(reinterpret_cast<const double*>(ptr)));
        }

        static Vectorized loadu(const double* ptr) {
            return Vectorized(vld1q_f64(ptr));
        }

        // 不连续内存加载函数
        template<typename T0, typename T1>
        static Vectorized loadu_unaligned(const T0* ptr0, const T1* ptr1) {
            float64x2_t result = vdupq_n_f64(0.0);
            result = vsetq_lane_f64(static_cast<double>(*ptr0), result, 0);
            result = vsetq_lane_f64(static_cast<double>(*ptr1), result, 1);
            return Vectorized(result);
        }


        static void prefetch(const double* ptr) {
            asm volatile("prfm pldl1keep, [%0]" : : "r" (ptr));
        }

        void store(double* ptr) const {
            vst1q_f64(ptr, values);
        }
    };

}  // namespace Breeze
#endif // USE_NEON

#endif // VECTORIZED_H