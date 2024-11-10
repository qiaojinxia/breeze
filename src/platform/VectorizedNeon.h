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

    template <>
    class Vectorized<i32> {
    public:
        static constexpr int size() { return 4; }  // NEON is 128-bit, so 4 x i32
        explicit Vectorized(const i32 v) : values(vdupq_n_s32(v)) {}
        explicit Vectorized(const float32x4_t v) : values(vcvtq_s32_f32(v)) {}
        explicit Vectorized(const int32x4_t v) : values(v) {}

        const i32& operator[](const index_t idx) const {
            return reinterpret_cast<const i32*>(&values)[idx];
        }

        i32& operator[](const index_t idx) {
            return reinterpret_cast<i32*>(&values)[idx];
        }

        static Vectorized arange(const i32 start) {
            return Vectorized(vcombine_s32(
                vcreate_s32(static_cast<uint64_t>(start) | static_cast<uint64_t>(start + 1) << 32),
                vcreate_s32(static_cast<uint64_t>(start + 2) | static_cast<uint64_t>(start + 3) << 32)
            ));
        }

        static Vectorized blendv(const Vectorized& a, const Vectorized& b, const Vectorized& mask) {
            const uint32x4_t mask_bits = vcltq_s32(vdupq_n_s32(0), mask.values);
            return Vectorized(vbslq_s32(mask_bits, b.values, a.values));
        }

    private:
        int32x4_t values;
    };

    template <>
    class Vectorized<i64> {
    public:
        static constexpr int size() { return 2; }
        Vectorized() : values(vdupq_n_s64(0)) {}
        explicit Vectorized(const i64 v) : values(vdupq_n_s64(v)) {}
        explicit Vectorized(const float32x4_t v) : values(vreinterpretq_s64_f32(v)) {}
        explicit Vectorized(const float64x2_t v) : values(vreinterpretq_s64_f64(v)) {}
        explicit Vectorized(const int64x2_t v) : values(v) {}

        [[nodiscard]] const int64x2_t& get_values() const { return values; }
        int64x2_t& get_values() { return values; }

        const i64& operator[](const index_t idx) const {
            return reinterpret_cast<const i64*>(&values)[idx];
        }

        i64& operator[](const index_t idx) {
            return reinterpret_cast<i64*>(&values)[idx];
        }

        [[nodiscard]] i64 horizontal_sum() const {
            return vgetq_lane_s64(values, 0) + vgetq_lane_s64(values, 1);
        }

        static Vectorized blendv(const Vectorized& a, const Vectorized& b, const Vectorized& mask) {
            const uint64x2_t mask_bits = vcltq_s64(vdupq_n_s64(0), mask.values);
            return Vectorized(vbslq_s64(mask_bits, b.values, a.values));
        }

        Vectorized operator+(const i64 scalar) const {
            return Vectorized(vaddq_s64(values, vdupq_n_s64(scalar)));
        }

        Vectorized operator>(const Vectorized& other) const {
            return Vectorized(vreinterpretq_s64_u64(vcgtq_s64(values, other.values)));
        }

        Vectorized operator<(const Vectorized& other) const {
            return Vectorized(vreinterpretq_s64_u64(vcltq_s64(values, other.values)));
        }

        Vectorized operator+(const Vectorized& other) const {
            return Vectorized(vaddq_s64(values, other.values));
        }

        Vectorized operator-(const Vectorized& other) const {
            return Vectorized(vsubq_s64(values, other.values));
        }

        Vectorized operator*(const Vectorized& other) const {
            alignas(16) i64 a_raw[2], b_raw[2], result_raw[2];
            vst1q_s64(a_raw, values);
            vst1q_s64(b_raw, other.values);
            for(int i = 0; i < 2; i++) {
                result_raw[i] = a_raw[i] * b_raw[i];
            }
            return Vectorized(vld1q_s64(result_raw));
        }

        Vectorized operator/(const Vectorized& other) const {
            alignas(16) i64 a_raw[2], b_raw[2], result_raw[2];
            vst1q_s64(a_raw, values);
            vst1q_s64(b_raw, other.values);
            for(int i = 0; i < 2; i++) {
                result_raw[i] = a_raw[i] / b_raw[i];
            }
            return Vectorized(vld1q_s64(result_raw));
        }

        [[nodiscard]] Vectorized abs() const {
            return Vectorized(vabsq_s64(values));
        }

        // todo i64矩阵运算
        [[nodiscard]]  Vectorized sin() const { return Vectorized(1ll); }
        [[nodiscard]] Vectorized cos() const { return Vectorized(1ll); }
        [[nodiscard]] Vectorized pow(const Vectorized& exponents) const { (void)exponents; return Vectorized(1ll); }
        [[nodiscard]] Vectorized tan() const { return Vectorized(1ll); }
        [[nodiscard]] Vectorized atan() const { return Vectorized(1ll); }
        [[nodiscard]] Vectorized log() const { return Vectorized(1ll); }
        [[nodiscard]] Vectorized log2() const { return Vectorized(1ll); }
        [[nodiscard]] Vectorized log10() const { return Vectorized(1ll); }
        [[nodiscard]] Vectorized exp() const { return Vectorized(1ll); }
        [[nodiscard]] Vectorized sqrt() const { return Vectorized(1ll); }

        static void prefetch(const i64* ptr) {
            __builtin_prefetch(ptr, 0, 3);
        }

        [[nodiscard]] Vectorized max(const Vectorized& other) const {
            const int64x2_t gt = vcgtq_s64(values, other.values);
            return Vectorized(vbslq_s64(gt, values, other.values));
        }

        static Vectorized max(const Vectorized& a, const Vectorized& b) {
            const int64x2_t gt = vcgtq_s64(a.values, b.values);
            return Vectorized(vbslq_s64(gt, a.values, b.values));
        }

        [[nodiscard]] Vectorized min(const Vectorized& other) const {
            const int64x2_t lt = vcltq_s64(values, other.values);  // 比较大小
            return Vectorized(vbslq_s64(lt, values, other.values));
        }


        static Vectorized min(const Vectorized& a, const Vectorized& b) {
            const int64x2_t lt = vcltq_s64(a.values, b.values);
            return Vectorized(vbslq_s64(lt, a.values, b.values));
        }

        static Vectorized loadu(const char* ptr) {
            return Vectorized(vld1q_s64(reinterpret_cast<const i64*>(ptr)));
        }

        static Vectorized loadu(const i64* ptr) {
            return Vectorized(vld1q_s64(ptr));
        }

        template<typename T0, typename T1>
        static Vectorized loadu_unaligned(const T0* ptr0, const T1* ptr1) {
            return Vectorized(vcombine_s64(
                vcreate_s64(static_cast<i64>(*ptr0)),
                vcreate_s64(static_cast<i64>(*ptr1))
            ));
        }

        static Vectorized arange(const i64 start) {
            return Vectorized(vcombine_s64(
                vcreate_s64(start),
                vcreate_s64(start + 1)
            ));
        }

        void store(i64* ptr) const {
            vst1q_s64(ptr, values);
        }

    private:
        int64x2_t values;
    };

    // Specialization for float
    template <>
    class Vectorized<float> {
    public:
        static constexpr int size() { return 4; }

        Vectorized() : values(vdupq_n_f32(0.0f)) {}
        explicit Vectorized(const float v) : values(vdupq_n_f32(v)) {}
        explicit Vectorized(const float32x4_t v) : values(v) {}

        [[nodiscard]] const float32x4_t& get_values() const { return values; }
        float32x4_t& get_values() { return values; }

        // 下标访问 const 版本
        float operator[](const index_t idx) const {
            return reinterpret_cast<const float*>(&values)[idx];
        }

        // 下标访问非 const 版本
        float& operator[](const index_t idx) {
            return reinterpret_cast<float*>(&values)[idx];
        }

        static Vectorized blendv(const Vectorized& a, const Vectorized& b, const Vectorized& mask) {
            // NEON 中使用 vbslq_f32 进行条件选择
            const uint32x4_t mask_bits = vreinterpretq_u32_f32(mask.values);
            return Vectorized(vbslq_f32(mask_bits, b.values, a.values));
        }

        // 加上一个标量
        Vectorized operator+(const float scalar) const {
            // vdupq_n_f32 创建一个所有元素都是scalar的向量
            return Vectorized(vaddq_f32(values, vdupq_n_f32(scalar)));
        }

        Vectorized operator>(const Vectorized& other) const {
            // vcgtq_f32 返回大于比较的结果
            return Vectorized(vreinterpretq_f32_u32(vcgtq_f32(values, other.values)));
        }

        Vectorized operator<(const Vectorized& other) const {
            // vcltq_f32 返回小于比较的结果
            return Vectorized(vreinterpretq_f32_u32(vcltq_f32(values, other.values)));
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

        void store(float* ptr) const {
            vst1q_f32(ptr, values);
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

        static void prefetch(const float* ptr) {
            asm volatile("prfm pldl1keep, [%0]" : : "r" (ptr));
        }
    private:
        float32x4_t values;
    };

    // Specialization for double
    template <>
    class Vectorized<double> {

    public:
        static constexpr int size() { return 2; }

        Vectorized() : values(vdupq_n_f64(0.0)) {}
        explicit Vectorized(const double v) : values(vdupq_n_f64(v)) {}
        explicit Vectorized(const float64x2_t v) : values(v) {}

        [[nodiscard]] const float64x2_t& get_values() const { return values; }
        float64x2_t& get_values() { return values; }

        double operator[](const index_t idx) const {
            return reinterpret_cast<const double*>(&values)[idx];
        }

        double& operator[](const index_t idx) {
            return reinterpret_cast<double*>(&values)[idx];
        }

        static Vectorized blendv(const Vectorized& a, const Vectorized& b, const Vectorized& mask) {
            // NEON 中使用 vbslq_f64 进行条件选择
            // mask 中的每个位为1选择b的值，为0选择a的值
            const uint64x2_t mask_bits = vreinterpretq_u64_f64(mask.values);
            return Vectorized(vbslq_f64(mask_bits, b.values, a.values));
        }

        // 加上一个标量
        Vectorized operator+(const double scalar) const {
            // vdupq_n_f64 创建一个所有元素都是scalar的向量
            return Vectorized(vaddq_f64(values, vdupq_n_f64(scalar)));
        }

        Vectorized operator>(const Vectorized& other) const {
            // vcgtq_f64 返回大于比较的结果
            // 结果中大于为全1 (0xFFFFFFFFFFFFFFFF)，否则为0
            return Vectorized(vreinterpretq_f64_u64(vcgtq_f64(values, other.values)));
        }

        Vectorized operator<(const Vectorized& other) const {
            // vcltq_f64 返回小于比较的结果
            // 结果中小于为全1 (0xFFFFFFFFFFFFFFFF)，否则为0
            return Vectorized(vreinterpretq_f64_u64(vcltq_f64(values, other.values)));
        }
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

    private:
        float64x2_t values;
    };

}  // namespace Breeze
#endif // USE_NEON

#endif // VECTORIZED_H