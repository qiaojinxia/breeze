#ifndef VECTORIZED_H
#define VECTORIZED_H

#include <immintrin.h>
#include <sleef.h>
#ifdef USE_AVX2
namespace Breeze {

    template <typename scalar_t>
    class Vectorized;

    // Specialization for float
    template <>
    class Vectorized<float> {
    public:
        static constexpr int size() { return 8; }

        Vectorized() : values(_mm256_setzero_ps()) {}
        explicit Vectorized(const float v) : values(_mm256_set1_ps(v)) {}
        explicit Vectorized(const __m256 v) : values(v) {}

        // 水平求和
        [[nodiscard]] float horizontal_sum() const {
            const __m128 low = _mm256_castps256_ps128(values);
            const auto high = _mm256_extractf128_ps(values, 1);
            __m128 sum = _mm_add_ps(low, high);
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            return _mm_cvtss_f32(sum);
        }

        Vectorized operator+(const Vectorized& other) const {
            return Vectorized(_mm256_add_ps(values, other.values));
        }


        Vectorized operator-(const Vectorized& other) const {
            return Vectorized(_mm256_sub_ps(values, other.values));
        }

        Vectorized operator*(const Vectorized& other) const {
            return Vectorized(_mm256_mul_ps(values, other.values));
        }

        Vectorized operator/(const Vectorized& other) const {
            return Vectorized(_mm256_div_ps(values, other.values));
        }

        [[nodiscard]] Vectorized sin() const {
            return Vectorized(Sleef_sinf8_u10avx2(values));
        }

        [[nodiscard]] Vectorized cos() const {
            return Vectorized(Sleef_cosf8_u10avx2(values));
        }

        [[nodiscard]] Vectorized pow(const Vectorized& exponents) const {
            return Vectorized(Sleef_powf8_u10avx2(values, exponents.values));
        }

        [[nodiscard]] Vectorized tan() const {
            return Vectorized(Sleef_tanf8_u10avx2(values));
        }

        [[nodiscard]] Vectorized atan() const {
            return Vectorized(Sleef_atanf8_u10avx2(values));
        }

        [[nodiscard]] Vectorized log() const {
            return Vectorized(Sleef_logf8_u10avx2(values));
        }

        [[nodiscard]] Vectorized log2() const {
            return Vectorized(Sleef_log2f8_u10avx2(values));
        }

        [[nodiscard]] Vectorized log10() const {
            return Vectorized(Sleef_log10f8_u10avx2(values));
        }

        [[nodiscard]] Vectorized exp() const {
            return Vectorized(Sleef_expf8_u10avx2(values));
        }

        [[nodiscard]] Vectorized abs() const {
            return Vectorized(Sleef_fabsf8_avx2(values));
        }

        [[nodiscard]] Vectorized sqrt() const {
            return Vectorized(Sleef_sqrtf8_avx2(values));
        }


        static void prefetch(const float* ptr) {
            __builtin_prefetch(ptr, 0, 3);
        }

        [[nodiscard]] static Vectorized randn(const Vectorized& v1, const Vectorized& v2) {
            const __m256 vec_two_pi = _mm256_set1_ps(2.0f * M_PI);
            const __m256 vec_minus_two = _mm256_set1_ps(-2.0f);
            const __m256 vec_log_u1 = Sleef_logf8_u10avx2(v1.values);
            const __m256 vec_sqrt = _mm256_sqrt_ps(_mm256_mul_ps(vec_minus_two, vec_log_u1));
            const __m256 vec_cos = Sleef_logf8_u10avx2(_mm256_mul_ps(vec_two_pi, v2.values));
            const __m256 vec_result = _mm256_mul_ps(vec_sqrt, vec_cos);
            return Vectorized(vec_result);
        }


        [[nodiscard]] Vectorized max(const Vectorized& other) const {
            return Vectorized(_mm256_max_ps(values, other.values));
        }

        static Vectorized max(const Vectorized& a, const Vectorized& b) {
            return Vectorized(_mm256_max_ps(a.values, b.values));
        }

        [[nodiscard]] Vectorized min(const Vectorized& other) const {
            return Vectorized(_mm256_min_ps(values, other.values));
        }

        static Vectorized min(const Vectorized& a, const Vectorized& b) {
            return Vectorized(_mm256_min_ps(a.values, b.values));
        }


        static Vectorized loadu(const char* ptr) {
            return Vectorized(_mm256_loadu_ps(reinterpret_cast<const float*>(ptr)));
        }

        static Vectorized loadu(const float* ptr) {
            return Vectorized(_mm256_loadu_ps(ptr));
        }


        template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
        static Vectorized loadu_unaligned(const T0* ptr0, const T1* ptr1, const T2* ptr2, const T3* ptr3,
                                          const T4* ptr4, const T5* ptr5, const T6* ptr6, const T7* ptr7) {
            const __m256 result = _mm256_set_ps(
                static_cast<float>(*ptr7),
                static_cast<float>(*ptr6),
                static_cast<float>(*ptr5),
                static_cast<float>(*ptr4),
                static_cast<float>(*ptr3),
                static_cast<float>(*ptr2),
                static_cast<float>(*ptr1),
                static_cast<float>(*ptr0)
            );
            return Vectorized(result);
        }

        void store(float* ptr) const {
            _mm256_storeu_ps(ptr, values);
        }
    private:
        __m256 values;
    };

    // Specialization for double
    template <>
    class Vectorized<double> {
    public:
        static constexpr int size() { return 4; }

        Vectorized() : values(_mm256_setzero_pd()) {}
        explicit Vectorized(const double v) : values(_mm256_set1_pd(v)) {}
        explicit Vectorized(const __m256d v) : values(v) {}

        [[nodiscard]] double horizontal_sum() const {
            // 1. 分离高低 128 位
            const __m128d low = _mm256_castpd256_pd128(values);
            const __m128d high = _mm256_extractf128_pd(values, 1);

            // 2. 将高低位相加
            __m128d sum = _mm_add_pd(low, high);

            // 3. 将结果的两个双精度数相加
            sum = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));

            // 4. 提取结果
            return _mm_cvtsd_f64(sum);
        }

        Vectorized operator+(const Vectorized& other) const {
            return Vectorized(_mm256_add_pd(values, other.values));
        }

        Vectorized operator-(const Vectorized& other) const {
            return Vectorized(_mm256_sub_pd(values, other.values));
        }

        Vectorized operator*(const Vectorized& other) const {
            return Vectorized(_mm256_mul_pd(values, other.values));
        }

        Vectorized operator/(const Vectorized& other) const {
            return Vectorized(_mm256_div_pd(values, other.values));
        }

        [[nodiscard]] Vectorized sin() const {
            return Vectorized(Sleef_sinf8_u10avx2(values));
        }

        [[nodiscard]] Vectorized cos() const {
            return Vectorized(Sleef_cosd4_u10avx2(values));
        }

        [[nodiscard]] Vectorized pow(const Vectorized& exponents) const {
            return Vectorized(Sleef_powd4_u10avx2(values, exponents.values));
        }

        [[nodiscard]] Vectorized tan() const {
            return Vectorized(Sleef_tand4_u10avx2(values));
        }

        [[nodiscard]] Vectorized atan() const {
            return Vectorized(Sleef_atand4_u10avx2(values));
        }


        [[nodiscard]] Vectorized log() const {
            return Vectorized(Sleef_logd4_u10avx2(values));
        }

        [[nodiscard]] Vectorized log2() const {
            return Vectorized(Sleef_log2d4_u10avx2(values));
        }

        [[nodiscard]] Vectorized log10() const {
            return Vectorized(Sleef_log10d4_u10avx2(values));
        }

        [[nodiscard]] Vectorized exp() const {
            return Vectorized(Sleef_expd4_u10avx2(values));
        }

        [[nodiscard]] Vectorized abs() const {
            return Vectorized(Sleef_fabsd4_avx2(values));
        }

        [[nodiscard]] Vectorized sqrt() const {
            return Vectorized(Sleef_sqrtd4_avx2(values));
        }

        static void prefetch(const double* ptr) {
            __builtin_prefetch(ptr, 0, 3);
        }

        [[nodiscard]] static Vectorized randn(const Vectorized& v1, const Vectorized& v2) {
            const __m256d vec_two_pi = _mm256_set1_pd(2.0 * M_PI);
            const __m256d vec_minus_two = _mm256_set1_pd(-2.0);
            const __m256d vec_log_u1 = Sleef_logd4_u10avx2(v1.values);
            const __m256d vec_sqrt = _mm256_sqrt_pd(_mm256_mul_pd(vec_minus_two, vec_log_u1));
            const __m256d vec_cos = Sleef_cosd4_u10avx2(_mm256_mul_pd(vec_two_pi, v2.values));
            const __m256d vec_result = _mm256_mul_pd(vec_sqrt, vec_cos);
            return Vectorized(vec_result);
        }

        [[nodiscard]] Vectorized max(const Vectorized& other) const {
            return Vectorized(_mm256_max_pd(values, other.values));
        }

        static Vectorized max(const Vectorized& a, const Vectorized& b) {
            return Vectorized(_mm256_max_pd(a.values, b.values));
        }


        [[nodiscard]] Vectorized min(const Vectorized& other) const {
            return Vectorized(_mm256_min_pd(values, other.values));
        }

        static Vectorized min(const Vectorized& a, const Vectorized& b) {
            return Vectorized(_mm256_min_pd(a.values, b.values));
        }

        static Vectorized loadu(const char* ptr) {
            return Vectorized(_mm256_loadu_pd(reinterpret_cast<const double*>(ptr)));
        }

        static Vectorized loadu(const double* ptr) {
            return Vectorized(_mm256_loadu_pd(ptr));
        }

        template<typename T0, typename T1, typename T2, typename T3>
        static Vectorized loadu_unaligned(const T0* ptr0, const T1* ptr1, const T2* ptr2, const T3* ptr3) {
            const __m256d vec_result = _mm256_set_pd(
                static_cast<double>(*ptr3),
                static_cast<double>(*ptr2),
                static_cast<double>(*ptr1),
                static_cast<double>(*ptr0)
            );
            return Vectorized(vec_result);
        }

        void store(double* ptr) const {
            _mm256_storeu_pd(ptr, values);
        }
    private:
        __m256d values;
    };

}  // namespace Breeze
#endif // USE_AVX2

#endif // VECTORIZED_H