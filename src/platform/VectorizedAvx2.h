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

        [[nodiscard]] static Vectorized randn(const Vectorized& v1, const Vectorized& v2) {
            const __m256 vec_two_pi = _mm256_set1_ps(2.0f * M_PI);
            const __m256 vec_minus_two = _mm256_set1_ps(-2.0f);
            const __m256 vec_log_u1 = Sleef_logf8_u10avx2(v1.values);
            const __m256 vec_sqrt = _mm256_sqrt_ps(_mm256_mul_ps(vec_minus_two, vec_log_u1));
            const __m256 vec_cos = Sleef_logf8_u10avx2(_mm256_mul_ps(vec_two_pi, v2.values));
            const __m256 vec_result = _mm256_mul_ps(vec_sqrt, vec_cos);
            return Vectorized(vec_result);
        }

        static Vectorized loadu(const char* ptr) {
            return Vectorized(_mm256_loadu_ps(reinterpret_cast<const float*>(ptr)));
        }

        static Vectorized loadu(const float* ptr) {
            return Vectorized(_mm256_loadu_ps(ptr));
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

        [[nodiscard]] static Vectorized randn(const Vectorized& v1, const Vectorized& v2) {
            const __m256d vec_two_pi = _mm256_set1_pd(2.0 * M_PI);
            const __m256d vec_minus_two = _mm256_set1_pd(-2.0);
            const __m256d vec_log_u1 = Sleef_logd4_u10avx2(v1.values);
            const __m256d vec_sqrt = _mm256_sqrt_pd(_mm256_mul_pd(vec_minus_two, vec_log_u1));
            const __m256d vec_cos = Sleef_cosd4_u10avx2(_mm256_mul_pd(vec_two_pi, v2.values));
            const __m256d vec_result = _mm256_mul_pd(vec_sqrt, vec_cos);
            return Vectorized(vec_result);
        }

        static Vectorized loadu(const char* ptr) {
            return Vectorized(_mm256_loadu_pd(reinterpret_cast<const double*>(ptr)));
        }

        static Vectorized loadu(const double* ptr) {
            return Vectorized(_mm256_loadu_pd(ptr));
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