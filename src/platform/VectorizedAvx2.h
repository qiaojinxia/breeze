#ifndef VECTORIZED_H
#define VECTORIZED_H

#include <immintrin.h>
#include <array>
#include <type_traits>

#ifdef USE_AVX2
namespace Breeze {

    template <typename scalar_t>
    class Vectorized;

    // Specialization for float
    template <>
    class Vectorized<float> {
    private:
        __m256 values;

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

        static Vectorized loadu(const char* ptr) {
            return Vectorized(_mm256_loadu_ps(reinterpret_cast<const float*>(ptr)));
        }

        void store(float* ptr) const {
            _mm256_storeu_ps(ptr, values);
        }
    };

    // Specialization for double
    template <>
    class Vectorized<double> {
    private:
        __m256d values;

    public:
        static constexpr int size() { return 4; }  // AVX2 can store 4 doubles in a __m256d

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

        static Vectorized loadu(const char* ptr) {
            return Vectorized(_mm256_loadu_pd(reinterpret_cast<const double*>(ptr)));
        }

        void store(double* ptr) const {
            _mm256_storeu_pd(ptr, values);
        }
    };

}  // namespace Breeze
#endif // USE_AVX2

#endif // VECTORIZED_H
