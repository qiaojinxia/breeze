#include "tests/test_cases.h"
#include <omp.h>
#include <cstdio>
#include <thread>
#include <sstream>
using namespace Breeze;
int main() {
#pragma omp parallel
    {
        std::stringstream ss;
        ss << std::this_thread::get_id();
        printf("%s, Hello, world.\n", ss.str().c_str());
    }
    TensorTest::run_all_tests();

    return 0;

}
