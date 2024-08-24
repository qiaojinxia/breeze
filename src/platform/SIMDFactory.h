//
// Created by caomaobay on 2024/8/24.
//

#ifndef BREEZE_SIMDFACTORY_H
#define BREEZE_SIMDFACTORY_H

#include "SIMDOps.h"

#ifdef USE_AVX2
#include "AVX2Ops.h"
#elif defined(USE_NEON)
#include "NEONOps.h"
#endif

namespace Breeze {

    template<typename T>
    const SIMDOps<T>& getSIMDOps() {
        #ifdef USE_AVX2
                return AVX2Ops<T>::getInstance();
        #elif defined(USE_NEON)
                return NEONOps<T>::getInstance();
        #endif
    }

} // namespace Breeze

#endif // BREEZE_SIMDFACTORY_H