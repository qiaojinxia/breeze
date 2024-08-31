//
// Created by caomaobay on 2024/8/24.
//

#ifndef BREEZE_SIMDFACTORY_H
#define BREEZE_SIMDFACTORY_H


#ifdef USE_AVX2
#include "VectorizedAvx2.h"
#elif defined(USE_NEON)
#include "VectorizedNeon.h"
#endif

#endif // BREEZE_SIMDFACTORY_H