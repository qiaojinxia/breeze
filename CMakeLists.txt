cmake_minimum_required(VERSION 3.28)
project(breeze)

set(CMAKE_CXX_STANDARD 17)

# 查找 Armadillo 库
find_package(Armadillo REQUIRED)

add_executable(breeze src/main.cpp
        src/node.cpp
        src/node.h
        src/loss_functions.h)

# 链接 Armadillo 库
target_link_libraries(breeze ${ARMADILLO_LIBRARIES})

# 包含 Armadillo 头文件
target_include_directories(breeze PRIVATE ${ARMADILLO_INCLUDE_DIRS})