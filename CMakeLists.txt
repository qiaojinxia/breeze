cmake_minimum_required(VERSION 3.28)
project(breeze)

set(CMAKE_CXX_STANDARD 17)

add_executable(breeze
        src/node.cpp
        src/node.h
        src/main.cpp
)
