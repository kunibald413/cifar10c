cmake_minimum_required(VERSION 3.10)

project(cifar_train C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(cifar_train src/main.c)
