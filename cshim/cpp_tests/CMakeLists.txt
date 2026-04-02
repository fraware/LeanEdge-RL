cmake_minimum_required(VERSION 3.20)
project(leanrl_cpp_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(leanrl_cpp_smoke smoke_test.cpp)
target_link_libraries(leanrl_cpp_smoke GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(leanrl_cpp_smoke)
