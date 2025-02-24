cmake_minimum_required(VERSION 3.22)

project(
  diffuse
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  set(DIFFUSE_CPP_COMPILE_OPTIONS /W4 /bigobj)
else()
  set(DIFFUSE_CPP_COMPILE_OPTIONS -Wall -Wextra -pedantic -march=native)
endif()

include(FetchContent)

# Fetch noarr
FetchContent_Declare(
  noarr_structures
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/noarr-structures)
FetchContent_MakeAvailable(noarr_structures)

file(GLOB src_files "src/*")

add_executable(diffuse ${src_files})

target_compile_options(
  diffuse PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${DIFFUSE_CPP_COMPILE_OPTIONS}>)

target_include_directories(diffuse
                           PUBLIC ${noarr_structures_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
target_link_libraries(diffuse PUBLIC OpenMP::OpenMP_CXX)
