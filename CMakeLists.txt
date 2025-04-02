cmake_minimum_required(VERSION 3.22)

project(
  diffuse
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_ASAN OFF)

if(USE_ASAN)
  add_compile_options(-fsanitize=address -g)
  add_link_options(-fsanitize=address)
endif()

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

# Fetch argparse
FetchContent_Declare(
  argparse
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/argparse)
FetchContent_MakeAvailable(argparse)

# Fetch json
FetchContent_Declare(
  json
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/json)
FetchContent_MakeAvailable(json)

# Fetch highway
FetchContent_Declare(
  highway
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/highway)

option(HWY_ENABLE_TESTS OFF)
option(HWY_ENABLE_CONTRIB OFF)
option(HWY_ENABLE_INSTALL OFF)
option(HWY_ENABLE_EXAMPLES OFF)

FetchContent_MakeAvailable(highway)

find_package(OpenMP REQUIRED)
find_package(LAPACK REQUIRED)

file(GLOB src_files "src/*")

set(mpi_src_files "${CMAKE_CURRENT_SOURCE_DIR}/src/MPI_1D_blocking.cpp")

list(REMOVE_ITEM src_files ${mpi_src_files})

add_executable(diffuse ${src_files})

target_compile_options(
  diffuse PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DIFFUSE_CPP_COMPILE_OPTIONS}>)

target_include_directories(
  diffuse PRIVATE ${noarr_structures_SOURCE_DIR}/include
                  ${argparse_SOURCE_DIR}/include ${highway_SOURCE_DIR})

target_link_libraries(diffuse nlohmann_json::nlohmann_json hwy
                      OpenMP::OpenMP_CXX LAPACK::LAPACK)

find_package(MPI)

if(MPI_FOUND)
  add_executable(diffuse_mpi ${src_files} ${mpi_src_files})

  target_compile_options(
    diffuse_mpi
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DIFFUSE_CPP_COMPILE_OPTIONS}>)

  target_include_directories(
    diffuse_mpi PRIVATE ${noarr_structures_SOURCE_DIR}/include
                        ${argparse_SOURCE_DIR}/include ${highway_SOURCE_DIR})

  target_link_libraries(diffuse_mpi nlohmann_json::nlohmann_json hwy
                        OpenMP::OpenMP_CXX LAPACK::LAPACK MPI::MPI_CXX)
endif()
