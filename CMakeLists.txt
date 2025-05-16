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
  set(DIFFUSE_CPP_COMPILE_OPTIONS -Wall -Wextra -pedantic -march=native -lpapi -O3)
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

file(GLOB src_files "src/*")

add_executable(diffuse ${src_files})

target_compile_options(
  diffuse PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DIFFUSE_CPP_COMPILE_OPTIONS}>)

target_include_directories(
  diffuse PRIVATE ${noarr_structures_SOURCE_DIR}/include
                  ${argparse_SOURCE_DIR}/include ${highway_SOURCE_DIR}
                  ${PAPI_INCLUDE_DIRS})

target_link_libraries(diffuse nlohmann_json::nlohmann_json)

target_link_libraries(diffuse hwy)

find_package(OpenMP REQUIRED)
target_link_libraries(diffuse OpenMP::OpenMP_CXX)

find_package(LAPACK REQUIRED)
target_link_libraries(diffuse LAPACK::LAPACK)

# Find and link PAPI
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAPI REQUIRED papi)

include_directories(BEFORE ${PAPI_INCLUDE_DIRS})
target_include_directories(diffuse PUBLIC ${PAPI_INCLUDE_DIRS})
target_compile_options(diffuse PUBLIC ${PAPI_CFLAGS_OTHER})
target_link_libraries(diffuse ${PAPI_LIBRARIES})

message(PAPI_INCLUDE_DIR="${PAPI_INCLUDE_DIRS}")
message(PAPI_LIBRARIES="${PAPI_LIBRARIES}")
message(PAPI_COMPILE_OPTIONS="${PAPI_CFLAGS_OTHER}")

find_package(MPI)
if (MPI_FOUND)

  file(GLOB mpi_src_files "src/mpi/*")

  set(SHARED_SOURCES
      src/solver_utils.h
      src/algorithms.cpp
      src/omp_helper.h
      src/problem.cpp
      src/reference_thomas_solver.cpp)

  add_executable(diffuse_mpi ${mpi_src_files} ${SHARED_SOURCES})
  target_compile_options(
    diffuse_mpi PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DIFFUSE_CPP_COMPILE_OPTIONS}>)
  target_include_directories(
    diffuse_mpi PRIVATE ${noarr_structures_SOURCE_DIR}/include
                        ${argparse_SOURCE_DIR}/include ${highway_SOURCE_DIR}
                        ${PAPI_INCLUDE_DIRS})

  target_link_libraries(diffuse_mpi nlohmann_json::nlohmann_json)

  target_link_libraries(diffuse_mpi hwy)

  find_package(OpenMP REQUIRED)
  target_link_libraries(diffuse_mpi OpenMP::OpenMP_CXX)

  find_package(LAPACK REQUIRED)
  target_link_libraries(diffuse_mpi LAPACK::LAPACK)

  # Find and link PAPI
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PAPI REQUIRED papi)

  target_include_directories(diffuse_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(diffuse_mpi MPI::MPI_CXX)


  include_directories(BEFORE ${PAPI_INCLUDE_DIRS})
  target_include_directories(diffuse_mpi PUBLIC ${PAPI_INCLUDE_DIRS})
  target_compile_options(diffuse_mpi PUBLIC ${PAPI_CFLAGS_OTHER})
  target_link_libraries(diffuse_mpi ${PAPI_LIBRARIES})
  target_compile_definitions(diffuse_mpi PUBLIC USE_MPI)


else()
  message(STATUS "MPI not found, not compiling MPI version")
endif()