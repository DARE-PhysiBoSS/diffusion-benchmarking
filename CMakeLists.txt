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
  set(DIFFUSE_CPP_COMPILE_OPTIONS -Wall -Wextra -pedantic)
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

set(base_src_files "src/main.cpp" "src/algorithms.cpp" "src/problem.cpp" "src/reference_thomas_solver.cpp")

file(GLOB src_files "src/*")

set(src_files "src/least_compute_thomas_solver_t.cpp" "src/least_memory_thomas_solver_d_t.cpp")

add_executable(diffuse ${src_files} ${base_src_files})

target_compile_options(
  diffuse PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DIFFUSE_CPP_COMPILE_OPTIONS}>)

target_include_directories(
  diffuse PRIVATE ${noarr_structures_SOURCE_DIR}/include
                  ${argparse_SOURCE_DIR}/include ${highway_SOURCE_DIR})

target_link_libraries(diffuse nlohmann_json::nlohmann_json $ENV{SDV_TRACE_C_LINK})

# target_link_libraries(diffuse hwy)

# find_package(OpenMP REQUIRED)
# target_link_libraries(diffuse OpenMP::OpenMP_CXX)

# find_package(LAPACK REQUIRED)
# target_link_libraries(diffuse LAPACK::LAPACK)

# Find and link PAPI
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAPI IMPORTED_TARGET papi)

if(PAPI_FOUND)

  target_link_libraries(diffuse PkgConfig::PAPI)

else()

  message(STATUS "'papi' will be built from source:")

  # Fetch papi
  FetchContent_Declare(
    papi
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/papi)
  FetchContent_MakeAvailable(papi)

  execute_process(COMMAND ./configure --prefix=${papi_BINARY_DIR}
                  WORKING_DIRECTORY ${papi_SOURCE_DIR}/src)

  add_custom_target(
    papi_build
    BYPRODUCTS ${papi_BINARY_DIR}/lib/libpapi.a
    COMMAND make -j
    COMMAND make install
    WORKING_DIRECTORY ${papi_SOURCE_DIR}/src)

  add_dependencies(diffuse papi_build)

  target_include_directories(diffuse PRIVATE ${papi_BINARY_DIR}/include)
  target_link_libraries(diffuse ${papi_BINARY_DIR}/lib/libpapi.a)

  set(CMAKE_PREFIX_PATH
      ${papi_BINARY_DIR}
      CACHE PATH "Path to installed packages" FORCE)

endif()
