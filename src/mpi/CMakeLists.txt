

file(GLOB mpi_src_files "src/mpi/*")

add_executable(diffuse_mpi ${mpi_src_files})
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
