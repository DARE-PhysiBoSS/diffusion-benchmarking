find_package(MPI)

if(MPI_FOUND)
  message(STATUS "MPI found! Building diffuse_mpi.")
  
  set(CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "MPI C++ Compiler" FORCE)

  file(GLOB mpi_sources "../src/*")

  # Create a separate source list for diffuse_mpi
  set(diffuse_mpi_sources ${mpi_sources})

  message(STATUS "MPI Files: ${mpi_sources}")

  # Define a custom target to force recompilation
  add_custom_target(force_recompile_mpi
    COMMAND ${CMAKE_COMMAND} -E touch ${diffuse_mpi_sources}
    COMMENT "Forcing recompilation of all MPI sources"
  )

  add_executable(diffuse_mpi ${diffuse_mpi_sources})

  target_compile_options(diffuse_mpi PRIVATE ${DIFFUSE_CPP_COMPILE_OPTIONS})

  target_include_directories(diffuse_mpi PRIVATE
    ${noarr_structures_SOURCE_DIR}/include
    ${argparse_SOURCE_DIR}/include
    ${highway_SOURCE_DIR})

  target_link_libraries(diffuse_mpi MPI::MPI_CXX OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json LAPACK::LAPACK hwy)

  target_compile_definitions(diffuse_mpi PRIVATE -DUSE_MPI)

  # Ensure recompilation of all sources for MPI
  add_dependencies(diffuse_mpi force_recompile_mpi)

else()
  message(WARNING "MPI not found! The diffuse_mpi target will NOT be built.")
endif()