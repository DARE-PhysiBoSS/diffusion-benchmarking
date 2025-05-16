#pragma once

#include <nlohmann/json.hpp>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "problem.h"

class diffusion_solver
{
public:

	#ifdef USE_MPI
		using index_t = std::int32_t; 
		index_t x_min, y_min, z_min;
		index_t x_max, y_max, z_max;
		MPI_Comm mpi_comm;
	#endif
	// Allocates common resources
	virtual void prepare(const max_problem_t& problem) = 0;

	// Sets the solver specific parameters
	virtual void tune(const nlohmann::json&) {};

	// Allocates solver specific resources
	virtual void initialize() = 0;

	// Solves the diffusion problem
	virtual void solve() = 0;

	// Saves data to a file in human readable format with the following structure:
	// Each line contains a space-separated list of values for a single point in the grid (so all substrates)
	// The points are ordered in x, y, z order
	virtual void save(std::ostream& out) const = 0;

	// Accesses the value at the given coordinates
	virtual double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const = 0;

	virtual ~diffusion_solver() = default;
};
