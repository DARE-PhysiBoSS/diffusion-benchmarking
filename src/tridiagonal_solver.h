#pragma once

#include <nlohmann/json.hpp>

#include "problem.h"

class tridiagonal_solver
{
public:
	// Allocates common resources
	virtual void prepare(const max_problem_t& problem) = 0;

	// Sets the solver specific parameters
	virtual void tune(const nlohmann::json&) {};

	// Allocates solver specific resources
	virtual void initialize() = 0;

	virtual void solve_x() = 0;
	virtual void solve_y() = 0;
	virtual void solve_z() = 0;

	// Saves data to a file in human readable format with the following structure:
	// Each line contains a space-separated list of values for a single point in the grid (so all substrates)
	// The points are ordered in x, y, z order
	virtual void save(const std::string& file) = 0;

	// Accesses the value at the given coordinates
	virtual double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) = 0;
};
