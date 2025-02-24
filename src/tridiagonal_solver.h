#pragma once

#include "problem.h"
#include "tunable_params.h"

class tridiagonal_solver
{
public:
	// Allocates common resources
	virtual void prepare(const max_problem_t& problem) = 0;

	// Sets the solver specific parameters
	virtual void tune(const tunable_params_t&) {};

	// Allocates solver specific resources
	virtual void initialize() = 0;

	virtual void solve_x() = 0;
	virtual void solve_y() = 0;
	virtual void solve_z() = 0;
};
