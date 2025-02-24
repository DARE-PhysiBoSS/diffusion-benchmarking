#pragma once

#include <map>
#include <memory>
#include <string>

#include "tridiagonal_solver.h"

class algorithms
{
	std::map<std::string, std::unique_ptr<tridiagonal_solver>> solvers_;

	double warmup_time_s_ = 3.;

public:
	algorithms(bool double_precision);

	// Run the algorithm on the given problem for specified number of iterations
	void run(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params,
			 const std::string& output_file);

	// Validate one iteration of the algorithm with the reference implementation
	void validate(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params);

	// Measure the algorithm performance
	void measure(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params);
};
