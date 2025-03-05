#pragma once

#include <map>
#include <memory>
#include <string>

#include "tridiagonal_solver.h"

class algorithms
{
	std::map<std::string, std::unique_ptr<tridiagonal_solver>> solvers_;

	static constexpr double relative_difference_print_threshold_ = 0.01;

	double max_relative_diff_x_, max_relative_diff_y_, max_relative_diff_z_;

	void common_validate(tridiagonal_solver& alg, tridiagonal_solver& ref, const max_problem_t& problem,
						 double& max_relative_difference);

	void benchmark_inner(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params);

public:
	algorithms(bool double_precision);

	// Run the algorithm on the given problem for specified number of iterations
	void run(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params,
			 const std::string& output_file);

	// Validate one iteration of the algorithm with the reference implementation
	void validate(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params);

	// Measure the algorithm performance
	void benchmark(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params);
};
