#pragma once

#include <map>
#include <memory>
#include <string>

#include "diffusion_solver.h"

enum class benchmark_kind
{
	full_solve,
	per_dimension
};

class algorithms
{
	std::map<std::string, std::unique_ptr<diffusion_solver>> solvers_;

	bool verbose_;

	static constexpr double relative_difference_print_threshold_ = 0.01;
	static constexpr double absolute_difference_print_threshold_ = 1e-6;

	std::pair<double, double> common_validate(diffusion_solver& alg, diffusion_solver& ref,
											  const max_problem_t& problem);

	void benchmark_inner(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params,
						 benchmark_kind kind);

public:
	algorithms(bool double_precision, bool verbose);

	// Run the algorithm on the given problem for specified number of iterations
	void run(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params,
			 const std::string& output_file);

	// Validate one iteration of the algorithm with the reference implementation
	void validate(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params);

	// Measure the algorithm performance
	void benchmark(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params);
};
