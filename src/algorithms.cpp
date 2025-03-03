#include "algorithms.h"

#include <iostream>

#include "least_compute_thomas_solver.h"
#include "least_memory_thomas_solver.h"
#include "reference_thomas_solver.h"
#include "tridiagonal_solver.h"

template <typename real_t>
std::map<std::string, std::unique_ptr<tridiagonal_solver>> get_solvers_map()
{
	std::map<std::string, std::unique_ptr<tridiagonal_solver>> solvers;

	solvers.emplace("ref", std::make_unique<reference_thomas_solver<real_t>>());
	solvers.emplace("lstc", std::make_unique<least_compute_thomas_solver<real_t>>());
	solvers.emplace("lstm", std::make_unique<least_memory_thomas_solver<real_t>>());

	return solvers;
}

algorithms::algorithms(bool double_precision)
{
	if (double_precision)
		solvers_ = get_solvers_map<double>();
	else
		solvers_ = get_solvers_map<float>();
}

void algorithms::run(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params,
					 const std::string& output_file)
{
	auto& solver = solvers_.at(alg);

	solver->prepare(problem);
	solver->tune(params);
	solver->initialize();

	for (std::size_t i = 0; i < problem.iterations; i++)
	{
		solver->solve_x();

		if (problem.dims > 1)
			solver->solve_y();

		if (problem.dims > 2)
			solver->solve_z();
	}

	solver->save(output_file);
}

void common_prepare(tridiagonal_solver& alg, tridiagonal_solver& ref, const max_problem_t& problem,
					const nlohmann::json& params)
{
	alg.prepare(problem);
	alg.tune(params);
	alg.initialize();

	ref.prepare(problem);
	ref.tune(params);
	ref.initialize();
}

void algorithms::common_validate(tridiagonal_solver& alg, tridiagonal_solver& ref, const max_problem_t& problem,
								 double& max_relative_difference)
{
	for (std::size_t z = 0; z < problem.nz; z++)
		for (std::size_t y = 0; y < problem.ny; y++)
			for (std::size_t x = 0; x < problem.nx; x++)
				for (std::size_t s = 0; s < problem.substrates_count; s++)
				{
					auto ref_val = ref.access(s, x, y, z);
					auto val = alg.access(s, x, y, z);

					auto relative_diff = std::abs(val - ref_val) / std::abs(ref_val);
					max_relative_difference = std::max(max_relative_difference, relative_diff);

					if (relative_diff > relative_difference_print_threshold_)
						std::cout << "At [" << s << ", " << x << ", " << y << ", " << z << "]: " << val
								  << " != " << ref_val << std::endl;
				}
}

void algorithms::validate(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params)
{
	auto& solver = solvers_.at(alg);
	auto& ref_solver = solvers_.at("ref");

	max_relative_diff_x_ = 0.;
	max_relative_diff_y_ = 0.;
	max_relative_diff_z_ = 0.;

	// validate solve_x
	{
		common_prepare(*solver, *ref_solver, problem, params);
		solver->solve_x();
		ref_solver->solve_x();

		common_validate(*solver, *ref_solver, problem, max_relative_diff_x_);
	}

	if (problem.dims > 1)
	{
		// validate solve_y
		{
			common_prepare(*solver, *ref_solver, problem, params);
			solver->solve_y();
			ref_solver->solve_y();

			common_validate(*solver, *ref_solver, problem, max_relative_diff_y_);
		}
	}

	if (problem.dims > 2)
	{
		// validate solve_z
		{
			common_prepare(*solver, *ref_solver, problem, params);
			solver->solve_z();
			ref_solver->solve_z();

			common_validate(*solver, *ref_solver, problem, max_relative_diff_z_);
		}
	}

	std::cout << "Maximal relative difference in x: " << max_relative_diff_x_ * 100 << "%" << std::endl;
	std::cout << "Maximal relative difference in y: " << max_relative_diff_y_ * 100 << "%" << std::endl;
	std::cout << "Maximal relative difference in z: " << max_relative_diff_z_ * 100 << "%" << std::endl;
}

template <typename func_t>
std::vector<std::size_t> warmup_and_measure(tridiagonal_solver& solver, const max_problem_t& problem,
											const nlohmann::json& params, double warmup_time_s, func_t&& func)
{
	// warmup
	{
		auto start = std::chrono::high_resolution_clock::now();
		auto end = start;
		do
		{
			func();
			end = std::chrono::high_resolution_clock::now();
		} while ((double)std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < warmup_time_s);
	}

	// measure
	std::vector<std::size_t> times;
	{
		for (std::size_t i = 0; i < 10; i++)
		{
			solver.prepare(problem);
			solver.tune(params);
			solver.initialize();

			auto start = std::chrono::high_resolution_clock::now();
			func();
			auto end = std::chrono::high_resolution_clock::now();

			times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		}
	}

	return times;
}

void algorithms::measure(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params)
{
	auto& solver = solvers_.at(alg);

	std::cout << "algorithm,dims,s,nx,ny,nz,init_time,repetitions,x_time,y_time,z_time,x_std,y_std,z_std" << std::endl;

	solver->prepare(problem);

	std::size_t init_time_us;
	{
		auto start = std::chrono::high_resolution_clock::now();

		solver->initialize();

		auto end = std::chrono::high_resolution_clock::now();

		init_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	}

	auto x_times = warmup_and_measure(*solver, problem, params, warmup_time_s_, [&solver]() { solver->solve_x(); });
	auto y_times = warmup_and_measure(*solver, problem, params, warmup_time_s_, [&solver]() { solver->solve_y(); });
	auto z_times = warmup_and_measure(*solver, problem, params, warmup_time_s_, [&solver]() { solver->solve_z(); });

	auto compute_mean_and_std = [](const std::vector<std::size_t>& times) {
		double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
		double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
		double std_dev = std::sqrt(sq_sum / times.size() - mean * mean);
		return std::make_pair(mean, std_dev);
	};

	auto [x_mean, x_std] = compute_mean_and_std(x_times);
	auto [y_mean, y_std] = compute_mean_and_std(y_times);
	auto [z_mean, z_std] = compute_mean_and_std(z_times);

	std::cout << alg << "," << problem.dims << "," << problem.substrates_count << "," << problem.nx << "," << problem.ny
			  << "," << problem.nz << "," << init_time_us << "," << 10 << "," << x_mean << "," << y_mean << ","
			  << z_mean << "," << x_std << "," << y_std << "," << z_std << std::endl;
}
