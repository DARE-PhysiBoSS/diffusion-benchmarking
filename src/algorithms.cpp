#include "algorithms.h"

#include <iostream>

#include "full_lapack_solver.h"
#include "general_lapack_thomas_solver.h"
#include "lapack_thomas_solver.h"
#include "least_compute_thomas_solver.h"
#include "least_memory_thomas_solver.h"
#include "reference_thomas_solver.h"
#include "tridiagonal_solver.h"

template <typename real_t>
std::map<std::string, std::unique_ptr<diffusion_solver>> get_solvers_map()
{
	std::map<std::string, std::unique_ptr<diffusion_solver>> solvers;

	solvers.emplace("ref", std::make_unique<reference_thomas_solver<real_t>>());
	solvers.emplace("lstc", std::make_unique<least_compute_thomas_solver<real_t>>());
	solvers.emplace("lstm", std::make_unique<least_memory_thomas_solver<real_t>>());
	solvers.emplace("lapack", std::make_unique<lapack_thomas_solver<real_t>>());
	solvers.emplace("lapack2", std::make_unique<general_lapack_thomas_solver<real_t>>());
	solvers.emplace("full_lapack", std::make_unique<full_lapack_solver<real_t>>());

	return solvers;
}

algorithms::algorithms(bool double_precision, bool verbose) : verbose_(verbose)
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
		solver->solve();
	}

	solver->save(output_file);
}

void common_prepare(diffusion_solver& alg, diffusion_solver& ref, const max_problem_t& problem,
					const nlohmann::json& params)
{
	alg.prepare(problem);
	alg.tune(params);
	alg.initialize();

	ref.prepare(problem);
	ref.tune(params);
	ref.initialize();
}

std::pair<double, double> algorithms::common_validate(diffusion_solver& alg, diffusion_solver& ref,
													  const max_problem_t& problem)
{
	double maximum_absolute_difference = 0.;
	double rmse = 0.;

	for (std::size_t z = 0; z < problem.nz; z++)
		for (std::size_t y = 0; y < problem.ny; y++)
			for (std::size_t x = 0; x < problem.nx; x++)
				for (std::size_t s = 0; s < problem.substrates_count; s++)
				{
					auto ref_val = ref.access(s, x, y, z);
					auto val = alg.access(s, x, y, z);

					auto diff = std::abs(val - ref_val);
					maximum_absolute_difference = std::max(maximum_absolute_difference, diff);
					rmse += diff * diff;

					auto relative_diff = diff / std::abs(ref_val);
					if (diff > absolute_difference_print_threshold_
						&& relative_diff > relative_difference_print_threshold_ && verbose_)
						std::cout << "At [" << s << ", " << x << ", " << y << ", " << z << "]: " << val
								  << " is not close to the reference " << ref_val << std::endl;
				}

	rmse = std::sqrt(rmse / (problem.nz * problem.ny * problem.nx * problem.substrates_count));

	return { maximum_absolute_difference, rmse };
}

void algorithms::validate(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params)
{
	auto ref_solver = dynamic_cast<tridiagonal_solver*>(solvers_.at("ref").get());
	auto solver = solvers_.at(alg).get();

	tridiagonal_solver* adi_solver = dynamic_cast<tridiagonal_solver*>(solver);

	if (adi_solver)
	{
		double max_absolute_diff_x = 0.;
		double max_absolute_diff_y = 0.;
		double max_absolute_diff_z = 0.;

		double rmse_x = 0.;
		double rmse_y = 0.;
		double rmse_z = 0.;

		// validate solve_x
		{
			common_prepare(*solver, *ref_solver, problem, params);
			adi_solver->solve_x();
			ref_solver->solve_x();

			std::tie(max_absolute_diff_x, rmse_x) = common_validate(*solver, *ref_solver, problem);
		}

		if (problem.dims > 1)
		{
			// validate solve_y
			{
				common_prepare(*solver, *ref_solver, problem, params);
				adi_solver->solve_y();
				ref_solver->solve_y();

				std::tie(max_absolute_diff_y, rmse_y) = common_validate(*solver, *ref_solver, problem);
			}
		}

		if (problem.dims > 2)
		{
			// validate solve_z
			{
				common_prepare(*solver, *ref_solver, problem, params);
				adi_solver->solve_z();
				ref_solver->solve_z();

				std::tie(max_absolute_diff_z, rmse_z) = common_validate(*solver, *ref_solver, problem);
			}
		}

		std::cout << "X - Maximal absolute difference: " << max_absolute_diff_x << ", RMSE:" << rmse_x << std::endl;
		std::cout << "Y - Maximal absolute difference: " << max_absolute_diff_y << ", RMSE:" << rmse_y << std::endl;
		std::cout << "Z - Maximal absolute difference: " << max_absolute_diff_z << ", RMSE:" << rmse_z << std::endl;
	}
	else
	{
		common_prepare(*solver, *ref_solver, problem, params);
		solver->solve();
		ref_solver->solve();

		auto [max_absolute_diff, rmse] = common_validate(*solver, *ref_solver, problem);

		std::cout << "Maximal absolute difference: " << max_absolute_diff << ", RMSE:" << rmse << std::endl;
	}
}

void algorithms::benchmark_inner(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params,
								 benchmark_kind kind)
{
	auto inner_iterations = params.contains("inner_iterations") ? (std::size_t)params["inner_iterations"] : 10;

	auto solver = solvers_.at(alg).get();

	solver->prepare(problem);
	solver->tune(params);

	std::size_t init_time_us;

	{
		auto start = std::chrono::high_resolution_clock::now();

		solver->initialize();

		auto end = std::chrono::high_resolution_clock::now();

		init_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	}

	auto compute_mean_and_std = [](const std::vector<std::size_t>& times) {
		double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
		double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
		double std_dev = std::sqrt(sq_sum / times.size() - mean * mean);
		return std::make_pair(mean, std_dev);
	};

	if (auto adi_solver = dynamic_cast<tridiagonal_solver*>(solver);
		adi_solver && kind == benchmark_kind::per_dimension)
	{
		std::vector<std::size_t> times_x, times_y, times_z;

		for (std::size_t i = 0; i < inner_iterations; i++)
		{
			{
				auto start = std::chrono::high_resolution_clock::now();
				adi_solver->solve_x();
				auto end = std::chrono::high_resolution_clock::now();

				times_x.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
			}

			if (problem.dims > 1)
			{
				auto start = std::chrono::high_resolution_clock::now();
				adi_solver->solve_y();
				auto end = std::chrono::high_resolution_clock::now();

				times_y.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
			}

			if (problem.dims > 2)
			{
				auto start = std::chrono::high_resolution_clock::now();
				adi_solver->solve_z();
				auto end = std::chrono::high_resolution_clock::now();

				times_z.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
			}
		}

		auto [x_mean, x_std] = compute_mean_and_std(times_x);
		auto [y_mean, y_std] = compute_mean_and_std(times_y);
		auto [z_mean, z_std] = compute_mean_and_std(times_z);

		std::cout << alg << "," << problem.dims << "," << problem.substrates_count << "," << problem.nx << ","
				  << problem.ny << "," << problem.nz << "," << init_time_us << "," << 10 << "," << x_mean << ","
				  << y_mean << "," << z_mean << "," << x_std << "," << y_std << "," << z_std << std::endl;
	}
	else
	{
		std::vector<std::size_t> times;

		for (std::size_t i = 0; i < inner_iterations; i++)
		{
			auto start = std::chrono::high_resolution_clock::now();
			solver->solve();
			auto end = std::chrono::high_resolution_clock::now();

			times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		}

		auto [mean, std_dev] = compute_mean_and_std(times);

		std::cout << alg << "," << problem.dims << "," << problem.substrates_count << "," << problem.nx << ","
				  << problem.ny << "," << problem.nz << "," << init_time_us << "," << 10 << "," << mean << ","
				  << std_dev << std::endl;
	}
}

void algorithms::benchmark(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params)
{
	auto& solver = solvers_.at(alg);

	benchmark_kind kind = benchmark_kind::per_dimension;

	if (params.contains("benchmark_kind"))
	{
		auto kind_str = params["benchmark_kind"].get<std::string>();
		if (kind_str == "full_solve")
			kind = benchmark_kind::full_solve;
		else if (kind_str == "per_dimension")
			kind = benchmark_kind::per_dimension;
	}

	if (dynamic_cast<tridiagonal_solver*>(solver.get()) && kind == benchmark_kind::per_dimension)
		std::cout << "algorithm,dims,s,nx,ny,nz,init_time,repetitions,x_time,y_time,z_time,x_std,y_std,z_std"
				  << std::endl;
	else
		std::cout << "algorithm,dims,s,nx,ny,nz,init_time,repetitions,time,std_dev" << std::endl;

	solver->prepare(problem);
	solver->tune(params);
	solver->initialize();

	// warmup
	{
		auto warmup_time_s = params.contains("warmup_time") ? (double)params["warmup_time"] : 3.0;
		auto start = std::chrono::high_resolution_clock::now();
		auto end = start;
		do
		{
			solver->initialize();
			solver->solve();
			end = std::chrono::high_resolution_clock::now();
		} while ((double)std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < warmup_time_s);
	}

	// measure outer
	{
		auto outer_iterations = params.contains("outer_iterations") ? (std::size_t)params["outer_iterations"] : 1;

		for (std::size_t i = 0; i < outer_iterations; i++)
			benchmark_inner(alg, problem, params, kind);
	}
}
