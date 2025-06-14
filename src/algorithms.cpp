#include "algorithms.h"

#include <fstream>
#include <iostream>
#include <papi.h>

#include "biofvm.h"
#include "blocked_thomas_solver.h"
#include "blocked_thomas_solver_t.h"
#include "cubed_thomas_solver_t.h"
#include "cyclic_reduction_solver.h"
#include "cyclic_reduction_solver_t.h"
#include "full_lapack_solver.h"
#include "general_lapack_thomas_solver.h"
#include "lapack_thomas_solver.h"
#include "least_compute_thomas_solver.h"
#include "least_compute_thomas_solver_m.h"
#include "least_compute_thomas_solver_s.h"
#include "least_compute_thomas_solver_s_t.h"
#include "least_compute_thomas_solver_t.h"
#include "least_memory_thomas_solver.h"
#include "least_memory_thomas_solver_d_t.h"
#include "least_memory_thomas_solver_t.h"
#include "reference_thomas_solver.h"
#include "serial_blocked_thomas_solver.h"
#include "simd.h"
#include "tridiagonal_solver.h"


template <typename real_t>
std::map<std::string, std::function<std::unique_ptr<diffusion_solver>()>> get_solvers_map()
{
	std::map<std::string, std::function<std::unique_ptr<diffusion_solver>()>> solvers;

	solvers.emplace("ref", []() { return std::make_unique<reference_thomas_solver<real_t>>(); });
	solvers.emplace("lstc", []() { return std::make_unique<least_compute_thomas_solver<real_t>>(); });
	solvers.emplace("lstcm", []() { return std::make_unique<least_compute_thomas_solver_m<real_t, false>>(); });
	solvers.emplace("lstcma", []() { return std::make_unique<least_compute_thomas_solver_m<real_t, true>>(); });
	solvers.emplace("lstct", []() { return std::make_unique<least_compute_thomas_solver_t<real_t, false>>(); });
	solvers.emplace("lstcta", []() { return std::make_unique<least_compute_thomas_solver_t<real_t, true>>(); });
	solvers.emplace("lstcs", []() { return std::make_unique<least_compute_thomas_solver_s<real_t>>(); });
	solvers.emplace("lstcst", []() { return std::make_unique<least_compute_thomas_solver_s_t<real_t, false>>(false); });
	solvers.emplace("lstcsta", []() { return std::make_unique<least_compute_thomas_solver_s_t<real_t, true>>(false); });
	solvers.emplace("lstcstai", []() { return std::make_unique<least_compute_thomas_solver_s_t<real_t, true>>(true); });
	solvers.emplace("lstm", []() { return std::make_unique<least_memory_thomas_solver<real_t>>(); });
	solvers.emplace("lstmdt",
					[]() { return std::make_unique<least_memory_thomas_solver_d_t<real_t, false>>(false, false); });
	solvers.emplace("lstmdta",
					[]() { return std::make_unique<least_memory_thomas_solver_d_t<real_t, true>>(false, false); });
	solvers.emplace("lstmdtai",
					[]() { return std::make_unique<least_memory_thomas_solver_d_t<real_t, true>>(true, false); });
	solvers.emplace("lstmdtfa",
					[]() { return std::make_unique<least_memory_thomas_solver_d_t<real_t, true>>(false, true); });
	solvers.emplace("lstmdtfai",
					[]() { return std::make_unique<least_memory_thomas_solver_d_t<real_t, true>>(true, true); });
	solvers.emplace("lstmt", []() { return std::make_unique<least_memory_thomas_solver_t<real_t, false>>(false); });
	solvers.emplace("lstmta", []() { return std::make_unique<least_memory_thomas_solver_t<real_t, true>>(false); });
	solvers.emplace("lstmtai", []() { return std::make_unique<least_memory_thomas_solver_t<real_t, true>>(true); });
	solvers.emplace("lapack", []() { return std::make_unique<lapack_thomas_solver<real_t>>(); });
	solvers.emplace("lapack2", []() { return std::make_unique<general_lapack_thomas_solver<real_t>>(); });
	solvers.emplace("full_lapack", []() { return std::make_unique<full_lapack_solver<real_t>>(); });
	solvers.emplace("avx256d", []() { return std::make_unique<simd<double>>(); });
	solvers.emplace("biofvm", []() { return std::make_unique<biofvm<real_t>>(); });
	solvers.emplace("cr", []() { return std::make_unique<cyclic_reduction_solver<real_t, false>>(); });
	solvers.emplace("crt", []() { return std::make_unique<cyclic_reduction_solver_t<real_t, false>>(); });
	solvers.emplace("sblocked", []() { return std::make_unique<serial_blocked_thomas_solver<real_t, false>>(); });
	solvers.emplace("blocked", []() { return std::make_unique<blocked_thomas_solver<real_t, false>>(); });
	solvers.emplace("blockedt", []() { return std::make_unique<blocked_thomas_solver_t<real_t, false>>(); });
	solvers.emplace("blockedta", []() { return std::make_unique<blocked_thomas_solver_t<real_t, true>>(); });
	solvers.emplace("cubed", []() { return std::make_unique<cubed_thomas_solver_t<real_t, true>>(); });
	return solvers;
}

algorithms::algorithms(bool double_precision, bool verbose) : verbose_(verbose)
{
	if (double_precision)
		solvers_ = get_solvers_map<double>();
	else
		solvers_ = get_solvers_map<float>();
}

std::unique_ptr<diffusion_solver> algorithms::get_solver(const std::string& alg)
{
	auto solver = solvers_.at(alg)();
	return solver;
}

std::unique_ptr<locally_onedimensional_solver> algorithms::try_get_adi_solver(const std::string& alg)
{
	auto solver = get_solver(alg);
	if (dynamic_cast<locally_onedimensional_solver*>(solver.get()))
	{
		return std::unique_ptr<locally_onedimensional_solver>(
			dynamic_cast<locally_onedimensional_solver*>(solver.release()));
	}
	return nullptr;
}

void algorithms::append_params(std::ostream& os, const nlohmann::json& params, bool header)
{
	std::vector<std::string> keys_to_skip = { "warmup_time", "outer_iterations", "inner_iterations", "benchmark_kind", "papi_counters" };
	if (header)
	{
		for (auto it = params.begin(); it != params.end(); ++it)
		{
			if (std::find(keys_to_skip.begin(), keys_to_skip.end(), it.key()) == keys_to_skip.end())
				os << it.key() << ",";
		}
	}
	else
	{
		for (auto it = params.begin(); it != params.end(); ++it)
		{
			if (std::find(keys_to_skip.begin(), keys_to_skip.end(), it.key()) == keys_to_skip.end())
				os << it.value() << ",";
		}
	}
}

std::ofstream open_file(const std::string& file_name)
{
	std::ofstream file(file_name);
	if (!file.is_open())
		throw std::runtime_error("Cannot open file " + file_name);
	return file;
}

void algorithms::run(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params,
					 const std::string& output_file)
{
	auto solver = get_solver(alg);

	solver->tune(params);
	solver->prepare(problem);
	solver->initialize();

	if (!output_file.empty())
	{
		std::filesystem::path output_path(output_file);
		auto init_output_path = (output_path.parent_path() / ("initial_" + output_path.filename().string()));

		auto init_output = open_file(init_output_path.string());
		solver->save(init_output);
	}

	solver->solve();

	if (!output_file.empty())
	{
		auto output = open_file(output_file);
		solver->save(output);
	}
}

void common_prepare(diffusion_solver& alg, diffusion_solver& ref, const max_problem_t& problem,
					const nlohmann::json& params)
{
	alg.tune(params);
	alg.prepare(problem);
	alg.initialize();

	ref.tune(params);
	ref.prepare(problem);
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
	benchmark_kind kind = get_benchmark_kind(params);

	if (try_get_adi_solver(alg) && kind == benchmark_kind::per_dimension)
	{
		double max_absolute_diff_x = 0.;
		double max_absolute_diff_y = 0.;
		double max_absolute_diff_z = 0.;

		double rmse_x = 0.;
		double rmse_y = 0.;
		double rmse_z = 0.;

		// validate solve_x
		{
			auto ref_solver = try_get_adi_solver("ref");
			auto adi_solver = try_get_adi_solver(alg);

			common_prepare(*adi_solver, *ref_solver, problem, params);

			adi_solver->solve_x();
			ref_solver->solve_x();

			std::tie(max_absolute_diff_x, rmse_x) = common_validate(*adi_solver, *ref_solver, problem);
		}

		if (problem.dims > 1)
		{
			// validate solve_y
			{
				auto ref_solver = try_get_adi_solver("ref");
				auto adi_solver = try_get_adi_solver(alg);

				common_prepare(*adi_solver, *ref_solver, problem, params);

				adi_solver->solve_y();
				ref_solver->solve_y();

				std::tie(max_absolute_diff_y, rmse_y) = common_validate(*adi_solver, *ref_solver, problem);
			}
		}

		if (problem.dims > 2)
		{
			// validate solve_z
			{
				auto ref_solver = try_get_adi_solver("ref");
				auto adi_solver = try_get_adi_solver(alg);

				common_prepare(*adi_solver, *ref_solver, problem, params);

				adi_solver->solve_z();
				ref_solver->solve_z();

				std::tie(max_absolute_diff_z, rmse_z) = common_validate(*adi_solver, *ref_solver, problem);
			}
		}

		std::cout << "X - Maximal absolute difference: " << max_absolute_diff_x << ", RMSE:" << rmse_x << std::endl;
		std::cout << "Y - Maximal absolute difference: " << max_absolute_diff_y << ", RMSE:" << rmse_y << std::endl;
		std::cout << "Z - Maximal absolute difference: " << max_absolute_diff_z << ", RMSE:" << rmse_z << std::endl;
	}
	else
	{
		auto ref_solver = get_solver("ref");
		auto solver = get_solver(alg);

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

	auto solver = get_solver(alg);

	solver->tune(params);
	solver->prepare(problem);

	std::size_t init_time_us;

	{
		auto start = std::chrono::high_resolution_clock::now();

		solver->initialize();

		auto end = std::chrono::high_resolution_clock::now();

		init_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	}

	auto compute_mean_and_std = [](const std::vector<std::size_t>& times) {
		double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
		std::vector<double> variances(times.size());
		std::transform(times.begin(), times.end(), variances.begin(),
					   [mean](double x) { return (x - mean) * (x - mean); });
		double variance = std::accumulate(variances.begin(), variances.end(), 0.0) / times.size();
		double std_dev = std::sqrt(variance);
		return std::make_pair(mean, std_dev);
	};

	std::cout << alg << "," << problem.dims << "," << problem.substrates_count << "," << problem.nx << "," << problem.ny
			  << "," << problem.nz << ",";
	append_params(std::cout, params, false);
	std::cout << init_time_us << ",";

	if (auto adi_solver = dynamic_cast<locally_onedimensional_solver*>(solver.get());
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

		std::cout << x_mean << "," << y_mean << "," << z_mean << "," << x_std << "," << y_std << "," << z_std
				  << std::endl;
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

		std::cout << mean << "," << std_dev << std::endl;
	}
}

void algorithms::benchmark(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params)
{
	benchmark_kind kind = get_benchmark_kind(params);

	// make header
	{
		std::cout << "algorithm,dims,s,nx,ny,nz,";
		append_params(std::cout, params, true);

		if (try_get_adi_solver(alg) && kind == benchmark_kind::per_dimension)
			std::cout << "init_time,x_time,y_time,z_time,x_std,y_std,z_std" << std::endl;
		else
			std::cout << "init_time,time,std_dev" << std::endl;
	}

	// warmup
	{
		auto warmup_time_s = params.contains("warmup_time") ? (double)params["warmup_time"] : 3.0;
		auto start = std::chrono::high_resolution_clock::now();
		auto end = start;
		do
		{
			auto solver = get_solver(alg);
			solver->tune(params);
			solver->prepare(problem);
			solver->initialize();

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

struct EventData
{
	uint32_t Code;
	std::string Deriv;
	std::string Description;
};

void readPapiJson(std::map<std::string, EventData>& events, const std::string& filename)
{
	std::ifstream file(filename);
	if (!file)
	{
		throw std::runtime_error("Cannot open file: " + filename + "\n" + "Try running: $./scripts/obtain_counters.sh");
	}

	nlohmann::json j;
	file >> j;

	for (auto& [name, data] : j.items())
	{
		events[name] = EventData(data["Code"].get<uint32_t>(), data["Deriv"].get<std::string>(),
								 data["Description"].get<std::string>());
	}
}



void algorithms::profile(const std::string& alg, const max_problem_t& problem, const nlohmann::json& params)
{
	std::string csv_path;
	if (params.contains("profile_output"))
	{
		csv_path = params["profile_output"];
	}
	else
	{
		csv_path = "./profile.csv";
	}
	std::ofstream file(csv_path, std::ios::app);

	auto solver = get_solver(alg);

	solver->tune(params);
	solver->prepare(problem);
	solver->initialize();

	std::map<std::string, EventData> counters_info;

	readPapiJson(counters_info, "./example-problems/counters.json");

	if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
	{
		std::cerr << "PAPI library init error!" << std::endl;
		return;
	}

	// Events to monitor
	int events = PAPI_NULL;
	std::vector<std::string> counters_names = params["papi_counters"].get<std::vector<std::string>>();
	const int NUM_EVENTS = counters_names.size();
	std::vector<long long> counters(NUM_EVENTS);

	// add events
	PAPI_create_eventset(&events);
	for (int i = 0; i < NUM_EVENTS; ++i)
	{
		PAPI_add_event(events, counters_info[counters_names[i]].Code);
	}

	file << counters_names[0];
	for (int i = 1; i < NUM_EVENTS; ++i)
	{
		file << "," << counters_names[i];
	}
	file << std::endl;

	// Start counters
	if (PAPI_start(events) != PAPI_OK)
	{
		std::cerr << "PAPI failed to start counters." << std::endl;
		return;
	}
	solver->solve();

	// Stop counters
	if (PAPI_stop(events, counters.data()) != PAPI_OK)
	{
		std::cerr << "PAPI failed to stop counters." << std::endl;
		return;
	}

	file << counters[0];
	for (int i = 1; i < NUM_EVENTS; ++i)
	{
		file << "," << counters[i];
	}
	file << std::endl;
	PAPI_reset(events);

	file.close();
}

benchmark_kind algorithms::get_benchmark_kind(const nlohmann::json& params)
{
	benchmark_kind kind = benchmark_kind::per_dimension;

	if (params.contains("benchmark_kind"))
	{
		auto kind_str = params["benchmark_kind"].get<std::string>();
		if (kind_str == "full_solve")
			kind = benchmark_kind::full_solve;
		else if (kind_str == "per_dimension")
			kind = benchmark_kind::per_dimension;
	}

	return kind;
}
