#include "problem.h"

#include <fstream>

#include <nlohmann/json.hpp>

max_problem_t problems::read_problem(const std::string& file)
{
	std::ifstream ifs(file);
	if (!ifs)
		throw std::runtime_error("Cannot open file " + file);

	nlohmann::json j;
	ifs >> j;

	max_problem_t problem;
	problem.dims = j["dims"];
	problem.dx = j["dx"];
	problem.dy = j["dy"];
	problem.dz = j["dz"];
	problem.nx = j["nx"];
	problem.ny = j["ny"];
	problem.nz = j["nz"];
	problem.substrates_count = j["substrates_count"];
	problem.iterations = j["iterations"];
	problem.dt = j["dt"];
	problem.diffusion_coefficients = j["diffusion_coefficients"].get<std::vector<double>>();
	problem.decay_rates = j["decay_rates"].get<std::vector<double>>();
	problem.initial_conditions = j["initial_conditions"].get<std::vector<double>>();

	if (problem.diffusion_coefficients.size() != problem.substrates_count)
		throw std::runtime_error("diffusion_coefficients size does not match substrates_count");

	if (problem.decay_rates.size() != problem.substrates_count)
		throw std::runtime_error("decay_rates size does not match substrates_count");

	if (problem.initial_conditions.size() != problem.substrates_count)
		throw std::runtime_error("initial_conditions size does not match substrates_count");

	if (problem.dims < 1 || problem.dims > 3)
		throw std::runtime_error("dims must be in range [1, 3]");

	if (problem.nx < 2 || problem.ny < 2 || problem.nz < 2)
		throw std::runtime_error("nx, ny, nz must be at least 2");

	return problem;
}
