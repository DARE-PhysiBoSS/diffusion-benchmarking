#include <vector>

template <typename num_t, typename real_t>
struct problem_t
{
	num_t dims;
	num_t dx, dy, dz;
	num_t nx, ny, nz;
	num_t substrates_count;

	real_t dt;
	std::vector<real_t> diffusion_coefficient;
	std::vector<real_t> decay_rate;
	std::vector<real_t> initial_conditions;

	template <typename other_num_t, typename other_real_t>
	problem_t<other_num_t, other_real_t> cast() const
	{
		problem_t<other_num_t, other_real_t> other_problem;
		other_problem.dims = static_cast<other_num_t>(dims);
		other_problem.dx = static_cast<other_num_t>(dx);
		other_problem.dy = static_cast<other_num_t>(dy);
		other_problem.dz = static_cast<other_num_t>(dz);
		other_problem.nx = static_cast<other_num_t>(nx);
		other_problem.ny = static_cast<other_num_t>(ny);
		other_problem.nz = static_cast<other_num_t>(nz);
		other_problem.substrates_count = static_cast<other_num_t>(substrates_count);
		other_problem.dt = static_cast<other_real_t>(dt);
		other_problem.diffusion_coefficient =
			std::vector<other_real_t>(diffusion_coefficient.begin(), diffusion_coefficient.end());
		other_problem.decay_rate = std::vector<other_real_t>(decay_rate.begin(), decay_rate.end());
		other_problem.initial_conditions =
			std::vector<other_real_t>(initial_conditions.begin(), initial_conditions.end());
		return other_problem;
	}
};

using max_problem_t = problem_t<std::size_t, double>;
