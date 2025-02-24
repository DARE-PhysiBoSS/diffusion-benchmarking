#include "reference_thomas_solver.h"

#include <noarr/traversers.hpp>

#include "omp_helper.h"

template <typename real_t>
void reference_thomas_solver<real_t>::precompute_values(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b,
														std::unique_ptr<real_t[]>& b0, index_t shape, index_t dims,
														index_t n)
{
	a = std::make_unique<real_t[]>(problem_.substrates_count);
	b = std::make_unique<real_t[]>(problem_.substrates_count * n);
	b0 = std::make_unique<real_t[]>(problem_.substrates_count);

	auto l = noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(problem_.substrates_count, n);

	// compute a_i, b0_i
	for (index_t s = 0; s < problem_.substrates_count; s++)
	{
		a[s] = -problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);
		b0[s] = 1 + problem_.dt * problem_.decay_rates[s] / dims
				+ problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);
	}

	// compute b_i
	for (index_t i = 0; i < n; i++)
		for (index_t s = 0; s < problem_.substrates_count; s++)
			if (i == 0)
				(l | noarr::get_at<'s', 'i'>(b.get(), s, i)) = b0[s];
			else if (i != n - 1)
				(l | noarr::get_at<'s', 'i'>(b.get(), s, i)) =
					(b0[s] - a[s]) - (a[s] * a[s]) / (l | noarr::get_at<'s', 'i'>(b.get(), s, i - 1));
			else
				(l | noarr::get_at<'s', 'i'>(b.get(), s, i)) =
					(b0[s]) - (a[s] * a[s]) / (l | noarr::get_at<'s', 'i'>(b.get(), s, i - 1));
}

template <typename num_t, typename real_t>
auto get_substrates_layout(const problem_t<num_t, real_t>& problem)
{
	return noarr::scalar<real_t>()
		   ^ noarr::vectors<'s', 'x', 'y', 'z'>(problem.substrates_count, problem.nz, problem.ny, problem.nx);
}

template <typename real_t>
void reference_thomas_solver<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problem_cast<std::int32_t, real_t>(problem);
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout(problem_);

	omp_trav_for_each(noarr::traverser(substrates_layout), [&](auto state) {
		auto s_idx = noarr::get_index<'s'>(state);

		(substrates_layout | noarr::get_at(substrates_.get(), state)) = problem_.initial_conditions[s_idx];
	});
}

template <typename real_t>
void reference_thomas_solver<real_t>::initialize()
{
	if (problem_.dims >= 1)
		precompute_values(a_, bx_, b0_, problem_.dx, problem_.dims, problem_.nx);
	if (problem_.dims >= 2)
		precompute_values(a_, by_, b0_, problem_.dy, problem_.dims, problem_.ny);
	if (problem_.dims >= 3)
		precompute_values(a_, bz_, b0_, problem_.dz, problem_.dims, problem_.nz);
}

template <typename real_t>
void reference_thomas_solver<real_t>::solve_x()
{
	auto dens_l = get_substrates_layout(problem_);
	auto diag_l = noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(problem_.substrates_count, problem_.nx);

	for (index_t z = 0; z < problem_.nz; z++)
	{
		for (index_t y = 0; y < problem_.ny; y++)
		{
			for (index_t s = 0; s < problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, 1, y, z)) -=
					(a_[s] / (b0_[s] - a_[s]))
					* (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, 0, y, z));
			}

			for (index_t x = 2; x < problem_.nx; x++)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) -=
						(a_[s] / b0_[s])
						* (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x - 1, y, z));
				}
			}

			for (index_t s = 0; s < problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, problem_.nx - 1, y, z)) /=
					(diag_l | noarr::get_at<'s', 'i'>(bx_.get(), s, problem_.nx - 1));
			}

			for (index_t x = problem_.nx - 2; x >= 0; x--)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) =
						((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z))
						 - a_[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x + 1, y, z)))
						/ (diag_l | noarr::get_at<'s', 'i'>(bx_.get(), s, x));
				}
			}
		}
	}
}

template <typename real_t>
void reference_thomas_solver<real_t>::solve_y()
{
	auto dens_l = get_substrates_layout(problem_);
	auto diag_l = noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(problem_.substrates_count, problem_.ny);

	for (index_t z = 0; z < problem_.nz; z++)
	{
		for (index_t x = 0; x < problem_.nx; x++)
		{
			for (index_t s = 0; s < problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, 1, z)) -=
					(a_[s] / (b0_[s] - a_[s]))
					* (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, 0, z));
			}
		}
	}

	for (index_t z = 0; z < problem_.nz; z++)
	{
		for (index_t y = 2; y < problem_.ny; y++)
		{
			for (index_t x = 0; x < problem_.nx; x++)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) -=
						(a_[s] / b0_[s])
						* (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y - 1, z));
				}
			}
		}
	}

	for (index_t z = 0; z < problem_.nz; z++)
	{
		for (index_t x = 0; x < problem_.nx; x++)
		{
			for (index_t s = 0; s < problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, problem_.ny - 1, z)) /=
					(diag_l | noarr::get_at<'s', 'i'>(by_.get(), s, problem_.ny - 1));
			}
		}
	}

	for (index_t z = 0; z < problem_.nz; z++)
	{
		for (index_t y = problem_.ny - 2; y >= 0; y--)
		{
			for (index_t x = 0; x < problem_.nx; x++)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) =
						((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z))
						 - a_[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y + 1, z)))
						/ (diag_l | noarr::get_at<'s', 'i'>(by_.get(), s, y));
				}
			}
		}
	}
}

template <typename real_t>
void reference_thomas_solver<real_t>::solve_z()
{
	auto dens_l = get_substrates_layout(problem_);
	auto diag_l = noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(problem_.substrates_count, problem_.nz);

	for (index_t y = 0; y < problem_.ny; y++)
	{
		for (index_t x = 0; x < problem_.nx; x++)
		{
			for (index_t s = 0; s < problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, 1)) -=
					(a_[s] / (b0_[s] - a_[s]))
					* (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, 0));
			}
		}
	}

	for (index_t z = 2; z < problem_.nz; z++)
	{
		for (index_t y = 0; y < problem_.ny; y++)
		{
			for (index_t x = 0; x < problem_.nx; x++)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) -=
						(a_[s] / b0_[s])
						* (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z - 1));
				}
			}
		}
	}

	for (index_t y = 0; y < problem_.ny; y++)
	{
		for (index_t x = 0; x < problem_.nx; x++)
		{
			for (index_t s = 0; s < problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, problem_.nz - 1)) /=
					(diag_l | noarr::get_at<'s', 'i'>(bz_.get(), s, problem_.nz - 1));
			}
		}
	}

	for (index_t z = problem_.nz - 2; z >= 0; z--)
	{
		for (index_t y = 0; y < problem_.ny; y++)
		{
			for (index_t x = 0; x < problem_.nx; x++)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) =
						((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z))
						 - a_[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z + 1)))
						/ (diag_l | noarr::get_at<'s', 'i'>(bz_.get(), s, z));
				}
			}
		}
	}
}

template class reference_thomas_solver<float>;
template class reference_thomas_solver<double>;
