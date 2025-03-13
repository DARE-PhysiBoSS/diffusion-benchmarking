#include "least_memory_thomas_solver.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <omp.h>

#include "solver_utils.h"

template <typename real_t>
void least_memory_thomas_solver<real_t>::precompute_values(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b1,
														   std::unique_ptr<real_t[]>& b, index_t shape, index_t dims,
														   index_t n)
{
	a = std::make_unique<real_t[]>(problem_.substrates_count);
	b1 = std::make_unique<real_t[]>(problem_.substrates_count);
	b = std::make_unique<real_t[]>(n * problem_.substrates_count);

	auto layout = get_diagonal_layout(problem_, n);

	auto b_diag = noarr::make_bag(layout, b.get());

	// compute a
	for (index_t s = 0; s < problem_.substrates_count; s++)
		a[s] = -problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b1
	for (index_t s = 0; s < problem_.substrates_count; s++)
		b1[s] = 1 + problem_.decay_rates[s] * problem_.dt / dims
				+ 2 * problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t s = 0; s < problem_.substrates_count; s++)
				b_diag.template at<'i', 's'>(i, s) =
					1 + problem_.decay_rates[s] * problem_.dt / dims
					+ problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t s = 0; s < problem_.substrates_count; s++)
				b_diag.template at<'i', 's'>(i, s) =
					1 + problem_.decay_rates[s] * problem_.dt / dims
					+ 2 * problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);
	}

	// compute b_i'
	{
		for (index_t s = 0; s < problem_.substrates_count; s++)
			b_diag.template at<'i', 's'>(0, s) = 1 / b_diag.template at<'i', 's'>(0, s);

		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < problem_.substrates_count; s++)
			{
				b_diag.template at<'i', 's'>(i, s) =
					1 / (b_diag.template at<'i', 's'>(i, s) - a[s] * a[s] * b_diag.template at<'i', 's'>(i - 1, s));
			}
	}
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout<3>(problem_);

	solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_);
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"]
												: (problem_.nx + omp_get_num_threads() - 1) / omp_get_num_threads();
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::initialize()
{
	if (problem_.dims >= 1)
		precompute_values(ax_, b1x_, bx_, problem_.dx, problem_.dims, problem_.nx);
	if (problem_.dims >= 2)
		precompute_values(ay_, b1y_, by_, problem_.dy, problem_.dims, problem_.ny);
	if (problem_.dims >= 3)
		precompute_values(az_, b1z_, bz_, problem_.dz, problem_.dims, problem_.nz);
}

template <typename real_t>
template <std::size_t dims>
auto least_memory_thomas_solver<real_t>::get_substrates_layout(const problem_t<index_t, real_t>& problem)
{
	if constexpr (dims == 1)
		return noarr::scalar<real_t>() ^ noarr::vectors<'x', 's'>(problem.nx, problem.substrates_count);
	else if constexpr (dims == 2)
		return noarr::scalar<real_t>()
			   ^ noarr::vectors<'x', 'y', 's'>(problem.nx, problem.ny, problem.substrates_count);
	else if constexpr (dims == 3)
		return noarr::scalar<real_t>()
			   ^ noarr::vectors<'x', 'y', 'z', 's'>(problem.nx, problem.ny, problem.nz, problem.substrates_count);
}

template <typename real_t>
auto least_memory_thomas_solver<real_t>::get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n)
{
	return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(n, problem.substrates_count);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b1,
					  const real_t* __restrict__ b, const density_layout_t dens_l, const diagonal_layout_t diag_l,
					  std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static, work_items) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];
		real_t b_tmp = a_s / (b1_s + a_s);

		for (index_t i = 1; i < n; i++)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) -=
				(dens_l | noarr::get_at<'x', 's'>(densities, i - 1, s)) * b_tmp;

			b_tmp = a_s / (b1_s - a_s * b_tmp);

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}

		{
			(dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) =
				(dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s))
				* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));

			// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) =
				((dens_l | noarr::get_at<'x', 's'>(densities, i, s))
				 - a_s * (dens_l | noarr::get_at<'x', 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];

#pragma omp for schedule(static, work_items) nowait
		for (index_t yz = 0; yz < m; yz++)
		{
			real_t b_tmp = a_s / (b1_s + a_s);

			for (index_t i = 1; i < n; i++)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) -=
					(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i - 1, s)) * b_tmp;

				b_tmp = a_s / (b1_s - a_s * b_tmp);
			}

			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s)) =
					(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
					((dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
					 - a_s * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i + 1, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b1,
					  const real_t* __restrict__ b, const density_layout_t dens_l, const diagonal_layout_t diag_l,
					  std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];
		real_t b_tmp = a_s / (b1_s + a_s);

		for (index_t i = 1; i < n; i++)
		{
#pragma omp for schedule(static, work_items) nowait
			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i, s)) -=
					(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i - 1, s)) * b_tmp;
			}

			b_tmp = a_s / (b1_s - a_s * b_tmp);
		}

#pragma omp for schedule(static, work_items) nowait
		for (index_t x = 0; x < x_len; x++)
		{
			(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, n - 1, s)) =
				(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, n - 1, s))
				* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
#pragma omp for schedule(static, work_items) nowait
			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i, s)) =
					((dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i, s))
					 - a_s * (dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i + 1, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b1,
					  const real_t* __restrict__ b, const density_layout_t dens_l, const diagonal_layout_t diag_l,
					  std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];

#pragma omp for schedule(static, work_items) nowait
		for (index_t z = 0; z < z_len; z++)
		{
			real_t b_tmp = a_s / (b1_s + a_s);

			for (index_t i = 1; i < n; i++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i, s)) -=
						(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i - 1, s)) * b_tmp;
				}

				b_tmp = a_s / (b1_s - a_s * b_tmp);
			}

			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, n - 1, s)) =
					(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, n - 1, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i, s)) =
						((dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i, s))
						 - a_s * (dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i + 1, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b1,
					  const real_t* __restrict__ b, const density_layout_t dens_l, const diagonal_layout_t diag_l,
					  std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];
		real_t b_tmp = a_s / (b1_s + a_s);

		for (index_t i = 1; i < n; i++)
		{
#pragma omp for schedule(static, work_items) nowait
			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i, s)) -=
						(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i - 1, s)) * b_tmp;
				}
			}

			b_tmp = a_s / (b1_s - a_s * b_tmp);
		}

#pragma omp for schedule(static, work_items) nowait
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, n - 1, s)) =
					(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, n - 1, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
#pragma omp for schedule(static, work_items) nowait
			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i, s)) =
						((dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i, s))
						 - a_s * (dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i + 1, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::solve_x()
{
	if (problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(substrates_.get(), ax_.get(), b1x_.get(), bx_.get(),
								  get_substrates_layout<1>(problem_), get_diagonal_layout(problem_, problem_.nx),
								  work_items_);
	}
	else if (problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(substrates_.get(), ax_.get(), b1x_.get(), bx_.get(),
										 get_substrates_layout<2>(problem_) ^ noarr::rename<'y', 'm'>(),
										 get_diagonal_layout(problem_, problem_.nx), work_items_);
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(substrates_.get(), ax_.get(), b1x_.get(), bx_.get(),
										 get_substrates_layout<3>(problem_) ^ noarr::merge_blocks<'z', 'y', 'm'>(),
										 get_diagonal_layout(problem_, problem_.nx), work_items_);
	}
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::solve_y()
{
	if (problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(substrates_.get(), ay_.get(), b1y_.get(), by_.get(),
								  get_substrates_layout<2>(problem_), get_diagonal_layout(problem_, problem_.ny),
								  work_items_);
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(substrates_.get(), ay_.get(), b1y_.get(), by_.get(),
								  get_substrates_layout<3>(problem_), get_diagonal_layout(problem_, problem_.ny),
								  work_items_);
	}
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(substrates_.get(), az_.get(), b1z_.get(), bz_.get(), get_substrates_layout<3>(problem_),
							  get_diagonal_layout(problem_, problem_.nz), work_items_);
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::solve()
{
	if (problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(substrates_.get(), ax_.get(), b1x_.get(), bx_.get(),
								  get_substrates_layout<1>(problem_), get_diagonal_layout(problem_, problem_.nx),
								  work_items_);
	}
	if (problem_.dims == 2)
	{
#pragma omp parallel
		{
			solve_slice_x_2d_and_3d<index_t>(substrates_.get(), ax_.get(), b1x_.get(), bx_.get(),
											 get_substrates_layout<2>(problem_) ^ noarr::rename<'y', 'm'>(),
											 get_diagonal_layout(problem_, problem_.nx), work_items_);
#pragma omp barrier
			solve_slice_y_2d<index_t>(substrates_.get(), ay_.get(), b1y_.get(), by_.get(),
									  get_substrates_layout<2>(problem_), get_diagonal_layout(problem_, problem_.ny),
									  work_items_);
		}
	}
	if (problem_.dims == 3)
	{
#pragma omp parallel
		{
			solve_slice_x_2d_and_3d<index_t>(substrates_.get(), ax_.get(), b1x_.get(), bx_.get(),
											 get_substrates_layout<3>(problem_) ^ noarr::merge_blocks<'z', 'y', 'm'>(),
											 get_diagonal_layout(problem_, problem_.nx), work_items_);
#pragma omp barrier
			solve_slice_y_3d<index_t>(substrates_.get(), ay_.get(), b1y_.get(), by_.get(),
									  get_substrates_layout<3>(problem_), get_diagonal_layout(problem_, problem_.ny),
									  work_items_);
#pragma omp barrier
			solve_slice_z_3d<index_t>(substrates_.get(), az_.get(), b1z_.get(), bz_.get(),
									  get_substrates_layout<3>(problem_), get_diagonal_layout(problem_, problem_.nz),
									  work_items_);
		}
	}
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::save(const std::string& file) const
{
	auto dens_l = get_substrates_layout<3>(problem_);

	std::ofstream out(file);

	for (index_t z = 0; z < problem_.nz; z++)
		for (index_t y = 0; y < problem_.ny; y++)
			for (index_t x = 0; x < problem_.nx; x++)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
					out << (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) << " ";
				out << std::endl;
			}

	out.close();
}

template <typename real_t>
double least_memory_thomas_solver<real_t>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
	auto dens_l = get_substrates_layout<3>(problem_);

	return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
}

template <>
float least_memory_thomas_solver<float>::limit_threshold_ = 1e-6f;

template <>
double least_memory_thomas_solver<double>::limit_threshold_ = 1e-12;

template class least_memory_thomas_solver<float>;
template class least_memory_thomas_solver<double>;
