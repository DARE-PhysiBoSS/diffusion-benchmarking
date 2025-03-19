#include "least_compute_thomas_solver_s_t.h"

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <omp.h>

#include "noarr/structures/extra/funcs.hpp"
#include "noarr/structures/structs/setters.hpp"
#include "solver_utils.h"

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::precompute_values(std::unique_ptr<real_t[]>& b,
																		   std::unique_ptr<real_t[]>& c,
																		   std::unique_ptr<real_t[]>& e, index_t shape,
																		   index_t dims, index_t n)
{
	b = std::make_unique<real_t[]>(n * problem_.substrates_count);
	e = std::make_unique<real_t[]>(n * problem_.substrates_count);
	c = std::make_unique<real_t[]>(problem_.substrates_count);

	auto layout = get_diagonal_layout(problem_, n);

	auto b_diag = noarr::make_bag(layout, b.get());
	auto e_diag = noarr::make_bag(layout, e.get());

	// compute c_i
	for (index_t s = 0; s < problem_.substrates_count; s++)
		c[s] = -problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);

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

	// compute b_i' and e_i
	{
		for (index_t s = 0; s < problem_.substrates_count; s++)
			b_diag.template at<'i', 's'>(0, s) = 1 / b_diag.template at<'i', 's'>(0, s);

		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < problem_.substrates_count; s++)
			{
				b_diag.template at<'i', 's'>(i, s) =
					1 / (b_diag.template at<'i', 's'>(i, s) - c[s] * c[s] * b_diag.template at<'i', 's'>(i - 1, s));

				e_diag.template at<'i', 's'>(i - 1, s) = c[s] * b_diag.template at<'i', 's'>(i - 1, s);
			}
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);

	auto substrates_layout = get_substrates_layout<3>(problem_);

	if (aligned_x)
		substrates_ = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
	else
		substrates_ = (real_t*)std::malloc((substrates_layout | noarr::get_size()));

	// Initialize substrates

	solver_utils::initialize_substrate(substrates_layout, substrates_, problem_);
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"]
												: (problem_.nx + omp_get_num_threads() - 1) / omp_get_num_threads();
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 1;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::initialize()
{
	if (problem_.dims >= 1)
		precompute_values(bx_, cx_, ex_, problem_.dx, problem_.dims, problem_.nx);
	if (problem_.dims >= 2)
		precompute_values(by_, cy_, ey_, problem_.dy, problem_.dims, problem_.ny);
	if (problem_.dims >= 3)
		precompute_values(bz_, cz_, ez_, problem_.dz, problem_.dims, problem_.nz);
}

template <typename real_t, bool aligned_x>
template <std::size_t dims>
auto least_compute_thomas_solver_s_t<real_t, aligned_x>::get_substrates_layout(
	const problem_t<index_t, real_t>& problem) const
{
	if constexpr (!aligned_x)
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
	else
	{
		std::size_t x_size = problem.nx * sizeof(real_t);
		std::size_t x_size_padded = (x_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		x_size_padded /= sizeof(real_t);

		if constexpr (dims == 1)
			return noarr::scalar<real_t>() ^ noarr::vectors<'x', 's'>(x_size_padded, problem.substrates_count)
				   ^ noarr::slice<'x'>(problem.nx);
		else if constexpr (dims == 2)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 's'>(x_size_padded, problem.ny, problem.substrates_count)
				   ^ noarr::slice<'x'>(problem.nx);
		else if constexpr (dims == 3)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 'z', 's'>(x_size_padded, problem.ny, problem.nz, problem.substrates_count)
				   ^ noarr::slice<'x'>(problem.nx);
	}
}

template <typename real_t, bool aligned_x>
auto least_compute_thomas_solver_s_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
																			 index_t n)
{
	return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(n, problem.substrates_count);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					  const real_t* __restrict__ e, const density_layout_t dens_l, const diagonal_layout_t diag_l,
					  std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static, work_items) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t i = 1; i < n; i++)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) =
				(dens_l | noarr::get_at<'x', 's'>(densities, i, s))
				- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
					  * (dens_l | noarr::get_at<'x', 's'>(densities, i - 1, s));

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
				 - c[s] * (dens_l | noarr::get_at<'x', 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static, work_items) nowait collapse(2)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t yz = 0; yz < m; yz++)
		{
			for (index_t i = 1; i < n; i++)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
					(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
					- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
						  * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i - 1, s));
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
					 - c[s] * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i + 1, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					  const real_t* __restrict__ e, const density_layout_t dens_l, const diagonal_layout_t diag_l,
					  std::size_t x_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

	for (index_t s = 0; s < substrates_count; s++)
	{
		// body
		{
			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t x_len = body_dens_l | noarr::get_length<'x'>();
			const index_t X_len = body_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait
			for (index_t X = 0; X < X_len; X++)
			{
				for (index_t i = 1; i < n; i++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, X, x, s)) =
							(body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, X, x, s))
							- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
								  * (body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i - 1, X, x, s));
					}
				}

				for (index_t x = 0; x < x_len; x++)
				{
					(body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, n - 1, X, x, s)) =
						(body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, n - 1, X, x, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, X, x, s)) =
							((body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, X, x, s))
							 - c[s] * (body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i + 1, X, x, s)))
							* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
					}
				}
			}
		}

// remainder
#pragma omp single
		{
			auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t x_len = border_dens_l | noarr::get_length<'x'>();

			for (index_t i = 1; i < n; i++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(border_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, noarr::lit<0>, x, s)) =
						(border_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, noarr::lit<0>, x, s))
						- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
							  * (border_dens_l
								 | noarr::get_at<'y', 'X', 'x', 's'>(densities, i - 1, noarr::lit<0>, x, s));
				}
			}

			for (index_t x = 0; x < x_len; x++)
			{
				(border_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, n - 1, noarr::lit<0>, x, s)) =
					(border_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, n - 1, noarr::lit<0>, x, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(border_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, noarr::lit<0>, x, s)) =
						((border_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, noarr::lit<0>, x, s))
						 - c[s]
							   * (border_dens_l
								  | noarr::get_at<'y', 'X', 'x', 's'>(densities, i + 1, noarr::lit<0>, x, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					  const real_t* __restrict__ e, const density_layout_t dens_l, const diagonal_layout_t diag_l,
					  std::size_t work_items, std::size_t x_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

	for (index_t s = 0; s < substrates_count; s++)
	{
#pragma omp for schedule(static, work_items) nowait
		for (index_t z = 0; z < z_len; z++)
		{
			// body
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t x_len = body_dens_l | noarr::get_length<'x'>();
				const index_t X_len = body_dens_l | noarr::get_length<'X'>();

				for (index_t X = 0; X < X_len; X++)
				{
					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, X, x, s)) =
								(body_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, X, x, s))
								- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
									  * (body_dens_l
										 | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i - 1, X, x, s));
						}
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, n - 1, X, x, s)) =
							(body_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, n - 1, X, x, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, X, x, s)) =
								((body_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, X, x, s))
								 - c[s]
									   * (body_dens_l
										  | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i + 1, X, x, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}

			// border
			{
				auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t x_len = border_dens_l | noarr::get_length<'x'>();

				for (index_t i = 1; i < n; i++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, noarr::lit<0>, x, s)) =
							(border_dens_l
							 | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, noarr::lit<0>, x, s))
							- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
								  * (border_dens_l
									 | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i - 1, noarr::lit<0>, x,
																			  s));
					}
				}

				for (index_t x = 0; x < x_len; x++)
				{
					(border_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, n - 1, noarr::lit<0>, x, s)) =
						(border_dens_l
						 | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, n - 1, noarr::lit<0>, x, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, noarr::lit<0>, x, s)) =
							((border_dens_l
							  | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, noarr::lit<0>, x, s))
							 - c[s]
								   * (border_dens_l
									  | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i + 1, noarr::lit<0>, x,
																			   s)))
							* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					  const real_t* __restrict__ e, const density_layout_t dens_l, const diagonal_layout_t diag_l,
					  std::size_t work_items, std::size_t x_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

	for (index_t s = 0; s < substrates_count; s++)
	{
#pragma omp for schedule(static, work_items) nowait
		for (index_t y = 0; y < y_len; y++)
		{
			// body
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t x_len = body_dens_l | noarr::get_length<'x'>();
				const index_t X_len = body_dens_l | noarr::get_length<'X'>();

				for (index_t X = 0; X < X_len; X++)
				{
					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, i, y, x, s)) =
								(body_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, i, y, x, s))
								- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
									  * (body_dens_l
										 | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, i - 1, y, x, s));
						}
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, n - 1, y, x, s)) =
							(body_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, n - 1, y, x, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, i, y, x, s)) =
								((body_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, i, y, x, s))
								 - c[s]
									   * (body_dens_l
										  | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, i + 1, y, x, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}

			// remainder
			{
				auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t x_len = border_dens_l | noarr::get_length<'x'>();

				for (index_t i = 1; i < n; i++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, noarr::lit<0>, i, y, x, s)) =
							(border_dens_l
							 | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, noarr::lit<0>, i, y, x, s))
							- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
								  * (border_dens_l
									 | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, noarr::lit<0>, i - 1, y, x,
																			  s));
					}
				}

				for (index_t x = 0; x < x_len; x++)
				{
					(border_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, noarr::lit<0>, n - 1, y, x, s)) =
						(border_dens_l
						 | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, noarr::lit<0>, n - 1, y, x, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, noarr::lit<0>, i, y, x, s)) =
							((border_dens_l
							  | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, noarr::lit<0>, i, y, x, s))
							 - c[s]
								   * (border_dens_l
									  | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, noarr::lit<0>, i + 1, y, x,
																			   s)))
							* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
					}
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve_x()
{
	if (problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(substrates_, bx_.get(), cx_.get(), ex_.get(), get_substrates_layout<1>(problem_),
								  get_diagonal_layout(problem_, problem_.nx), work_items_);
	}
	else if (problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(substrates_, bx_.get(), cx_.get(), ex_.get(),
										 get_substrates_layout<2>(problem_) ^ noarr::rename<'y', 'm'>(),
										 get_diagonal_layout(problem_, problem_.nx), work_items_);
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(substrates_, bx_.get(), cx_.get(), ex_.get(),
										 get_substrates_layout<3>(problem_) ^ noarr::merge_blocks<'z', 'y', 'm'>(),
										 get_diagonal_layout(problem_, problem_.nx), work_items_);
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve_y()
{
	if (problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<2>(problem_),
								  get_diagonal_layout(problem_, problem_.ny), x_tile_size_);
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<3>(problem_),
								  get_diagonal_layout(problem_, problem_.ny), work_items_, x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(substrates_, bz_.get(), cz_.get(), ez_.get(), get_substrates_layout<3>(problem_),
							  get_diagonal_layout(problem_, problem_.nz), work_items_, x_tile_size_);
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve()
{
	if (problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(substrates_, bx_.get(), cx_.get(), ex_.get(), get_substrates_layout<1>(problem_),
								  get_diagonal_layout(problem_, problem_.nx), work_items_);
	}
	else if (problem_.dims == 2)
	{
#pragma omp parallel
		{
			solve_slice_x_2d_and_3d<index_t>(substrates_, bx_.get(), cx_.get(), ex_.get(),
											 get_substrates_layout<2>(problem_) ^ noarr::rename<'y', 'm'>(),
											 get_diagonal_layout(problem_, problem_.nx), work_items_);
#pragma omp barrier
			solve_slice_y_2d<index_t>(substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<2>(problem_),
									  get_diagonal_layout(problem_, problem_.ny), work_items_);
		}
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		{
			solve_slice_x_2d_and_3d<index_t>(substrates_, bx_.get(), cx_.get(), ex_.get(),
											 get_substrates_layout<3>(problem_) ^ noarr::merge_blocks<'z', 'y', 'm'>(),
											 get_diagonal_layout(problem_, problem_.nx), work_items_);
#pragma omp barrier
			solve_slice_y_3d<index_t>(substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<3>(problem_),
									  get_diagonal_layout(problem_, problem_.ny), work_items_, x_tile_size_);
#pragma omp barrier
			solve_slice_z_3d<index_t>(substrates_, bz_.get(), cz_.get(), ez_.get(), get_substrates_layout<3>(problem_),
									  get_diagonal_layout(problem_, problem_.nz), work_items_, x_tile_size_);
		}
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::save(const std::string& file) const
{
	auto dens_l = get_substrates_layout<3>(problem_);

	std::ofstream out(file);

	for (index_t z = 0; z < problem_.nz; z++)
		for (index_t y = 0; y < problem_.ny; y++)
			for (index_t x = 0; x < problem_.nx; x++)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
					out << (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_, s, x, y, z)) << " ";
				out << std::endl;
			}

	out.close();
}

template <typename real_t, bool aligned_x>
double least_compute_thomas_solver_s_t<real_t, aligned_x>::access(std::size_t s, std::size_t x, std::size_t y,
																  std::size_t z) const
{
	auto dens_l = get_substrates_layout<3>(problem_);

	return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_, s, x, y, z));
}

template <typename real_t, bool aligned_x>
least_compute_thomas_solver_s_t<real_t, aligned_x>::~least_compute_thomas_solver_s_t()
{
	std::free(substrates_);
}

template class least_compute_thomas_solver_s_t<float, false>;
template class least_compute_thomas_solver_s_t<double, false>;

template class least_compute_thomas_solver_s_t<float, true>;
template class least_compute_thomas_solver_s_t<double, true>;
