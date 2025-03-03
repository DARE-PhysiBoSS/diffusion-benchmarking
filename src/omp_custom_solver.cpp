#include "omp_custom_solver.h"

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>

#include <noarr/traversers.hpp>

#include "omp_helper.h"

template <typename real_t>
void omp_custom_solver<real_t>::precompute_values(std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& c,
												  std::unique_ptr<real_t[]>& e, index_t shape, index_t dims, index_t n,
												  index_t copies)
{
	b = std::make_unique<real_t[]>(n * problem_.substrates_count * copies);
	e = std::make_unique<real_t[]>((n - 1) * problem_.substrates_count * copies);
	c = std::make_unique<real_t[]>(problem_.substrates_count * copies);

	auto layout = noarr::scalar<real_t>() ^ noarr::vector<'s'>() ^ noarr::vector<'x'>() ^ noarr::vector<'i'>()
				  ^ noarr::set_length<'i'>(n) ^ noarr::set_length<'x'>(copies)
				  ^ noarr::set_length<'s'>(problem_.substrates_count);

	auto b_diag = noarr::make_bag(layout, b.get());
	auto e_diag = noarr::make_bag(layout, e.get());

	// compute c_i
	for (index_t x = 0; x < copies; x++)
		for (index_t s = 0; s < problem_.substrates_count; s++)
			c[x * problem_.substrates_count + s] = -problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < problem_.substrates_count; s++)
					b_diag.template at<'i', 'x', 's'>(i, x, s) =
						1 + problem_.decay_rates[s] * problem_.dt / dims
						+ problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < problem_.substrates_count; s++)
					b_diag.template at<'i', 'x', 's'>(i, x, s) =
						1 + problem_.decay_rates[s] * problem_.dt / dims
						+ 2 * problem_.dt * problem_.diffusion_coefficients[s] / (shape * shape);
	}

	// compute b_i' and e_i
	{
		for (index_t x = 0; x < copies; x++)
			for (index_t s = 0; s < problem_.substrates_count; s++)
				b_diag.template at<'i', 'x', 's'>(0, x, s) = 1 / b_diag.template at<'i', 'x', 's'>(0, x, s);

		for (index_t i = 1; i < n; i++)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < problem_.substrates_count; s++)
				{
					b_diag.template at<'i', 'x', 's'>(i, x, s) =
						1
						/ (b_diag.template at<'i', 'x', 's'>(i, x, s)
						   - c[x * problem_.substrates_count + s] * c[x * problem_.substrates_count + s]
								 * b_diag.template at<'i', 'x', 's'>(i - 1, x, s));

					e_diag.template at<'i', 'x', 's'>(i - 1, x, s) =
						c[x * problem_.substrates_count + s] * b_diag.template at<'i', 'x', 's'>(i - 1, x, s);
				}
	}
}

template <typename real_t>
void omp_custom_solver<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout<3>(problem_);

	omp_trav_for_each(noarr::traverser(substrates_layout), [&](auto state) {
		auto s_idx = noarr::get_index<'s'>(state);

		(substrates_layout | noarr::get_at(substrates_.get(), state)) = problem_.initial_conditions[s_idx];
	});
}

template <typename real_t>
void omp_custom_solver<real_t>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"] : 1;
}

template <typename real_t>
void omp_custom_solver<real_t>::initialize()
{
	if (problem_.dims >= 1)
		precompute_values(bx_, cx_, ex_, problem_.dx, problem_.dims, problem_.nx, 1);
	if (problem_.dims >= 2)
		precompute_values(by_, cy_, ey_, problem_.dy, problem_.dims, problem_.ny, 1);
	if (problem_.dims >= 3)
		precompute_values(bz_, cz_, ez_, problem_.dz, problem_.dims, problem_.nz, 1);
}

template <std::size_t dims, typename num_t, typename real_t>
auto get_substrates_layout(const problem_t<num_t, real_t>& problem)
{
	if constexpr (dims == 1)
		return noarr::scalar<real_t>() ^ noarr::vectors<'s', 'x'>(problem.substrates_count, problem.nx);
	else if constexpr (dims == 2)
		return noarr::scalar<real_t>()
			   ^ noarr::vectors<'s', 'x', 'y'>(problem.substrates_count, problem.nx, problem.ny);
	else if constexpr (dims == 3)
		return noarr::scalar<real_t>()
			   ^ noarr::vectors<'s', 'x', 'y', 'z'>(problem.substrates_count, problem.nx, problem.ny, problem.nz);
}

template <typename index_t, typename real_t, typename density_layout_t>
void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					  const real_t* __restrict__ e, const density_layout_t dens_l, std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count) ^ noarr::vector<'i'>(n);

	for (index_t i = 1; i < n; i++)
	{
#pragma omp for schedule(static, work_items) nowait
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) =
				(dens_l | noarr::get_at<'x', 's'>(densities, i, s))
				- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
					  * (dens_l | noarr::get_at<'x', 's'>(densities, i - 1, s));
		}
	}

#pragma omp for schedule(static, work_items) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		(dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) =
			(dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) * (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
#pragma omp for schedule(static, work_items) nowait
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) =
				((dens_l | noarr::get_at<'x', 's'>(densities, i, s))
				 - c[s] * (dens_l | noarr::get_at<'x', 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
		}
	}

#pragma omp barrier
}

template <typename index_t, typename real_t, typename density_layout_t>
void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l, std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count) ^ noarr::vector<'i'>(n);

#pragma omp for schedule(static, work_items)
	for (index_t yz = 0; yz < m; yz++)
	{
		for (index_t i = 1; i < n; i++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
					(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
					- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
						  * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i - 1, s));
			}
		}

		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s)) =
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s))
				* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
					((dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
					 - c[s] * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i + 1, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					  const real_t* __restrict__ e, const density_layout_t dens_l, std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count) ^ noarr::vector<'i'>(n);

	for (index_t i = 1; i < n; i++)
	{
#pragma omp for collapse(2) schedule(static, work_items) nowait
		for (index_t x = 0; x < x_len; x++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'y', 'x', 's'>(densities, i, x, s)) =
					(dens_l | noarr::get_at<'y', 'x', 's'>(densities, i, x, s))
					- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
						  * (dens_l | noarr::get_at<'y', 'x', 's'>(densities, i - 1, x, s));
			}
		}
	}

#pragma omp for collapse(2) schedule(static, work_items) nowait
	for (index_t x = 0; x < x_len; x++)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'y', 'x', 's'>(densities, n - 1, x, s)) =
				(dens_l | noarr::get_at<'y', 'x', 's'>(densities, n - 1, x, s))
				* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
		}
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
#pragma omp for collapse(2) schedule(static, work_items) nowait
		for (index_t x = 0; x < x_len; x++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'y', 'x', 's'>(densities, i, x, s)) =
					((dens_l | noarr::get_at<'y', 'x', 's'>(densities, i, x, s))
					 - c[s] * (dens_l | noarr::get_at<'y', 'x', 's'>(densities, i + 1, x, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}

#pragma omp barrier
}

template <typename index_t, typename real_t, typename density_layout_t>
void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					  const real_t* __restrict__ e, const density_layout_t dens_l, std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count) ^ noarr::vector<'i'>(n);

#pragma omp for schedule(static, work_items)
	for (index_t z = 0; z < z_len; z++)
	{
		for (index_t i = 1; i < n; i++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				for (index_t s = 0; s < substrates_count; s++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i, x, s)) =
						(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i, x, s))
						- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
							  * (dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i - 1, x, s));
				}
			}
		}

		for (index_t x = 0; x < x_len; x++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, n - 1, x, s)) =
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, n - 1, x, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				for (index_t s = 0; s < substrates_count; s++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i, x, s)) =
						((dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i, x, s))
						 - c[s] * (dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i + 1, x, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					  const real_t* __restrict__ e, const density_layout_t dens_l, std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count) ^ noarr::vector<'i'>(n);

	for (index_t i = 1; i < n; i++)
	{
#pragma omp for collapse(3) schedule(static, work_items) nowait
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				for (index_t s = 0; s < substrates_count; s++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i, y, x, s)) =
						(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i, y, x, s))
						- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
							  * (dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i - 1, y, x, s));
				}
			}
		}
	}

#pragma omp for collapse(3) schedule(static, work_items) nowait
	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t x = 0; x < x_len; x++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, n - 1, y, x, s)) =
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, n - 1, y, x, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}
		}
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
#pragma omp for collapse(3) schedule(static, work_items) nowait
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				for (index_t s = 0; s < substrates_count; s++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i, y, x, s)) =
						((dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i, y, x, s))
						 - c[s] * (dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i + 1, y, x, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}

#pragma omp barrier
}

template <typename real_t>
void omp_custom_solver<real_t>::solve_x()
{
	if (problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
								  get_substrates_layout<1>(problem_), work_items_);
	}
	else if (problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
										 get_substrates_layout<2>(problem_) ^ noarr::rename<'y', 'm'>(), work_items_);
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
										 get_substrates_layout<3>(problem_) ^ noarr::merge_blocks<'z', 'y', 'm'>(),
										 work_items_);
	}
}

template <typename real_t>
void omp_custom_solver<real_t>::solve_y()
{
	if (problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(substrates_.get(), by_.get(), cy_.get(), ey_.get(),
								  get_substrates_layout<2>(problem_), work_items_);
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(substrates_.get(), by_.get(), cy_.get(), ey_.get(),
								  get_substrates_layout<3>(problem_), work_items_);
	}
}

template <typename real_t>
void omp_custom_solver<real_t>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(substrates_.get(), bz_.get(), cz_.get(), ez_.get(), get_substrates_layout<3>(problem_),
							  work_items_);
}

template <typename real_t>
void omp_custom_solver<real_t>::save(const std::string& file) const
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
double omp_custom_solver<real_t>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
	auto dens_l = get_substrates_layout<3>(problem_);

	return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
}

template class omp_custom_solver<float>;
template class omp_custom_solver<double>;
