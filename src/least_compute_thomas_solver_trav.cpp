#include "least_compute_thomas_solver_trav.h"

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <omp.h>

#include "solver_utils.h"

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::precompute_values(std::unique_ptr<real_t[]>& b,
																 std::unique_ptr<real_t[]>& c,
																 std::unique_ptr<real_t[]>& e, index_t shape,
																 index_t dims, index_t n, index_t copies)
{
	b = std::make_unique<real_t[]>(n * problem_.substrates_count * copies);
	e = std::make_unique<real_t[]>((n - 1) * problem_.substrates_count * copies);
	c = std::make_unique<real_t[]>(problem_.substrates_count * copies);

	auto layout = noarr::scalar<real_t>() ^ noarr::vector<'s'>(problem_.substrates_count) ^ noarr::vector<'x'>(copies)
				  ^ noarr::vector<'i'>(n);

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
void least_compute_thomas_solver_trav<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout<3>(problem_);

	solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_);
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"]
												: (problem_.nx + omp_get_num_threads() - 1) / omp_get_num_threads();
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::initialize()
{
	if (problem_.dims >= 1)
		precompute_values(bx_, cx_, ex_, problem_.dx, problem_.dims, problem_.nx, 1);
	if (problem_.dims >= 2)
		precompute_values(by_, cy_, ey_, problem_.dy, problem_.dims, problem_.ny, 1);
	if (problem_.dims >= 3)
		precompute_values(bz_, cz_, ez_, problem_.dz, problem_.dims, problem_.nz, 1);
}

template <typename real_t>
template <std::size_t dims>
auto least_compute_thomas_solver_trav<real_t>::get_substrates_layout(const problem_t<index_t, real_t>& problem)
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

template <char slice_dim, char para_dim, typename index_t, typename real_t, typename density_layout_t>
void solve_slice(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
				 const real_t* __restrict__ e, const density_layout_t dens_l, std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<slice_dim>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count) ^ noarr::vector<slice_dim>(n);
	auto c_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count);

	auto para_dens_l = dens_l ^ noarr::into_blocks_static<para_dim, 'b', 'P', para_dim>(work_items)
					   ^ noarr::step<'P'>(omp_get_thread_num(), omp_get_num_threads());

	noarr::traverser(para_dens_l).order(noarr::shift<slice_dim>(noarr::lit<1>)).for_each([=](auto state) {
		auto prev_state = noarr::neighbor<slice_dim>(state, -1);
		(para_dens_l | noarr::get_at(densities, state)) =
			(para_dens_l | noarr::get_at(densities, state))
			- (diag_l | noarr::get_at(e, prev_state)) * (para_dens_l | noarr::get_at(densities, prev_state));
	});

	noarr::traverser(para_dens_l).order(noarr::fix<slice_dim>(n - 1)).for_each([=](auto state) {
		(para_dens_l | noarr::get_at(densities, state)) =
			(para_dens_l | noarr::get_at(densities, state)) * (diag_l | noarr::get_at(b, state));
	});

	noarr::traverser(para_dens_l)
		.order(noarr::reverse<slice_dim>() ^ noarr::shift<slice_dim>(noarr::lit<1>))
		.for_each([=](auto state) {
			auto next_state = noarr::neighbor<slice_dim>(state, 1);
			(para_dens_l | noarr::get_at(densities, state)) =
				((para_dens_l | noarr::get_at(densities, state))
				 - (c_l | noarr::get_at(c, state)) * (para_dens_l | noarr::get_at(densities, next_state)))
				* (diag_l | noarr::get_at(b, state));
		});
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::solve_x()
{
	if (problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice<'x', 's', index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
									   get_substrates_layout<1>(problem_), work_items_);
	}
	else if (problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice<'x', 'y', index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
									   get_substrates_layout<2>(problem_), work_items_);
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice<'x', 'z', index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
									   get_substrates_layout<3>(problem_), work_items_);
	}
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::solve_y()
{
	if (problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice<'y', 'x', index_t>(substrates_.get(), by_.get(), cy_.get(), ey_.get(),
									   get_substrates_layout<2>(problem_), work_items_);
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice<'y', 'z', index_t>(substrates_.get(), by_.get(), cy_.get(), ey_.get(),
									   get_substrates_layout<3>(problem_), work_items_);
	}
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::solve_z()
{
#pragma omp parallel
	solve_slice<'z', 'y', index_t>(substrates_.get(), bz_.get(), cz_.get(), ez_.get(),
								   get_substrates_layout<3>(problem_), work_items_);
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::solve()
{
	if (problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice<'x', 's', index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
									   get_substrates_layout<1>(problem_), work_items_);
	}
	else if (problem_.dims == 2)
	{
#pragma omp parallel
		{
			solve_slice<'x', 'y', index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
										   get_substrates_layout<2>(problem_), work_items_);
#pragma omp barrier
			solve_slice<'y', 'x', index_t>(substrates_.get(), by_.get(), cy_.get(), ey_.get(),
										   get_substrates_layout<2>(problem_), work_items_);
		}
	}
	else if (problem_.dims == 3)
	{
#pragma omp parallel
		{
			solve_slice<'x', 'z', index_t>(substrates_.get(), bx_.get(), cx_.get(), ex_.get(),
										   get_substrates_layout<3>(problem_), work_items_);
#pragma omp barrier
			solve_slice<'y', 'x', index_t>(substrates_.get(), by_.get(), cy_.get(), ey_.get(),
										   get_substrates_layout<3>(problem_), work_items_);
#pragma omp barrier
			solve_slice<'z', 'y', index_t>(substrates_.get(), bz_.get(), cz_.get(), ez_.get(),
										   get_substrates_layout<3>(problem_), work_items_);
		}
	}
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::save(const std::string& file) const
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
double least_compute_thomas_solver_trav<real_t>::access(std::size_t s, std::size_t x, std::size_t y,
														std::size_t z) const
{
	auto dens_l = get_substrates_layout<3>(problem_);

	return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
}

template class least_compute_thomas_solver_trav<float>;
template class least_compute_thomas_solver_trav<double>;
