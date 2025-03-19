#include "least_compute_thomas_solver_trav.h"

#include <cstddef>
#include <omp.h>

template <typename real_t>
template <char dim>
auto least_compute_thomas_solver_trav<real_t>::get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n)
{
	return noarr::scalar<real_t>() ^ noarr::vectors<'s', dim>(problem.substrates_count, n);
}

template <char slice_dim, char para_dim, typename index_t, typename real_t, typename density_layout_t,
		  typename diagonal_layout_t>
void solve_slice(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
				 const real_t* __restrict__ e, const density_layout_t dens_l, const diagonal_layout_t diag_l,
				 std::size_t work_items)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<slice_dim>();

	auto c_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count);

	auto parallelism = noarr::into_blocks_static<para_dim, 'b', 'P', para_dim>(work_items)
					   ^ noarr::step<'P'>(omp_get_thread_num(), omp_get_num_threads());

	noarr::traverser(dens_l).order(parallelism ^ noarr::shift<slice_dim>(noarr::lit<1>)).for_each([=](auto state) {
		auto prev_state = noarr::neighbor<slice_dim>(state, -1);
		(dens_l | noarr::get_at(densities, state)) =
			(dens_l | noarr::get_at(densities, state))
			- (diag_l | noarr::get_at(e, prev_state)) * (dens_l | noarr::get_at(densities, prev_state));
	});

	noarr::traverser(dens_l).order(parallelism ^ noarr::fix<slice_dim>(n - 1)).for_each([=](auto state) {
		(dens_l | noarr::get_at(densities, state)) =
			(dens_l | noarr::get_at(densities, state)) * (diag_l | noarr::get_at(b, state));
	});

	noarr::traverser(dens_l)
		.order(parallelism ^ noarr::reverse<slice_dim>() ^ noarr::shift<slice_dim>(noarr::lit<1>))
		.for_each([=](auto state) {
			auto next_state = noarr::neighbor<slice_dim>(state, 1);
			(dens_l | noarr::get_at(densities, state)) =
				((dens_l | noarr::get_at(densities, state))
				 - (c_l | noarr::get_at(c, state)) * (dens_l | noarr::get_at(densities, next_state)))
				* (diag_l | noarr::get_at(b, state));
		});
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::solve_x()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice<'x', 's', index_t>(this->substrates_, this->bx_.get(), this->cx_.get(), this->ex_.get(),
									   get_substrates_layout<1>(),
									   get_diagonal_layout<'x'>(this->problem_, this->problem_.nx), this->work_items_);
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice<'x', 'y', index_t>(this->substrates_, this->bx_.get(), this->cx_.get(), this->ex_.get(),
									   get_substrates_layout<2>(),
									   get_diagonal_layout<'x'>(this->problem_, this->problem_.nx), this->work_items_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice<'x', 'z', index_t>(this->substrates_, this->bx_.get(), this->cx_.get(), this->ex_.get(),
									   get_substrates_layout<3>(),
									   get_diagonal_layout<'x'>(this->problem_, this->problem_.nx), this->work_items_);
	}
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice<'y', 'x', index_t>(this->substrates_, this->by_.get(), this->cy_.get(), this->ey_.get(),
									   get_substrates_layout<2>(),
									   get_diagonal_layout<'y'>(this->problem_, this->problem_.ny), this->work_items_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice<'y', 'z', index_t>(this->substrates_, this->by_.get(), this->cy_.get(), this->ey_.get(),
									   get_substrates_layout<3>(),
									   get_diagonal_layout<'y'>(this->problem_, this->problem_.ny), this->work_items_);
	}
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::solve_z()
{
#pragma omp parallel
	solve_slice<'z', 'y', index_t>(this->substrates_, this->bz_.get(), this->cz_.get(), this->ez_.get(),
								   get_substrates_layout<3>(),
								   get_diagonal_layout<'z'>(this->problem_, this->problem_.nz), this->work_items_);
}

template <typename real_t>
void least_compute_thomas_solver_trav<real_t>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice<'x', 's', index_t>(this->substrates_, this->bx_.get(), this->cx_.get(), this->ex_.get(),
									   get_substrates_layout<1>(),
									   get_diagonal_layout<'x'>(this->problem_, this->problem_.nx), this->work_items_);
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			solve_slice<'x', 'y', index_t>(
				this->substrates_, this->bx_.get(), this->cx_.get(), this->ex_.get(), get_substrates_layout<2>(),
				get_diagonal_layout<'x'>(this->problem_, this->problem_.nx), this->work_items_);
#pragma omp barrier
			solve_slice<'y', 'x', index_t>(
				this->substrates_, this->by_.get(), this->cy_.get(), this->ey_.get(), get_substrates_layout<2>(),
				get_diagonal_layout<'y'>(this->problem_, this->problem_.ny), this->work_items_);
		}
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			solve_slice<'x', 'z', index_t>(
				this->substrates_, this->bx_.get(), this->cx_.get(), this->ex_.get(), get_substrates_layout<3>(),
				get_diagonal_layout<'x'>(this->problem_, this->problem_.nx), this->work_items_);
#pragma omp barrier
			solve_slice<'y', 'x', index_t>(
				this->substrates_, this->by_.get(), this->cy_.get(), this->ey_.get(), get_substrates_layout<3>(),
				get_diagonal_layout<'y'>(this->problem_, this->problem_.ny), this->work_items_);
#pragma omp barrier
			solve_slice<'z', 'y', index_t>(
				this->substrates_, this->bz_.get(), this->cz_.get(), this->ez_.get(), get_substrates_layout<3>(),
				get_diagonal_layout<'z'>(this->problem_, this->problem_.nz), this->work_items_);
		}
	}
}

template class least_compute_thomas_solver_trav<float>;
template class least_compute_thomas_solver_trav<double>;
