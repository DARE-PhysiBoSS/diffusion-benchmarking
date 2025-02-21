#include "omp_custom_solver.h"

#include <array>

#include <noarr/traversers.hpp>

template <typename T, typename F>
inline void omp_trav_for_each(const T& trav, const F& f)
{
#pragma omp parallel for
	for (auto trav_inner : trav)
		trav_inner.for_each(f);
}

template <typename real_t>
void omp_custom_solver<real_t>::precompute_values(std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& c,
												  std::unique_ptr<real_t[]>& e, index_t shape, index_t dims, index_t n,
												  index_t copies)
{
	if (n == 1) // special case
	{
		b = std::make_unique<real_t[]>(problem_.substrates_count * copies);

		for (index_t x = 0; x < copies; x++)
			for (index_t s = 0; s < problem_.substrates_count; s++)
				b[x * problem_.substrates_count + s] =
					1 / (1 + problem_.decay_rates[s] * problem_.diffusion_time_step / dims);

		return;
	}

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
			c[x * problem_.substrates_count + s] =
				-problem_.diffusion_time_step * problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < problem_.substrates_count; s++)
					b_diag.template at<'i', 'x', 's'>(i, x, s) =
						1 + problem_.decay_rates[s] * problem_.diffusion_time_step / dims
						+ problem_.diffusion_time_step * problem_.diffusion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < problem_.substrates_count; s++)
					b_diag.template at<'i', 'x', 's'>(i, x, s) =
						1 + problem_.decay_rates[s] * problem_.diffusion_time_step / dims
						+ 2 * problem_.diffusion_time_step * problem_.diffusion_coefficients[s] / (shape * shape);
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
	problem_ = problem.cast<std::int32_t, real_t>();
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout();

	omp_trav_for_each(noarr::traverser(substrates_layout), [&](auto state) {
		auto s_idx = noarr::get_index<'s'>(state);

		(substrates_layout | noarr::get_at(substrates_.get(), state)) = problem_.initial_conditions[s_idx];
	});
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

template <typename real_t>
auto omp_custom_solver<real_t>::get_substrates_layout()
{
	return noarr::scalar<real_t>()
		   ^ noarr::vectors<'s', 'x', 'y', 'z'>(problem_.substrates_count, problem_.nx, problem_.ny, problem_.nz);
}

template <char swipe_dim, typename index_t, typename real_t, typename density_layout_t>
void solve_slice_omp(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
					 const real_t* __restrict__ e, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<swipe_dim>();

	auto diag_l = noarr::scalar<real_t>() ^ noarr::vector<'s'>(substrates_count) ^ noarr::vector<'i'>(n);

#pragma omp for
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t i = 1; i < n; i++)
		{
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s)) =
				(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s))
				- (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
					  * (dens_l | noarr::get_at<swipe_dim, 's'>(densities, i - 1, s));
		}
	}

#pragma omp for
	for (index_t s = 0; s < substrates_count; s++)
	{
		(dens_l | noarr::get_at<swipe_dim, 's'>(densities, n - 1, s)) =
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, n - 1, s))
			* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
	}

#pragma omp for
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t i = n - 2; i >= 0; i--)
		{
			(dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s)) =
				((dens_l | noarr::get_at<swipe_dim, 's'>(densities, i, s))
				 - c[s] * (dens_l | noarr::get_at<swipe_dim, 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
		}
	}
}

void solve_x() {}
