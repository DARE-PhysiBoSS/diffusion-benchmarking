#include "cyclic_reduction_solver.h"

#include <cstddef>
#include <cstdlib>

#include "omp_helper.h"

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, index_t shape, index_t dims)
{
	// allocate memory for a and b1
	a = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));
	b1 = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));

	// compute a
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		a[s] = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b1
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		b1[s] = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
				+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::prepare(const max_problem_t& problem)
{
	this->problem_ = problems::cast<std::int32_t, real_t>(problem);

	auto substrates_layout = get_substrates_layout<3>();

	if (aligned_x)
		this->substrates_ = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
	else
		this->substrates_ = (real_t*)std::malloc((substrates_layout | noarr::get_size()));

	// Initialize substrates
	solver_utils::initialize_substrate(substrates_layout, this->substrates_, this->problem_);
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, this->problem_.dx, this->problem_.dims);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, this->problem_.dy, this->problem_.dims);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, this->problem_.dz, this->problem_.dims);

	auto max_n = std::max({ this->problem_.nx, this->problem_.ny, this->problem_.nz });

	for (index_t i = 0; i < get_max_threads(); i++)
	{
		a_scratch_.push_back((real_t*)std::malloc(max_n * sizeof(real_t)));
		b_scratch_.push_back((real_t*)std::malloc(max_n * sizeof(real_t)));
		c_scratch_.push_back((real_t*)std::malloc(max_n * sizeof(real_t)));
	}
}

template <typename real_t, bool aligned_x>
auto cyclic_reduction_solver<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
																	 index_t n)
{
	if constexpr (aligned_x)
	{
		std::size_t size = n * sizeof(real_t);
		std::size_t size_padded = (size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		size_padded /= sizeof(real_t);

		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(size_padded, problem.substrates_count)
			   ^ noarr::slice<'i'>(n);
	}
	else
	{
		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(n, problem.substrates_count);
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_l = dens_l ^ noarr::step<'x'>(1, 2);

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		// prepare inner arrays a, b, c
		for (index_t i = 1; i < n; i += 2)
		{
			const auto a_tmp = ac[s];
			const auto a_low_tmp = a_tmp * (i == 1 ? 0 : 1);

			const auto c_tmp = a_tmp;
			const auto c_high_tmp = a_tmp * (i == n - 2 ? 0 : 1);

			const auto b_low_tmp = b1[s] + ((i == 1) ? a_tmp : 0);
			const auto b_tmp = b1[s] + ((i == n - 1) ? a_tmp : 0);
			const auto b_high_tmp = b1[s] + ((i == n - 2) ? a_tmp : 0);

			const real_t alpha = a_tmp / b_low_tmp;
			const real_t beta = (i == n - 1) ? 0 : (c_tmp / b_high_tmp);

			b[i / 2] = b_tmp - (alpha + beta) * c_tmp;

			(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) -=
				alpha * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - 1))
				+ beta * ((i == n - 1) ? 0 : (dens_l | noarr::get_at<'s', 'x'>(densities, s, i + 1)));

			a[i / 2] = -alpha * a_low_tmp;
			c[i / 2] = -beta * c_high_tmp;
		}

		for (index_t step = 0; step < inner_steps; step++)
		{
			index_t stride = 1 << step;
			for (index_t i = 2 * stride - 1; i < inner_n; i += 2 * stride)
			{
				if (i + stride < inner_n)
				{
					real_t alpha = a[i] / b[i - stride];
					real_t beta = c[i] / b[i + stride];

					b[i] -= alpha * c[i - stride] + beta * a[i + stride];

					(inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) -=
						alpha * (inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride))
						+ beta * (inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride));

					a[i] = -alpha * a[i - stride];
					c[i] = -beta * c[i + stride];
				}
				else
				{
					real_t alpha = a[i] / b[i - stride];

					b[i] -= alpha * c[i - stride];

					(inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) -=
						alpha * (inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride));

					a[i] = -alpha * a[i - stride];
					c[i] = 0;
				}
			}
		}

		// the first solved unknown
		{
			index_t i = (1 << inner_steps) - 1;
			(inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
				(inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) / b[i];
		}

		for (index_t step = inner_steps - 1; step >= 0; step--)
		{
			index_t stride = 1 << step;

			index_t i = stride - 1;
			// the first unknown of each step does not have (i - stride) dependency
			{
				(inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
					((inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
					 - c[i] * (inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride)))
					/ b[i];
			}

			i += 2 * stride;

			for (; i < inner_n; i += 2 * stride)
			{
				if (i + stride < inner_n)
				{
					(inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
						((inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
						 - a[i] * (inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride))
						 - c[i] * (inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride)))
						/ b[i];
				}
				else
				{
					(inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
						((inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
						 - a[i] * (inner_dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride)))
						/ b[i];
				}
			}
		}

		{
			// the first unknown of each step does not have (i - stride) dependency
			{
				(dens_l | noarr::get_at<'s', 'x'>(densities, s, 0)) =
					((dens_l | noarr::get_at<'s', 'x'>(densities, s, 0))
					 - ac[s] * (dens_l | noarr::get_at<'s', 'x'>(densities, s, 1)))
					/ (b1[s] + ac[s]);
			}

			for (index_t i = 2; i < n; i += 2)
			{
				if (i + 1 < n)
				{
					(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
						((dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
						 - ac[s] * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - 1))
						 - ac[s] * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i + 1)))
						/ b1[s];
				}
				else
				{
					(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
						((dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
						 - a[i] * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - 1)))
						/ (b1[s] + ac[s]);
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
									const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
									real_t* __restrict__ c, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_l = dens_l ^ noarr::step<'x'>(1, 2);

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t yz = 0; yz < m; yz++)
		{
			// prepare inner arrays a, b, c
			for (index_t i = 1; i < n; i += 2)
			{
				const auto a_tmp = ac[s];
				const auto a_low_tmp = a_tmp * (i == 1 ? 0 : 1);

				const auto c_tmp = a_tmp;
				const auto c_high_tmp = a_tmp * (i == n - 2 ? 0 : 1);

				const auto b_low_tmp = b1[s] + ((i == 1) ? a_tmp : 0);
				const auto b_tmp = b1[s] + ((i == n - 1) ? a_tmp : 0);
				const auto b_high_tmp = b1[s] + ((i == n - 2) ? a_tmp : 0);

				const real_t alpha = a_tmp / b_low_tmp;
				const real_t beta = (i == n - 1) ? 0 : (c_tmp / b_high_tmp);

				b[i / 2] = b_tmp - (alpha + beta) * c_tmp;

				(dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) -=
					alpha * (dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i - 1))
					+ beta * ((i == n - 1) ? 0 : (dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i + 1)));

				a[i / 2] = -alpha * a_low_tmp;
				c[i / 2] = -beta * c_high_tmp;
			}

			for (index_t step = 0; step < inner_steps; step++)
			{
				index_t stride = 1 << step;
				for (index_t i = 2 * stride - 1; i < inner_n; i += 2 * stride)
				{
					if (i + stride < inner_n)
					{
						real_t alpha = a[i] / b[i - stride];
						real_t beta = c[i] / b[i + stride];

						b[i] -= alpha * c[i - stride] + beta * a[i + stride];

						(inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) -=
							alpha * (inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i - stride))
							+ beta * (inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i + stride));

						a[i] = -alpha * a[i - stride];
						c[i] = -beta * c[i + stride];
					}
					else
					{
						real_t alpha = a[i] / b[i - stride];

						b[i] -= alpha * c[i - stride];

						(inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) -=
							alpha * (inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i - stride));

						a[i] = -alpha * a[i - stride];
						c[i] = 0;
					}
				}
			}

			// the first solved unknown
			{
				index_t i = (1 << inner_steps) - 1;
				(inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) =
					(inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) / b[i];
			}

			for (index_t step = inner_steps - 1; step >= 0; step--)
			{
				index_t stride = 1 << step;

				index_t i = stride - 1;
				// the first unknown of each step does not have (i - stride) dependency
				{
					(inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) =
						((inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i))
						 - c[i] * (inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i + stride)))
						/ b[i];
				}

				i += 2 * stride;

				for (; i < inner_n; i += 2 * stride)
				{
					if (i + stride < inner_n)
					{
						(inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) =
							((inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i))
							 - a[i] * (inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i - stride))
							 - c[i] * (inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i + stride)))
							/ b[i];
					}
					else
					{
						(inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) =
							((inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i))
							 - a[i] * (inner_dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i - stride)))
							/ b[i];
					}
				}
			}

			{
				// the first unknown of each step does not have (i - stride) dependency
				{
					(dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, 0)) =
						((dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, 0))
						 - ac[s] * (dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, 1)))
						/ (b1[s] + ac[s]);
				}

				for (index_t i = 2; i < n; i += 2)
				{
					if (i + 1 < n)
					{
						(dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) =
							((dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i))
							 - ac[s] * (dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i - 1))
							 - ac[s] * (dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i + 1)))
							/ b1[s];
					}
					else
					{
						(dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i)) =
							((dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i))
							 - a[i] * (dens_l | noarr::get_at<'s', 'm', 'x'>(densities, s, yz, i - 1)))
							/ (b1[s] + ac[s]);
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_l = dens_l ^ noarr::step<'y'>(1, 2);

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		// prepare inner arrays a, b, c
		for (index_t i = 1; i < n; i += 2)
		{
			const auto a_tmp = ac[s];
			const auto a_low_tmp = a_tmp * (i == 1 ? 0 : 1);

			const auto c_tmp = a_tmp;
			const auto c_high_tmp = a_tmp * (i == n - 2 ? 0 : 1);

			const auto b_low_tmp = b1[s] + ((i == 1) ? a_tmp : 0);
			const auto b_tmp = b1[s] + ((i == n - 1) ? a_tmp : 0);
			const auto b_high_tmp = b1[s] + ((i == n - 2) ? a_tmp : 0);

			const real_t alpha = a_tmp / b_low_tmp;
			const real_t beta = (i == n - 1) ? 0 : (c_tmp / b_high_tmp);

			b[i / 2] = b_tmp - (alpha + beta) * c_tmp;

			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) -=
					alpha * (dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i - 1))
					+ beta * ((i == n - 1) ? 0 : (dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i + 1)));
			}

			a[i / 2] = -alpha * a_low_tmp;
			c[i / 2] = -beta * c_high_tmp;
		}

		for (index_t step = 0; step < inner_steps; step++)
		{
			index_t stride = 1 << step;
			for (index_t i = 2 * stride - 1; i < inner_n; i += 2 * stride)
			{
				if (i + stride < inner_n)
				{
					real_t alpha = a[i] / b[i - stride];
					real_t beta = c[i] / b[i + stride];

					b[i] -= alpha * c[i - stride] + beta * a[i + stride];

					for (index_t x = 0; x < x_len; x++)
					{
						(inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) -=
							alpha * (inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i - stride))
							+ beta * (inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i + stride));
					}

					a[i] = -alpha * a[i - stride];
					c[i] = -beta * c[i + stride];
				}
				else
				{
					real_t alpha = a[i] / b[i - stride];

					b[i] -= alpha * c[i - stride];

					for (index_t x = 0; x < x_len; x++)
					{
						(inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) -=
							alpha * (inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i - stride));
					}

					a[i] = -alpha * a[i - stride];
					c[i] = 0;
				}
			}
		}

		// the first solved unknown
		for (index_t x = 0; x < x_len; x++)
		{
			index_t i = (1 << inner_steps) - 1;
			(inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) =
				(inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) / b[i];
		}

		for (index_t step = inner_steps - 1; step >= 0; step--)
		{
			index_t stride = 1 << step;

			index_t i = stride - 1;
			// the first unknown of each step does not have (i - stride) dependency
			for (index_t x = 0; x < x_len; x++)
			{
				(inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) =
					((inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i))
					 - c[i] * (inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i + stride)))
					/ b[i];
			}

			i += 2 * stride;

			for (; i < inner_n; i += 2 * stride)
			{
				if (i + stride < inner_n)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) =
							((inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i))
							 - a[i] * (inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i - stride))
							 - c[i] * (inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i + stride)))
							/ b[i];
					}
				}
				else
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) =
							((inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i))
							 - a[i] * (inner_dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i - stride)))
							/ b[i];
					}
				}
			}
		}

		{
			// the first unknown of each step does not have (i - stride) dependency

			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, 0)) =
					((dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, 0))
					 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, 1)))
					/ (b1[s] + ac[s]);
			}

			for (index_t i = 2; i < n; i += 2)
			{
				if (i + 1 < n)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) =
							((dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i))
							 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i - 1))
							 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i + 1)))
							/ b1[s];
					}
				}
				else
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i)) =
							((dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i))
							 - a[i] * (dens_l | noarr::get_at<'s', 'x', 'y'>(densities, s, x, i - 1)))
							/ (b1[s] + ac[s]);
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_l = dens_l ^ noarr::step<'y'>(1, 2);

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			// prepare inner arrays a, b, c
			for (index_t i = 1; i < n; i += 2)
			{
				const auto a_tmp = ac[s];
				const auto a_low_tmp = a_tmp * (i == 1 ? 0 : 1);

				const auto c_tmp = a_tmp;
				const auto c_high_tmp = a_tmp * (i == n - 2 ? 0 : 1);

				const auto b_low_tmp = b1[s] + ((i == 1) ? a_tmp : 0);
				const auto b_tmp = b1[s] + ((i == n - 1) ? a_tmp : 0);
				const auto b_high_tmp = b1[s] + ((i == n - 2) ? a_tmp : 0);

				const real_t alpha = a_tmp / b_low_tmp;
				const real_t beta = (i == n - 1) ? 0 : (c_tmp / b_high_tmp);

				b[i / 2] = b_tmp - (alpha + beta) * c_tmp;

				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) -=
						alpha * (dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i - 1))
						+ beta
							  * ((i == n - 1)
									 ? 0
									 : (dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i + 1)));
				}

				a[i / 2] = -alpha * a_low_tmp;
				c[i / 2] = -beta * c_high_tmp;
			}

			for (index_t step = 0; step < inner_steps; step++)
			{
				index_t stride = 1 << step;
				for (index_t i = 2 * stride - 1; i < inner_n; i += 2 * stride)
				{
					if (i + stride < inner_n)
					{
						real_t alpha = a[i] / b[i - stride];
						real_t beta = c[i] / b[i + stride];

						b[i] -= alpha * c[i - stride] + beta * a[i + stride];

						for (index_t x = 0; x < x_len; x++)
						{
							(inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) -=
								alpha
									* (inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i - stride))
								+ beta
									  * (inner_dens_l
										 | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i + stride));
						}

						a[i] = -alpha * a[i - stride];
						c[i] = -beta * c[i + stride];
					}
					else
					{
						real_t alpha = a[i] / b[i - stride];

						b[i] -= alpha * c[i - stride];

						for (index_t x = 0; x < x_len; x++)
						{
							(inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) -=
								alpha
								* (inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i - stride));
						}

						a[i] = -alpha * a[i - stride];
						c[i] = 0;
					}
				}
			}

			// the first solved unknown
			for (index_t x = 0; x < x_len; x++)
			{
				index_t i = (1 << inner_steps) - 1;
				(inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) =
					(inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) / b[i];
			}

			for (index_t step = inner_steps - 1; step >= 0; step--)
			{
				index_t stride = 1 << step;

				index_t i = stride - 1;
				// the first unknown of each step does not have (i - stride) dependency
				for (index_t x = 0; x < x_len; x++)
				{
					(inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) =
						((inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i))
						 - c[i] * (inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i + stride)))
						/ b[i];
				}

				i += 2 * stride;

				for (; i < inner_n; i += 2 * stride)
				{
					if (i + stride < inner_n)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) =
								((inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i))
								 - a[i]
									   * (inner_dens_l
										  | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i - stride))
								 - c[i]
									   * (inner_dens_l
										  | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i + stride)))
								/ b[i];
						}
					}
					else
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) =
								((inner_dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i))
								 - a[i]
									   * (inner_dens_l
										  | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i - stride)))
								/ b[i];
						}
					}
				}
			}

			{
				// the first unknown of each step does not have (i - stride) dependency

				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, 0)) =
						((dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, 0))
						 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, 1)))
						/ (b1[s] + ac[s]);
				}

				for (index_t i = 2; i < n; i += 2)
				{
					if (i + 1 < n)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) =
								((dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i))
								 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i - 1))
								 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i + 1)))
								/ b1[s];
						}
					}
					else
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i)) =
								((dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i))
								 - a[i] * (dens_l | noarr::get_at<'s', 'x', 'z', 'y'>(densities, s, x, z, i - 1)))
								/ (b1[s] + ac[s]);
						}
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_l = dens_l ^ noarr::step<'z'>(1, 2);

	for (index_t s = 0; s < substrates_count; s++)
	{
		// prepare inner arrays a, b, c
		for (index_t i = 1; i < n; i += 2)
		{
			const auto a_tmp = ac[s];
			const auto a_low_tmp = a_tmp * (i == 1 ? 0 : 1);

			const auto c_tmp = a_tmp;
			const auto c_high_tmp = a_tmp * (i == n - 2 ? 0 : 1);

			const auto b_low_tmp = b1[s] + ((i == 1) ? a_tmp : 0);
			const auto b_tmp = b1[s] + ((i == n - 1) ? a_tmp : 0);
			const auto b_high_tmp = b1[s] + ((i == n - 2) ? a_tmp : 0);

			const real_t alpha = a_tmp / b_low_tmp;
			const real_t beta = (i == n - 1) ? 0 : (c_tmp / b_high_tmp);

			b[i / 2] = b_tmp - (alpha + beta) * c_tmp;

#pragma omp for schedule(static) nowait
			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) -=
						alpha * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i - 1))
						+ beta
							  * ((i == n - 1)
									 ? 0
									 : (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i + 1)));
				}
			}

			a[i / 2] = -alpha * a_low_tmp;
			c[i / 2] = -beta * c_high_tmp;
		}

		for (index_t step = 0; step < inner_steps; step++)
		{
			index_t stride = 1 << step;
			for (index_t i = 2 * stride - 1; i < inner_n; i += 2 * stride)
			{
				if (i + stride < inner_n)
				{
					real_t alpha = a[i] / b[i - stride];
					real_t beta = c[i] / b[i + stride];

					b[i] -= alpha * c[i - stride] + beta * a[i + stride];

#pragma omp for schedule(static) nowait
					for (index_t y = 0; y < y_len; y++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) -=
								alpha
									* (inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i - stride))
								+ beta
									  * (inner_dens_l
										 | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i + stride));
						}
					}

					a[i] = -alpha * a[i - stride];
					c[i] = -beta * c[i + stride];
				}
				else
				{
					real_t alpha = a[i] / b[i - stride];

					b[i] -= alpha * c[i - stride];

#pragma omp for schedule(static) nowait
					for (index_t y = 0; y < y_len; y++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) -=
								alpha
								* (inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i - stride));
						}
					}

					a[i] = -alpha * a[i - stride];
					c[i] = 0;
				}
			}
		}

		// the first solved unknown
#pragma omp for schedule(static) nowait
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				index_t i = (1 << inner_steps) - 1;
				(inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) =
					(inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) / b[i];
			}
		}

		for (index_t step = inner_steps - 1; step >= 0; step--)
		{
			index_t stride = 1 << step;

			index_t i = stride - 1;
			// the first unknown of each step does not have (i - stride) dependency
#pragma omp for schedule(static) nowait
			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) =
						((inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i))
						 - c[i] * (inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i + stride)))
						/ b[i];
				}
			}

			i += 2 * stride;

			for (; i < inner_n; i += 2 * stride)
			{
				if (i + stride < inner_n)
				{
#pragma omp for schedule(static) nowait
					for (index_t y = 0; y < y_len; y++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) =
								((inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i))
								 - a[i]
									   * (inner_dens_l
										  | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i - stride))
								 - c[i]
									   * (inner_dens_l
										  | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i + stride)))
								/ b[i];
						}
					}
				}
				else
				{
#pragma omp for schedule(static) nowait
					for (index_t y = 0; y < y_len; y++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) =
								((inner_dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i))
								 - a[i]
									   * (inner_dens_l
										  | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i - stride)))
								/ b[i];
						}
					}
				}
			}
		}

		{
			// the first unknown of each step does not have (i - stride) dependency

#pragma omp for schedule(static) nowait
			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, 0)) =
						((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, 0))
						 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, 1)))
						/ (b1[s] + ac[s]);
				}
			}
			for (index_t i = 2; i < n; i += 2)
			{
				if (i + 1 < n)
				{
#pragma omp for schedule(static) nowait
					for (index_t y = 0; y < y_len; y++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) =
								((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i))
								 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i - 1))
								 - ac[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i + 1)))
								/ b1[s];
						}
					}
				}
				else
				{
#pragma omp for schedule(static) nowait
					for (index_t y = 0; y < y_len; y++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i)) =
								((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i))
								 - a[i] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(densities, s, x, y, i - 1)))
								/ (b1[s] + ac[s]);
						}
					}
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::solve_x()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
								  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
								  get_substrates_layout<1>());
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
										 b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
										 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>());
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
										 b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
										 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>());
	}
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, a_scratch_[get_thread_num()],
								  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
								  get_substrates_layout<2>());
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, a_scratch_[get_thread_num()],
								  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
								  get_substrates_layout<3>());
	}
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, a_scratch_[get_thread_num()], b_scratch_[get_thread_num()],
							  c_scratch_[get_thread_num()], get_substrates_layout<3>());
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
								  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
								  get_substrates_layout<1>());
	}
	// 	if (this->problem_.dims == 2)
	// 	{
	// #pragma omp parallel
	// 		{
	// 			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
	// 											 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
	// 											 get_diagonal_layout(this->problem_, this->problem_.nx));
	// #pragma omp barrier
	// 			solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<2>(),
	// 									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	// 		}
	// 	}
	// 	if (this->problem_.dims == 3)
	// 	{
	// #pragma omp parallel
	// 		{
	// 			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
	// 											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y',
	// 'm'>(), 											 get_diagonal_layout(this->problem_, this->problem_.nx));
	// #pragma omp barrier 			solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, cy_,
	// get_substrates_layout<3>(), 									  get_diagonal_layout(this->problem_,
	// this->problem_.ny), x_tile_size_); #pragma omp barrier 			solve_slice_z_3d<index_t>(this->substrates_,
	// az_, b1z_, cz_, get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nz),
	// x_tile_size_);
	// 		}
	// 	}
}

template <typename real_t, bool aligned_x>
cyclic_reduction_solver<real_t, aligned_x>::cyclic_reduction_solver()
	: ax_(nullptr), b1x_(nullptr), ay_(nullptr), b1y_(nullptr), az_(nullptr), b1z_(nullptr)
{}

template <typename real_t, bool aligned_x>
cyclic_reduction_solver<real_t, aligned_x>::~cyclic_reduction_solver()
{
	if (b1x_)
	{
		std::free(ax_);
		std::free(b1x_);
	}
	if (b1y_)
	{
		std::free(ay_);
		std::free(b1y_);
	}
	if (b1z_)
	{
		std::free(az_);
		std::free(b1z_);
	}

	for (std::size_t i = 0; i < b_scratch_.size(); i++)
	{
		std::free(a_scratch_[i]);
		std::free(b_scratch_[i]);
		std::free(c_scratch_[i]);
	}
}

template class cyclic_reduction_solver<float, false>;
template class cyclic_reduction_solver<double, false>;

template class cyclic_reduction_solver<float, true>;
template class cyclic_reduction_solver<double, true>;
