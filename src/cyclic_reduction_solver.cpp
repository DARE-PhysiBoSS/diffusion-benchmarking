#include "cyclic_reduction_solver.h"

#include <cstddef>
#include <cstdlib>

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
	a_scratch_ = (real_t*)std::malloc(max_n * sizeof(real_t));
	b_scratch_ = (real_t*)std::malloc(max_n * sizeof(real_t));
	c_scratch_ = (real_t*)std::malloc(max_n * sizeof(real_t));
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

// template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
// static void solve_slice_x_1d(real_t* __restrict__ densities, real_t* __restrict__ a, real_t* __restrict__ b,
// 							 real_t* __restrict__ c, const density_layout_t dens_l, const diagonal_layout_t diag_l)
// {
// 	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
// 	const index_t n = dens_l | noarr::get_length<'x'>();

// 	index_t log_n = (int)std::log2(n);

// 	for (index_t s = 0; s < substrates_count; s++)
// 	{
// 		for (index_t step = 0; step < log_n; step++)
// 		{
// 			index_t stride = 1 << step;
// 			for (index_t i = 2 * stride - 1; i < n; i += 2 * stride)
// 			{
// 				if (i + stride < n)
// 				{
// 					real_t alpha = (diag_l | noarr::get_at<'s', 'i'>(a, s, i))
// 								   / (diag_l | noarr::get_at<'s', 'i'>(b, s, i - stride));
// 					real_t beta = (diag_l | noarr::get_at<'s', 'i'>(c, s, i))
// 								  / (diag_l | noarr::get_at<'s', 'i'>(b, s, i + stride));

// 					(diag_l | noarr::get_at<'s', 'i'>(b, s, i)) -=
// 						alpha * (diag_l | noarr::get_at<'s', 'i'>(c, s, i - stride))
// 						+ beta * (diag_l | noarr::get_at<'s', 'i'>(a, s, i + stride));

// 					(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) -=
// 						alpha * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride))
// 						+ beta * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride));

// 					(diag_l | noarr::get_at<'s', 'i'>(a, s, i)) =
// 						-alpha * (diag_l | noarr::get_at<'s', 'i'>(a, s, i - stride));
// 					(diag_l | noarr::get_at<'s', 'i'>(c, s, i)) =
// 						-beta * (diag_l | noarr::get_at<'s', 'i'>(c, s, i + stride));

// 					// std::cout << "b[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(b, s, i)) <<
// 					// ", "
// 					// 		  << "a[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(a, s, i)) << ", "
// 					// 		  << "c[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(c, s, i)) << ", "
// 					// 		  << "densities[" << s << "][" << i
// 					// 		  << "] = " << (dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) << std::endl;
// 				}
// 				else
// 				{
// 					real_t alpha = (diag_l | noarr::get_at<'s', 'i'>(a, s, i))
// 								   / (diag_l | noarr::get_at<'s', 'i'>(b, s, i - stride));

// 					(diag_l | noarr::get_at<'s', 'i'>(b, s, i)) -=
// 						alpha * (diag_l | noarr::get_at<'s', 'i'>(c, s, i - stride));

// 					(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) -=
// 						alpha * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride));

// 					(diag_l | noarr::get_at<'s', 'i'>(a, s, i)) =
// 						-alpha * (diag_l | noarr::get_at<'s', 'i'>(a, s, i - stride));


// 					// std::cout << "b[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(b, s, i)) <<
// 					// ", "
// 					// 		  << "a[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(a, s, i)) << ", "
// 					// 		  << "c[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(c, s, i)) << ", "
// 					// 		  << "densities[" << s << "][" << i
// 					// 		  << "] = " << (dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) << std::endl;
// 				}
// 			}
// 		}

// 		// the first solved unknown
// 		{
// 			index_t i = (1 << log_n) - 1;
// 			(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
// 				(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) / (diag_l | noarr::get_at<'s', 'i'>(b, s, i));
// 		}

// 		for (index_t step = log_n - 1; step >= 0; step--)
// 		{
// 			index_t stride = 1 << step;

// 			index_t i = stride - 1;
// 			// the first unknown of each step does not have (i - stride) dependency
// 			{
// 				(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
// 					((dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
// 					 - (diag_l | noarr::get_at<'s', 'i'>(c, s, i))
// 						   * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride])
// 					/ (diag_l | noarr::get_at<'s', 'i'>(b, s, i));
// 			}
// 			i += 2 * stride;

// 			for (; i < n; i += 2 * stride)
// 			{
// 				if (i + stride < n)
// 				{
// 					(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
// 						((dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
// 						 - (diag_l | noarr::get_at<'s', 'i'>(a, s, i))
// 							   * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride]
// 						 - (diag_l | noarr::get_at<'s', 'i'>(c, s, i))
// 							   * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride])
// 						/ (diag_l | noarr::get_at<'s', 'i'>(b, s, i));
// 				}
// 				else
// 				{
// 					(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
// 						((dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
// 						 - (diag_l | noarr::get_at<'s', 'i'>(a, s, i))
// 							   * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride])
// 						/ (diag_l | noarr::get_at<'s', 'i'>(b, s, i));
// 				}
// 			}
// 		}
// 	}
// }


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b,c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_l = dens_l ^ noarr::step<'x'>(1, 2);

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

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, real_t* __restrict__ a, real_t* __restrict__ b,
									real_t* __restrict__ c, const density_layout_t dens_l,
									const diagonal_layout_t diag_l)
{
	// const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	// const index_t n = dens_l | noarr::get_length<'x'>();
	// const index_t m = dens_l | noarr::get_length<'m'>();

	// index_t log_n = (int)std::log2(n);

	// for (index_t s = 0; s < substrates_count; s++)
	// {
	// 	for (index_t step = 0; step < log_n; step++)
	// 	{
	// 		index_t stride = 1 << step;
	// 		for (index_t i = 2 * stride - 1; i < n; i += 2 * stride)
	// 		{
	// 			if (i + stride < n)
	// 			{
	// 				real_t alpha = (diag_l | noarr::get_at<'s', 'i'>(a, s, i))
	// 							   / (diag_l | noarr::get_at<'s', 'i'>(b, s, i - stride];
	// 				real_t beta = (diag_l | noarr::get_at<'s', 'i'>(c, s, i))
	// 							  / (diag_l | noarr::get_at<'s', 'i'>(b, s, i + stride];

	// 				(diag_l | noarr::get_at<'s', 'i'>(b, s, i)) -=
	// 					alpha * (diag_l | noarr::get_at<'s', 'i'>(c, s, i - stride]
	// 					+ beta * (diag_l | noarr::get_at<'s', 'i'>(a, s, i + stride];

	// 				(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) -=
	// 					alpha * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride]
	// 					+ beta * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride];

	// 				(diag_l | noarr::get_at<'s', 'i'>(a, s, i)) =
	// 					-alpha * (diag_l | noarr::get_at<'s', 'i'>(a, s, i - stride];
	// 				(diag_l | noarr::get_at<'s', 'i'>(c, s, i)) =
	// 					-beta * (diag_l | noarr::get_at<'s', 'i'>(c, s, i + stride];

	// 				// std::cout << "b[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(b, s, i)) <<
	// 				// ", "
	// 				// 		  << "a[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(a, s, i)) << ", "
	// 				// 		  << "c[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(c, s, i)) << ", "
	// 				// 		  << "densities[" << s << "][" << i
	// 				// 		  << "] = " << (dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) << std::endl;
	// 			}
	// 			else
	// 			{
	// 				real_t alpha = (diag_l | noarr::get_at<'s', 'i'>(a, s, i))
	// 							   / (diag_l | noarr::get_at<'s', 'i'>(b, s, i - stride];

	// 				(diag_l | noarr::get_at<'s', 'i'>(b, s, i)) -=
	// 					alpha * (diag_l | noarr::get_at<'s', 'i'>(c, s, i - stride];

	// 				(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) -=
	// 					alpha * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride];

	// 				(diag_l | noarr::get_at<'s', 'i'>(a, s, i)) =
	// 					-alpha * (diag_l | noarr::get_at<'s', 'i'>(a, s, i - stride];


	// 				// std::cout << "b[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(b, s, i)) <<
	// 				// ", "
	// 				// 		  << "a[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(a, s, i)) << ", "
	// 				// 		  << "c[" << s << "][" << i << "] = " << (diag_l | noarr::get_at<'s', 'i'>(c, s, i)) << ", "
	// 				// 		  << "densities[" << s << "][" << i
	// 				// 		  << "] = " << (dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) << std::endl;
	// 			}
	// 		}
	// 	}

	// 	// the first solved unknown
	// 	{
	// 		index_t i = (1 << log_n) - 1;
	// 		(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
	// 			(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) / (diag_l | noarr::get_at<'s', 'i'>(b, s, i));
	// 	}

	// 	for (index_t step = log_n - 1; step >= 0; step--)
	// 	{
	// 		index_t stride = 1 << step;

	// 		index_t i = stride - 1;
	// 		// the first unknown of each step does not have (i - stride) dependency
	// 		{
	// 			(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
	// 				((dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
	// 				 - (diag_l | noarr::get_at<'s', 'i'>(c, s, i))
	// 					   * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride])
	// 				/ (diag_l | noarr::get_at<'s', 'i'>(b, s, i));
	// 		}
	// 		i += 2 * stride;

	// 		for (; i < n; i += 2 * stride)
	// 		{
	// 			if (i + stride < n)
	// 			{
	// 				(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
	// 					((dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
	// 					 - (diag_l | noarr::get_at<'s', 'i'>(a, s, i))
	// 						   * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride]
	// 					 - (diag_l | noarr::get_at<'s', 'i'>(c, s, i))
	// 						   * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i + stride])
	// 					/ (diag_l | noarr::get_at<'s', 'i'>(b, s, i));
	// 			}
	// 			else
	// 			{
	// 				(dens_l | noarr::get_at<'s', 'x'>(densities, s, i)) =
	// 					((dens_l | noarr::get_at<'s', 'x'>(densities, s, i))
	// 					 - (diag_l | noarr::get_at<'s', 'i'>(a, s, i))
	// 						   * (dens_l | noarr::get_at<'s', 'x'>(densities, s, i - stride])
	// 					/ (diag_l | noarr::get_at<'s', 'i'>(b, s, i));
	// 			}
	// 		}
	// 	}
	// }
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, std::size_t x_tile_size)
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
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				real_t b_tmp = a_s / (b1_s + a_s);

				for (index_t i = 1; i < n; i++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i, s)) -=
							(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i - 1, s)) * b_tmp;
					}

					b_tmp = a_s / (b1_s - a_s * b_tmp);
				}

				for (index_t x = 0; x < x_len; x++)
				{
					(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, n - 1, s)) =
						(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, n - 1, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i, s)) =
							((body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i, s))
							 - a_s * (body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i + 1, s)))
							* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
					}
				}
			}
		}

		// remainder
		{
			auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t x_len = border_dens_l | noarr::get_length<'x'>();

#pragma omp single
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				real_t b_tmp = a_s / (b1_s + a_s);

				for (index_t i = 1; i < n; i++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i, s)) -=
							(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i - 1, s))
							* b_tmp;
					}

					b_tmp = a_s / (b1_s - a_s * b_tmp);
				}

				for (index_t x = 0; x < x_len; x++)
				{
					(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, n - 1, s)) =
						(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, n - 1, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i, s)) =
							((border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i, s))
							 - a_s
								   * (border_dens_l
									  | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i + 1, s)))
							* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, std::size_t x_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			// body
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t x_len = body_dens_l | noarr::get_length<'x'>();
				const index_t X_len = body_dens_l | noarr::get_length<'X'>();

				for (index_t X = 0; X < X_len; X++)
				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					real_t b_tmp = a_s / (b1_s + a_s);

					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i, s)) -=
								(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i - 1, s))
								* b_tmp;
						}

						b_tmp = a_s / (b1_s - a_s * b_tmp);
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, n - 1, s)) =
							(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, n - 1, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i, s)) =
								((body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i, s))
								 - a_s
									   * (body_dens_l
										  | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i + 1, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}

			// remainder
			{
				auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t x_len = border_dens_l | noarr::get_length<'x'>();

				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					real_t b_tmp = a_s / (b1_s + a_s);

					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(border_dens_l
							 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, i, s)) -=
								(border_dens_l
								 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, i - 1, s))
								* b_tmp;
						}

						b_tmp = a_s / (b1_s - a_s * b_tmp);
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l
						 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, n - 1, s)) =
							(border_dens_l
							 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, n - 1, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(border_dens_l
							 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, i, s)) =
								((border_dens_l
								  | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, i, s))
								 - a_s
									   * (border_dens_l
										  | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x,
																				   i + 1, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, std::size_t x_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			// body
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t x_len = body_dens_l | noarr::get_length<'x'>();
				const index_t X_len = body_dens_l | noarr::get_length<'X'>();

				for (index_t X = 0; X < X_len; X++)
				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					real_t b_tmp = a_s / (b1_s + a_s);

					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i, s)) -=
								(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i - 1, s))
								* b_tmp;
						}

						b_tmp = a_s / (b1_s - a_s * b_tmp);
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, n - 1, s)) =
							(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, n - 1, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i, s)) =
								((body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i, s))
								 - a_s
									   * (body_dens_l
										  | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i + 1, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}

			// remainder
			{
				auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t x_len = border_dens_l | noarr::get_length<'x'>();

				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					real_t b_tmp = a_s / (b1_s + a_s);

					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(border_dens_l
							 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, i, s)) -=
								(border_dens_l
								 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, i - 1, s))
								* b_tmp;
						}

						b_tmp = a_s / (b1_s - a_s * b_tmp);
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l
						 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, n - 1, s)) =
							(border_dens_l
							 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, n - 1, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(border_dens_l
							 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, i, s)) =
								((border_dens_l
								  | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, i, s))
								 - a_s
									   * (border_dens_l
										  | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x,
																				   i + 1, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
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
		// #pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, a_scratch_, b_scratch_, c_scratch_,
								  get_substrates_layout<1>(), get_diagonal_layout(this->problem_, this->problem_.nx));
	}
	// 	else if (this->problem_.dims == 2)
	// 	{
	// #pragma omp parallel
	// 		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
	// 										 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
	// 										 get_diagonal_layout(this->problem_, this->problem_.nx));
	// 	}
	// 	else if (this->problem_.dims == 3)
	// 	{
	// #pragma omp parallel
	// 		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
	// 										 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
	// 										 get_diagonal_layout(this->problem_, this->problem_.nx));
	// 	}
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::solve_y()
{
	// 	if (this->problem_.dims == 2)
	// 	{
	// #pragma omp parallel
	// 		solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<2>(),
	// 								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	// 	}
	// 	else if (this->problem_.dims == 3)
	// 	{
	// #pragma omp parallel
	// 		solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
	// 								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	// }
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::solve_z()
{
	// #pragma omp parallel
	// 	solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
	// 							  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_);
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
		// #pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, a_scratch_, b_scratch_, c_scratch_,
								  get_substrates_layout<1>(), get_diagonal_layout(this->problem_, this->problem_.nx));
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
	// 											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
	// 											 get_diagonal_layout(this->problem_, this->problem_.nx));
	// #pragma omp barrier
	// 			solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
	// 									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	// #pragma omp barrier
	// 			solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
	// 									  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_);
	// 		}
	// 	}
}

template <typename real_t, bool aligned_x>
cyclic_reduction_solver<real_t, aligned_x>::cyclic_reduction_solver()
	: ax_(nullptr),
	  b1x_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  a_scratch_(nullptr),
	  b_scratch_(nullptr),
	  c_scratch_(nullptr)
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

	if (a_scratch_)
	{
		std::free(a_scratch_);
		std::free(b_scratch_);
		std::free(c_scratch_);
	}
}

template class cyclic_reduction_solver<float, false>;
template class cyclic_reduction_solver<double, false>;

template class cyclic_reduction_solver<float, true>;
template class cyclic_reduction_solver<double, true>;
