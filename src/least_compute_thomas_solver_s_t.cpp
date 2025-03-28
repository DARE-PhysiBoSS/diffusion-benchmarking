#include "least_compute_thomas_solver_s_t.h"

#include <cstddef>
#include <immintrin.h>
#include <type_traits>

#include <hwy/highway.h>


namespace hn = hwy::HWY_NAMESPACE;

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::precompute_values(real_t*& b, real_t*& c, real_t*& e,
																		   index_t shape, index_t dims, index_t n,
																		   bool prepended_e)
{
	auto layout = get_diagonal_layout(this->problem_, n);


	if (aligned_x)
	{
		b = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
		e = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
	}
	else
	{
		b = (real_t*)std::malloc((layout | noarr::get_size()));
		e = (real_t*)std::malloc((layout | noarr::get_size()));
	}

	c = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));

	auto b_diag = noarr::make_bag(layout, b);
	auto e_diag = noarr::make_bag(layout, e);

	// compute c_i
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		c[s] = -1 * -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
				b_diag.template at<'i', 's'>(i, s) =
					1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
					+ this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
				b_diag.template at<'i', 's'>(i, s) =
					1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
					+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);
	}

	// compute b_i'
	{
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			b_diag.template at<'i', 's'>(0, s) = 1 / b_diag.template at<'i', 's'>(0, s);

		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				b_diag.template at<'i', 's'>(i, s) =
					1 / (b_diag.template at<'i', 's'>(i, s) - c[s] * c[s] * b_diag.template at<'i', 's'>(i - 1, s));

				e_diag.template at<'i', 's'>(i - 1, s) = c[s] * b_diag.template at<'i', 's'>(i - 1, s);
			}
	}

	// compute e_i
	if (prepended_e)
	{
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			e_diag.template at<'i', 's'>(0, s) = 0;

		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				e_diag.template at<'i', 's'>(i, s) = c[s] * b_diag.template at<'i', 's'>(i - 1, s);
			}
	}
	else
	{
		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				e_diag.template at<'i', 's'>(i - 1, s) = c[s] * b_diag.template at<'i', 's'>(i - 1, s);
			}
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void least_compute_thomas_solver_s_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(bx_, cx_, ex_, this->problem_.dx, this->problem_.dims, this->problem_.nx,
						  vectorized_x_ == true);
	if (this->problem_.dims >= 2)
		precompute_values(by_, cy_, ey_, this->problem_.dy, this->problem_.dims, this->problem_.ny, false);
	if (this->problem_.dims >= 3)
		precompute_values(bz_, cz_, ez_, this->problem_.dz, this->problem_.dims, this->problem_.nz, false);
}

template <typename real_t, bool aligned_x>
auto least_compute_thomas_solver_s_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
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

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t i = 1; i < n; i++)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) =
				(dens_l | noarr::get_at<'x', 's'>(densities, i, s))
				+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
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
				 + c[s] * (dens_l | noarr::get_at<'x', 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}
	}
}

// 16xfloat
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 16, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t rows[16])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::ConcatLowerLower(d, rows[8], rows[0]);
	auto t1 = hn::ConcatLowerLower(d, rows[9], rows[1]);
	auto t2 = hn::ConcatLowerLower(d, rows[10], rows[2]);
	auto t3 = hn::ConcatLowerLower(d, rows[11], rows[3]);
	auto t4 = hn::ConcatLowerLower(d, rows[12], rows[4]);
	auto t5 = hn::ConcatLowerLower(d, rows[13], rows[5]);
	auto t6 = hn::ConcatLowerLower(d, rows[14], rows[6]);
	auto t7 = hn::ConcatLowerLower(d, rows[15], rows[7]);
	auto t8 = hn::ConcatUpperUpper(d, rows[8], rows[0]);
	auto t9 = hn::ConcatUpperUpper(d, rows[9], rows[1]);
	auto t10 = hn::ConcatUpperUpper(d, rows[10], rows[2]);
	auto t11 = hn::ConcatUpperUpper(d, rows[11], rows[3]);
	auto t12 = hn::ConcatUpperUpper(d, rows[12], rows[4]);
	auto t13 = hn::ConcatUpperUpper(d, rows[13], rows[5]);
	auto t14 = hn::ConcatUpperUpper(d, rows[14], rows[6]);
	auto t15 = hn::ConcatUpperUpper(d, rows[15], rows[7]);

	auto tt0 = hn::InterleaveEvenBlocks(d, t0, t4);
	auto tt1 = hn::InterleaveEvenBlocks(d, t1, t5);
	auto tt2 = hn::InterleaveEvenBlocks(d, t2, t6);
	auto tt3 = hn::InterleaveEvenBlocks(d, t3, t7);
	auto tt4 = hn::InterleaveOddBlocks(d, t0, t4);
	auto tt5 = hn::InterleaveOddBlocks(d, t1, t5);
	auto tt6 = hn::InterleaveOddBlocks(d, t2, t6);
	auto tt7 = hn::InterleaveOddBlocks(d, t3, t7);
	auto tt8 = hn::InterleaveEvenBlocks(d, t8, t12);
	auto tt9 = hn::InterleaveEvenBlocks(d, t9, t13);
	auto tt10 = hn::InterleaveEvenBlocks(d, t10, t14);
	auto tt11 = hn::InterleaveEvenBlocks(d, t11, t15);
	auto tt12 = hn::InterleaveOddBlocks(d, t8, t12);
	auto tt13 = hn::InterleaveOddBlocks(d, t9, t13);
	auto tt14 = hn::InterleaveOddBlocks(d, t10, t14);
	auto tt15 = hn::InterleaveOddBlocks(d, t11, t15);

	auto u0 = hn::InterleaveLower(d, tt0, tt2);
	auto u1 = hn::InterleaveLower(d, tt1, tt3);
	auto u2 = hn::InterleaveUpper(d, tt0, tt2);
	auto u3 = hn::InterleaveUpper(d, tt1, tt3);
	auto u4 = hn::InterleaveLower(d, tt4, tt6);
	auto u5 = hn::InterleaveLower(d, tt5, tt7);
	auto u6 = hn::InterleaveUpper(d, tt4, tt6);
	auto u7 = hn::InterleaveUpper(d, tt5, tt7);
	auto u8 = hn::InterleaveLower(d, tt8, tt10);
	auto u9 = hn::InterleaveLower(d, tt9, tt11);
	auto u10 = hn::InterleaveUpper(d, tt8, tt10);
	auto u11 = hn::InterleaveUpper(d, tt9, tt11);
	auto u12 = hn::InterleaveLower(d, tt12, tt14);
	auto u13 = hn::InterleaveLower(d, tt13, tt15);
	auto u14 = hn::InterleaveUpper(d, tt12, tt14);
	auto u15 = hn::InterleaveUpper(d, tt13, tt15);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
	rows[4] = hn::InterleaveLower(d, u4, u5);
	rows[5] = hn::InterleaveUpper(d, u4, u5);
	rows[6] = hn::InterleaveLower(d, u6, u7);
	rows[7] = hn::InterleaveUpper(d, u6, u7);
	rows[8] = hn::InterleaveLower(d, u8, u9);
	rows[9] = hn::InterleaveUpper(d, u8, u9);
	rows[10] = hn::InterleaveLower(d, u10, u11);
	rows[11] = hn::InterleaveUpper(d, u10, u11);
	rows[12] = hn::InterleaveLower(d, u12, u13);
	rows[13] = hn::InterleaveUpper(d, u12, u13);
	rows[14] = hn::InterleaveLower(d, u14, u15);
	rows[15] = hn::InterleaveUpper(d, u14, u15);
}

// 8xdouble
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 8, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, double>, bool> = true>
HWY_INLINE void transpose(vec_t rows[8])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::ConcatLowerLower(d, rows[4], rows[0]);
	auto t1 = hn::ConcatLowerLower(d, rows[5], rows[1]);
	auto t2 = hn::ConcatLowerLower(d, rows[6], rows[2]);
	auto t3 = hn::ConcatLowerLower(d, rows[7], rows[3]);
	auto t4 = hn::ConcatUpperUpper(d, rows[4], rows[0]);
	auto t5 = hn::ConcatUpperUpper(d, rows[5], rows[1]);
	auto t6 = hn::ConcatUpperUpper(d, rows[6], rows[2]);
	auto t7 = hn::ConcatUpperUpper(d, rows[7], rows[3]);

	auto u0 = hn::InterleaveEvenBlocks(d, t0, t2);
	auto u1 = hn::InterleaveEvenBlocks(d, t1, t3);
	auto u2 = hn::InterleaveOddBlocks(d, t0, t2);
	auto u3 = hn::InterleaveOddBlocks(d, t1, t3);
	auto u4 = hn::InterleaveEvenBlocks(d, t4, t6);
	auto u5 = hn::InterleaveEvenBlocks(d, t5, t7);
	auto u6 = hn::InterleaveOddBlocks(d, t4, t6);
	auto u7 = hn::InterleaveOddBlocks(d, t5, t7);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
	rows[4] = hn::InterleaveLower(d, u4, u5);
	rows[5] = hn::InterleaveUpper(d, u4, u5);
	rows[6] = hn::InterleaveLower(d, u6, u7);
	rows[7] = hn::InterleaveUpper(d, u6, u7);
}

// 8xfloat
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 8, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t rows[8])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveEvenBlocks(d, rows[0], rows[4]);
	auto t1 = hn::InterleaveEvenBlocks(d, rows[1], rows[5]);
	auto t2 = hn::InterleaveEvenBlocks(d, rows[2], rows[6]);
	auto t3 = hn::InterleaveEvenBlocks(d, rows[3], rows[7]);
	auto t4 = hn::InterleaveOddBlocks(d, rows[0], rows[4]);
	auto t5 = hn::InterleaveOddBlocks(d, rows[1], rows[5]);
	auto t6 = hn::InterleaveOddBlocks(d, rows[2], rows[6]);
	auto t7 = hn::InterleaveOddBlocks(d, rows[3], rows[7]);

	auto u0 = hn::InterleaveLower(d, t0, t2);
	auto u1 = hn::InterleaveLower(d, t1, t3);
	auto u2 = hn::InterleaveUpper(d, t0, t2);
	auto u3 = hn::InterleaveUpper(d, t1, t3);
	auto u4 = hn::InterleaveLower(d, t4, t6);
	auto u5 = hn::InterleaveLower(d, t5, t7);
	auto u6 = hn::InterleaveUpper(d, t4, t6);
	auto u7 = hn::InterleaveUpper(d, t5, t7);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
	rows[4] = hn::InterleaveLower(d, u4, u5);
	rows[5] = hn::InterleaveUpper(d, u4, u5);
	rows[6] = hn::InterleaveLower(d, u6, u7);
	rows[7] = hn::InterleaveUpper(d, u6, u7);
}

// 4xfloat
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 4, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t rows[4])
{
	hn::DFromV<vec_t> d;

	auto u0 = hn::InterleaveLower(d, rows[0], rows[2]);
	auto u1 = hn::InterleaveLower(d, rows[1], rows[3]);
	auto u2 = hn::InterleaveUpper(d, rows[0], rows[2]);
	auto u3 = hn::InterleaveUpper(d, rows[1], rows[3]);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
}

// 4xdouble
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 4, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, double>, bool> = true>
HWY_INLINE void transpose(vec_t rows[4])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveEvenBlocks(d, rows[0], rows[2]);
	auto t1 = hn::InterleaveEvenBlocks(d, rows[1], rows[3]);
	auto t2 = hn::InterleaveOddBlocks(d, rows[0], rows[2]);
	auto t3 = hn::InterleaveOddBlocks(d, rows[1], rows[3]);

	rows[0] = hn::InterleaveLower(d, t0, t1);
	rows[1] = hn::InterleaveUpper(d, t0, t1);
	rows[2] = hn::InterleaveLower(d, t2, t3);
	rows[3] = hn::InterleaveUpper(d, t2, t3);
}

// 2xfloat & 2xdouble
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 2, bool> = true>
HWY_INLINE void transpose(vec_t rows[2])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveLower(d, rows[0], rows[1]);
	auto t1 = hn::InterleaveUpper(d, rows[0], rows[1]);

	rows[0] = t0;
	rows[1] = t1;
}

template <int First, int Last>
struct static_for
{
	template <typename Fn>
	void operator()(Fn const& fn) const
	{
		if (First < Last)
		{
			fn(std::integral_constant<int, First> {});
			static_for<First + 1, Last>()(fn);
		}
	}
};

template <int N>
struct static_for<N, N>
{
	template <typename Fn>
	void operator()(Fn const&) const
	{}
};

template <int First, int Last>
struct static_rfor
{
	template <typename Fn>
	void operator()(Fn const& fn) const
	{
		if (First >= Last)
		{
			fn(std::integral_constant<int, First> {});
			static_rfor<First - 1, Last>()(fn);
		}
	}
};

template <int N>
struct static_rfor<N - 1, N>
{
	template <typename Fn>
	void operator()(Fn const&) const
	{}
};

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ b,
											  const real_t* __restrict__ c, const real_t* __restrict__ e,
											  const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'m', 'b', 'm', 'v'>(noarr::lit<simd_length>);

	for (index_t s = 0; s < substrates_count; s++)
	{
		// vectorized body
		{
			const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t m = body_dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) nowait
			for (index_t yz = 0; yz < m; yz++)
			{
				simd_t prev;

				// forward substitution until last simd_length elements
				for (index_t i = 0; i < full_n - simd_length; i += simd_length)
				{
					// vector registers that hold the to be transposed x*yz plane
					simd_t rows[simd_length];

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v] =
							hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));

					// transposition to enable vectorization
					transpose(rows);

					// actual forward substitution (vectorized)
					{
						auto e_tmp = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(e, i, s)));

						rows[0] = hn::MulAdd(prev, hn::BroadcastLane<0>(e_tmp), rows[0]);
						static_for<1, simd_length>()(
							[&](auto v) { rows[v] = hn::MulAdd(rows[v - 1], hn::BroadcastLane<v>(e_tmp), rows[v]); });

						prev = rows[simd_length - 1];
					}

					// transposition back to the original form
					transpose(rows);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v], d,
								  &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}

				// we are aligned to the vector size, so we can safely continue
				// here we fuse the end of forward substitution and the beginning of backwards propagation
				{
					// vector registers that hold the to be transposed x*yz plane
					simd_t rows[simd_length];

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v] = hn::Load(
							d, &(body_dens_l
								 | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));

					// transposition to enable vectorization
					transpose(rows);

					index_t remainder_work = n % simd_length;
					remainder_work += remainder_work == 0 ? simd_length : 0;

					// the rest of forward part
					{
						auto e_tmp = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(e, full_n - simd_length, s)));

						rows[0] = hn::MulAdd(prev, hn::BroadcastLane<0>(e_tmp), rows[0]);
						for (index_t v = 1; v < remainder_work; v++)
						{
							e_tmp = hn::Slide1Down(d, e_tmp);
							rows[v] = hn::MulAdd(rows[v - 1], hn::BroadcastLane<0>(e_tmp), rows[v]);
						}
					}

					// the begin of backward part
					{
						auto cs = hn::Set(d, c[s]);
						auto b_tmp = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(b, full_n - simd_length, s)));

						rows[simd_length - 1] =
							hn::Mul(rows[simd_length - 1], hn::BroadcastLane<simd_length - 1>(b_tmp));
						for (index_t v = simd_length - 2; v >= 0; v--)
						{
							b_tmp = hn::Slide1Up(d, b_tmp);
							rows[v] = hn::Mul(hn::MulAdd(rows[v + 1], cs, rows[v]),
											  hn::BroadcastLane<simd_length - 1>(b_tmp));
						}

						prev = rows[0];
					}

					// transposition back to the original form
					transpose(rows);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v], d,
								  &(body_dens_l
									| noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));
				}

				// we continue with backwards substitution
				for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
				{
					// vector registers that hold the to be transposed x*yz plane
					simd_t rows[simd_length];

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v] =
							hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));

					// transposition to enable vectorization
					transpose(rows);

					// backward propagation
					{
						auto cs = hn::Set(d, c[s]);
						auto b_tmp = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(b, i, s)));

						rows[simd_length - 1] = hn::Mul(hn::MulAdd(prev, cs, rows[simd_length - 1]),
														hn::BroadcastLane<simd_length - 1>(b_tmp));
						static_rfor<simd_length - 2, 0>()([&](auto v) {
							rows[v] = hn::Mul(hn::MulAdd(rows[v + 1], cs, rows[v]), hn::BroadcastLane<v>(b_tmp));
						});

						prev = rows[0];
					}

					// transposition back to the original form
					transpose(rows);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v], d,
								  &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}
			}
		}

		// yz remainder
		{
			auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t v_len = rem_dens_l | noarr::get_length<'v'>();

#pragma omp for schedule(static) nowait
			for (index_t yz = 0; yz < v_len; yz++)
			{
				for (index_t i = 1; i < n; i++)
				{
					(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i, s)) =
						(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i, s))
						+ (diag_l | noarr::get_at<'i', 's'>(e, i, s))
							  * (rem_dens_l
								 | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i - 1, s));
				}

				{
					(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, n - 1, s)) =
						(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, n - 1, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i, s)) =
						((rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i, s))
						 + c[s]
							   * (rem_dens_l
								  | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i + 1, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ b,
									const real_t* __restrict__ c, const real_t* __restrict__ e,
									const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t yz = 0; yz < m; yz++)
		{
			for (index_t i = 1; i < n; i++)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
					(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
					+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
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
					 + c[s] * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i + 1, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
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
				for (index_t i = 1; i < n; i++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, X, x, s)) =
							(body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i, X, x, s))
							+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
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
							 + c[s] * (body_dens_l | noarr::get_at<'y', 'X', 'x', 's'>(densities, i + 1, X, x, s)))
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
						+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
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
						 + c[s]
							   * (border_dens_l
								  | noarr::get_at<'y', 'X', 'x', 's'>(densities, i + 1, noarr::lit<0>, x, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
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
					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, X, x, s)) =
								(body_dens_l | noarr::get_at<'z', 'y', 'X', 'x', 's'>(densities, z, i, X, x, s))
								+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
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
								 + c[s]
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
							+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
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
							 + c[s]
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
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
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
					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, i, y, x, s)) =
								(body_dens_l | noarr::get_at<'X', 'z', 'y', 'x', 's'>(densities, X, i, y, x, s))
								+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
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
								 + c[s]
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
							+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
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
							 + c[s]
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
	if (vectorized_x_)
	{
		if (this->problem_.dims == 1)
		{
#pragma omp parallel
			solve_slice_x_1d<index_t>(this->substrates_, bx_, cx_, ex_, get_substrates_layout<1>(),
									  get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, bx_, cx_, ex_,
													   get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
													   get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d_transpose<index_t>(
				this->substrates_, bx_, cx_, ex_, get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
				get_diagonal_layout(this->problem_, this->problem_.nx));
		}
	}
	else
	{
		if (this->problem_.dims == 1)
		{
#pragma omp parallel
			solve_slice_x_1d<index_t>(this->substrates_, bx_, cx_, ex_, get_substrates_layout<1>(),
									  get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_, cx_, ex_,
											 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
											 get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_, cx_, ex_,
											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
											 get_diagonal_layout(this->problem_, this->problem_.nx));
		}
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(this->substrates_, by_, cy_, ey_, get_substrates_layout<2>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, by_, cy_, ey_, get_substrates_layout<3>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, bz_, cz_, ez_, get_substrates_layout<3>(),
							  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_);
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, bx_, cx_, ex_, get_substrates_layout<1>(),
								  get_diagonal_layout(this->problem_, this->problem_.nx));
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			if (vectorized_x_)
				solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, bx_, cx_, ex_,
														   get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
														   get_diagonal_layout(this->problem_, this->problem_.nx));
			else

				solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_, cx_, ex_,
												 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
			solve_slice_y_2d<index_t>(this->substrates_, by_, cy_, ey_, get_substrates_layout<2>(),
									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
		}
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			if (vectorized_x_)
				solve_slice_x_2d_and_3d_transpose<index_t>(
					this->substrates_, bx_, cx_, ex_, get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
					get_diagonal_layout(this->problem_, this->problem_.nx));
			else
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_, cx_, ex_,
												 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
			solve_slice_y_3d<index_t>(this->substrates_, by_, cy_, ey_, get_substrates_layout<3>(),
									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
#pragma omp barrier
			solve_slice_z_3d<index_t>(this->substrates_, bz_, cz_, ez_, get_substrates_layout<3>(),
									  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_);
		}
	}
}

template <typename real_t, bool aligned_x>
least_compute_thomas_solver_s_t<real_t, aligned_x>::least_compute_thomas_solver_s_t(bool vectorized_x)
	: bx_(nullptr),
	  cx_(nullptr),
	  ex_(nullptr),
	  by_(nullptr),
	  cy_(nullptr),
	  ey_(nullptr),
	  bz_(nullptr),
	  cz_(nullptr),
	  ez_(nullptr),
	  vectorized_x_(vectorized_x)
{}


template <typename real_t, bool aligned_x>
least_compute_thomas_solver_s_t<real_t, aligned_x>::~least_compute_thomas_solver_s_t()
{
	if (bx_)
	{
		std::free(bx_);
		std::free(cx_);
		std::free(ex_);
	}
	if (by_)
	{
		std::free(by_);
		std::free(cy_);
		std::free(ey_);
	}
	if (bz_)
	{
		std::free(bz_);
		std::free(cz_);
		std::free(ez_);
	}
}

template class least_compute_thomas_solver_s_t<float, false>;
template class least_compute_thomas_solver_s_t<double, false>;

template class least_compute_thomas_solver_s_t<float, true>;
template class least_compute_thomas_solver_s_t<double, true>;
