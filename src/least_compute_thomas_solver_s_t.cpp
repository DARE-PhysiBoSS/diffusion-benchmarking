#include "least_compute_thomas_solver_s_t.h"

#include <cstddef>
#include <immintrin.h>
#include <iostream>

#include <hwy/highway.h>


namespace hn = hwy::HWY_NAMESPACE;

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::precompute_values(std::unique_ptr<real_t[]>& b,
																		   std::unique_ptr<real_t[]>& c,
																		   std::unique_ptr<real_t[]>& e, index_t shape,
																		   index_t dims, index_t n)
{
	b = std::make_unique<real_t[]>(n * this->problem_.substrates_count);
	e = std::make_unique<real_t[]>(n * this->problem_.substrates_count);
	c = std::make_unique<real_t[]>(this->problem_.substrates_count);

	auto layout = get_diagonal_layout(this->problem_, n);

	auto b_diag = noarr::make_bag(layout, b.get());
	auto e_diag = noarr::make_bag(layout, e.get());

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

	// compute b_i' and e_i
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
		precompute_values(bx_, cx_, ex_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(by_, cy_, ey_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(bz_, cz_, ez_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
}

template <typename real_t, bool aligned_x>
auto least_compute_thomas_solver_s_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
																			 index_t n)
{
	return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(n, problem.substrates_count);
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

template <typename SFINAE = hn::Vec<hn::FixedTag<float, 8>>>
void transpose(hn::Vec<hn::FixedTag<float, 8>> rows[8])
{
	hn::FixedTag<float, 8> d;

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

void transpose(hn::Vec<hn::FixedTag<float, 4>> rows[4])
{
	hn::FixedTag<float, 4> d;

	auto u0 = hn::InterleaveLower(d, rows[0], rows[2]);
	auto u1 = hn::InterleaveLower(d, rows[1], rows[3]);
	auto u2 = hn::InterleaveUpper(d, rows[0], rows[2]);
	auto u3 = hn::InterleaveUpper(d, rows[1], rows[3]);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
}

void transpose(hn::Vec<hn::FixedTag<double, 4>> rows[4])
{
	hn::FixedTag<double, 4> d;

	auto t0 = hn::InterleaveEvenBlocks(d, rows[0], rows[2]);
	auto t1 = hn::InterleaveEvenBlocks(d, rows[1], rows[3]);
	auto t2 = hn::InterleaveOddBlocks(d, rows[0], rows[2]);
	auto t3 = hn::InterleaveOddBlocks(d, rows[1], rows[3]);

	rows[0] = hn::InterleaveLower(d, t0, t1);
	rows[1] = hn::InterleaveUpper(d, t0, t1);
	rows[2] = hn::InterleaveLower(d, t2, t3);
	rows[3] = hn::InterleaveUpper(d, t2, t3);
}

template <typename real_t>
void transpose(hn::Vec<hn::FixedTag<real_t, 2>> rows[2])
{
	hn::FixedTag<real_t, 2> d;

	auto t0 = hn::InterleaveLower(d, rows[0], rows[1]);
	auto t1 = hn::InterleaveUpper(d, rows[0], rows[1]);

	rows[0] = t0;
	rows[1] = t1;
}

#define TRANSPOSE_8x8__(row0, row1, row2, row3, row4, row5, row6, row7)                                                \
	do                                                                                                                 \
	{                                                                                                                  \
		__m256 t0 = _mm256_unpacklo_ps(row0, row1);                                                                    \
		__m256 t1 = _mm256_unpackhi_ps(row0, row1);                                                                    \
		__m256 t2 = _mm256_unpacklo_ps(row2, row3);                                                                    \
		__m256 t3 = _mm256_unpackhi_ps(row2, row3);                                                                    \
		__m256 t4 = _mm256_unpacklo_ps(row4, row5);                                                                    \
		__m256 t5 = _mm256_unpackhi_ps(row4, row5);                                                                    \
		__m256 t6 = _mm256_unpacklo_ps(row6, row7);                                                                    \
		__m256 t7 = _mm256_unpackhi_ps(row6, row7);                                                                    \
                                                                                                                       \
		__m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);                                                                  \
		__m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);                                                                  \
		__m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);                                                                  \
		__m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);                                                                  \
		__m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);                                                                  \
		__m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);                                                                  \
		__m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);                                                                  \
		__m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);                                                                  \
                                                                                                                       \
		row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);                                                                 \
		row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);                                                                 \
		row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);                                                                 \
		row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);                                                                 \
		row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);                                                                 \
		row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);                                                                 \
		row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);                                                                 \
		row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);                                                                 \
	} while (0)

#define TRANSPOSE_8x8_(row0, row1, row2, row3, row4, row5, row6, row7)                                                 \
	do                                                                                                                 \
	{                                                                                                                  \
		__m256 t0 = _mm256_permute2f128_ps(row0, row4, 0x20);                                                          \
		__m256 t1 = _mm256_permute2f128_ps(row1, row5, 0x20);                                                          \
		__m256 t2 = _mm256_permute2f128_ps(row2, row6, 0x20);                                                          \
		__m256 t3 = _mm256_permute2f128_ps(row3, row7, 0x20);                                                          \
		__m256 t4 = _mm256_permute2f128_ps(row0, row4, 0x31);                                                          \
		__m256 t5 = _mm256_permute2f128_ps(row1, row5, 0x31);                                                          \
		__m256 t6 = _mm256_permute2f128_ps(row2, row6, 0x31);                                                          \
		__m256 t7 = _mm256_permute2f128_ps(row3, row7, 0x31);                                                          \
                                                                                                                       \
		__m256 tt0 = _mm256_unpacklo_ps(t0, t2);                                                                       \
		__m256 tt1 = _mm256_unpacklo_ps(t1, t3);                                                                       \
		__m256 tt2 = _mm256_unpackhi_ps(t0, t2);                                                                       \
		__m256 tt3 = _mm256_unpackhi_ps(t1, t3);                                                                       \
		__m256 tt4 = _mm256_unpacklo_ps(t4, t6);                                                                       \
		__m256 tt5 = _mm256_unpacklo_ps(t5, t7);                                                                       \
		__m256 tt6 = _mm256_unpackhi_ps(t4, t6);                                                                       \
		__m256 tt7 = _mm256_unpackhi_ps(t5, t7);                                                                       \
                                                                                                                       \
		row0 = _mm256_unpacklo_ps(tt0, tt1);                                                                           \
		row1 = _mm256_unpackhi_ps(tt0, tt1);                                                                           \
		row2 = _mm256_unpacklo_ps(tt2, tt3);                                                                           \
		row3 = _mm256_unpackhi_ps(tt2, tt3);                                                                           \
		row4 = _mm256_unpacklo_ps(tt4, tt5);                                                                           \
		row5 = _mm256_unpackhi_ps(tt4, tt5);                                                                           \
		row6 = _mm256_unpacklo_ps(tt6, tt7);                                                                           \
		row7 = _mm256_unpackhi_ps(tt6, tt7);                                                                           \
	} while (0)

#define TRANSPOSE_8x8(row0, row1, row2, row3, row4, row5, row6, row7)                                                  \
	do                                                                                                                 \
	{                                                                                                                  \
		auto t0 = hn::InterleaveEvenBlocks(d, row0, row4);                                                             \
		auto t1 = hn::InterleaveEvenBlocks(d, row1, row5);                                                             \
		auto t2 = hn::InterleaveEvenBlocks(d, row2, row6);                                                             \
		auto t3 = hn::InterleaveEvenBlocks(d, row3, row7);                                                             \
		auto t4 = hn::InterleaveOddBlocks(d, row0, row4);                                                              \
		auto t5 = hn::InterleaveOddBlocks(d, row1, row5);                                                              \
		auto t6 = hn::InterleaveOddBlocks(d, row2, row6);                                                              \
		auto t7 = hn::InterleaveOddBlocks(d, row3, row7);                                                              \
                                                                                                                       \
		auto u0 = hn::InterleaveLower(d, t0, t2);                                                                      \
		auto u1 = hn::InterleaveLower(d, t1, t3);                                                                      \
		auto u2 = hn::InterleaveUpper(d, t0, t2);                                                                      \
		auto u3 = hn::InterleaveUpper(d, t1, t3);                                                                      \
		auto u4 = hn::InterleaveLower(d, t4, t6);                                                                      \
		auto u5 = hn::InterleaveLower(d, t5, t7);                                                                      \
		auto u6 = hn::InterleaveUpper(d, t4, t6);                                                                      \
		auto u7 = hn::InterleaveUpper(d, t5, t7);                                                                      \
                                                                                                                       \
		row0 = hn::InterleaveLower(d, u0, u1);                                                                         \
		row1 = hn::InterleaveUpper(d, u0, u1);                                                                         \
		row2 = hn::InterleaveLower(d, u2, u3);                                                                         \
		row3 = hn::InterleaveUpper(d, u2, u3);                                                                         \
		row4 = hn::InterleaveLower(d, u4, u5);                                                                         \
		row5 = hn::InterleaveUpper(d, u4, u5);                                                                         \
		row6 = hn::InterleaveLower(d, u6, u7);                                                                         \
		row7 = hn::InterleaveUpper(d, u6, u7);                                                                         \
	} while (0)


#define TRANSPOSE_8x8_____(row0, row1, row2, row3, row4, row5, row6, row7)                                             \
	do                                                                                                                 \
	{                                                                                                                  \
		auto t0 = hn::InterleaveLower(d, row0, row1);                                                                  \
		auto t1 = hn::InterleaveUpper(d, row0, row1);                                                                  \
		auto t2 = hn::InterleaveLower(d, row2, row3);                                                                  \
		auto t3 = hn::InterleaveUpper(d, row2, row3);                                                                  \
		auto t4 = hn::InterleaveLower(d, row4, row5);                                                                  \
		auto t5 = hn::InterleaveUpper(d, row4, row5);                                                                  \
		auto t6 = hn::InterleaveLower(d, row6, row7);                                                                  \
		auto t7 = hn::InterleaveUpper(d, row6, row7);                                                                  \
                                                                                                                       \
		auto u0 = hn::InterleaveLower(d, t0, t2);                                                                      \
		auto u1 = hn::InterleaveLower(d, t1, t3);                                                                      \
		auto u2 = hn::InterleaveUpper(d, t0, t2);                                                                      \
		auto u3 = hn::InterleaveUpper(d, t1, t3);                                                                      \
		auto u4 = hn::InterleaveLower(d, t4, t6);                                                                      \
		auto u5 = hn::InterleaveLower(d, t5, t7);                                                                      \
		auto u6 = hn::InterleaveUpper(d, t4, t6);                                                                      \
		auto u7 = hn::InterleaveUpper(d, t5, t7);                                                                      \
                                                                                                                       \
		row0 = hn::InterleaveEvenBlocks(d, u0, u4);                                                                    \
		row1 = hn::InterleaveEvenBlocks(d, u1, u5);                                                                    \
		row2 = hn::InterleaveEvenBlocks(d, u2, u6);                                                                    \
		row3 = hn::InterleaveEvenBlocks(d, u3, u7);                                                                    \
		row4 = hn::InterleaveOddBlocks(d, u0, u4);                                                                     \
		row5 = hn::InterleaveOddBlocks(d, u1, u5);                                                                     \
		row6 = hn::InterleaveOddBlocks(d, u2, u6);                                                                     \
		row7 = hn::InterleaveOddBlocks(d, u3, u7);                                                                     \
	} while (0)

#define TRANSPOSE_8x8____(row0, row1, row2, row3, row4, row5, row6, row7)                                              \
	do                                                                                                                 \
	{                                                                                                                  \
	} while (0)


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ b,
											  const real_t* __restrict__ c, const real_t* __restrict__ e,
											  const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

	using simd_tag = hn::FixedTag<real_t, 8>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	// 	auto myprint = [](auto row) {
	// 		for (index_t i = 0; i < 8; i++)
	// 			std::cout << hn::ExtractLane(row, i) << " ";
	// 		std::cout << std::endl;
	// 	};
	// #pragma omp single
	// 	{
	// 		auto row0 = hn::Iota(d, 0);
	// 		auto row1 = hn::Iota(d, 8);
	// 		auto row2 = hn::Iota(d, 8 * 2);
	// 		auto row3 = hn::Iota(d, 8 * 3);
	// 		auto row4 = hn::Iota(d, 8 * 4);
	// 		auto row5 = hn::Iota(d, 8 * 5);
	// 		auto row6 = hn::Iota(d, 8 * 6);
	// 		auto row7 = hn::Iota(d, 8 * 7);

	// 		myprint(row0);
	// 		myprint(row1);
	// 		myprint(row2);
	// 		myprint(row3);
	// 		myprint(row4);
	// 		myprint(row5);
	// 		myprint(row6);
	// 		myprint(row7);

	// 		__m256 t0 = _mm256_blend_ps(row0.raw, row1.raw, 0b10101010);
	// 		__m256 t1 = _mm256_blend_ps(row0.raw, row1.raw, 0b01010101);
	// 		__m256 t2 = _mm256_blend_ps(row2.raw, row3.raw, 0b10101010);
	// 		__m256 t3 = _mm256_blend_ps(row2.raw, row3.raw, 0b01010101);
	// 		__m256 t4 = _mm256_blend_ps(row4.raw, row5.raw, 0b10101010);
	// 		__m256 t5 = _mm256_blend_ps(row4.raw, row5.raw, 0b01010101);
	// 		__m256 t6 = _mm256_blend_ps(row6.raw, row7.raw, 0b10101010);
	// 		__m256 t7 = _mm256_blend_ps(row6.raw, row7.raw, 0b01010101);


	// 		std::cout << "Transposed" << std::endl;

	// 		myprint(simd_t { t0 });
	// 		myprint(simd_t { t1 });
	// 		myprint(simd_t { t2 });
	// 		myprint(simd_t { t3 });
	// 		myprint(simd_t { t4 });
	// 		myprint(simd_t { t5 });
	// 		myprint(simd_t { t6 });
	// 		myprint(simd_t { t7 });

	// 		auto u0 = _mm256_shuffle_ps(t0, t2, 0x44);
	// 		auto u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
	// 		auto u2 = _mm256_shuffle_ps(t1, t3, 0x44);
	// 		auto u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
	// 		auto u4 = _mm256_shuffle_ps(t4, t6, 0x44);
	// 		auto u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
	// 		auto u6 = _mm256_shuffle_ps(t5, t7, 0x44);
	// 		auto u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

	// 		std::cout << "Transposed" << std::endl;
	// 		myprint(simd_t { u0 });
	// 		myprint(simd_t { u1 });
	// 		myprint(simd_t { u2 });
	// 		myprint(simd_t { u3 });
	// 		myprint(simd_t { u4 });
	// 		myprint(simd_t { u5 });
	// 		myprint(simd_t { u6 });
	// 		myprint(simd_t { u7 });

	// 		row0.raw = _mm256_permute2f128_ps(u0, u4, 0x20);
	// 		row1.raw = _mm256_permute2f128_ps(u1, u5, 0x20);
	// 		row2.raw = _mm256_permute2f128_ps(u2, u6, 0x20);
	// 		row3.raw = _mm256_permute2f128_ps(u3, u7, 0x20);
	// 		row4.raw = _mm256_permute2f128_ps(u0, u4, 0x31);
	// 		row5.raw = _mm256_permute2f128_ps(u1, u5, 0x31);
	// 		row6.raw = _mm256_permute2f128_ps(u2, u6, 0x31);
	// 		row7.raw = _mm256_permute2f128_ps(u3, u7, 0x31);

	// 		std::cout << "Transposed" << std::endl;


	// 		myprint(row0);
	// 		myprint(row1);
	// 		myprint(row2);
	// 		myprint(row3);
	// 		myprint(row4);
	// 		myprint(row5);
	// 		myprint(row6);
	// 		myprint(row7);
	// 	}

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t yz = 0; yz < m; yz += simd_length)
		{
			// begin
			simd_t prev;
			{
				simd_t rows[simd_length];

				for (index_t r = 0; r < simd_length; r++)
					rows[r] = hn::Load(d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + r, 0, s)));

				TRANSPOSE_8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]);

				auto e_tmp = hn::LoadU(d, &(diag_l | noarr::get_at<'i', 's'>(e, 0, s)));

				rows[1] = hn::MulAdd(rows[0], hn::BroadcastLane<0>(e_tmp), rows[1]);
				rows[2] = hn::MulAdd(rows[1], hn::BroadcastLane<1>(e_tmp), rows[2]);
				rows[3] = hn::MulAdd(rows[2], hn::BroadcastLane<2>(e_tmp), rows[3]);
				rows[4] = hn::MulAdd(rows[3], hn::BroadcastLane<3>(e_tmp), rows[4]);
				rows[5] = hn::MulAdd(rows[4], hn::BroadcastLane<4>(e_tmp), rows[5]);
				rows[6] = hn::MulAdd(rows[5], hn::BroadcastLane<5>(e_tmp), rows[6]);
				rows[7] = hn::MulAdd(rows[6], hn::BroadcastLane<6>(e_tmp), rows[7]);

				prev = rows[7];

				TRANSPOSE_8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]);


				for (index_t r = 0; r < simd_length; r++)
					hn::Store(rows[r], d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + r, 0, s)));
			}

			for (index_t i = simd_length; i < n - simd_length; i += simd_length)
			{
				simd_t rows[simd_length];

				for (index_t r = 0; r < simd_length; r++)
					rows[r] = hn::Load(d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + r, i, s)));

				TRANSPOSE_8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]);

				auto e_tmp = hn::LoadU(d, &(diag_l | noarr::get_at<'i', 's'>(e, i - 1, s)));

				rows[0] = hn::MulAdd(prev, hn::BroadcastLane<0>(e_tmp), rows[0]);
				rows[1] = hn::MulAdd(rows[0], hn::BroadcastLane<0>(e_tmp), rows[1]);
				rows[2] = hn::MulAdd(rows[1], hn::BroadcastLane<1>(e_tmp), rows[2]);
				rows[3] = hn::MulAdd(rows[2], hn::BroadcastLane<2>(e_tmp), rows[3]);
				rows[4] = hn::MulAdd(rows[3], hn::BroadcastLane<3>(e_tmp), rows[4]);
				rows[5] = hn::MulAdd(rows[4], hn::BroadcastLane<4>(e_tmp), rows[5]);
				rows[6] = hn::MulAdd(rows[5], hn::BroadcastLane<5>(e_tmp), rows[6]);
				rows[7] = hn::MulAdd(rows[6], hn::BroadcastLane<6>(e_tmp), rows[7]);

				prev = rows[7];

				TRANSPOSE_8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]);


				for (index_t r = 0; r < simd_length; r++)
					hn::Store(rows[r], d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + r, i, s)));
			}

			// fuse last
			{
				simd_t rows[simd_length];

				for (index_t r = 0; r < simd_length; r++)
					rows[r] = hn::Load(d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + r, n - 8, s)));

				TRANSPOSE_8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]);

				auto e_tmp = hn::LoadU(d, &(diag_l | noarr::get_at<'i', 's'>(e, n - 9, s)));

				rows[0] = hn::MulAdd(prev, hn::BroadcastLane<0>(e_tmp), rows[0]);
				rows[1] = hn::MulAdd(rows[0], hn::BroadcastLane<0>(e_tmp), rows[1]);
				rows[2] = hn::MulAdd(rows[1], hn::BroadcastLane<1>(e_tmp), rows[2]);
				rows[3] = hn::MulAdd(rows[2], hn::BroadcastLane<2>(e_tmp), rows[3]);
				rows[4] = hn::MulAdd(rows[3], hn::BroadcastLane<3>(e_tmp), rows[4]);
				rows[5] = hn::MulAdd(rows[4], hn::BroadcastLane<4>(e_tmp), rows[5]);
				rows[6] = hn::MulAdd(rows[5], hn::BroadcastLane<5>(e_tmp), rows[6]);
				rows[7] = hn::MulAdd(rows[6], hn::BroadcastLane<6>(e_tmp), rows[7]);

				auto cs = hn::Set(d, c[s]);
				auto b_tmp = hn::LoadU(d, &(diag_l | noarr::get_at<'i', 's'>(b, n - 8, s)));

				rows[7] = hn::Mul(rows[7], hn::BroadcastLane<7>(b_tmp));
				rows[6] = hn::Mul(hn::MulAdd(rows[7], cs, rows[6]), hn::BroadcastLane<6>(b_tmp));
				rows[5] = hn::Mul(hn::MulAdd(rows[6], cs, rows[5]), hn::BroadcastLane<5>(b_tmp));
				rows[4] = hn::Mul(hn::MulAdd(rows[5], cs, rows[4]), hn::BroadcastLane<4>(b_tmp));
				rows[3] = hn::Mul(hn::MulAdd(rows[4], cs, rows[3]), hn::BroadcastLane<3>(b_tmp));
				rows[2] = hn::Mul(hn::MulAdd(rows[3], cs, rows[2]), hn::BroadcastLane<2>(b_tmp));
				rows[1] = hn::Mul(hn::MulAdd(rows[2], cs, rows[1]), hn::BroadcastLane<1>(b_tmp));
				rows[0] = hn::Mul(hn::MulAdd(rows[1], cs, rows[0]), hn::BroadcastLane<0>(b_tmp));

				prev = rows[0];

				TRANSPOSE_8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]);

				for (index_t r = 0; r < simd_length; r++)
					hn::Store(rows[r], d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + r, n - 8, s)));
			}

			for (index_t i = n - 16; i >= 0; i -= 8)
			{
				simd_t rows[simd_length];

				for (index_t r = 0; r < simd_length; r++)
					rows[r] = hn::Load(d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + r, i, s)));

				TRANSPOSE_8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]);

				auto cs = hn::Set(d, c[s]);
				auto b_tmp = hn::LoadU(d, &(diag_l | noarr::get_at<'i', 's'>(b, i, s)));

				rows[7] = hn::Mul(hn::MulAdd(prev, cs, rows[7]), hn::BroadcastLane<6>(b_tmp));
				rows[6] = hn::Mul(hn::MulAdd(rows[7], cs, rows[6]), hn::BroadcastLane<6>(b_tmp));
				rows[5] = hn::Mul(hn::MulAdd(rows[6], cs, rows[5]), hn::BroadcastLane<5>(b_tmp));
				rows[4] = hn::Mul(hn::MulAdd(rows[5], cs, rows[4]), hn::BroadcastLane<4>(b_tmp));
				rows[3] = hn::Mul(hn::MulAdd(rows[4], cs, rows[3]), hn::BroadcastLane<3>(b_tmp));
				rows[2] = hn::Mul(hn::MulAdd(rows[3], cs, rows[2]), hn::BroadcastLane<2>(b_tmp));
				rows[1] = hn::Mul(hn::MulAdd(rows[2], cs, rows[1]), hn::BroadcastLane<1>(b_tmp));
				rows[0] = hn::Mul(hn::MulAdd(rows[1], cs, rows[0]), hn::BroadcastLane<0>(b_tmp));

				prev = rows[0];

				TRANSPOSE_8x8(rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]);

				for (index_t r = 0; r < simd_length; r++)
					hn::Store(rows[r], d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + r, i, s)));
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
			solve_slice_x_1d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(), get_substrates_layout<1>(),
									  get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
													   get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
													   get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
													   get_substrates_layout<3>()
														   ^ noarr::merge_blocks<'z', 'y', 'm'>(),
													   get_diagonal_layout(this->problem_, this->problem_.nx));
		}
	}
	else
	{
		if (this->problem_.dims == 1)
		{
#pragma omp parallel
			solve_slice_x_1d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(), get_substrates_layout<1>(),
									  get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
											 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
											 get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
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
		solve_slice_y_2d<index_t>(this->substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<2>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<3>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, bz_.get(), cz_.get(), ez_.get(), get_substrates_layout<3>(),
							  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_);
}

template <typename real_t, bool aligned_x>
void least_compute_thomas_solver_s_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(), get_substrates_layout<1>(),
								  get_diagonal_layout(this->problem_, this->problem_.nx));
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
											 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
											 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
			solve_slice_y_2d<index_t>(this->substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<2>(),
									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
		}
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
											 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
			solve_slice_y_3d<index_t>(this->substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<3>(),
									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
#pragma omp barrier
			solve_slice_z_3d<index_t>(this->substrates_, bz_.get(), cz_.get(), ez_.get(), get_substrates_layout<3>(),
									  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_);
		}
	}
}

template <typename real_t, bool aligned_x>
least_compute_thomas_solver_s_t<real_t, aligned_x>::least_compute_thomas_solver_s_t(bool vectorized_x)
	: vectorized_x_(vectorized_x)
{}

template class least_compute_thomas_solver_s_t<float, false>;
// template class least_compute_thomas_solver_s_t<double, false>;

template class least_compute_thomas_solver_s_t<float, true>;
// template class least_compute_thomas_solver_s_t<double, true>;
