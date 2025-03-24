#include "least_compute_thomas_solver_s_t.h"

#include <cstddef>
#include <immintrin.h>

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
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 1;
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

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ b,
											  const real_t* __restrict__ c, const real_t* __restrict__ e,
											  const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t yz = 0; yz < m; yz += 8)
		{
			// begin
			__m256 prev;
			{
				__m256 row0 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, 0, s)));
				__m256 row1 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 1, 0, s)));
				__m256 row2 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 2, 0, s)));
				__m256 row3 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 3, 0, s)));
				__m256 row4 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 4, 0, s)));
				__m256 row5 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 5, 0, s)));
				__m256 row6 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 6, 0, s)));
				__m256 row7 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 7, 0, s)));

				__m256 t0 = _mm256_unpacklo_ps(row0, row1); // [a00, a10, a01, a11, a02, a12, a03, a13]
				__m256 t1 = _mm256_unpackhi_ps(row0, row1);
				__m256 t2 = _mm256_unpacklo_ps(row2, row3);
				__m256 t3 = _mm256_unpackhi_ps(row2, row3);
				__m256 t4 = _mm256_unpacklo_ps(row4, row5);
				__m256 t5 = _mm256_unpackhi_ps(row4, row5);
				__m256 t6 = _mm256_unpacklo_ps(row6, row7);
				__m256 t7 = _mm256_unpackhi_ps(row6, row7);

				__m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // [a00, a10, a20, a30, a01, a11, a21, a31]
				__m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
				__m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
				__m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
				__m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
				__m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
				__m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
				__m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

				__m256 tr0 = _mm256_permute2f128_ps(tt0, tt4, 0x20); // [a00, a10, a20, a30, a40, a50, a60, a70]
				__m256 tr1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				__m256 tr2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				__m256 tr3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				__m256 tr4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				__m256 tr5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				__m256 tr6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				__m256 tr7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				row0 = tr0;
				row1 = _mm256_fmadd_ps(row0, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, 0, s))), tr1);
				row2 = _mm256_fmadd_ps(row1, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, 1, s))), tr2);
				row3 = _mm256_fmadd_ps(row2, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, 2, s))), tr3);
				row4 = _mm256_fmadd_ps(row3, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, 3, s))), tr4);
				row5 = _mm256_fmadd_ps(row4, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, 4, s))), tr5);
				row6 = _mm256_fmadd_ps(row5, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, 5, s))), tr6);
				row7 = _mm256_fmadd_ps(row6, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, 6, s))), tr7);

				prev = row7;

				t0 = _mm256_unpacklo_ps(row0, row1); // [a00, a10, a01, a11, a02, a12, a03, a13]
				t1 = _mm256_unpackhi_ps(row0, row1);
				t2 = _mm256_unpacklo_ps(row2, row3);
				t3 = _mm256_unpackhi_ps(row2, row3);
				t4 = _mm256_unpacklo_ps(row4, row5);
				t5 = _mm256_unpackhi_ps(row4, row5);
				t6 = _mm256_unpacklo_ps(row6, row7);
				t7 = _mm256_unpackhi_ps(row6, row7);

				tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // [a00, a10, a20, a30, a01, a11, a21, a31]
				tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
				tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
				tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
				tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
				tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
				tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
				tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

				tr0 = _mm256_permute2f128_ps(tt0, tt4, 0x20); // [a00, a10, a20, a30, a40, a50, a60, a70]
				tr1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				tr2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				tr3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				tr4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				tr5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				tr6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				tr7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, 0, s)), tr0);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 1, 0, s)), tr1);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 2, 0, s)), tr2);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 3, 0, s)), tr3);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 4, 0, s)), tr4);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 5, 0, s)), tr5);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 6, 0, s)), tr6);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 7, 0, s)), tr7);
			}

			for (index_t i = 8; i < n - 8; i += 8)
			{
				__m256 row0 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)));
				__m256 row1 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 1, i, s)));
				__m256 row2 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 2, i, s)));
				__m256 row3 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 3, i, s)));
				__m256 row4 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 4, i, s)));
				__m256 row5 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 5, i, s)));
				__m256 row6 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 6, i, s)));
				__m256 row7 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 7, i, s)));

				__m256 t0 = _mm256_unpacklo_ps(row0, row1); // [a00, a10, a01, a11, a02, a12, a03, a13]
				__m256 t1 = _mm256_unpackhi_ps(row0, row1);
				__m256 t2 = _mm256_unpacklo_ps(row2, row3);
				__m256 t3 = _mm256_unpackhi_ps(row2, row3);
				__m256 t4 = _mm256_unpacklo_ps(row4, row5);
				__m256 t5 = _mm256_unpackhi_ps(row4, row5);
				__m256 t6 = _mm256_unpacklo_ps(row6, row7);
				__m256 t7 = _mm256_unpackhi_ps(row6, row7);

				__m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // [a00, a10, a20, a30, a01, a11, a21, a31]
				__m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
				__m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
				__m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
				__m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
				__m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
				__m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
				__m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

				__m256 tr0 = _mm256_permute2f128_ps(tt0, tt4, 0x20); // [a00, a10, a20, a30, a40, a50, a60, a70]
				__m256 tr1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				__m256 tr2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				__m256 tr3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				__m256 tr4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				__m256 tr5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				__m256 tr6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				__m256 tr7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				row0 = _mm256_fmadd_ps(prev, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))), tr0);
				row1 = _mm256_fmadd_ps(row0, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, i, s))), tr1);
				row2 = _mm256_fmadd_ps(row1, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, i + 1, s))), tr2);
				row3 = _mm256_fmadd_ps(row2, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, i + 2, s))), tr3);
				row4 = _mm256_fmadd_ps(row3, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, i + 3, s))), tr4);
				row5 = _mm256_fmadd_ps(row4, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, i + 4, s))), tr5);
				row6 = _mm256_fmadd_ps(row5, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, i + 5, s))), tr6);
				row7 = _mm256_fmadd_ps(row6, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, i + 6, s))), tr7);

				prev = row7;

				t0 = _mm256_unpacklo_ps(row0, row1); // [a00, a10, a01, a11, a02, a12, a03, a13]
				t1 = _mm256_unpackhi_ps(row0, row1);
				t2 = _mm256_unpacklo_ps(row2, row3);
				t3 = _mm256_unpackhi_ps(row2, row3);
				t4 = _mm256_unpacklo_ps(row4, row5);
				t5 = _mm256_unpackhi_ps(row4, row5);
				t6 = _mm256_unpacklo_ps(row6, row7);
				t7 = _mm256_unpackhi_ps(row6, row7);

				tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // [a00, a10, a20, a30, a01, a11, a21, a31]
				tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
				tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
				tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
				tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
				tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
				tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
				tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

				tr0 = _mm256_permute2f128_ps(tt0, tt4, 0x20); // [a00, a10, a20, a30, a40, a50, a60, a70]
				tr1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				tr2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				tr3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				tr4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				tr5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				tr6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				tr7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)), tr0);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 1, i, s)), tr1);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 2, i, s)), tr2);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 3, i, s)), tr3);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 4, i, s)), tr4);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 5, i, s)), tr5);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 6, i, s)), tr6);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 7, i, s)), tr7);
			}

			// for (index_t i = 1; i < n; i++)
			// {
			// 	(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
			// 		(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
			// 		+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
			// 			  * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i - 1, s));
			// }

			// fuse last
			{
				__m256 row0 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 8, s)));
				__m256 row1 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 1, n - 8, s)));
				__m256 row2 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 2, n - 8, s)));
				__m256 row3 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 3, n - 8, s)));
				__m256 row4 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 4, n - 8, s)));
				__m256 row5 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 5, n - 8, s)));
				__m256 row6 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 6, n - 8, s)));
				__m256 row7 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 7, n - 8, s)));

				__m256 t0 = _mm256_unpacklo_ps(row0, row1); // [a00, a10, a01, a11, a02, a12, a03, a13]
				__m256 t1 = _mm256_unpackhi_ps(row0, row1);
				__m256 t2 = _mm256_unpacklo_ps(row2, row3);
				__m256 t3 = _mm256_unpackhi_ps(row2, row3);
				__m256 t4 = _mm256_unpacklo_ps(row4, row5);
				__m256 t5 = _mm256_unpackhi_ps(row4, row5);
				__m256 t6 = _mm256_unpacklo_ps(row6, row7);
				__m256 t7 = _mm256_unpackhi_ps(row6, row7);

				__m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // [a00, a10, a20, a30, a01, a11, a21, a31]
				__m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
				__m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
				__m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
				__m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
				__m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
				__m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
				__m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

				__m256 tr0 = _mm256_permute2f128_ps(tt0, tt4, 0x20); // [a00, a10, a20, a30, a40, a50, a60, a70]
				__m256 tr1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				__m256 tr2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				__m256 tr3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				__m256 tr4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				__m256 tr5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				__m256 tr6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				__m256 tr7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				row0 = _mm256_fmadd_ps(prev, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, n - 9, s))), tr0);
				row1 = _mm256_fmadd_ps(row0, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, n - 8, s))), tr1);
				row2 = _mm256_fmadd_ps(row1, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, n - 7, s))), tr2);
				row3 = _mm256_fmadd_ps(row2, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, n - 6, s))), tr3);
				row4 = _mm256_fmadd_ps(row3, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, n - 5, s))), tr4);
				row5 = _mm256_fmadd_ps(row4, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, n - 4, s))), tr5);
				row6 = _mm256_fmadd_ps(row5, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, n - 3, s))), tr6);
				row7 = _mm256_fmadd_ps(row6, _mm256_set1_ps((diag_l | noarr::get_at<'i', 's'>(e, n - 2, s))), tr7);

				__m256 cs = _mm256_set1_ps(c[s]);

				row7 = _mm256_mul_ps(row7, _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, n - 1, s)));
				row6 = _mm256_mul_ps(_mm256_fmadd_ps(row7, cs, row6),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, n - 2, s)));
				row5 = _mm256_mul_ps(_mm256_fmadd_ps(row6, cs, row5),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, n - 3, s)));
				row4 = _mm256_mul_ps(_mm256_fmadd_ps(row5, cs, row4),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, n - 4, s)));
				row3 = _mm256_mul_ps(_mm256_fmadd_ps(row4, cs, row3),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, n - 5, s)));
				row2 = _mm256_mul_ps(_mm256_fmadd_ps(row3, cs, row2),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, n - 6, s)));
				row1 = _mm256_mul_ps(_mm256_fmadd_ps(row2, cs, row1),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, n - 7, s)));
				row0 = _mm256_mul_ps(_mm256_fmadd_ps(row1, cs, row0),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, n - 8, s)));

				prev = row0;

				t0 = _mm256_unpacklo_ps(row0, row1); // [a00, a10, a01, a11, a02, a12, a03, a13]
				t1 = _mm256_unpackhi_ps(row0, row1);
				t2 = _mm256_unpacklo_ps(row2, row3);
				t3 = _mm256_unpackhi_ps(row2, row3);
				t4 = _mm256_unpacklo_ps(row4, row5);
				t5 = _mm256_unpackhi_ps(row4, row5);
				t6 = _mm256_unpacklo_ps(row6, row7);
				t7 = _mm256_unpackhi_ps(row6, row7);

				tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // [a00, a10, a20, a30, a01, a11, a21, a31]
				tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
				tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
				tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
				tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
				tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
				tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
				tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

				tr0 = _mm256_permute2f128_ps(tt0, tt4, 0x20); // [a00, a10, a20, a30, a40, a50, a60, a70]
				tr1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				tr2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				tr3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				tr4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				tr5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				tr6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				tr7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 8, s)), tr0);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 1, n - 8, s)), tr1);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 2, n - 8, s)), tr2);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 3, n - 8, s)), tr3);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 4, n - 8, s)), tr4);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 5, n - 8, s)), tr5);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 6, n - 8, s)), tr6);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 7, n - 8, s)), tr7);
			}

			// {
			// 	(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s)) =
			// 		(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s))
			// 		* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			// }

			for (index_t i = n - 16; i >= 0; i -= 8)
			{
				__m256 row0 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)));
				__m256 row1 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 1, i, s)));
				__m256 row2 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 2, i, s)));
				__m256 row3 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 3, i, s)));
				__m256 row4 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 4, i, s)));
				__m256 row5 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 5, i, s)));
				__m256 row6 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 6, i, s)));
				__m256 row7 = _mm256_loadu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 7, i, s)));

				__m256 t0 = _mm256_unpacklo_ps(row0, row1); // [a00, a10, a01, a11, a02, a12, a03, a13]
				__m256 t1 = _mm256_unpackhi_ps(row0, row1);
				__m256 t2 = _mm256_unpacklo_ps(row2, row3);
				__m256 t3 = _mm256_unpackhi_ps(row2, row3);
				__m256 t4 = _mm256_unpacklo_ps(row4, row5);
				__m256 t5 = _mm256_unpackhi_ps(row4, row5);
				__m256 t6 = _mm256_unpacklo_ps(row6, row7);
				__m256 t7 = _mm256_unpackhi_ps(row6, row7);

				__m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // [a00, a10, a20, a30, a01, a11, a21, a31]
				__m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
				__m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
				__m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
				__m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
				__m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
				__m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
				__m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

				__m256 tr0 = _mm256_permute2f128_ps(tt0, tt4, 0x20); // [a00, a10, a20, a30, a40, a50, a60, a70]
				__m256 tr1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				__m256 tr2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				__m256 tr3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				__m256 tr4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				__m256 tr5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				__m256 tr6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				__m256 tr7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				__m256 cs = _mm256_set1_ps(c[s]);

				row7 = _mm256_mul_ps(_mm256_fmadd_ps(prev, cs, tr7),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, i + 7, s)));
				row6 = _mm256_mul_ps(_mm256_fmadd_ps(row7, cs, tr6),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, i + 6, s)));
				row5 = _mm256_mul_ps(_mm256_fmadd_ps(row6, cs, tr5),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, i + 5, s)));
				row4 = _mm256_mul_ps(_mm256_fmadd_ps(row5, cs, tr4),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, i + 4, s)));
				row3 = _mm256_mul_ps(_mm256_fmadd_ps(row4, cs, tr3),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, i + 3, s)));
				row2 = _mm256_mul_ps(_mm256_fmadd_ps(row3, cs, tr2),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, i + 2, s)));
				row1 = _mm256_mul_ps(_mm256_fmadd_ps(row2, cs, tr1),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, i + 1, s)));
				row0 = _mm256_mul_ps(_mm256_fmadd_ps(row1, cs, tr0),
									 _mm256_set1_ps(diag_l | noarr::get_at<'i', 's'>(b, i + 0, s)));

				prev = row0;

				t0 = _mm256_unpacklo_ps(row0, row1); // [a00, a10, a01, a11, a02, a12, a03, a13]
				t1 = _mm256_unpackhi_ps(row0, row1);
				t2 = _mm256_unpacklo_ps(row2, row3);
				t3 = _mm256_unpackhi_ps(row2, row3);
				t4 = _mm256_unpacklo_ps(row4, row5);
				t5 = _mm256_unpackhi_ps(row4, row5);
				t6 = _mm256_unpacklo_ps(row6, row7);
				t7 = _mm256_unpackhi_ps(row6, row7);

				tt0 = _mm256_shuffle_ps(t0, t2, 0x44); // [a00, a10, a20, a30, a01, a11, a21, a31]
				tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
				tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
				tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
				tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
				tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
				tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
				tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

				tr0 = _mm256_permute2f128_ps(tt0, tt4, 0x20); // [a00, a10, a20, a30, a40, a50, a60, a70]
				tr1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				tr2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				tr3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				tr4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				tr5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				tr6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				tr7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)), tr0);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 1, i, s)), tr1);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 2, i, s)), tr2);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 3, i, s)), tr3);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 4, i, s)), tr4);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 5, i, s)), tr5);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 6, i, s)), tr6);
				_mm256_storeu_ps(&(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz + 7, i, s)), tr7);
			}

			// for (index_t i = n - 2; i >= 0; i--)
			// {
			// 	(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
			// 		((dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
			// 		 + c[s] * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i + 1, s)))
			// 		* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			// }
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

template class least_compute_thomas_solver_s_t<float, false>;
template class least_compute_thomas_solver_s_t<double, false>;

template class least_compute_thomas_solver_s_t<float, true>;
template class least_compute_thomas_solver_s_t<double, true>;
