#include "least_memory_thomas_solver_d_t.h"

#include <cstddef>
#include <iostream>

#include "vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, real_t*& c,
																		  index_t shape, index_t dims, index_t n)
{
	auto layout = get_diagonal_layout(this->problem_, n);

	if (aligned_x)
	{
		c = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
	}
	else
	{
		c = (real_t*)std::malloc((layout | noarr::get_size()));
	}

	a = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));
	b1 = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));

	auto c_diag = noarr::make_bag(layout, c);

	// compute a
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		a[s] = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b1
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		b1[s] = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
				+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute c_i'
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
	{
		c_diag.template at<'i', 's'>(0, s) = a[s] / (b1[s] + a[s]);

		for (index_t i = 1; i < n - 1; i++)
		{
			const real_t r = 1 / (b1[s] - a[s] * c_diag.template at<'i', 's'>(i - 1, s));
			c_diag.template at<'i', 's'>(i, s) = a[s] * r;
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void least_memory_thomas_solver_d_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, cx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, cy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, cz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
}

template <typename real_t, bool aligned_x>
auto least_memory_thomas_solver_d_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
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
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];
		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);
		const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

		real_t a_tmp = 0;
		real_t b_tmp = b1_s + a_s;
		real_t c_tmp = a_s;
		real_t prev = 0;

		for (index_t i = 0; i < n; i++)
		{
			const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

			real_t curr = d.template at<'x'>(i);
			curr = r * (curr - a_tmp * prev);
			d.template at<'x'>(i) = curr;

			c_tmp = a_s * r;

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << " c: " << c_tmp
			// 		  << " curr: " << curr << " prev: " << prev << std::endl;

			a_tmp = a_s;
			b_tmp = b1_s + (i == n - 2 ? a_s : 0);
			prev = curr;
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			real_t curr = d.template at<'x'>(i);
			curr = curr - c.template at<'i'>(i) * prev;
			d.template at<'x'>(i) = curr;

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s))
			// 		  << " c: " << c.template at<'i'>(i) << " curr: " << curr << " prev: " << prev << std::endl;

			prev = curr;
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ a,
											  const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
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
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];

		// vectorized body
		{
			const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;


			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t m = body_dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) nowait
			for (index_t yz = 0; yz < m; yz++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				simd_t prev = hn::Zero(d);

				// forward substitution until last simd_length elements
				for (index_t i = 0; i < full_n - simd_length; i += simd_length)
				{
					// vector registers that hold the to be transposed x*yz plane
					simd_t rows[simd_length + 1];

					rows[0] = prev;

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v + 1] =
							hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));

					// transposition to enable vectorization
					transpose(rows + 1);

					// actual forward substitution (vectorized)
					{
						for (index_t v = 1; v < simd_length + 1; v++)
						{
							const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

							rows[v] = hn::Mul(hn::MulAdd(rows[v - 1], hn::Set(d, -a_tmp), rows[v]), hn::Set(d, r));

							a_tmp = a_s;
							b_tmp = b1_s;
							c_tmp = a_s * r;
						};

						prev = rows[simd_length];
					}

					// transposition back to the original form
					transpose(rows + 1);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v + 1], d,
								  &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}

				// we are aligned to the vector size, so we can safely continue
				// here we fuse the end of forward substitution and the beginning of backwards propagation
				{
					// vector registers that hold the to be transposed x*yz plane
					simd_t rows[simd_length + 1];

					rows[0] = prev;

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v + 1] = hn::Load(
							d, &(body_dens_l
								 | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));

					// transposition to enable vectorization
					transpose(rows + 1);

					index_t remainder_work = n % simd_length;
					remainder_work += remainder_work == 0 ? simd_length : 0;

					// the rest of forward part
					{
						for (index_t v = 0; v < remainder_work; v++)
						{
							const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

							rows[v + 1] = hn::Mul(hn::MulAdd(rows[v], hn::Set(d, -a_tmp), rows[v + 1]), hn::Set(d, r));

							a_tmp = a_s;
							b_tmp = b1_s + (v == remainder_work - 2 ? a_s : 0);
							c_tmp = a_s * r;
						}
					}

					// the begin of backward part
					{
						auto c = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(back_c, full_n - simd_length, s)));

						for (index_t v = simd_length - 2; v >= 0; v--)
						{
							c = hn::Slide1Up(d, c);
							rows[v + 1] =
								hn::NegMulAdd(rows[v + 2], hn::BroadcastLane<simd_length - 1>(c), rows[v + 1]);
						}

						prev = rows[1];
					}

					// transposition back to the original form
					transpose(rows + 1);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v + 1], d,
								  &(body_dens_l
									| noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));
				}

				// we continue with backwards substitution
				for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
				{
					// vector registers that hold the to be transposed x*yz plane
					simd_t rows[simd_length + 1];

					rows[simd_length] = prev;

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v] =
							hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));

					// transposition to enable vectorization
					transpose(rows);

					// backward propagation
					{
						auto c = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(back_c, i, s)));

						static_rfor<simd_length - 1, 0>()([&](auto v) {
							rows[v] = hn::NegMulAdd(rows[v + 1], hn::BroadcastLane<v>(c), rows[v]);
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
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				const auto d = noarr::make_bag(rem_dens_l ^ noarr::fix<'s', 'm', 'v'>(s, noarr::lit<0>, yz), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				real_t prev = 0;

				for (index_t i = 0; i < n; i++)
				{
					const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

					real_t curr = d.template at<'x'>(i);
					curr = r * (curr - a_tmp * prev);
					d.template at<'x'>(i) = curr;

					a_tmp = a_s;
					b_tmp = b1_s + (i == n - 2 ? a_s : 0);
					c_tmp = a_s * r;
					prev = curr;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					real_t curr = d.template at<'x'>(i);
					curr = curr - c.template at<'i'>(i) * prev;
					d.template at<'x'>(i) = curr;

					prev = curr;
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
									const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
									const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	constexpr char dim = 'x';
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<dim>();
	const index_t m = dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t yz = 0; yz < m; yz++)
		{
			const real_t a_s = a[s];
			const real_t b1_s = b1[s];
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'm'>(s, yz), densities);
			const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

			real_t a_tmp = 0;
			real_t b_tmp = b1_s + a_s;
			real_t c_tmp = a_s;
			real_t prev = 0;

			for (index_t i = 0; i < n; i++)
			{
				const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

				real_t curr = d.template at<dim>(i);
				curr = r * (curr - a_tmp * prev);
				d.template at<dim>(i) = curr;

				a_tmp = a_s;
				b_tmp = b1_s + (i == n - 2 ? a_s : 0);
				c_tmp = a_s * r;
				prev = curr;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				real_t curr = d.template at<dim>(i);
				curr = curr - c.template at<'i'>(i) * prev;
				d.template at<dim>(i) = curr;

				prev = curr;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t x_tile_size)
{
	constexpr char dim = 'y';
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const real_t a_s = a[s];
			const real_t b1_s = b1[s];
			const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'b', 'X'>(s, noarr::lit<0>, X), densities);
			const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

			const index_t remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
			const index_t x_len_remainder = remainder == 0 ? x_tile_size : remainder;
			const index_t x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

			real_t c_tmp = a_s;

			{
				const real_t r = 1 / (b1_s + a_s);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(0, x) *= r;
				}

				c_tmp = a_s * r;
			}

			for (index_t i = 1; i < n; i++)
			{
				const real_t b_tmp = b1_s + (i == n - 1 ? a_s : 0);
				const real_t r = 1 / (b_tmp - a_s * c_tmp);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
				}

				c_tmp = a_s * r;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				const auto back_c = c.template at<'i'>(i);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t x_tile_size)
{
	constexpr char dim = 'y';
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(3)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			for (index_t X = 0; X < X_len; X++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				const auto d =
					noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'z', 'b', 'X'>(s, z, noarr::lit<0>, X), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				const index_t remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
				const index_t x_len_remainder = remainder == 0 ? x_tile_size : remainder;
				const index_t x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

				real_t c_tmp = a_s;

				{
					const real_t r = 1 / (b1_s + a_s);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<dim, 'x'>(0, x) *= r;
					}

					c_tmp = a_s * r;
				}

				for (index_t i = 1; i < n; i++)
				{
					const real_t b_tmp = b1_s + (i == n - 1 ? a_s : 0);
					const real_t r = 1 / (b_tmp - a_s * c_tmp);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<dim, 'x'>(i, x) =
							r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
					}

					c_tmp = a_s * r;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					const auto back_c = c.template at<'i'>(i);

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t x_tile_size)
{
	constexpr char dim = 'z';
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(3)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t X = 0; X < X_len; X++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				const auto d =
					noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'y', 'b', 'X'>(s, y, noarr::lit<0>, X), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				const index_t remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
				const index_t x_len_remainder = remainder == 0 ? x_tile_size : remainder;
				const index_t x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

				real_t c_tmp = a_s;

				{
					const real_t r = 1 / (b1_s + a_s);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<dim, 'x'>(0, x) *= r;
					}

					c_tmp = a_s * r;
				}

				for (index_t i = 1; i < n; i++)
				{
					const real_t b_tmp = b1_s + (i == n - 1 ? a_s : 0);
					const real_t r = 1 / (b_tmp - a_s * c_tmp);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<dim, 'x'>(i, x) =
							r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
					}

					c_tmp = a_s * r;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					const auto back_c = c.template at<'i'>(i);

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_x()
{
	if (vectorized_x_)
	{
		if (this->problem_.dims == 1)
		{
#pragma omp parallel
			solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, cx_, get_substrates_layout<1>(),
									  get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, ax_, b1x_, cx_,
													   get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
													   get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d_transpose<index_t>(
				this->substrates_, ax_, b1x_, cx_, get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
				get_diagonal_layout(this->problem_, this->problem_.nx));
		}
	}
	else
	{
		if (this->problem_.dims == 1)
		{
#pragma omp parallel
			solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, cx_, get_substrates_layout<1>(),
									  get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
											 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
											 get_diagonal_layout(this->problem_, this->problem_.nx));
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
											 get_diagonal_layout(this->problem_, this->problem_.nx));
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<2>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
							  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_);
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, cx_, get_substrates_layout<1>(),
								  get_diagonal_layout(this->problem_, this->problem_.nx));
	}
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			if (vectorized_x_)
				solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, ax_, b1x_, cx_,
														   get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
														   get_diagonal_layout(this->problem_, this->problem_.nx));
			else
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
												 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
			solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<2>(),
									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
		}
	}
	if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			if (vectorized_x_)
				solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, ax_, b1x_, cx_,
														   get_substrates_layout<3>()
															   ^ noarr::merge_blocks<'z', 'y', 'm'>(),
														   get_diagonal_layout(this->problem_, this->problem_.nx));
			else
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
												 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
			solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);
#pragma omp barrier
			solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
									  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_);
		}
	}
}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_t<real_t, aligned_x>::least_memory_thomas_solver_d_t(bool vectorized_x)
	: ax_(nullptr),
	  b1x_(nullptr),
	  cx_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  cy_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  cz_(nullptr),
	  vectorized_x_(vectorized_x)
{}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_t<real_t, aligned_x>::~least_memory_thomas_solver_d_t()
{
	if (cx_)
	{
		std::free(cx_);
		std::free(ax_);
		std::free(b1x_);
	}
	if (cy_)
	{
		std::free(cy_);
		std::free(ay_);
		std::free(b1y_);
	}
	if (cz_)
	{
		std::free(cz_);
		std::free(az_);
		std::free(b1z_);
	}
}

template class least_memory_thomas_solver_d_t<float, false>;
template class least_memory_thomas_solver_d_t<double, false>;

template class least_memory_thomas_solver_d_t<float, true>;
template class least_memory_thomas_solver_d_t<double, true>;
