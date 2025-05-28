#include "cubed_mix_thomas_solver_t.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "noarr/structures/base/structs_common.hpp"
#include "noarr/structures/extra/funcs.hpp"
#include "noarr/structures/extra/shortcuts.hpp"
#include "omp_helper.h"
#include "problem.h"
#include "vector_transpose_helper.h"

// a helper using for accessing static constexpr variables
using alg = cubed_mix_thomas_solver_t<double, true>;

template <typename real_t, bool aligned_x>
void cubed_mix_thomas_solver_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, index_t shape,
																	 index_t dims, index_t n, index_t counters_count,
																	 std::unique_ptr<aligned_atomic<long>[]>& counters,
																	 index_t group_size, index_t& block_size,
																	 std::vector<index_t>& group_block_lengths,
																	 std::vector<index_t>& group_block_offsets)
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

	counters = std::make_unique<aligned_atomic<long>[]>(counters_count);

	block_size = n / group_size;

	group_block_lengths.clear();
	group_block_offsets.clear();

	for (index_t i = 0; i < group_size; i++)
	{
		if (i < n % group_size)
			group_block_lengths.push_back(block_size + 1);
		else
			group_block_lengths.push_back(block_size);
	}

	group_block_offsets.resize(group_size);

	for (index_t i = 0; i < group_size; i++)
	{
		if (i == 0)
			group_block_offsets[i] = 0;
		else
			group_block_offsets[i] = group_block_offsets[i - 1] + group_block_lengths[i - 1];
	}
}


template <typename real_t, bool aligned_x>
void cubed_mix_thomas_solver_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, real_t*& c, index_t shape,
																	 index_t dims, index_t n)
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
void cubed_mix_thomas_solver_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void cubed_mix_thomas_solver_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	cores_division_ = params.contains("cores_division") ? (std::array<index_t, 3>)params["cores_division"]
														: std::array<index_t, 3> { 2, 2, 2 };

	if (vectorized_x_)
	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		x_tile_size_ = (x_tile_size_ + hn::Lanes(d) - 1) / hn::Lanes(d) * hn::Lanes(d);
		std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
		alignment_size_ = std::max(alignment_size_, vector_length * x_tile_size_ / hn::Lanes(d));
	}

	substrate_step_ = params.contains("substrate_step") ? (index_t)params["substrate_step"] : 1;
}

template <typename real_t, bool aligned_x>
void cubed_mix_thomas_solver_t<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims == 2)
		cores_division_[2] = 1;


	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, cx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
	{
		if (cores_division_[1] == 1)
		{
			countersy_count_ = 0;
			precompute_values(ay_, b1y_, cy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
		}
		else
		{
			countersy_count_ = cores_division_[0] * cores_division_[2];

			precompute_values(ay_, b1y_, this->problem_.dy, this->problem_.dims, this->problem_.ny, countersy_count_,
							  countersy_, cores_division_[1], group_blocks_[1], group_block_lengthsy_,
							  group_block_offsetsy_);
		}
	}
	if (this->problem_.dims >= 3)
	{
		if (cores_division_[2] == 1)
		{
			countersz_count_ = 1;

			precompute_values(az_, b1z_, cz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
		}
		else
		{
			countersz_count_ = cores_division_[0] * cores_division_[1];

			precompute_values(az_, b1z_, this->problem_.dz, this->problem_.dims, this->problem_.nz, countersz_count_,
							  countersz_, cores_division_[2], group_blocks_[2], group_block_lengthsz_,
							  group_block_offsetsz_);
		}
	}

	auto scratch_layouty = get_scratch_layout(this->problem_.ny, this->problem_.nx * this->problem_.nz);
	auto scratch_layoutz = get_scratch_layout(this->problem_.nz, this->problem_.nx * this->problem_.ny);

	if (aligned_x)
	{
		a_scratchy_ = (real_t*)std::aligned_alloc(alignment_size_, (scratch_layouty | noarr::get_size()));
		c_scratchy_ = (real_t*)std::aligned_alloc(alignment_size_, (scratch_layouty | noarr::get_size()));

		a_scratchz_ = (real_t*)std::aligned_alloc(alignment_size_, (scratch_layoutz | noarr::get_size()));
		c_scratchz_ = (real_t*)std::aligned_alloc(alignment_size_, (scratch_layoutz | noarr::get_size()));
	}
	else
	{
		a_scratchy_ = (real_t*)std::malloc((scratch_layouty | noarr::get_size()));
		c_scratchy_ = (real_t*)std::malloc((scratch_layouty | noarr::get_size()));

		a_scratchz_ = (real_t*)std::malloc((scratch_layoutz | noarr::get_size()));
		c_scratchz_ = (real_t*)std::malloc((scratch_layoutz | noarr::get_size()));
	}
}

template <typename real_t, bool aligned_x>
auto cubed_mix_thomas_solver_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
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
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ a,
											  const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
											  const density_layout_t dens_l, const diagonal_layout_t diag_l,
											  const index_t y_begin, const index_t y_end, const index_t z_begin,
											  const index_t z_end, const index_t s_begin, const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'x'>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];

		for (index_t z = z_begin; z < z_end; z++)
		{
			// vectorized body
			{
				const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

				auto y_remainder = (y_end - y_begin) % simd_length;

				for (index_t y = y_begin; y < y_end - y_remainder; y += simd_length)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1_s + a_s;
					real_t c_tmp = a_s;
					simd_t prev = hn::Zero(d);
					// vector registers that hold the to be transposed x*yz plane
					simd_t* rows = new simd_t[simd_length + 1];

					// forward substitution until last simd_length elements
					for (index_t i = 0; i < full_n - simd_length; i += simd_length)
					{
						rows[0] = prev;

						// aligned loads
						for (index_t v = 0; v < simd_length; v++)
							rows[v + 1] =
								hn::Load(d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, i, s)));

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
									  &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, i, s)));
					}

					// we are aligned to the vector size, so we can safely continue
					// here we fuse the end of forward substitution and the beginning of backwards propagation
					{
						rows[0] = prev;

						// aligned loads
						for (index_t v = 0; v < simd_length; v++)
							rows[v + 1] = hn::Load(d, &(dens_l
														| noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v,
																							full_n - simd_length, s)));

						// transposition to enable vectorization
						transpose(rows + 1);

						index_t remainder_work = n % simd_length;
						remainder_work += remainder_work == 0 ? simd_length : 0;

						// the rest of forward part
						{
							for (index_t v = 0; v < remainder_work; v++)
							{
								const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

								rows[v + 1] =
									hn::Mul(hn::MulAdd(rows[v], hn::Set(d, -a_tmp), rows[v + 1]), hn::Set(d, r));

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
								rows[v + 1] =
									hn::NegMulAdd(rows[v + 2], hn::Set(d, hn::ExtractLane(c, v)), rows[v + 1]);
							}

							prev = rows[1];
						}

						// transposition back to the original form
						transpose(rows + 1);

						// aligned stores
						for (index_t v = 0; v < simd_length; v++)
							hn::Store(
								rows[v + 1], d,
								&(dens_l
								  | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, full_n - simd_length, s)));
					}

					// we continue with backwards substitution
					for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
					{
						rows[simd_length] = prev;

						// aligned loads
						for (index_t v = 0; v < simd_length; v++)
							rows[v] =
								hn::Load(d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, i, s)));

						// transposition to enable vectorization
						transpose(rows);

						// backward propagation
						{
							auto c = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(back_c, i, s)));

							for (index_t v = simd_length - 1; v >= 0; v--)
							{
								rows[v] = hn::NegMulAdd(rows[v + 1], hn::Set(d, hn::ExtractLane(c, v)), rows[v]);
							}

							prev = rows[0];
						}

						// transposition back to the original form
						transpose(rows);

						// aligned stores
						for (index_t v = 0; v < simd_length; v++)
							hn::Store(rows[v], d,
									  &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, i, s)));
					}

					delete[] rows;
				}
			}

			// yz remainder
			{
				auto y_remainder = (y_end - y_begin) % simd_length;

				for (index_t y = y_end - y_remainder; y < y_end; y++)
				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y', 'z'>(s, y, z), densities);
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
}

template <int x_tile_multiple, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d_intrinsics(real_t* __restrict__ densities, const real_t* __restrict__ a,
										const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
										const density_layout_t dens_l, const diagonal_layout_t diag_l,
										const index_t z_begin, const index_t z_end, const index_t s_begin,
										const index_t s_end)
{
	constexpr char dim = 'y';
	const index_t n = dens_l | noarr::get_length<dim>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length * x_tile_multiple);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t z = z_begin; z < z_end; z++)
		{
			for (index_t X = 0; X < X_len; X++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				const auto d =
					noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'z', 'b', 'X'>(s, z, noarr::lit<0>, X), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				simd_t prev[x_tile_multiple];

				for (index_t x = 0; x < x_tile_multiple; x++)
					prev[x] = hn::Zero(t);

				for (index_t i = 0; i < n; i++)
				{
					const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

					for (index_t x = 0; x < x_tile_multiple; x++)
					{
						simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, x * simd_length));
						curr = hn::Mul(hn::MulAdd(hn::Set(t, -a_tmp), prev[x], curr), hn::Set(t, r));
						hn::Store(curr, t, &d.template at<dim, 'x'>(i, x * simd_length));

						prev[x] = curr;
					}

					a_tmp = a_s;
					b_tmp = b1_s + (i == n - 2 ? a_s : 0);
					c_tmp = a_s * r;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_tile_multiple; x++)
					{
						simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, x * simd_length));
						curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev[x], curr);
						hn::Store(curr, t, &d.template at<dim, 'x'>(i, x * simd_length));

						prev[x] = curr;
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d_intrinsics_dispatch(real_t* __restrict__ densities, const real_t* __restrict__ a,
												 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
												 const density_layout_t dens_l, const diagonal_layout_t diag_l,
												 const index_t z_begin, const index_t z_end, const index_t s_begin,
												 const index_t s_end, const index_t x_tile_size)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);

	const index_t x_tile_multiple = x_tile_size / simd_length;

	if (x_tile_multiple == 1)
	{
		solve_slice_y_3d_intrinsics<1>(densities, a, b1, back_c, dens_l, diag_l, z_begin, z_end, s_begin, s_end);
	}
	else if (x_tile_multiple == 2)
	{
		solve_slice_y_3d_intrinsics<2>(densities, a, b1, back_c, dens_l, diag_l, z_begin, z_end, s_begin, s_end);
	}
	else if (x_tile_multiple == 3)
	{
		solve_slice_y_3d_intrinsics<3>(densities, a, b1, back_c, dens_l, diag_l, z_begin, z_end, s_begin, s_end);
	}
	else if (x_tile_multiple == 4)
	{
		solve_slice_y_3d_intrinsics<4>(densities, a, b1, back_c, dens_l, diag_l, z_begin, z_end, s_begin, s_end);
	}
	else
	{
		throw std::runtime_error("Unsupported x_tile_size for intrinsics");
	}
}


template <int x_tile_multiple, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d_intrinsics(real_t* __restrict__ densities, const real_t* __restrict__ a,
										const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
										const density_layout_t dens_l, const diagonal_layout_t diag_l,
										const index_t s_begin, const index_t s_end)
{
	constexpr char dim = 'z';
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length * x_tile_multiple);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t s = s_begin; s < s_end; s++)
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

				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				simd_t prev[x_tile_multiple];

				for (index_t x = 0; x < x_tile_multiple; x++)
					prev[x] = hn::Zero(t);

				for (index_t i = 0; i < n; i++)
				{
					const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

					for (index_t x = 0; x < x_tile_multiple; x++)
					{
						simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, x * simd_length));
						curr = hn::Mul(hn::MulAdd(hn::Set(t, -a_tmp), prev[x], curr), hn::Set(t, r));
						hn::Store(curr, t, &d.template at<dim, 'x'>(i, x * simd_length));

						prev[x] = curr;
					}

					a_tmp = a_s;
					b_tmp = b1_s + (i == n - 2 ? a_s : 0);
					c_tmp = a_s * r;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_tile_multiple; x++)
					{
						simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, x * simd_length));
						curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev[x], curr);
						hn::Store(curr, t, &d.template at<dim, 'x'>(i, x * simd_length));

						prev[x] = curr;
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d_intrinsics_dispatch(real_t* __restrict__ densities, const real_t* __restrict__ a,
												 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
												 const density_layout_t dens_l, const diagonal_layout_t diag_l,
												 const index_t s_begin, const index_t s_end, const index_t x_tile_size)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);

	const index_t x_tile_multiple = x_tile_size / simd_length;

	if (x_tile_multiple == 1)
	{
		solve_slice_z_3d_intrinsics<1>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 2)
	{
		solve_slice_z_3d_intrinsics<2>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 3)
	{
		solve_slice_z_3d_intrinsics<3>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 4)
	{
		solve_slice_z_3d_intrinsics<4>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else
	{
		throw std::runtime_error("Unsupported x_tile_size for intrinsics");
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_y_start(real_t* __restrict__ densities, const real_t* __restrict__ ac,
								const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
								index_t& epoch, std::atomic<long>& counter, const index_t coop_size,
								const density_layout_t dens_l, const scratch_layout_t scratch_l, const index_t s_begin,
								const index_t s_end, const index_t x_begin, const index_t x_end, const index_t y_begin,
								const index_t y_end, const index_t z_begin, const index_t z_end)
{
	constexpr char dim = 'y';
	const index_t n = dens_l | noarr::get_length<dim>();

	auto a = noarr::make_bag(scratch_l, a_data);
	auto c = noarr::make_bag(scratch_l, c_data);
	auto d = noarr::make_bag(dens_l, densities);

	for (index_t s = s_begin; s < s_end; s++)
	{
		epoch++;

		for (index_t z = z_begin; z < z_end; z++)
		{
			// Normalize the first and the second equation
			index_t i;
			for (i = y_begin; i < y_begin + 2; i++)
			{
				const auto state = noarr::idx<'s', 'i'>(s, i);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + ((i == 0) || (i == n - 1) ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				for (index_t x = x_begin; x < x_end; x++)
				{
					d.template at<'s', 'x', 'z', dim>(s, x, z, i) /= b_tmp;
				}
			}

			// Process the lower diagonal (forward)
			for (; i < y_end; i++)
			{
				const auto state = noarr::idx<'s', 'i'>(s, i);
				const auto prev_state = noarr::idx<'s', 'i'>(s, i - 1);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				for (index_t x = x_begin; x < x_end; x++)
				{
					d.template at<'s', 'x', 'z', dim>(s, x, z, i) =
						r
						* (d.template at<'s', 'x', 'z', dim>(s, x, z, i)
						   - a_tmp * d.template at<'s', 'x', 'z', dim>(s, x, z, i - 1));
				}
			}

			// Process the upper diagonal (backward)
			for (i = y_end - 3; i >= y_begin + 1; i--)
			{
				const auto state = noarr::idx<'s', 'i'>(s, i);
				const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

				for (index_t x = x_begin; x < x_end; x++)
				{
					d.template at<'s', 'x', 'z', dim>(s, x, z, i) -=
						c[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, i + 1);
				}

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<'s', 'i'>(s, i);
				const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				for (index_t x = x_begin; x < x_end; x++)
				{
					d.template at<'s', 'x', 'z', dim>(s, x, z, i) =
						r
						* (d.template at<'s', 'x', 'z', dim>(s, x, z, i)
						   - c[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, i + 1));
				}

				a[state] = r * a[state];
				c[state] = r * (0 - c[state] * c[next_state]);
			}
		}

		// Second part of modified thomas algorithm
		// We solve the system of equations that are composed of the first and last row of each block
		// We do it using Thomas Algorithm
		{
			const index_t block_size = n / coop_size;
			const index_t epoch_size = coop_size + 1;
			// std::cout << blocks_count << std::endl;

			// increment the counter
			auto current_value = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

			if (current_value == epoch * epoch_size - 1)
			{
				auto get_i = [block_size, n](index_t equation_idx) {
					const index_t block_idx = equation_idx / 2;
					const auto block_start = block_idx * block_size + std::min(block_idx, n % block_size);
					const auto actual_block_size = (block_idx < n % block_size) ? block_size + 1 : block_size;
					const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
					return i;
				};

				for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
				{
					const index_t i = get_i(equation_idx);
					const index_t prev_i = get_i(equation_idx - 1);
					const auto state = noarr::idx<'s', 'i'>(s, i);
					const auto prev_state = noarr::idx<'s', 'i'>(s, prev_i);

					const auto r = 1 / (1 - a[state] * c[prev_state]);

					c[state] *= r;

					for (index_t z = z_begin; z < z_end; z++)
					{
						for (index_t x = x_begin; x < x_end; x++)
						{
							d.template at<'s', 'x', 'z', dim>(s, x, z, i) =
								r
								* (d.template at<'s', 'x', 'z', dim>(s, x, z, i)
								   - a[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, prev_i));
						}
					}
				}

				for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
				{
					const index_t i = get_i(equation_idx);
					const index_t next_i = get_i(equation_idx + 1);
					const auto state = noarr::idx<'s', 'i'>(s, i);

					for (index_t z = z_begin; z < z_end; z++)
					{
						for (index_t x = x_begin; x < x_end; x++)
						{
							d.template at<'s', 'x', 'z', dim>(s, x, z, i) =
								d.template at<'s', 'x', 'z', dim>(s, x, z, i)
								- c[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, next_i);
						}
					}
				}

				// notify all threads that the last thread has finished
				counter.fetch_add(1, std::memory_order_acq_rel);
				// counter.notify_all();
			}
			else
			{
				// wait for the last thread to finish
				while (current_value < epoch * epoch_size)
				{
					// counter.wait(current_value, std::memory_order_acquire);
					current_value = counter.load(std::memory_order_acquire);
				}
			}
		}

		for (index_t z = z_begin; z < z_end; z++)
		{
			// Final part of modified thomas algorithm
			// Solve the rest of the unknowns
			{
				for (index_t i = y_begin + 1; i < y_end - 1; i++)
				{
					const auto state = noarr::idx<'s', 'i'>(s, i);

					for (index_t x = x_begin; x < x_end; x++)
					{
						d.template at<'s', 'x', 'z', dim>(s, x, z, i) =
							d.template at<'s', 'x', 'z', dim>(s, x, z, i)
							- a[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, y_begin)
							- c[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, y_end - 1);
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_z_start(real_t* __restrict__ densities, const real_t* __restrict__ ac,
								const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
								index_t& epoch, std::atomic<long>& counter, const index_t coop_size,
								const density_layout_t dens_l, const scratch_layout_t scratch_l, const index_t s_begin,
								const index_t s_end, const index_t x_begin, const index_t x_end, const index_t y_begin,
								const index_t y_end, const index_t z_begin, const index_t z_end,
								const index_t x_tile_size)
{
	constexpr char dim = 'z';
	const index_t n = dens_l | noarr::get_length<dim>();

	auto a = noarr::make_bag(scratch_l, a_data);
	auto c = noarr::make_bag(scratch_l, c_data);
	auto d = noarr::make_bag(dens_l, densities);

	for (index_t s = s_begin; s < s_end; s++)
	{
		epoch++;

		{
			// Normalize the first and the second equation
			index_t i;
			for (i = z_begin; i < z_begin + 2; i++)
			{
				const auto state = noarr::idx<'s', 'i'>(s, i);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + ((i == 0) || (i == n - 1) ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				for (index_t y = y_begin; y < y_end; y++)
					for (index_t x = x_begin; x < x_end; x++)
					{
						d.template at<'s', 'x', 'y', dim>(s, x, y, i) /= b_tmp;
					}
			}

			// Process the lower diagonal (forward)
			for (; i < z_end; i++)
			{
				const auto state = noarr::idx<'s', 'i'>(s, i);
				const auto prev_state = noarr::idx<'s', 'i'>(s, i - 1);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				for (index_t y = y_begin; y < y_end; y++)
					for (index_t x = x_begin; x < x_end; x++)
					{
						d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
							r
							* (d.template at<'s', 'x', 'y', dim>(s, x, y, i)
							   - a_tmp * d.template at<'s', 'x', 'y', dim>(s, x, y, i - 1));
					}
			}

			// Process the upper diagonal (backward)
			for (i = z_end - 3; i >= z_begin + 1; i--)
			{
				const auto state = noarr::idx<'s', 'i'>(s, i);
				const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

				for (index_t y = y_begin; y < y_end; y++)
					for (index_t x = x_begin; x < x_end; x++)
					{
						d.template at<'s', 'x', 'y', dim>(s, x, y, i) -=
							c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, i + 1);
					}

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<'s', 'i'>(s, i);
				const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				for (index_t y = y_begin; y < y_end; y++)
					for (index_t x = x_begin; x < x_end; x++)
					{
						d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
							r
							* (d.template at<'s', 'x', 'y', dim>(s, x, y, i)
							   - c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, i + 1));
					}

				a[state] = r * a[state];
				c[state] = r * (0 - c[state] * c[next_state]);
			}
		}

		// Second part of modified thomas algorithm
		// We solve the system of equations that are composed of the first and last row of each block
		// We do it using Thomas Algorithm
		{
			const index_t block_size = n / coop_size;
			const index_t epoch_size = coop_size + 1;
			// std::cout << blocks_count << std::endl;

			// increment the counter
			auto current_value = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

			if (current_value == epoch * epoch_size - 1)
			{
				auto get_i = [block_size, n](index_t equation_idx) {
					const index_t block_idx = equation_idx / 2;
					const auto block_start = block_idx * block_size + std::min(block_idx, n % block_size);
					const auto actual_block_size = (block_idx < n % block_size) ? block_size + 1 : block_size;
					const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
					return i;
				};

				for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
				{
					const index_t i = get_i(equation_idx);
					const index_t prev_i = get_i(equation_idx - 1);
					const auto state = noarr::idx<'s', 'i'>(s, i);
					const auto prev_state = noarr::idx<'s', 'i'>(s, prev_i);

					const auto r = 1 / (1 - a[state] * c[prev_state]);

					c[state] *= r;

					for (index_t y = y_begin; y < y_end; y++)
					{
						for (index_t x = x_begin; x < x_end; x++)
						{
							d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
								r
								* (d.template at<'s', 'x', 'y', dim>(s, x, y, i)
								   - a[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, prev_i));
						}
					}
				}

				for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
				{
					const index_t i = get_i(equation_idx);
					const index_t next_i = get_i(equation_idx + 1);
					const auto state = noarr::idx<'s', 'i'>(s, i);

					for (index_t y = y_begin; y < y_end; y++)
					{
						for (index_t x = x_begin; x < x_end; x++)
						{
							d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
								d.template at<'s', 'x', 'y', dim>(s, x, y, i)
								- c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, next_i);
						}
					}
				}

				// notify all threads that the last thread has finished
				counter.fetch_add(1, std::memory_order_acq_rel);
				// counter.notify_all();
			}
			else
			{
				// wait for the last thread to finish
				while (current_value < epoch * epoch_size)
				{
					// counter.wait(current_value, std::memory_order_acquire);
					current_value = counter.load(std::memory_order_acquire);
				}
			}
		}

		{
			// Final part of modified thomas algorithm
			// Solve the rest of the unknowns
			{
				for (index_t i = z_begin + 1; i < z_end - 1; i++)
				{
					const auto state = noarr::idx<'s', 'i'>(s, i);
					
					for (index_t y = y_begin; y < y_end; y++)
						for (index_t x = x_begin; x < x_end; x++)
						{
							d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
								d.template at<'s', 'x', 'y', dim>(s, x, y, i)
								- a[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, z_begin)
								- c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, z_end - 1);
						}
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void cubed_mix_thomas_solver_t<real_t, aligned_x>::solve_x()
{}

template <typename real_t, bool aligned_x>
void cubed_mix_thomas_solver_t<real_t, aligned_x>::solve_y()
{}

template <typename real_t, bool aligned_x>
void cubed_mix_thomas_solver_t<real_t, aligned_x>::solve_z()
{}

template <typename real_t, bool aligned_x>
void cubed_mix_thomas_solver_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
		return;


	for (index_t i = 0; i < countersy_count_; i++)
	{
		countersy_[i].value = 0;
	}

	for (index_t i = 0; i < countersz_count_; i++)
	{
		countersz_[i].value = 0;
	}

	{
		// solve_2d_and_3d(this->substrates_, get_substrates_layout<3>(), ax_, b1x_, ay_, b1y_, az_, b1z_, a_scratchx_,
		// 				c_scratchx_, a_scratchy_, c_scratchy_, a_scratchz_, c_scratchz_, countersx_.get(),
		// 				countersy_.get(), countersz_.get(),
		// 				get_scratch_layout(this->problem_.nx, this->problem_.ny * this->problem_.nz),
		// 				get_scratch_layout(this->problem_.ny, this->problem_.nx * this->problem_.nz),
		// 				get_scratch_layout(this->problem_.nz, this->problem_.ny * this->problem_.nx), cores_division_,
		// 				x_tile_size_, this->problem_.iterations);
	}

	auto dens_l = get_substrates_layout<3>();

	auto scratchy_l = get_scratch_layout(this->problem_.ny, this->problem_.nx * this->problem_.nz);
	auto scratchz_l = get_scratch_layout(this->problem_.nz, this->problem_.ny * this->problem_.nx);

	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

#pragma omp parallel num_threads(cores_division_[0] * cores_division_[1] * cores_division_[2])
	{
		const auto tid = get_thread_num();

		const auto tid_x = tid % cores_division_[0];
		const auto tid_y = (tid / cores_division_[0]) % cores_division_[1];
		const auto tid_z = tid / (cores_division_[0] * cores_division_[1]);

		const auto block_y_begin = cores_division_[1] == 1 ? 0 : group_block_offsetsy_[tid_y];
		const auto block_y_end = cores_division_[1] == 1 ? y_len : block_y_begin + group_block_lengthsy_[tid_y];

		const auto block_z_begin = cores_division_[2] == 1 ? 0 : group_block_offsetsz_[tid_z];
		const auto block_z_end = cores_division_[2] == 1 ? z_len : block_z_begin + group_block_lengthsz_[tid_z];

		// #pragma omp critical
		// 		{
		// 			std::cout << "Thread " << tid << " is working on blocks: "
		// 					  << "[" << 0 << ", " << x_len << ") x "
		// 					  << "[" << block_y_begin << ", " << block_y_end << ") x "
		// 					  << "[" << block_z_begin << ", " << block_z_end << ")" << std::endl;
		// 		}

		index_t epoch_y = 0;
		index_t epoch_z = 0;

		for (index_t s = 0; s < s_len; s += substrate_step_)
		{
			auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				// do x
				{
					solve_slice_x_2d_and_3d_transpose<index_t>(
						this->substrates_, ax_, b1x_, cx_, get_substrates_layout<3>(),
						get_diagonal_layout(this->problem_, this->problem_.nx), block_y_begin, block_y_end,
						block_z_begin, block_z_end, s, s + s_step_length);
				}

				// do Y
				{
					if (cores_division_[1] == 1)
					{
						solve_slice_y_3d_intrinsics_dispatch<index_t>(
							this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.ny), block_z_begin, block_z_end, s,
							s + s_step_length, x_tile_size_);
					}
					else
					{
						const auto lane_id = tid_x * cores_division_[2] + tid_z;
						const auto lane_scratch_l = scratchy_l ^ noarr::fix<'l'>(lane_id);

						solve_block_y_start<index_t>(this->substrates_, ay_, b1y_, a_scratchy_, c_scratchy_, epoch_y,
													 countersy_[lane_id].value, cores_division_[1], dens_l,
													 lane_scratch_l, s, s + s_step_length, 0, x_len, block_y_begin,
													 block_y_end, block_z_begin, block_z_end);
					}
				}

				if (z_len > 1)
				{
					// do Z
					{
						if (cores_division_[2] == 1)
						{
							solve_slice_z_3d_intrinsics_dispatch<index_t>(
								this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
								get_diagonal_layout(this->problem_, this->problem_.nz), s, s + s_step_length,
								x_tile_size_);
						}
						else
						{
							const auto lane_id = tid_x * cores_division_[1] + tid_y;
							const auto lane_scratch_l = scratchz_l ^ noarr::fix<'l'>(lane_id);

							solve_block_z_start<index_t>(this->substrates_, az_, b1z_, a_scratchz_, c_scratchz_,
														 epoch_z, countersz_[lane_id].value, cores_division_[2], dens_l,
														 lane_scratch_l, s, s + s_step_length, 0, x_len, block_y_begin,
														 block_y_end, block_z_begin, block_z_end, x_tile_size_);
						}
					}
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
cubed_mix_thomas_solver_t<real_t, aligned_x>::cubed_mix_thomas_solver_t(bool vectorized_x)
	: ax_(nullptr),
	  b1x_(nullptr),
	  cx_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  cy_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  cz_(nullptr),
	  a_scratchy_(nullptr),
	  c_scratchy_(nullptr),
	  a_scratchz_(nullptr),
	  c_scratchz_(nullptr),
	  vectorized_x_(vectorized_x),
	  group_block_lengthsy_({ 1 }),
	  group_block_lengthsz_({ 1 }),
	  group_block_offsetsy_({ 0 }),
	  group_block_offsetsz_({ 0 })
{}

template <typename real_t, bool aligned_x>
cubed_mix_thomas_solver_t<real_t, aligned_x>::~cubed_mix_thomas_solver_t()
{
	if (b1x_)
	{
		std::free(ax_);
		std::free(b1x_);
		std::free(cx_);
	}
	if (b1y_)
	{
		std::free(ay_);
		std::free(b1y_);
		std::free(cy_);
	}
	if (b1z_)
	{
		std::free(az_);
		std::free(b1z_);
		std::free(cz_);
	}

	if (a_scratchy_)
	{
		std::free(a_scratchy_);
		std::free(c_scratchy_);
	}
	if (a_scratchz_)
	{
		std::free(a_scratchz_);
		std::free(c_scratchz_);
	}
}

template class cubed_mix_thomas_solver_t<float, false>;
template class cubed_mix_thomas_solver_t<double, false>;

template class cubed_mix_thomas_solver_t<float, true>;
template class cubed_mix_thomas_solver_t<double, true>;
