#include "full_blocking.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>

#include "../barrier.h"
#include "../perf_utils.h"
#include "full_blocking_x.h"
#include "noarr/structures/extra/funcs.hpp"
#include "noarr/structures/structs/blocks.hpp"

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::precompute_values(
	index_t counters_count, std::unique_ptr<std::unique_ptr<aligned_atomic<index_t>>[]>& counters,
	std::unique_ptr<std::unique_ptr<std::barrier<>>[]>& barriers, index_t group_size, char dim)
{
	counters = std::make_unique<std::unique_ptr<aligned_atomic<index_t>>[]>(counters_count);
	barriers = std::make_unique<std::unique_ptr<std::barrier<>>[]>(counters_count);


#pragma omp parallel
	{
		auto tid = get_thread_id();

		index_t lane_id = get_lane_id(dim);

		index_t dim_id = (dim == 'x') ? tid.x : (dim == 'y' ? tid.y : tid.z);

		if (dim_id == 0)
		{
			counters[lane_id] = std::make_unique<aligned_atomic<index_t>>(0);
			barriers[lane_id] = std::make_unique<std::barrier<>>(group_size);
		}
	}
}

template <typename real_t, bool aligned_x>
template <char dim, bool fused_z>
void sdd_full_blocking<real_t, aligned_x>::precompute_values(std::unique_ptr<real_t*[]>& a,
															 std::unique_ptr<real_t*[]>& b,
															 std::unique_ptr<real_t*[]>& c, index_t shape, index_t n,
															 index_t dims)
{
	a = std::make_unique<real_t*[]>(get_max_threads());
	b = std::make_unique<real_t*[]>(get_max_threads());
	c = std::make_unique<real_t*[]>(get_max_threads());

#pragma omp parallel
	{
		auto tid = get_thread_id();

		auto arrays_layout = noarr::scalar<real_t*>() ^ get_thread_distribution_layout();

		real_t*& a_t = arrays_layout | noarr::get_at<'x', 'y', 'z', 'g'>(a.get(), tid.x, tid.y, tid.z, tid.group);
		real_t*& b_t = arrays_layout | noarr::get_at<'x', 'y', 'z', 'g'>(b.get(), tid.x, tid.y, tid.z, tid.group);
		real_t*& c_t = arrays_layout | noarr::get_at<'x', 'y', 'z', 'g'>(c.get(), tid.x, tid.y, tid.z, tid.group);

		auto get_layout = [&]() {
			if constexpr (dim == 'x')
			{
				if constexpr (fused_z)
					return get_diag_layout<'x'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group],
												y_sync_step_)
						   ^ noarr::merge_blocks<'Y', 'y'>()
						   ^ noarr::into_blocks_static<'y', 'b', 'z', 'y'>(group_block_lengthsy_[tid.y])
						   ^ noarr::fix<'b'>(noarr::lit<0>) ^ noarr::merge_blocks<'Z', 'z'>();
				else
					return get_diag_layout<'x', false>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													   group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group],
													   y_sync_step_)
						   ^ noarr::merge_blocks<'Y', 'y'>();
			}
			else // if constexpr (dim == 'y')
			{
				return get_diag_layout<dim>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
											group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group],
											y_sync_step_)
					   ^ noarr::merge_blocks<'X', 'x'>();
			}
			// else
			// {
			// 	return get_diag_layout<dim>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
			// 								group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group],
			// 								y_sync_step_)
			// 		   ^ noarr::merge_blocks<'X', 'x'>()
			// 		   ^ noarr::into_blocks_static<'x', 'b', 'y', 'x'>(group_block_lengthsy_[tid.x])
			// 		   ^ noarr::fix<'b'>(noarr::lit<0>);
			// }
		};

		auto diag_t_l = get_layout();


		a_t = (real_t*)std::aligned_alloc(alignment_size_, (diag_t_l | noarr::get_size()));
		b_t = (real_t*)std::aligned_alloc(alignment_size_, (diag_t_l | noarr::get_size()));
		c_t = (real_t*)std::aligned_alloc(alignment_size_, (diag_t_l | noarr::get_size()));

		auto a_bag = noarr::make_bag(diag_t_l, a_t);
		auto b_bag = noarr::make_bag(diag_t_l, b_t);
		auto c_bag = noarr::make_bag(diag_t_l, c_t);

		auto get_diffusion_coefficients = [&](index_t x, index_t y, index_t z, index_t s) {
			return this->problem_.diffusion_coefficients[s] + 10000 * x + 1000 * y + 100 * z;
		};

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s++)
			for (index_t x = 0; x < group_block_lengthsx_[tid.x]; x++)
				for (index_t y = 0; y < group_block_lengthsy_[tid.y]; y++)
					for (index_t z = 0; z < group_block_lengthsz_[tid.z]; z++)
					{
						auto idx = noarr::idx<'x', 'y', 'z', 's'>(x, y, z, s);

						const real_t dc = get_diffusion_coefficients(
							group_block_offsetsx_[tid.x] + x, group_block_offsetsy_[tid.y] + y,
							group_block_offsetsz_[tid.z] + z, group_block_offsetss_[tid.group] + s);

						auto dim_idx = dim == 'x' ? x : (dim == 'y' ? y : z);

						if (dim == 'x')
							dim_idx += group_block_offsetsx_[tid.x];
						else if (dim == 'y')
							dim_idx += group_block_offsetsy_[tid.y];
						else
							dim_idx += group_block_offsetsz_[tid.z];

						if (dim_idx == 0)
						{
							a_bag[idx] = 0;
							b_bag[idx] = 1 + this->problem_.dt * this->problem_.decay_rates[s] / dims
										 + 1 * this->problem_.dt * dc / (shape * shape);
							c_bag[idx] = -this->problem_.dt * dc / (shape * shape);
						}
						else if (dim_idx == n - 1)
						{
							a_bag[idx] = -this->problem_.dt * dc / (shape * shape);
							b_bag[idx] = 1 + this->problem_.dt * this->problem_.decay_rates[s] / dims
										 + 1 * this->problem_.dt * dc / (shape * shape);
							c_bag[idx] = 0;
						}
						else
						{
							a_bag[idx] = -this->problem_.dt * dc / (shape * shape);
							b_bag[idx] = 1 + this->problem_.dt * this->problem_.decay_rates[s] / dims
										 + 2 * this->problem_.dt * dc / (shape * shape);
							c_bag[idx] = -this->problem_.dt * dc / (shape * shape);
						}
					}
	}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::validate_restrictions()
{
	bool ok = this->problem_.nx / cores_division_[0] >= 3 && this->problem_.ny / cores_division_[1] >= 3
			  && this->problem_.nz / cores_division_[2] >= 3;

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;

	ok &= x_tile_size_ % hn::Lanes(t) == 0;

	if (!ok)
		throw std::runtime_error("Bad tunable params for this problem!");
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::prepare(const max_problem_t& problem)
{
	this->problem_ = problems::cast<std::int32_t, real_t>(problem);

	if (this->problem_.dims == 2)
		cores_division_[2] = 1;

	set_block_bounds(this->problem_.nx, cores_division_[0], group_blocks_[0], group_block_lengthsx_,
					 group_block_offsetsx_);

	set_block_bounds(this->problem_.ny, cores_division_[1], group_blocks_[1], group_block_lengthsy_,
					 group_block_offsetsy_);

	set_block_bounds(this->problem_.nz, cores_division_[2], group_blocks_[2], group_block_lengthsz_,
					 group_block_offsetsz_);

	{
		substrate_groups_ = get_max_threads() / (cores_division_[0] * cores_division_[1] * cores_division_[2]);

		auto ss_len = this->problem_.substrates_count;

		for (index_t group_id = 0; group_id < substrate_groups_; group_id++)
		{
			const auto [ss_begin, ss_end] = evened_work_distribution(ss_len, substrate_groups_, group_id);

			const auto s_begin = ss_begin;
			const auto s_end = std::min(this->problem_.substrates_count, ss_end);

			group_block_lengthss_.push_back(std::max(s_end - s_begin, 0));
			group_block_offsetss_.push_back(s_begin);
		}
	}

	validate_restrictions();

	thread_substrate_array_ = std::make_unique<real_t*[]>(get_max_threads());

#pragma omp parallel
	{
		const auto tid = get_thread_id();

		if (group_block_lengthss_[tid.group] != 0)
		{
			auto arrays_layout = noarr::scalar<real_t*>() ^ get_thread_distribution_layout();

			real_t*& substrates_t =
				arrays_layout
				| noarr::get_at<'x', 'y', 'z', 'g'>(thread_substrate_array_.get(), tid.x, tid.y, tid.z, tid.group);

			auto dens_t_l =
				get_blocked_substrate_layout(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
											 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

			substrates_t = (real_t*)std::aligned_alloc(alignment_size_, (dens_t_l | noarr::get_size()));

			if (problem.gaussian_pulse)
			{
				omp_trav_for_each(noarr::traverser(dens_t_l), [&](auto state) {
					index_t s = noarr::get_index<'s'>(state) + group_block_offsetss_[tid.group];
					index_t x = noarr::get_index<'x'>(state) + group_block_offsetsx_[tid.x];
					index_t y = noarr::get_index<'y'>(state) + group_block_offsetsy_[tid.y];
					index_t z = noarr::get_index<'z'>(state) + group_block_offsetsz_[tid.z];

					(dens_t_l | noarr::get_at(substrates_t, state)) =
						solver_utils::gaussian_analytical_solution(s, x, y, z, this->problem_);
				});
			}
			else
			{
				omp_trav_for_each(noarr::traverser(dens_t_l), [&](auto state) {
					auto s_idx = noarr::get_index<'s'>(state);

					(dens_t_l | noarr::get_at(substrates_t, state)) = problem.initial_conditions[s_idx];
				});
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;

	cores_division_ = params.contains("cores_division") ? (std::array<index_t, 3>)params["cores_division"]
														: std::array<index_t, 3> { 1, 2, 2 };

	fuse_z_ = params.contains("fuse_z") ? (bool)params["fuse_z"] : true;

	y_sync_step_ = params.contains("sync_step") ? (index_t)params["sync_step"] : 1;
	z_sync_step_ = params.contains("sync_step") ? (index_t)params["sync_step"] : 1;

	if (params.contains("sync_step_y"))
		y_sync_step_ = (index_t)params["sync_step_y"];

	if (params.contains("sync_step_z"))
		z_sync_step_ = (index_t)params["sync_step_z"];

	alt_blocked_ = params.contains("alt_blocked") ? (bool)params["alt_blocked"] : false;

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);

	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : vector_length;
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::initialize()
{
	if (!fuse_z_)
		precompute_values<'x', false>(ax_, bx_, cx_, this->problem_.dx, this->problem_.nx, this->problem_.dims);
	else
		precompute_values<'x'>(ax_, bx_, cx_, this->problem_.dx, this->problem_.nx, this->problem_.dims);

	precompute_values<'y'>(ay_, by_, cy_, this->problem_.dy, this->problem_.ny, this->problem_.dims);
	precompute_values<'z'>(az_, bz_, cz_, this->problem_.dz, this->problem_.nz, this->problem_.dims);

	// x counters
	{
		countersx_count_ = cores_division_[1] * cores_division_[2] * substrate_groups_;

		precompute_values(countersx_count_, countersx_, barriersx_, cores_division_[0], 'x');
	}

	// y counters
	{
		countersy_count_ = cores_division_[0] * cores_division_[2] * substrate_groups_;

		precompute_values(countersy_count_, countersy_, barriersy_, cores_division_[1], 'y');
	}

	{
		// z counters
		countersz_count_ = cores_division_[0] * cores_division_[1] * substrate_groups_;

		precompute_values(countersz_count_, countersz_, barriersz_, cores_division_[2], 'z');
	}

	a_scratch_ = std::make_unique<real_t*[]>(get_max_threads());
	c_scratch_ = std::make_unique<real_t*[]>(get_max_threads());

#pragma omp parallel
	{
		auto tid = get_thread_id();

		auto non_blocked_scratch_lx =
			get_non_blocked_scratch_layout<'x'>(group_block_lengthsx_[tid.x], alignment_size_ / sizeof(real_t));
		auto non_blocked_scratch_ly = get_non_blocked_scratch_layout<'y'>(
			group_block_lengthsy_[tid.y], std::min(x_tile_size_, group_block_lengthsx_[tid.x]));
		auto non_blocked_scratch_lz = get_non_blocked_scratch_layout<'z'>(
			group_block_lengthsz_[tid.z], std::min(x_tile_size_, group_block_lengthsx_[tid.x]));

		auto max_size_non_blocked =
			std::max({ non_blocked_scratch_lx | noarr::get_size(), non_blocked_scratch_ly | noarr::get_size(),
					   non_blocked_scratch_lz | noarr::get_size() });

		std::size_t max_size_blocked = 0;
		{
			auto scratch_lx =
				get_scratch_layout<'x', true>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y], y_sync_step_);
			auto scratch_ly =
				get_scratch_layout<'y', true>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y], y_sync_step_);
			auto scratch_lz =
				get_scratch_layout<'z', true>(group_block_lengthsx_[tid.x], z_sync_step_, group_block_lengthsz_[tid.z]);

			auto max = std::max(
				{ scratch_lx | noarr::get_size(), scratch_ly | noarr::get_size(), scratch_lz | noarr::get_size() });
			max_size_blocked = std::max(max, max_size_blocked);
		}

		{
			auto scratch_lx = get_scratch_layout<'x', true, false>(group_block_lengthsx_[tid.x],
																   group_block_lengthsy_[tid.y], y_sync_step_);
			auto scratch_ly = get_scratch_layout<'y', true, false>(group_block_lengthsx_[tid.x],
																   group_block_lengthsy_[tid.y], y_sync_step_);
			auto scratch_lz = get_scratch_layout<'z', true, false>(group_block_lengthsx_[tid.x], z_sync_step_,
																   group_block_lengthsz_[tid.z]);

			auto max = std::max(
				{ scratch_lx | noarr::get_size(), scratch_ly | noarr::get_size(), scratch_lz | noarr::get_size() });
			max_size_blocked = std::max(max, max_size_blocked);
		}

		auto linear_tid = get_thread_num();
		a_scratch_[linear_tid] =
			(real_t*)std::aligned_alloc(alignment_size_, std::max(max_size_blocked, max_size_non_blocked));
		c_scratch_[linear_tid] = (real_t*)std::aligned_alloc(alignment_size_, max_size_blocked);
	}
}
template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::set_block_bounds(index_t n, index_t group_size, index_t& block_size,
															std::vector<index_t>& group_block_lengths,
															std::vector<index_t>& group_block_offsets)
{
	if (group_size == 1)
	{
		block_size = n;
		group_block_lengths = { n };
		group_block_offsets = { 0 };
		return;
	}

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
thread_id_t<typename sdd_full_blocking<real_t, aligned_x>::index_t> sdd_full_blocking<real_t,
																					  aligned_x>::get_thread_id() const
{
	thread_id_t<typename sdd_full_blocking<real_t, aligned_x>::index_t> id;

	const index_t tid = get_thread_num();
	const index_t group_size = cores_division_[0] * cores_division_[1] * cores_division_[2];

	const index_t substrate_group_tid = tid % group_size;

	id.group = tid / group_size;

	id.x = substrate_group_tid % cores_division_[0];
	id.y = (substrate_group_tid / cores_division_[0]) % cores_division_[1];
	id.z = substrate_group_tid / (cores_division_[0] * cores_division_[1]);

	return id;
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename scratch_t>
constexpr static void x_forward(const density_bag_t d, const index_t y, const index_t z_begin, const index_t s,
								const index_t n, const diag_bag_t a, const diag_bag_t b, const diag_bag_t c,
								const scratch_t b_scratch)

{
	const index_t ny = d | noarr::get_length<'y'>();
	const index_t sync_step = (a | noarr::get_length<'Y'>()) * (a | noarr::get_length<'y'>()) / ny;
	const index_t z_end = std::min<index_t>(d | noarr::get_length<'z'>(), z_begin + sync_step);

	auto dens_l = d.structure() ^ noarr::slice<'z'>(z_begin, z_end - z_begin) ^ noarr::merge_blocks<'z', 'y'>();
	auto diag_l = a.structure() ^ noarr::fix<'Z'>(z_begin / sync_step) ^ noarr::merge_blocks<'Y', 'y'>();

	auto a_bag = noarr::make_bag(diag_l, a.data());
	auto b_bag = noarr::make_bag(diag_l, b.data());
	auto c_bag = noarr::make_bag(diag_l, c.data());
	auto d_bag = noarr::make_bag(dens_l, d.data());

	{
		auto idx = noarr::idx<'s', 'y', 'x', 'v'>(s, y, 0, noarr::lit<0>);
		b_scratch[idx] = 1 / b_bag[idx];
	}

	for (index_t i = 1; i < n; i++)
	{
		auto idx = noarr::idx<'s', 'y', 'x', 'v'>(s, y, i, noarr::lit<0>);
		auto prev_idx = noarr::idx<'s', 'y', 'x', 'v'>(s, y, i - 1, noarr::lit<0>);

		auto r = a_bag[idx] * b_scratch[prev_idx];

		b_scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

		d_bag[idx] -= r * d_bag[prev_idx];

		// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
	}
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename scratch_t>
constexpr static void x_backward(const density_bag_t d, const index_t y, const index_t z_begin, const index_t s,
								 const index_t n, const diag_bag_t c, scratch_t b_scratch)

{
	const index_t ny = d | noarr::get_length<'y'>();
	const index_t sync_step = (c | noarr::get_length<'Y'>()) * (c | noarr::get_length<'y'>()) / ny;
	const index_t z_end = std::min<index_t>(d | noarr::get_length<'z'>(), z_begin + sync_step);

	auto dens_l = d.structure() ^ noarr::slice<'z'>(z_begin, z_end - z_begin) ^ noarr::merge_blocks<'z', 'y'>();
	auto diag_l = c.structure() ^ noarr::fix<'Z'>(z_begin / sync_step) ^ noarr::merge_blocks<'Y', 'y'>();

	auto c_bag = noarr::make_bag(diag_l, c.data());
	auto d_bag = noarr::make_bag(dens_l, d.data());

	{
		auto idx = noarr::idx<'s', 'y', 'x', 'v'>(s, y, n - 1, noarr::lit<0>);

		d_bag[idx] *= b_scratch[idx];

		// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
		auto idx = noarr::idx<'s', 'y', 'x', 'v'>(s, y, i, noarr::lit<0>);
		auto next_idx = noarr::idx<'s', 'y', 'x', 'v'>(s, y, i + 1, noarr::lit<0>);

		d_bag[idx] = (d_bag[idx] - c_bag[idx] * d_bag[next_idx]) * b_scratch[idx];

		// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
	}
}

template <typename index_t, typename simd_t, typename simd_tag, typename diagonal_t, typename scratch_t,
		  typename... vec_pack_t>
constexpr static void x_forward_vectorized(simd_tag d, const index_t x, const index_t y, const index_t z,
										   const index_t s, const diagonal_t a, const diagonal_t b, const diagonal_t c,
										   const scratch_t b_scratch, simd_t& scratch_prev, simd_t& c_prev,
										   simd_t& d_prev, simd_t& d_first, vec_pack_t&... vec_pack)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<hn::TFromD<simd_tag>> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	const auto idx = noarr::idx<'s', 'Z', 'Y', 'y', 'x', 'v'>(s, z, y / simd_length, y % simd_length, x, noarr::lit<0>);

	simd_t a_curr = hn::Load(d, &a[idx]);
	simd_t b_curr = hn::Load(d, &b[idx]);

	auto r = hn::Mul(a_curr, scratch_prev);

	scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
	hn::Store(scratch_prev, d, &b_scratch[idx]);

	d_first = hn::NegMulAdd(d_prev, r, d_first);

	c_prev = hn::Load(d, &c[idx]);

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_forward_vectorized(d, x + 1, y, z, s, a, b, c, b_scratch, scratch_prev, c_prev, d_first, vec_pack...);
}

template <typename index_t, typename simd_t, typename simd_tag, typename diagonal_t, typename scratch_t,
		  typename... vec_pack_t>
constexpr static void x_forward_vectorized(simd_tag d, const index_t length, const index_t x, const index_t y,
										   const index_t z, const index_t s, const diagonal_t a, const diagonal_t b,
										   const diagonal_t c, const scratch_t b_scratch, simd_t& scratch_prev,
										   simd_t& c_prev, simd_t& d_prev, simd_t& d_first, vec_pack_t&... vec_pack)
{
	if (length == 0)
		return;

	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<hn::TFromD<simd_tag>> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	const auto idx = noarr::idx<'s', 'Z', 'Y', 'y', 'x', 'v'>(s, z, y / simd_length, y % simd_length, x, noarr::lit<0>);

	simd_t a_curr = hn::Load(d, &a[idx]);
	simd_t b_curr = hn::Load(d, &b[idx]);

	auto r = hn::Mul(a_curr, scratch_prev);

	scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
	hn::Store(scratch_prev, d, &b_scratch[idx]);

	d_first = hn::NegMulAdd(d_prev, r, d_first);

	c_prev = hn::Load(d, &c[idx]);

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_forward_vectorized(d, length - 1, x + 1, y, z, s, a, b, c, b_scratch, scratch_prev, c_prev, d_first,
							 vec_pack...);
}

template <typename index_t, typename simd_t, typename simd_tag, typename diagonal_t, typename scratch_t,
		  typename... vec_pack_t>
constexpr static simd_t x_backward_vectorized(simd_tag d, const index_t x, const index_t y, const index_t z,
											  const index_t s, const diagonal_t c, scratch_t b_scratch, simd_t& d_first,
											  simd_t& d_prev, vec_pack_t&... vec_pack)
{
	if constexpr (sizeof...(vec_pack_t) != 0)
		x_backward_vectorized(d, x + 1, y, z, s, c, b_scratch, d_prev, vec_pack...);

	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<hn::TFromD<simd_tag>> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	const auto idx = noarr::idx<'s', 'Z', 'Y', 'y', 'x', 'v'>(s, z, y / simd_length, y % simd_length, x, noarr::lit<0>);

	simd_t c_curr = hn::Load(d, &c[idx]);
	simd_t scratch = hn::Load(d, &b_scratch[idx]);

	d_first = hn::Mul(hn::NegMulAdd(d_prev, c_curr, d_first), scratch);

	return d_first;
}


template <typename index_t, typename simd_t, typename simd_tag, typename diagonal_t, typename scratch_t,
		  typename... vec_pack_t>
constexpr static simd_t x_backward_vectorized(simd_tag d, const index_t length, const index_t x, const index_t y,
											  const index_t z, const index_t s, const diagonal_t c, scratch_t b_scratch,
											  simd_t& d_first, simd_t& d_prev, vec_pack_t&... vec_pack)
{
	if (length == 0)
		return d_first;

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_backward_vectorized(d, length - 1, x + 1, y, z, s, c, b_scratch, d_prev, vec_pack...);

	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<hn::TFromD<simd_tag>> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	const auto idx = noarr::idx<'s', 'Z', 'Y', 'y', 'x', 'v'>(s, z, y / simd_length, y % simd_length, x, noarr::lit<0>);

	simd_t c_curr = hn::Load(d, &c[idx]);
	simd_t scratch = hn::Load(d, &b_scratch[idx]);

	d_first = hn::Mul(hn::NegMulAdd(d_prev, c_curr, d_first), scratch);

	return d_first;
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename... simd_pack_t>
constexpr static void load(const density_bag_t d, simd_tag t, const index_t x, const index_t y, const index_t z,
						   const index_t s, simd_t& first, simd_pack_t&... vec_pack)
{
	first = hn::Load(t, &(d.template at<'s', 'z', 'y', 'x'>(s, z, y, x)));
	if constexpr (sizeof...(vec_pack) > 0)
	{
		load(d, t, x, y + 1, z, s, vec_pack...);
	}
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename... simd_pack_t>
constexpr static void store(const density_bag_t d, simd_tag t, const index_t x, const index_t y, const index_t z,
							const index_t s, const simd_t& first, const simd_pack_t&... vec_pack)
{
	hn::Store(first, t, &(d.template at<'s', 'z', 'y', 'x'>(s, z, y, x)));
	if constexpr (sizeof...(vec_pack) > 0)
	{
		store(d, t, x, y + 1, z, s, vec_pack...);
	}
}

template <typename T>
constexpr inline decltype(auto) id(T&& t)
{
	return std::forward<T>(t);
}

template <typename... Args>
constexpr decltype(auto) get_last(Args&&... args)
{
	return (id(std::forward<Args>(args)), ...);
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, typename... simd_pack_t>
constexpr static void xy_fused_transpose_part(const density_bag_t d, simd_tag t, const index_t y_offset,
											  const index_t y_len, const index_t s, const index_t z_begin,
											  const index_t n, const diag_bag_t a, const diag_bag_t b,
											  const diag_bag_t c, const scratch_bag_t b_scratch,
											  simd_pack_t... vec_pack)
{
	constexpr index_t simd_length = sizeof...(vec_pack);

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	const index_t ny = d | noarr::get_length<'y'>();
	const index_t sync_step = (a | noarr::get_length<'Y'>()) * (a | noarr::get_length<'y'>()) / ny;
	const index_t diag_z = z_begin / sync_step;

	for (index_t y = y_offset; y < y_len; y += simd_length)
	{
		const index_t dens_z = z_begin + y / ny;
		const index_t dens_y = y % ny;

		simd_t c_prev = hn::Zero(t);
		simd_t d_prev = hn::Zero(t);
		simd_t scratch_prev = hn::Zero(t);

		// forward substitution until last simd_length elements
		for (index_t i = 0; i < full_n - simd_length; i += simd_length)
		{
			load(d, t, i, dens_y, dens_z, s, vec_pack...);

			// transposition to enable vectorization
			transpose(vec_pack...);

			// actual forward substitution (vectorized)
			x_forward_vectorized(t, i, y, diag_z, s, a, b, c, b_scratch, scratch_prev, c_prev, d_prev, vec_pack...);

			d_prev = get_last(vec_pack...);

			// transposition back to the original form
			// transpose(rows);

			store(d, t, i, dens_y, dens_z, s, vec_pack...);
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			load(d, t, full_n - simd_length, dens_y, dens_z, s, vec_pack...);

			// transposition to enable vectorization
			transpose(vec_pack...);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			x_forward_vectorized(t, remainder_work, full_n - simd_length, y, diag_z, s, a, b, c, b_scratch,
								 scratch_prev, c_prev, d_prev, vec_pack...);

			d_prev = hn::Zero(t);

			// the begin of backward part
			d_prev = x_backward_vectorized(t, remainder_work, full_n - simd_length, y, diag_z, s, c, b_scratch,
										   vec_pack..., d_prev);

			// transposition back to the original form
			transpose(vec_pack...);

			store(d, t, full_n - simd_length, dens_y, dens_z, s, vec_pack...);
		}

		// we continue with backwards substitution
		for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
		{
			load(d, t, i, dens_y, dens_z, s, vec_pack...);

			// transposition to enable vectorization
			// transpose(rows);

			// backward propagation
			d_prev = x_backward_vectorized(t, i, y, diag_z, s, c, b_scratch, vec_pack..., d_prev);

			// transposition back to the original form
			transpose(vec_pack...);

			store(d, t, i, dens_y, dens_z, s, vec_pack...);
		}
	}
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t>
constexpr void xy_fused_transpose_part_2(const density_bag_t d, const index_t s, const index_t z, const index_t n,
										 const index_t y_len, const diag_bag_t a, const diag_bag_t b,
										 const diag_bag_t c, const scratch_bag_t b_scratch, const index_t y_offset)
{
	constexpr index_t simd_length = 2;

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	if constexpr (!multi)
	{
		using simd_tag = hn::FixedTag<real_t, 2>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, vec0, vec1);
	}
	else
	{
		using simd_tag = hn::FixedTag<real_t, 2>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t avec0, avec1;
		simd_t bvec0, bvec1;
		simd_t cvec0, cvec1;
		simd_t dvec0, dvec1;

		xy_fused_transpose_part_multi<simd_t, simd_tag, index_t, density_bag_t, diag_bag_t, scratch_bag_t, simd_t,
									  simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, avec0,
											  avec1, bvec0, bvec1, cvec0, cvec1, dvec0, dvec1);
	}

	for (index_t y = y_offset + simd_y_len; y < y_len; y++)
	{
		x_forward<real_t>(d, y, z, s, n, a, b, c, b_scratch);

		x_backward<real_t>(d, y, z, s, n, c, b_scratch);
	}
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t>
constexpr void xy_fused_transpose_part_4(const density_bag_t d, const index_t s, const index_t z, const index_t n,
										 const index_t y_len, const diag_bag_t a, const diag_bag_t b,
										 const diag_bag_t c, const scratch_bag_t b_scratch, const index_t y_offset)
{
	constexpr index_t simd_length = 4;

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	if constexpr (!multi)
	{
		using simd_tag = hn::FixedTag<real_t, 4>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, vec0, vec1,
										vec2, vec3);
	}
	else
	{
		using simd_tag = hn::FixedTag<real_t, 4>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t avec0, avec1, avec2, avec3;
		simd_t bvec0, bvec1, bvec2, bvec3;
		simd_t cvec0, cvec1, cvec2, cvec3;
		simd_t dvec0, dvec1, dvec2, dvec3;

		xy_fused_transpose_part_multi<simd_t, simd_tag, index_t, density_bag_t, diag_bag_t, scratch_bag_t, simd_t,
									  simd_t, simd_t, simd_t>(
			d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, avec0, avec1, avec2, avec3, bvec0,
			bvec1, bvec2, bvec3, cvec0, cvec1, cvec2, cvec3, dvec0, dvec1, dvec2, dvec3);
	}

	if (y_offset + simd_y_len < y_len)
		xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, y_offset + simd_y_len);
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t>
constexpr void xy_fused_transpose_part_8(const density_bag_t d, const index_t s, const index_t z, const index_t n,
										 const index_t y_len, const diag_bag_t a, const diag_bag_t b,
										 const diag_bag_t c, const scratch_bag_t b_scratch, const index_t y_offset)
{
	constexpr index_t simd_length = 8;

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	if constexpr (!multi)
	{
		using simd_tag = hn::FixedTag<real_t, 8>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, vec0, vec1,
										vec2, vec3, vec4, vec5, vec6, vec7);
	}
	else
	{
		using simd_tag = hn::FixedTag<real_t, 8>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t avec0, avec1, avec2, avec3, avec4, avec5, avec6, avec7;
		simd_t bvec0, bvec1, bvec2, bvec3, bvec4, bvec5, bvec6, bvec7;
		simd_t cvec0, cvec1, cvec2, cvec3, cvec4, cvec5, cvec6, cvec7;
		simd_t dvec0, dvec1, dvec2, dvec3, dvec4, dvec5, dvec6, dvec7;

		xy_fused_transpose_part_multi<simd_t, simd_tag, index_t, density_bag_t, diag_bag_t, scratch_bag_t, simd_t,
									  simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t>(
			d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, avec0, avec1, avec2, avec3, avec4,
			avec5, avec6, avec7, bvec0, bvec1, bvec2, bvec3, bvec4, bvec5, bvec6, bvec7, cvec0, cvec1, cvec2, cvec3,
			cvec4, cvec5, cvec6, cvec7, dvec0, dvec1, dvec2, dvec3, dvec4, dvec5, dvec6, dvec7);
	}

	if (y_offset + simd_y_len < y_len)
		xy_fused_transpose_part_4<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, y_offset + simd_y_len);
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t>
constexpr void xy_fused_transpose_part_16(const density_bag_t d, const index_t s, const index_t z, const index_t n,
										  const index_t y_len, const diag_bag_t a, const diag_bag_t b,
										  const diag_bag_t c, const scratch_bag_t b_scratch, const index_t y_offset)
{
	constexpr index_t simd_length = 16;

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	if constexpr (!multi)
	{
		using simd_tag = hn::FixedTag<real_t, 16>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, vec0, vec1,
										vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, vec13,
										vec14, vec15);
	}
	else
	{
		using simd_tag = hn::FixedTag<real_t, 16>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t avec0, avec1, avec2, avec3, avec4, avec5, avec6, avec7, avec8, avec9, avec10, avec11, avec12, avec13,
			avec14, avec15;
		simd_t bvec0, bvec1, bvec2, bvec3, bvec4, bvec5, bvec6, bvec7, bvec8, bvec9, bvec10, bvec11, bvec12, bvec13,
			bvec14, bvec15;
		simd_t cvec0, cvec1, cvec2, cvec3, cvec4, cvec5, cvec6, cvec7, cvec8, cvec9, cvec10, cvec11, cvec12, cvec13,
			cvec14, cvec15;
		simd_t dvec0, dvec1, dvec2, dvec3, dvec4, dvec5, dvec6, dvec7, dvec8, dvec9, dvec10, dvec11, dvec12, dvec13,
			dvec14, dvec15;

		xy_fused_transpose_part_multi<simd_t, simd_tag, index_t, density_bag_t, diag_bag_t, scratch_bag_t, simd_t,
									  simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t,
									  simd_t, simd_t, simd_t, simd_t, simd_t>(
			d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, avec0, avec1, avec2, avec3, avec4,
			avec5, avec6, avec7, avec8, avec9, avec10, avec11, avec12, avec13, avec14, avec15, bvec0, bvec1, bvec2,
			bvec3, bvec4, bvec5, bvec6, bvec7, bvec8, bvec9, bvec10, bvec11, bvec12, bvec13, bvec14, bvec15, cvec0,
			cvec1, cvec2, cvec3, cvec4, cvec5, cvec6, cvec7, cvec8, cvec9, cvec10, cvec11, cvec12, cvec13, cvec14,
			cvec15, dvec0, dvec1, dvec2, dvec3, dvec4, dvec5, dvec6, dvec7, dvec8, dvec9, dvec10, dvec11, dvec12,
			dvec13, dvec14, dvec15);
	}

	if (y_offset + simd_y_len < y_len)
		xy_fused_transpose_part_8<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, y_offset + simd_y_len);
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 2, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t s, const index_t z,
												const index_t n, const index_t y_len, const diag_bag_t a,
												const diag_bag_t b, const diag_bag_t c, const scratch_bag_t b_scratch)
{
	xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 4, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t s, const index_t z,
												const index_t n, const index_t y_len, const diag_bag_t a,
												const diag_bag_t b, const diag_bag_t c, const scratch_bag_t b_scratch)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	if HWY_LANES_CONSTEXPR (simd_length == 2)
	{
		xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 4)
	{
		xy_fused_transpose_part_4<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 8, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t s, const index_t z,
												const index_t n, const index_t y_len, const diag_bag_t a,
												const diag_bag_t b, const diag_bag_t c, const scratch_bag_t b_scratch)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	if HWY_LANES_CONSTEXPR (simd_length == 2)
	{
		xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 4)
	{
		xy_fused_transpose_part_4<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 8)
	{
		xy_fused_transpose_part_8<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t,
		  std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) >= 16, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t s, const index_t z,
												const index_t n, const index_t y_len, const diag_bag_t a,
												const diag_bag_t b, const diag_bag_t c, const scratch_bag_t b_scratch)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	if HWY_LANES_CONSTEXPR (simd_length == 2)
	{
		xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 4)
	{
		xy_fused_transpose_part_4<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 8)
	{
		xy_fused_transpose_part_8<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 16)
	{
		xy_fused_transpose_part_16<multi, real_t>(d, s, z, n, y_len, a, b, c, b_scratch, 0);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l>
constexpr static void synchronize_x_blocked_distributed_remainder(
	real_t** __restrict__ densities, real_t** __restrict__ a_data, real_t** __restrict__ c_data,
	const density_layout_t dens_l, const diagonal_layout_t diag_l, const thread_distribution_l dist_l, const index_t n,
	const index_t n_alignment, const index_t y_begin, const index_t y_end, const index_t coop_size)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);

	const index_t block_size = n / coop_size;

	const auto ddiag_l = diag_l ^ noarr::fix<'Y'>(y_begin / simd_length);
	const auto ddesn_l = dens_l ^ noarr::slice<'y'>(y_begin, y_end - y_begin);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_simd_begin << " block_end: " << x_simd_end
	// 			  << " block_size: " << block_size_x << std::endl;

	auto get_i = [block_size, n, n_alignment, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		const auto actual_block_size_aligned = (actual_block_size + n_alignment - 1) / n_alignment * n_alignment;

		return std::make_tuple(block_idx, noarr::set_length<'x'>(actual_block_size_aligned) ^ noarr::fix<'x'>(offset),
							   noarr::set_length<'x'>(actual_block_size) ^ noarr::fix<'x'>(offset));
	};

	for (index_t y = 0; y < y_end - y_begin; y++)
	{
		real_t prev_c;
		real_t prev_d;

		{
			const auto [prev_block_idx, fix_dens, fix_diag] = get_i(0);
			const auto prev_c_bag =
				noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, prev_block_idx));
			const auto prev_d_bag =
				noarr::make_bag(ddesn_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, prev_block_idx));

			prev_c = prev_c_bag.template at<'y'>(y);
			prev_d = prev_d_bag.template at<'y'>(y);
		}

		for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
		{
			const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx);

			const auto a = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(a_data, block_idx));
			const auto c = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
			const auto d = noarr::make_bag(ddesn_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

			real_t curr_a = a.template at<'y'>(y);
			real_t curr_c = c.template at<'y'>(y);
			real_t curr_d = d.template at<'y'>(y);

			real_t r = 1 / (1 - prev_c * curr_a);

			curr_d = r * (curr_d - prev_d * curr_a);
			curr_c = r * curr_c;

			c.template at<'y'>(y) = curr_c;
			d.template at<'y'>(y) = curr_d;

			prev_c = curr_c;
			prev_d = curr_d;

			// #pragma omp critical
			// 				{
			// 					for (index_t l = 0; l < simd_length; l++)
			// 						std::cout << "mb " << z << " " << y + l << " " << equation_idx << " "
			// 								  << hn::ExtractLane(curr_a, l) << " " << hn::ExtractLane(r, l) << " "
			// 								  << hn::ExtractLane(curr_d, l) << std::endl;
			// 				}
		}

		for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx);

			const auto c = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
			const auto d = noarr::make_bag(ddesn_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

			real_t curr_c = c.template at<'y'>(y);
			real_t curr_d = d.template at<'y'>(y);

			curr_d = curr_d - prev_d * curr_c;

			d.template at<'y'>(y) = curr_d;

			prev_d = curr_d;

			// #pragma omp critical
			// 				{
			// 					for (index_t l = 0; l < simd_length; l++)
			// 						std::cout << "mf " << z << " " << y + l << " " << equation_idx << " "
			// 								  << hn::ExtractLane(curr_c, l) << " " << hn::ExtractLane(curr_d, l) <<
			// std::endl;
			// 				}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_x_blocked_distributed(real_t** __restrict__ densities, real_t** __restrict__ a_data,
														real_t** __restrict__ c_data, const density_layout_t dens_l,
														const diagonal_layout_t diag_l,
														const thread_distribution_l dist_l, const index_t n,
														const index_t n_alignment, const index_t z_begin,
														const index_t z_end, const index_t sync_step, const index_t tid,
														const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	auto ddens_l = dens_l ^ noarr::slice<'z'>(z_begin, z_end - z_begin) ^ noarr::merge_blocks<'z', 'y'>();
	auto ddiag_l = diag_l ^ noarr::fix<'Z'>(z_begin / sync_step);

	const index_t y_len = ddens_l | noarr::get_length<'y'>();
	const index_t simd_y_len = y_len / simd_length;

	const index_t block_size = n / coop_size;

	const index_t y_work = simd_y_len + 1;
	const index_t block_size_y = y_work / coop_size;
	const index_t t_y_begin = tid * block_size_y + std::min(tid, y_work % coop_size);
	const index_t t_y_end = t_y_begin + block_size_y + ((tid < y_work % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_simd_begin << " block_end: " << x_simd_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, n_alignment, coop_size](index_t equation_idx, index_t y) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);

		const auto transposed_y_offset = offset % simd_length;
		const auto transposed_x = offset - transposed_y_offset;

		const auto actual_block_size_aligned = (actual_block_size + n_alignment - 1) / n_alignment * n_alignment;

		return std::make_tuple(block_idx,
							   noarr::set_length<'x'>(actual_block_size_aligned)
								   ^ noarr::fix<'x', 'y'>(transposed_x, y * simd_length + transposed_y_offset),
							   noarr::set_length<'x'>(actual_block_size)
								   ^ noarr::fix<'x', 'Y', 'y'>(offset, y, noarr::lit<0>));
	};

	for (index_t y = t_y_begin; y < std::min(t_y_end, simd_y_len); y++)
	{
		simd_t prev_c;
		simd_t prev_d;

		{
			const auto [prev_block_idx, fix_dens, fix_diag] = get_i(0, y);
			const auto prev_c_bag =
				noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, prev_block_idx));
			const auto prev_d_bag =
				noarr::make_bag(ddens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, prev_block_idx));

			prev_c = hn::Load(t, &prev_c_bag.template at<>());
			prev_d = hn::Load(t, &prev_d_bag.template at<>());
		}

		for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
		{
			const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx, y);

			const auto a = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(a_data, block_idx));
			const auto c = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
			const auto d = noarr::make_bag(ddens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

			simd_t curr_a = hn::Load(t, &a.template at<>());
			simd_t curr_c = hn::Load(t, &c.template at<>());
			simd_t curr_d = hn::Load(t, &d.template at<>());

			simd_t r = hn::Div(hn::Set(t, 1), hn::NegMulAdd(prev_c, curr_a, hn::Set(t, 1)));

			curr_d = hn::Mul(r, hn::NegMulAdd(prev_d, curr_a, curr_d));
			curr_c = hn::Mul(r, curr_c);

			hn::Store(curr_c, t, &c.template at<>());
			hn::Store(curr_d, t, &d.template at<>());

			prev_c = curr_c;
			prev_d = curr_d;

			// #pragma omp critical
			// 				{
			// 					for (index_t l = 0; l < simd_length; l++)
			// 						std::cout << "mb " << z << " " << y + l << " " << equation_idx << " "
			// 								  << hn::ExtractLane(curr_a, l) << " " << hn::ExtractLane(r, l) << " "
			// 								  << hn::ExtractLane(curr_d, l) << std::endl;
			// 				}
		}

		for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx, y);

			const auto c = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
			const auto d = noarr::make_bag(ddens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

			simd_t curr_c = hn::Load(t, &c.template at<>());
			simd_t curr_d = hn::Load(t, &d.template at<>());

			curr_d = hn::NegMulAdd(prev_d, curr_c, curr_d);

			hn::Store(curr_d, t, &d.template at<>());

			prev_d = curr_d;

			// #pragma omp critical
			// 				{
			// 					for (index_t l = 0; l < simd_length; l++)
			// 						std::cout << "mf " << z << " " << y + l << " " << equation_idx << " "
			// 								  << hn::ExtractLane(curr_c, l) << " " << hn::ExtractLane(curr_d, l) <<
			// std::endl;
			// 				}
		}
	}

	if (t_y_end == y_work && t_y_begin < t_y_end)
		synchronize_x_blocked_distributed_remainder(densities, a_data, c_data, ddens_l, ddiag_l, dist_l, n, n_alignment,
													simd_y_len * simd_length, y_len, coop_size);

	barrier.arrive_and_wait();
}

template <bool begin, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_block_x_remainder(real_t* __restrict__ densities, const real_t* __restrict__ a,
									const real_t* __restrict__ b, const real_t* __restrict__ c,
									real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
									const density_layout_t dens_l, const diagonal_layout_t diag_l,
									const scratch_layout_t scratch_l, const index_t n, const index_t y_begin,
									const index_t y_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);

	auto diag_fix = noarr::fix<'Y'>(y_begin / simd_length);

	auto a_scratch_bag = noarr::make_bag(scratch_l ^ diag_fix, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l ^ diag_fix, c_scratch);

	auto a_bag = noarr::make_bag(diag_l ^ diag_fix, a);
	auto b_bag = noarr::make_bag(diag_l ^ diag_fix, b);
	auto c_bag = noarr::make_bag(diag_l ^ diag_fix, c);
	auto d_bag = noarr::make_bag(dens_l ^ noarr::slice<'y'>(y_begin, y_end - y_begin), densities);

	if constexpr (begin)
	{
		for (index_t y = 0; y < y_end - y_begin; y++)
		{
			// Normalize the first and the second equation
			for (index_t i = 0; i < 2; i++)
			{
				const auto idx = noarr::idx<'y', 'x'>(y, i);

				const auto r = 1 / b_bag[idx];

				a_scratch_bag[idx] = a_bag[idx] * r;
				c_scratch_bag[idx] = c_bag[idx] * r;
				d_bag[idx] = d_bag[idx] * r;

				// #pragma omp critical
				// 				std::cout << "f0: " << z_begin + z << " " << y_begin + i << " " << x << " " <<
				// d_bag[idx] << " "
				// 						  << b_bag[idx] << std::endl;
			}

			// Process the lower diagonal (forward)
			for (index_t i = 2; i < n; i++)
			{
				const auto prev_idx = noarr::idx<'y', 'x'>(y, i - 1);
				const auto idx = noarr::idx<'y', 'x'>(y, i);

				const auto r = 1 / (b_bag[idx] - a_bag[idx] * c_scratch_bag[prev_idx]);

				a_scratch_bag[idx] = r * (0 - a_bag[idx] * a_scratch_bag[prev_idx]);
				c_scratch_bag[idx] = r * c_bag[idx];

				d_bag[idx] = r * (d_bag[idx] - a_bag[idx] * d_bag[prev_idx]);


				// #pragma omp critical
				// 				std::cout << "f1: " << z_begin + z << " " << i + y_begin << " " << x << " " <<
				// d_bag[idx] << " "
				// 						  << a_bag[idx] << " " << b_bag[idx] << " " << c_scratch_bag[idx] << std::endl;
			}

			// Process the upper diagonal (backward)
			for (index_t i = n - 3; i >= 1; i--)
			{
				const auto idx = noarr::idx<'y', 'x'>(y, i);
				const auto next_idx = noarr::idx<'y', 'x'>(y, i + 1);

				d_bag[idx] = d_bag[idx] - c_scratch_bag[idx] * d_bag[next_idx];

				a_scratch_bag[idx] = a_scratch_bag[idx] - c_scratch_bag[idx] * a_scratch_bag[next_idx];
				c_scratch_bag[idx] = 0 - c_scratch_bag[idx] * c_scratch_bag[next_idx];


				// #pragma omp critical
				// 				std::cout << "b0: " << z_begin + z << " " << i + y_begin << " " << x << " " <<
				// d_bag[idx] << std::endl;
			}

			// Process the first row (backward)
			{
				const auto idx = noarr::idx<'y', 'x'>(y, 0);
				const auto next_idx = noarr::idx<'y', 'x'>(y, 1);

				const auto r = 1 / (1 - c_scratch_bag[idx] * a_scratch_bag[next_idx]);

				d_bag[idx] = r * (d_bag[idx] - c_scratch_bag[idx] * d_bag[next_idx]);

				a_scratch_bag[idx] = r * a_scratch_bag[idx];
				c_scratch_bag[idx] = r * (0 - c_scratch_bag[idx] * c_scratch_bag[next_idx]);


				// #pragma omp critical
				// 			std::cout << "b1: " << z_begin + z << " " << y_begin << " " << x << " " << d_bag[idx] <<
				// std::endl;
			}
		}
	}
	else
	{
		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		for (index_t y = 0; y < y_end - y_begin; y++)
		{
			for (index_t i = 1; i < n - 1; i++)
			{
				const auto idx_begin = noarr::idx<'y', 'x'>(y, 0);
				const auto idx = noarr::idx<'y', 'x'>(y, i);
				const auto idx_end = noarr::idx<'y', 'x'>(y, n - 1);

				d_bag[idx] = d_bag[idx] - a_scratch_bag[idx] * d_bag[idx_begin] - c_scratch_bag[idx] * d_bag[idx_end];

				// #pragma omp critical
				// 						std::cout << "l: " << z_begin +z << " " << i << " " << x << " "
				// 								  << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << " " <<
				// a[state] << " " << c[state]
				// 								  << std::endl;
			}
		}
	}
}

template <bool begin, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_block_x_remainder_alt(real_t* __restrict__ densities, const real_t* __restrict__ a,
										const real_t* __restrict__ b, const real_t* __restrict__ c,
										real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
										const density_layout_t dens_l, const diagonal_layout_t diag_l,
										const scratch_layout_t scratch_l, const index_t n, const index_t y_begin,
										const index_t y_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);

	auto diag_fix = noarr::fix<'Y'>(y_begin / simd_length);

	auto a_scratch_bag = noarr::make_bag(scratch_l ^ diag_fix, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l ^ diag_fix, c_scratch);

	auto a_bag = noarr::make_bag(diag_l ^ diag_fix, a);
	auto b_bag = noarr::make_bag(diag_l ^ diag_fix, b);
	auto c_bag = noarr::make_bag(diag_l ^ diag_fix, c);
	auto d_bag = noarr::make_bag(dens_l ^ noarr::slice<'y'>(y_begin, y_end - y_begin), densities);

	if constexpr (begin)
	{
		for (index_t y = 0; y < y_end - y_begin; y++)
		{
			// Normalize the first and the second equation
			for (index_t i = 0; i < 2; i++)
			{
				const auto idx = noarr::idx<'y', 'x'>(y, i);

				const auto r = 1 / b_bag[idx];

				a_scratch_bag[idx] = a_bag[idx] * r;
				c_scratch_bag[idx] = c_bag[idx] * r;
				d_bag[idx] = d_bag[idx] * r;

				if (i != 0 && i != n - 1)
				{
					const auto idx0 = noarr::idx<'y', 'x'>(y, 0);

					real_t r0 = 1 / (1 - c_scratch_bag[idx0] * a_scratch_bag[idx]);

					d_bag[idx0] = r0 * (d_bag[idx0] - c_scratch_bag[idx0] * d_bag[idx]);
					a_scratch_bag[idx0] = r0 * a_scratch_bag[idx0];
					c_scratch_bag[idx0] = r0 * (0 - c_scratch_bag[idx0] * c_scratch_bag[idx]);
				}

				// #pragma omp critical
				// 				std::cout << "f0: " << z_begin + z << " " << y_begin + i << " " << x << " " <<
				// d_bag[idx] << " "
				// 						  << b_bag[idx] << std::endl;
			}

			// Process the lower diagonal (forward)
			for (index_t i = 2; i < n; i++)
			{
				const auto prev_idx = noarr::idx<'y', 'x'>(y, i - 1);
				const auto idx = noarr::idx<'y', 'x'>(y, i);

				const auto r = 1 / (b_bag[idx] - a_bag[idx] * c_scratch_bag[prev_idx]);

				a_scratch_bag[idx] = r * (0 - a_bag[idx] * a_scratch_bag[prev_idx]);
				c_scratch_bag[idx] = r * c_bag[idx];

				d_bag[idx] = r * (d_bag[idx] - a_bag[idx] * d_bag[prev_idx]);

				if (i != 0 && i != n - 1)
				{
					const auto idx0 = noarr::idx<'y', 'x'>(y, 0);

					real_t r0 = 1 / (1 - c_scratch_bag[idx0] * a_scratch_bag[idx]);

					d_bag[idx0] = r0 * (d_bag[idx0] - c_scratch_bag[idx0] * d_bag[idx]);
					a_scratch_bag[idx0] = r0 * a_scratch_bag[idx0];
					c_scratch_bag[idx0] = r0 * (0 - c_scratch_bag[idx0] * c_scratch_bag[idx]);
				}

				// #pragma omp critical
				// 				std::cout << "f1: " << z_begin + z << " " << i + y_begin << " " << x << " " <<
				// d_bag[idx] << " "
				// 						  << a_bag[idx] << " " << b_bag[idx] << " " << c_scratch_bag[idx] << std::endl;
			}
		}
	}
	else
	{
		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		for (index_t y = 0; y < y_end - y_begin; y++)
		{
			for (index_t i = n - 2; i >= 1; i--)
			{
				const auto idx_begin = noarr::idx<'y', 'x'>(y, 0);
				const auto idx = noarr::idx<'y', 'x'>(y, i);
				const auto idx_end = noarr::idx<'y', 'x'>(y, i + 1);

				d_bag[idx] = d_bag[idx] - a_scratch_bag[idx] * d_bag[idx_begin] - c_scratch_bag[idx] * d_bag[idx_end];

				// #pragma omp critical
				// 						std::cout << "l: " << z_begin +z << " " << i << " " << x << " "
				// 								  << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << " " <<
				// a[state] << " " << c[state]
				// 								  << std::endl;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename sync_func_t>
static void solve_block_x_transpose(real_t* __restrict__ densities, const real_t* __restrict__ a,
									const real_t* __restrict__ b, const real_t* __restrict__ c,
									real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
									const density_layout_t dens_l, const diagonal_layout_t diag_l,
									const scratch_layout_t scratch_l, const index_t x_begin, const index_t x_end,
									const index_t z_begin, const index_t z_end, const index_t sync_step,
									const index_t s, const index_t, sync_func_t&& synchronize_blocked_x)
{
	const index_t n = x_end - x_begin;

	auto a_scratch_bag = noarr::make_bag(scratch_l ^ noarr::fix<'y'>(noarr::lit<0>), a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l ^ noarr::fix<'y'>(noarr::lit<0>), c_scratch);

	const auto step_len = z_end - z_begin;

	auto ddiag_l = diag_l ^ noarr::fix<'s', 'Z'>(s, z_begin / sync_step);
	auto a_bag = noarr::make_bag(ddiag_l ^ noarr::fix<'y'>(noarr::lit<0>), a);
	auto b_bag = noarr::make_bag(ddiag_l ^ noarr::fix<'y'>(noarr::lit<0>), b);
	auto c_bag = noarr::make_bag(ddiag_l ^ noarr::fix<'y'>(noarr::lit<0>), c);
	auto d_bag = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'z'>(z_begin, step_len)
									 ^ noarr::merge_blocks<'z', 'y'>(),
								 densities);

	const index_t y_len = d_bag | noarr::get_length<'y'>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	simd_t d_rows[simd_length];

	const index_t simd_y_len = y_len / simd_length;

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	for (index_t y = 0; y < simd_y_len; y++)
	{
		// vector registers that hold the to be transposed x*yz plane

		simd_t d_prev = hn::Zero(d);
		simd_t a_scratch_prev = hn::Set(d, -1);
		simd_t c_scratch_prev = hn::Zero(d);

		// forward substitution until last simd_length elements
		for (index_t i = 0; i < full_n - simd_length; i += simd_length)
		{
			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				d_rows[v] = hn::Load(d, &d_bag.template at<'y', 'x'>(y * simd_length + v, i));

			// transposition to enable vectorization
			transpose(d_rows);

			for (index_t v = 0; v < simd_length; v++)
			{
				const index_t x = i + v;

				const auto idx = noarr::idx<'Y', 'x'>(y, x);

				simd_t a_curr = hn::Load(d, &(a_bag[idx]));
				simd_t b_curr = hn::Load(d, &(b_bag[idx]));
				simd_t c_curr = hn::Load(d, &(c_bag[idx]));
				simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
				simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

				{
					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

					a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
					c_scratch_curr = hn::Mul(r, c_curr);
					d_rows[v] = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_rows[v]));

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "f " << z_begin + z << " " << y * simd_length + l
					// << " " << x_begin + x
					// 										  << " " << hn::ExtractLane(a_curr, l) << " " <<
					// hn::ExtractLane(b_curr, l)
					// 										  << " " << hn::ExtractLane(r, l) << " " <<
					// hn::ExtractLane(d_rows[v], l)
					// 										  << std::endl;
					// 						}
				}

				if (x >= 1)
				{
					d_prev = d_rows[v];
					a_scratch_prev = a_scratch_curr;
					c_scratch_prev = c_scratch_curr;
				}
				hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
				hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
			}

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(d_rows[v], d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, i)));
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			for (index_t v = 0; v < simd_length; v++)
				d_rows[v] = hn::Load(d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, full_n - simd_length)));

			// transposition to enable vectorization
			transpose(d_rows);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			for (index_t v = 0; v < remainder_work; v++)
			{
				const index_t x = full_n - simd_length + v;

				const auto idx = noarr::idx<'Y', 'x'>(y, x);

				simd_t a_curr = hn::Load(d, &(a_bag[idx]));
				simd_t b_curr = hn::Load(d, &(b_bag[idx]));
				simd_t c_curr = hn::Load(d, &(c_bag[idx]));
				simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
				simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

				// if (x < 2)
				// {
				// 	simd_t r = hn::Div(hn::Set(d, 1), b_curr);

				// 	a_scratch_curr = hn::Mul(a_curr, r);
				// 	c_scratch_curr = hn::Mul(c_curr, r);
				// 	d_rows[v] = hn::Mul(d_rows[v], r);
				// }
				// else
				{
					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

					a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
					c_scratch_curr = hn::Mul(r, c_curr);
					d_rows[v] = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_rows[v]));

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "f " << z_begin + z << " " << y * simd_length + l
					// << " " << x_begin + x
					// 										  << " " << hn::ExtractLane(a_curr, l) << " " <<
					// hn::ExtractLane(b_curr, l)
					// 										  << " " << hn::ExtractLane(r, l) << " " <<
					// hn::ExtractLane(d_rows[v], l)
					// 										  << std::endl;
					// 						}
				}

				if (x != 0 && x != n - 1)
				{
					d_prev = d_rows[v];
					a_scratch_prev = a_scratch_curr;
					c_scratch_prev = c_scratch_curr;
				}

				hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
				hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
			}

			// the begin of backward part
			for (index_t v = remainder_work - 3; v >= 0; v--)
			{
				const index_t x = full_n - simd_length + v;

				const auto idx = noarr::idx<'Y', 'x'>(y, x);

				simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
				simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

				if (x <= n - 3 && x >= 1)
				{
					d_rows[v] = hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]);

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "b " << z_begin + z << " " << y + l << " " <<
					// x_begin + x << " "
					// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
					// 										  << hn::ExtractLane(c_scratch_curr, l) << " " << 1 << "
					// "
					// 										  << hn::ExtractLane(d_rows[v], l) << std::endl;
					// 						}

					a_scratch_curr = hn::NegMulAdd(c_scratch_curr, a_scratch_prev, a_scratch_curr);
					c_scratch_curr = hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0));
				}
				else if (x == 0)
				{
					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_scratch_curr, a_scratch_prev, hn::Set(d, 1)));

					d_rows[v] = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]));

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "b " << z_begin + z << " " << y + l << " " <<
					// x_begin + x << " "
					// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
					// 										  << hn::ExtractLane(c_scratch_curr, l) << " " <<
					// hn::ExtractLane(r, l) << " "
					// 										  << hn::ExtractLane(d_rows[v], l) << std::endl;
					// 						}

					a_scratch_curr = hn::Mul(r, a_scratch_curr);
					c_scratch_curr = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0)));
				}

				d_prev = d_rows[v];
				a_scratch_prev = a_scratch_curr;
				c_scratch_prev = c_scratch_curr;
				hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
				hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
			}

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(d_rows[v], d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, full_n - simd_length)));
		}

		// we continue with backwards substitution
		for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
		{
			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				d_rows[v] = hn::Load(d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, i)));

			// backward propagation
			for (index_t v = simd_length - 1; v >= 0; v--)
			{
				const index_t x = i + v;

				const auto idx = noarr::idx<'Y', 'x'>(y, x);

				simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
				simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

				if (x <= n - 3 && x >= 1)
				{
					d_rows[v] = hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]);

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "b " << z_begin + z << " " << y + l << " " <<
					// x_begin + x << " "
					// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
					// 										  << hn::ExtractLane(c_scratch_curr, l) << " " << 1 << "
					// "
					// 										  << hn::ExtractLane(d_rows[v], l) << std::endl;
					// 						}

					a_scratch_curr = hn::NegMulAdd(c_scratch_curr, a_scratch_prev, a_scratch_curr);
					c_scratch_curr = hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0));
				}
				else if (x == 0)
				{
					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_scratch_curr, a_scratch_prev, hn::Set(d, 1)));
					d_rows[v] = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]));

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "b " << z_begin + z << " " << y + l << " " <<
					// x_begin + x << " "
					// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
					// 										  << hn::ExtractLane(c_scratch_curr, l) << " " <<
					// hn::ExtractLane(r, l) << " "
					// 										  << hn::ExtractLane(d_rows[v], l) << std::endl;
					// 						}

					a_scratch_curr = hn::Mul(r, a_scratch_curr);
					c_scratch_curr = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0)));
				}

				d_prev = d_rows[v];
				a_scratch_prev = a_scratch_curr;
				c_scratch_prev = c_scratch_curr;
				hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
				hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
			}

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(d_rows[v], d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, i)));
		}
	}

	solve_block_x_remainder<true>(densities, a, b, c, a_scratch, c_scratch, d_bag.structure(), ddiag_l, scratch_l, n,
								  simd_y_len * simd_length, y_len);

	synchronize_blocked_x();

	for (index_t y = 0; y < simd_y_len; y++)
	{
		const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

		const simd_t begin_unknowns = hn::Load(d, &(d_bag.template at<'y', 'x'>(y * simd_length, 0)));

		const auto transposed_y_offset = (n - 1) % simd_length;
		const auto transposed_x = (n - 1) - transposed_y_offset;

		const simd_t end_unknowns =
			hn::Load(d, &(d_bag.template at<'y', 'x'>(y * simd_length + transposed_y_offset, transposed_x)));

		for (index_t i = 0; i < full_n; i += simd_length)
		{
			for (index_t v = 0; v < simd_length; v++)
				d_rows[v] = hn::Load(d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, i)));

			for (index_t v = 0; v < simd_length; v++)
			{
				index_t x = i + v;

				const auto idx = noarr::idx<'Y', 'x'>(y, x);

				if (x > 0 && x < n - 1)
				{
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

					d_rows[v] = hn::NegMulAdd(a_scratch_curr, begin_unknowns, d_rows[v]);
					d_rows[v] = hn::NegMulAdd(c_scratch_curr, end_unknowns, d_rows[v]);

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "e " << z_begin + z << " " << y + l << " " <<
					// x_begin + x << " "
					// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
					// 										  << hn::ExtractLane(c_scratch_curr, l) << " " <<
					// hn::ExtractLane(d_rows[v], l)
					// 										  << std::endl;
					// 						}
				}
			}

			transpose(d_rows);

			for (index_t v = 0; v < simd_length; v++)
				hn::Store(d_rows[v], d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, i)));
		}
	}

	solve_block_x_remainder<false>(densities, a, b, c, a_scratch, c_scratch, d_bag.structure(), ddiag_l, scratch_l, n,
								   simd_y_len * simd_length, y_len);
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename sync_func_t>
static void solve_block_x_transpose_alt(real_t* __restrict__ densities, const real_t* __restrict__ a,
										const real_t* __restrict__ b, const real_t* __restrict__ c,
										real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
										const density_layout_t dens_l, const diagonal_layout_t diag_l,
										const scratch_layout_t scratch_l, const index_t x_begin, const index_t x_end,
										const index_t z_begin, const index_t z_end, const index_t sync_step,
										const index_t s, const index_t, sync_func_t&& synchronize_blocked_x)
{
	const index_t n = x_end - x_begin;

	auto a_scratch_bag = noarr::make_bag(scratch_l ^ noarr::fix<'y'>(noarr::lit<0>), a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l ^ noarr::fix<'y'>(noarr::lit<0>), c_scratch);

	const auto step_len = z_end - z_begin;

	auto ddiag_l = diag_l ^ noarr::fix<'s', 'Z'>(s, z_begin / sync_step);
	auto a_bag = noarr::make_bag(ddiag_l ^ noarr::fix<'y'>(noarr::lit<0>), a);
	auto b_bag = noarr::make_bag(ddiag_l ^ noarr::fix<'y'>(noarr::lit<0>), b);
	auto c_bag = noarr::make_bag(ddiag_l ^ noarr::fix<'y'>(noarr::lit<0>), c);
	auto d_bag = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'z'>(z_begin, step_len)
									 ^ noarr::merge_blocks<'z', 'y'>(),
								 densities);

	const index_t y_len = d_bag | noarr::get_length<'y'>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	simd_t d_rows[simd_length];

	const index_t simd_y_len = y_len / simd_length;

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	for (index_t y = 0; y < simd_y_len; y++)
	{
		// vector registers that hold the to be transposed x*yz plane

		simd_t d_prev = hn::Zero(d);
		simd_t a_scratch_prev = hn::Set(d, -1);
		simd_t c_scratch_prev = hn::Zero(d);


		const auto idx0 = noarr::idx<'Y', 'x'>(y, noarr::lit<0>);
		simd_t a_scratch_0;
		simd_t c_scratch_0;
		simd_t d_0;

		// forward substitution until last simd_length elements
		for (index_t i = 0; i < full_n - simd_length; i += simd_length)
		{
			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				d_rows[v] = hn::Load(d, &d_bag.template at<'y', 'x'>(y * simd_length + v, i));

			// transposition to enable vectorization
			transpose(d_rows);

			for (index_t v = 0; v < simd_length; v++)
			{
				const index_t x = i + v;

				const auto idx = noarr::idx<'Y', 'x'>(y, x);

				simd_t a_curr = hn::Load(d, &(a_bag[idx]));
				simd_t b_curr = hn::Load(d, &(b_bag[idx]));
				simd_t c_curr = hn::Load(d, &(c_bag[idx]));
				simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
				simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

				{
					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

					a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
					c_scratch_curr = hn::Mul(r, c_curr);
					d_rows[v] = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_rows[v]));

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "f " << z_begin + z << " " << y * simd_length + l
					// << " " << x_begin + x
					// 										  << " " << hn::ExtractLane(a_curr, l) << " " <<
					// hn::ExtractLane(b_curr, l)
					// 										  << " " << hn::ExtractLane(r, l) << " " <<
					// hn::ExtractLane(d_rows[v], l)
					// 										  << std::endl;
					// 						}
				}

				if (x != 0 && x != n - 1)
				{
					simd_t r0 = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_scratch_curr, c_scratch_0, hn::Set(d, 1)));

					d_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, d_rows[v], d_0));
					a_scratch_0 = hn::Mul(r0, a_scratch_0);
					c_scratch_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, c_scratch_curr, hn::Set(d, 0)));

					hn::Store(a_scratch_0, d, &a_scratch_bag[idx0]);
					hn::Store(c_scratch_0, d, &c_scratch_bag[idx0]);
				}

				if (x == 0)
				{
					d_0 = d_rows[0];
					a_scratch_0 = a_scratch_curr;
					c_scratch_0 = c_scratch_curr;
				}

				if (x >= 1)
				{
					d_prev = d_rows[v];
					a_scratch_prev = a_scratch_curr;
					c_scratch_prev = c_scratch_curr;
				}
				hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
				hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
			}

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(d_rows[v], d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, i)));
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			for (index_t v = 0; v < simd_length; v++)
				d_rows[v] = hn::Load(d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, full_n - simd_length)));

			// transposition to enable vectorization
			transpose(d_rows);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			for (index_t v = 0; v < remainder_work; v++)
			{
				const index_t x = full_n - simd_length + v;

				const auto idx = noarr::idx<'Y', 'x'>(y, x);

				simd_t a_curr = hn::Load(d, &(a_bag[idx]));
				simd_t b_curr = hn::Load(d, &(b_bag[idx]));
				simd_t c_curr = hn::Load(d, &(c_bag[idx]));
				simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
				simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

				// if (x < 2)
				// {
				// 	simd_t r = hn::Div(hn::Set(d, 1), b_curr);

				// 	a_scratch_curr = hn::Mul(a_curr, r);
				// 	c_scratch_curr = hn::Mul(c_curr, r);
				// 	d_rows[v] = hn::Mul(d_rows[v], r);
				// }
				// else
				{
					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

					a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
					c_scratch_curr = hn::Mul(r, c_curr);
					d_rows[v] = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_rows[v]));

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "f " << z_begin + z << " " << y * simd_length + l
					// << " " << x_begin + x
					// 										  << " " << hn::ExtractLane(a_curr, l) << " " <<
					// hn::ExtractLane(b_curr, l)
					// 										  << " " << hn::ExtractLane(r, l) << " " <<
					// hn::ExtractLane(d_rows[v], l)
					// 										  << std::endl;
					// 						}
				}

				if (x != 0 && x != n - 1)
				{
					simd_t r0 = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_scratch_curr, c_scratch_0, hn::Set(d, 1)));

					d_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, d_rows[v], d_0));
					a_scratch_0 = hn::Mul(r0, a_scratch_0);
					c_scratch_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, c_scratch_curr, hn::Set(d, 0)));

					hn::Store(a_scratch_0, d, &a_scratch_bag[idx0]);
					hn::Store(c_scratch_0, d, &c_scratch_bag[idx0]);
				}

				if (x == 0)
				{
					d_0 = d_rows[0];
					a_scratch_0 = a_scratch_curr;
					c_scratch_0 = c_scratch_curr;
				}

				if (x != 0 && x != n - 1)
				{
					d_prev = d_rows[v];
					a_scratch_prev = a_scratch_curr;
					c_scratch_prev = c_scratch_curr;
				}

				hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
				hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
			}

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(d_rows[v], d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, full_n - simd_length)));
		}

		hn::Store(d_0, d, &(d_bag.template at<'y', 'x'>(y * simd_length, 0)));
		hn::Store(a_scratch_0, d, &a_scratch_bag[idx0]);
		hn::Store(c_scratch_0, d, &c_scratch_bag[idx0]);
	}

	solve_block_x_remainder_alt<true>(densities, a, b, c, a_scratch, c_scratch, d_bag.structure(), ddiag_l, scratch_l,
									  n, simd_y_len * simd_length, y_len);

	synchronize_blocked_x();

	for (index_t y = 0; y < simd_y_len; y++)
	{
		const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

		const simd_t begin_unknowns = hn::Load(d, &(d_bag.template at<'y', 'x'>(y * simd_length, 0)));
		simd_t d_prev;

		for (index_t i = full_n - simd_length; i >= 0; i -= simd_length)
		{
			for (index_t v = 0; v < simd_length; v++)
				d_rows[v] = hn::Load(d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, i)));

			for (index_t v = simd_length - 1; v >= 0; v--)
			{
				index_t x = i + v;

				const auto idx = noarr::idx<'Y', 'x'>(y, x);

				if (x > 0 && x < n - 1)
				{
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

					d_rows[v] = hn::NegMulAdd(a_scratch_curr, begin_unknowns, d_rows[v]);
					d_rows[v] = hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]);

					// #pragma omp critical
					// 						{
					// 							for (index_t l = 0; l < simd_length; l++)
					// 								std::cout << "e " << z_begin + z << " " << y + l << " " <<
					// x_begin + x << " "
					// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
					// 										  << hn::ExtractLane(c_scratch_curr, l) << " " <<
					// hn::ExtractLane(d_rows[v], l)
					// 										  << std::endl;
					// 						}
				}
				d_prev = d_rows[v];
			}

			transpose(d_rows);

			for (index_t v = 0; v < simd_length; v++)
				hn::Store(d_rows[v], d, &(d_bag.template at<'y', 'x'>(y * simd_length + v, i)));
		}
	}

	solve_block_x_remainder_alt<false>(densities, a, b, c, a_scratch, c_scratch, d_bag.structure(), ddiag_l, scratch_l,
									   n, simd_y_len * simd_length, y_len);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_x_2d_and_3d_transpose_l(real_t* __restrict__ densities, const real_t* __restrict__ a,
												const real_t* __restrict__ b, const real_t* __restrict__ c,
												real_t* __restrict__ b_scratch, const density_layout_t dens_l,
												const diagonal_layout_t diag_l, const scratch_layout_t scratch_l,
												const index_t s, const index_t z_begin, const index_t z_end, index_t n)
{
	const auto a_bag = noarr::make_bag(diag_l, a);
	const auto b_bag = noarr::make_bag(diag_l, b);
	const auto c_bag = noarr::make_bag(diag_l, c);

	const auto d_bag = noarr::make_bag(dens_l, densities);

	const auto b_scratch_bag = noarr::make_bag(scratch_l, b_scratch);

	const index_t z_len = z_end - z_begin;
	const index_t y_len = z_len * (dens_l | noarr::get_length<'y'>());

	xy_fused_transpose_part_dispatch<false, real_t>(d_bag, s, z_begin, n, y_len, a_bag, b_bag, c_bag, b_scratch_bag);
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_y_blocked_distributed(real_t** __restrict__ densities, real_t** __restrict__ a_data,
														real_t** __restrict__ c_data, const density_layout_t dens_l,
														const scratch_layout_t scratch_l,
														const thread_distribution_l dist_l, const index_t n,
														const index_t z_begin, const index_t z_end, const index_t tid,
														const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto sscratch_l = scratch_l ^ noarr::merge_blocks<'X', 'x'>();

	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t z_len = z_end - z_begin;
	const index_t x_simd_len = (x_len + simd_length - 1) / simd_length;

	const index_t block_size = n / coop_size;

	const index_t work_len = z_len * x_simd_len;
	const index_t block_size_z = work_len / coop_size;
	const index_t work_begin = tid * block_size_z + std::min(tid, work_len % coop_size);
	const index_t work_end = work_begin + block_size_z + ((tid < work_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_simd_begin << " block_end: " << x_simd_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		return std::make_tuple(block_idx, noarr::set_length<'y'>(actual_block_size) ^ noarr::fix<'y'>(offset));
	};

	for (index_t w_idx = work_begin; w_idx < work_end; w_idx++)
	{
		const index_t z = z_begin + w_idx / x_simd_len;
		const index_t x = (w_idx % x_simd_len) * simd_length;

		simd_t prev_c;
		simd_t prev_d;

		{
			const auto [prev_block_idx, fix_l] = get_i(0);
			const auto prev_c_bag =
				noarr::make_bag(sscratch_l ^ fix_l, dist_l | noarr::get_at<'y'>(c_data, prev_block_idx));
			const auto prev_d_bag =
				noarr::make_bag(dens_l ^ fix_l, dist_l | noarr::get_at<'y'>(densities, prev_block_idx));

			prev_c = hn::Load(t, &prev_c_bag.template at<'x', 'z'>(x, z - z_begin));
			prev_d = hn::Load(t, &prev_d_bag.template at<'x', 'z'>(x, z));
		}

		for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
		{
			const auto [block_idx, fix_l] = get_i(equation_idx);

			const auto a = noarr::make_bag(sscratch_l ^ fix_l, dist_l | noarr::get_at<'y'>(a_data, block_idx));
			const auto c = noarr::make_bag(sscratch_l ^ fix_l, dist_l | noarr::get_at<'y'>(c_data, block_idx));
			const auto d = noarr::make_bag(dens_l ^ fix_l, dist_l | noarr::get_at<'y'>(densities, block_idx));

			simd_t curr_a = hn::Load(t, &a.template at<'x', 'z'>(x, z - z_begin));
			simd_t curr_c = hn::Load(t, &c.template at<'x', 'z'>(x, z - z_begin));
			simd_t curr_d = hn::Load(t, &d.template at<'x', 'z'>(x, z));

			simd_t r = hn::Div(hn::Set(t, 1), hn::NegMulAdd(prev_c, curr_a, hn::Set(t, 1)));

			curr_d = hn::Mul(r, hn::NegMulAdd(prev_d, curr_a, curr_d));
			curr_c = hn::Mul(r, curr_c);

			hn::Store(curr_c, t, &c.template at<'x', 'z'>(x, z - z_begin));
			hn::Store(curr_d, t, &d.template at<'x', 'z'>(x, z));

			prev_c = curr_c;
			prev_d = curr_d;

			// #pragma omp critical
			// 			std::cout << "mf " << z << " " << i << " " << x << " rf: " << rf[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}

		for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto [block_idx, fix_l] = get_i(equation_idx);

			const auto c = noarr::make_bag(sscratch_l ^ fix_l, dist_l | noarr::get_at<'y'>(c_data, block_idx));
			const auto d = noarr::make_bag(dens_l ^ fix_l, dist_l | noarr::get_at<'y'>(densities, block_idx));

			simd_t curr_c = hn::Load(t, &c.template at<'x', 'z'>(x, z - z_begin));
			simd_t curr_d = hn::Load(t, &d.template at<'x', 'z'>(x, z));

			curr_d = hn::NegMulAdd(prev_d, curr_c, curr_d);

			hn::Store(curr_d, t, &d.template at<'x', 'z'>(x, z));

			prev_d = curr_d;

			// #pragma omp critical
			// 			std::cout << "mb " << z << " " << i << " " << x << " b: " << b[state] << " c: " << c[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	barrier.arrive_and_wait();
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename sync_func_t>
static void solve_block_y(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
						  const real_t* __restrict__ c, real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
						  const density_layout_t dens_l, const diagonal_layout_t diag_l,
						  const scratch_layout_t scratch_l, const index_t y_begin, const index_t y_end,
						  const index_t z_begin, const index_t z_end, const index_t s, const index_t x_tile_size,
						  sync_func_t&& synchronize_blocked_y)
{
#define solve_block_y_vec

#ifdef solve_block_y_vec
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;
#endif

	const index_t n = y_end - y_begin;

	const index_t x_block_len = ((dens_l | noarr::get_length<'x'>()) + x_tile_size - 1) / x_tile_size;

	auto remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
	if (remainder == 0)
		remainder = x_tile_size;

	auto a_bag = noarr::make_bag(diag_l, a);
	auto b_bag = noarr::make_bag(diag_l, b);
	auto c_bag = noarr::make_bag(diag_l, c);
	auto d_bag = noarr::make_bag(dens_l ^ noarr::set_length<'y'>(n), densities);
	auto a_scratch_bag = noarr::make_bag(scratch_l, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l, c_scratch);

	for (index_t z = z_begin; z < z_end; z++)
	{
		for (index_t X = 0; X < x_block_len; X++)
		{
			const index_t tile_len = X == x_block_len - 1 ? remainder : x_tile_size;

			for (index_t i = 0; i < 2; i++)
			{
#ifdef solve_block_y_vec
				for (index_t x = 0; x < tile_len; x += simd_length)
#else
				for (index_t x = 0; x < tile_len; x++)
#endif
				{
					auto idx = noarr::idx<'s', 'y'>(s, i);

					auto scratch_idx = noarr::idx<'X', 'x', 'z'>(X, x, z - z_begin);
					auto diag_idx = noarr::idx<'X', 'x', 'z'>(X, x, z);
					auto dens_idx = noarr::idx<'x', 'z'>(X * x_tile_size + x, z);

#ifdef solve_block_y_vec
					simd_t a_curr = hn::Load(d, &(a_bag[idx & diag_idx]));
					simd_t b_curr = hn::Load(d, &(b_bag[idx & diag_idx]));
					simd_t c_curr = hn::Load(d, &(c_bag[idx & diag_idx]));
					simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);

					simd_t r = hn::Div(hn::Set(d, 1), b_curr);

					a_scratch_curr = hn::Mul(r, a_curr);
					c_scratch_curr = hn::Mul(r, c_curr);
					d_curr = hn::Mul(r, d_curr);

					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx & scratch_idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx & scratch_idx]);
					hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else
					real_t r = 1 / b_bag[idx & diag_idx];

					a_scratch_bag[idx & scratch_idx] = r * a_bag[idx & diag_idx];
					c_scratch_bag[idx & scratch_idx] = r * c_bag[idx & diag_idx];
					d_bag[idx & dens_idx] = r * d_bag[idx & dens_idx];
#endif

					// #pragma omp critical
					// 					std::cout << "f0: " << z_begin + i << " " << blocked_y + y << " " << x << "
					// " << d_bag[idx] << " " << b_bag[idx]
					// 							  << std::endl;
				}
			}

			// Process the lower diagonal (forward)
			for (index_t i = 2; i < n; i++)
			{
#ifdef solve_block_y_vec
				for (index_t x = 0; x < tile_len; x += simd_length)
#else
				for (index_t x = 0; x < tile_len; x++)
#endif
				{
					auto idx = noarr::idx<'s', 'y'>(s, i);
					auto prev_idx = noarr::idx<'s', 'y'>(s, i - 1);

					auto scratch_idx = noarr::idx<'X', 'x', 'z'>(X, x, z - z_begin);
					auto diag_idx = noarr::idx<'X', 'x', 'z'>(X, x, z);
					auto dens_idx = noarr::idx<'x', 'z'>(X * x_tile_size + x, z);

#ifdef solve_block_y_vec
					simd_t a_curr = hn::Load(d, &(a_bag[idx & diag_idx]));
					simd_t b_curr = hn::Load(d, &(b_bag[idx & diag_idx]));
					simd_t c_curr = hn::Load(d, &(c_bag[idx & diag_idx]));
					simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
					simd_t d_prev = hn::Load(d, &(d_bag[prev_idx & dens_idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);
					simd_t a_scratch_prev = hn::Load(d, &a_scratch_bag[prev_idx & scratch_idx]);
					simd_t c_scratch_prev = hn::Load(d, &c_scratch_bag[prev_idx & scratch_idx]);

					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

					a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
					c_scratch_curr = hn::Mul(r, c_curr);
					d_curr = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_curr));

					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx & scratch_idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx & scratch_idx]);
					hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else
					real_t r =
						1 / (b_bag[idx & diag_idx] - a_bag[idx & diag_idx] * c_scratch_bag[prev_idx & scratch_idx]);

					a_scratch_bag[idx & scratch_idx] =
						r * (-a_bag[idx & diag_idx] * a_scratch_bag[prev_idx & scratch_idx]);
					c_scratch_bag[idx & scratch_idx] = r * c_bag[idx & diag_idx];
					d_bag[idx & dens_idx] =
						r * (d_bag[idx & dens_idx] - a_bag[idx & diag_idx] * d_bag[prev_idx & dens_idx]);
#endif
					// #pragma omp critical
					// 			std::cout << "f1: " << z << " " << i + y_begin << " " << x << " " << d_bag[idx]
					// 					  << " " << a_bag[idx]  << " " << b_bag[idx]  << " " <<
					// c_scratch_bag[idx]
					// << std::endl;
				}
			}

			// Process the upper diagonal (backward)
			for (index_t i = n - 3; i >= 1; i--)
			{
#ifdef solve_block_y_vec
				for (index_t x = 0; x < tile_len; x += simd_length)
#else
				for (index_t x = 0; x < tile_len; x++)
#endif
				{
					auto idx = noarr::idx<'s', 'y'>(s, i);
					auto next_idx = noarr::idx<'s', 'y'>(s, i + 1);

					auto scratch_idx = noarr::idx<'X', 'x', 'z'>(X, x, z - z_begin);
					auto dens_idx = noarr::idx<'x', 'z'>(X * x_tile_size + x, z);

#ifdef solve_block_y_vec
					simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
					simd_t d_prev = hn::Load(d, &(d_bag[next_idx & dens_idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);
					simd_t a_scratch_prev = hn::Load(d, &a_scratch_bag[next_idx & scratch_idx]);
					simd_t c_scratch_prev = hn::Load(d, &c_scratch_bag[next_idx & scratch_idx]);

					d_curr = hn::NegMulAdd(c_scratch_curr, d_prev, d_curr);
					a_scratch_curr = hn::NegMulAdd(c_scratch_curr, a_scratch_prev, a_scratch_curr);
					c_scratch_curr = hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0));

					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx & scratch_idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx & scratch_idx]);
					hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else

					d_bag[idx & dens_idx] =
						d_bag[idx & dens_idx] - c_scratch_bag[idx & scratch_idx] * d_bag[next_idx & dens_idx];
					a_scratch_bag[idx & scratch_idx] =
						a_scratch_bag[idx & scratch_idx]
						- c_scratch_bag[idx & scratch_idx] * a_scratch_bag[next_idx & scratch_idx];
					c_scratch_bag[idx & scratch_idx] =
						-c_scratch_bag[idx & scratch_idx] * c_scratch_bag[next_idx & scratch_idx];
#endif
					// #pragma omp critical
					// 			std::cout << "b0: " << z << " " << i + y_begin << " " << x << " " << d_bag[idx] <<
					// std::endl;
				}
			}

			// Process the first row (backward)
			{
#ifdef solve_block_y_vec
				for (index_t x = 0; x < tile_len; x += simd_length)
#else
				for (index_t x = 0; x < tile_len; x++)
#endif
				{
					auto idx = noarr::idx<'s', 'y'>(s, 0);
					auto next_idx = noarr::idx<'s', 'y'>(s, 1);

					auto scratch_idx = noarr::idx<'X', 'x', 'z'>(X, x, z - z_begin);
					auto dens_idx = noarr::idx<'x', 'z'>(X * x_tile_size + x, z);

#ifdef solve_block_y_vec
					simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
					simd_t d_prev = hn::Load(d, &(d_bag[next_idx & dens_idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);
					simd_t a_scratch_prev = hn::Load(d, &a_scratch_bag[next_idx & scratch_idx]);
					simd_t c_scratch_prev = hn::Load(d, &c_scratch_bag[next_idx & scratch_idx]);

					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_scratch_curr, a_scratch_prev, hn::Set(d, 1)));

					d_curr = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, d_prev, d_curr));
					a_scratch_curr = hn::Mul(r, a_scratch_curr);
					c_scratch_curr = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0)));

					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx & scratch_idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx & scratch_idx]);
					hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else
					real_t r = 1 / (1 - c_scratch_bag[idx & scratch_idx] * a_scratch_bag[next_idx & scratch_idx]);

					d_bag[idx & dens_idx] =
						r * (d_bag[idx & dens_idx] - c_scratch_bag[idx & scratch_idx] * d_bag[next_idx & dens_idx]);
					a_scratch_bag[idx & scratch_idx] = r * a_scratch_bag[idx & scratch_idx];
					c_scratch_bag[idx & scratch_idx] =
						r * -c_scratch_bag[idx & scratch_idx] * c_scratch_bag[next_idx & scratch_idx];
#endif
					// #pragma omp critical
					// 			std::cout << "b1: " << z << " " << y_begin << " " << x << " " << d_bag[idx] <<
					// std::endl;
				}
			}
		}
	}

	synchronize_blocked_y(z_begin, z_end);

	for (index_t z = z_begin; z < z_end; z++)
	{
		for (index_t X = 0; X < x_block_len; X++)
		{
			const index_t tile_len = X == x_block_len - 1 ? remainder : x_tile_size;

			for (index_t i = 1; i < n - 1; i++)
			{
#ifdef solve_block_y_vec
				for (index_t x = 0; x < tile_len; x += simd_length)
#else
				for (index_t x = 0; x < tile_len; x++)
#endif
				{
					auto idx_begin = noarr::idx<'s', 'y'>(s, 0);
					auto idx = noarr::idx<'s', 'y'>(s, i);
					auto idx_end = noarr::idx<'s', 'y'>(s, n - 1);

					auto scratch_idx = noarr::idx<'X', 'x', 'z'>(X, x, z - z_begin);
					auto dens_idx = noarr::idx<'x', 'z'>(X * x_tile_size + x, z);

#ifdef solve_block_y_vec
					simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
					simd_t d_begin = hn::Load(d, &(d_bag[idx_begin & dens_idx]));
					simd_t d_end = hn::Load(d, &(d_bag[idx_end & dens_idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);

					d_curr = hn::NegMulAdd(a_scratch_curr, d_begin, d_curr);
					d_curr = hn::NegMulAdd(c_scratch_curr, d_end, d_curr);

					hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else
					d_bag[idx & dens_idx] = d_bag[idx & dens_idx]
											- a_scratch_bag[idx & scratch_idx] * d_bag[idx_begin & dens_idx]
											- c_scratch_bag[idx & scratch_idx] * d_bag[idx_end & dens_idx];
#endif
					// #pragma omp critical
					// 						std::cout << "l: " << z << " " << i << " " << x << " "
					// 								  << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << " " <<
					// a[state] << " " << c[state]
					// 								  << std::endl;
				}
			}
		}
	}
}



template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename sync_func_t>
static void solve_block_y_alt(real_t* __restrict__ densities, const real_t* __restrict__ a,
							  const real_t* __restrict__ b, const real_t* __restrict__ c,
							  real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
							  const density_layout_t dens_l, const diagonal_layout_t diag_l,
							  const scratch_layout_t scratch_l, const index_t y_begin, const index_t y_end,
							  const index_t z_begin, const index_t z_end, const index_t s, const index_t x_len,
							  const index_t x_tile_size, sync_func_t&& synchronize_blocked_y)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s) ^ noarr::set_length<'y'>(y_end - y_begin)
						  ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size)
						  ^ noarr::fix<'b'>(noarr::lit<0>);

	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();
	const index_t y_len = blocked_dens_l | noarr::get_length<'y'>();

	auto remainder = ((x_len + simd_length - 1) / simd_length * simd_length) % x_tile_size;
	if (remainder == 0)
		remainder = x_tile_size;

	auto a_scratch_bag = noarr::make_bag(scratch_l, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l, c_scratch);

	const auto step_len = z_end - z_begin;

	auto a_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'z'>(z_begin, step_len), a);
	auto b_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'z'>(z_begin, step_len), b);
	auto c_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'z'>(z_begin, step_len), c);
	auto d_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), densities);

	for (index_t z = 0; z < step_len; z++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const index_t tile_len = X == X_len - 1 ? remainder : x_tile_size;

			// Normalize the first and the second equation
			for (index_t i = 0; i < 2; i++)
				for (index_t x = 0; x < tile_len; x += simd_length)
				{
					const auto idx = noarr::idx<'z', 'X', 'x', 'y'>(z, X, x, i);

					simd_t a_curr = hn::Load(d, &(a_bag[idx]));
					simd_t b_curr = hn::Load(d, &(b_bag[idx]));
					simd_t c_curr = hn::Load(d, &(c_bag[idx]));
					simd_t d_curr = hn::Load(d, &(d_bag[idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

					simd_t r = hn::Div(hn::Set(d, 1), b_curr);

					a_scratch_curr = hn::Mul(r, a_curr);
					c_scratch_curr = hn::Mul(r, c_curr);
					d_curr = hn::Mul(r, d_curr);

					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
					hn::Store(d_curr, d, &d_bag[idx]);

					if (i != 0 && i != y_len - 1)
					{
						const auto idx0 = noarr::idx<'z', 'X', 'x', 'y'>(z, X, x, noarr::lit<0>);

						simd_t a_scratch_0 = hn::Load(d, &a_scratch_bag[idx0]);
						simd_t c_scratch_0 = hn::Load(d, &c_scratch_bag[idx0]);
						simd_t r0 = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_scratch_curr, c_scratch_0, hn::Set(d, 1)));

						simd_t d_0 = hn::Load(d, &d_bag[idx0]);

						d_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, d_curr, d_0));
						a_scratch_0 = hn::Mul(r0, a_scratch_0);
						c_scratch_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, c_scratch_curr, hn::Set(d, 0)));

						hn::Store(a_scratch_0, d, &a_scratch_bag[idx0]);
						hn::Store(c_scratch_0, d, &c_scratch_bag[idx0]);
						hn::Store(d_0, d, &d_bag[idx0]);
					}

					// #pragma omp critical
					// 				std::cout << "f0: " << z_begin + z << " " << y_begin + i << " " << x << " " <<
					// d_bag[idx] << " "
					// 						  << b_bag[idx] << std::endl;
				}

			// Process the lower diagonal (forward)
			for (index_t i = 2; i < y_len; i++)
				for (index_t x = 0; x < tile_len; x += simd_length)
				{
					const auto prev_idx = noarr::idx<'z', 'X', 'x', 'y'>(z, X, x, i - 1);
					const auto idx = noarr::idx<'z', 'X', 'x', 'y'>(z, X, x, i);

					simd_t a_curr = hn::Load(d, &(a_bag[idx]));
					simd_t b_curr = hn::Load(d, &(b_bag[idx]));
					simd_t c_curr = hn::Load(d, &(c_bag[idx]));
					simd_t d_curr = hn::Load(d, &(d_bag[idx]));
					simd_t d_prev = hn::Load(d, &(d_bag[prev_idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);
					simd_t a_scratch_prev = hn::Load(d, &a_scratch_bag[prev_idx]);
					simd_t c_scratch_prev = hn::Load(d, &c_scratch_bag[prev_idx]);

					simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

					a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
					c_scratch_curr = hn::Mul(r, c_curr);
					d_curr = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_curr));

					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
					hn::Store(d_curr, d, &d_bag[idx]);

					if (i != 0 && i != y_len - 1)
					{
						const auto idx0 = noarr::idx<'z', 'X', 'x', 'y'>(z, X, x, noarr::lit<0>);

						simd_t a_scratch_0 = hn::Load(d, &a_scratch_bag[idx0]);
						simd_t c_scratch_0 = hn::Load(d, &c_scratch_bag[idx0]);
						simd_t r0 = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_scratch_curr, c_scratch_0, hn::Set(d, 1)));

						simd_t d_0 = hn::Load(d, &d_bag[idx0]);

						d_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, d_curr, d_0));
						a_scratch_0 = hn::Mul(r0, a_scratch_0);
						c_scratch_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, c_scratch_curr, hn::Set(d, 0)));

						hn::Store(a_scratch_0, d, &a_scratch_bag[idx0]);
						hn::Store(c_scratch_0, d, &c_scratch_bag[idx0]);
						hn::Store(d_0, d, &d_bag[idx0]);
					}

					// #pragma omp critical
					// 				std::cout << "f1: " << z_begin + z << " " << i + y_begin << " " << x << " " <<
					// d_bag[idx] << " "
					// 						  << a_bag[idx] << " " << b_bag[idx] << " " << c_scratch_bag[idx] <<
					// std::endl;
				}
		}
	}

	synchronize_blocked_y(z_begin, z_end);

	for (index_t z = 0; z < step_len; z++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const index_t tile_len = X == X_len - 1 ? remainder : x_tile_size;

			// Final part of modified thomas algorithm
			// Solve the rest of the unknowns
			for (index_t i = y_len - 2; i >= 1; i--)
				for (index_t x = 0; x < tile_len; x += simd_length)
				{
					const auto idx_begin = noarr::idx<'z', 'X', 'x', 'y'>(z, X, x, 0);
					const auto idx = noarr::idx<'z', 'X', 'x', 'y'>(z, X, x, i);
					const auto idx_prev = noarr::idx<'z', 'X', 'x', 'y'>(z, X, x, i + 1);

					simd_t d_curr = hn::Load(d, &(d_bag[idx]));
					simd_t d_begin = hn::Load(d, &(d_bag[idx_begin]));
					simd_t d_end = hn::Load(d, &(d_bag[idx_prev]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

					d_curr = hn::NegMulAdd(a_scratch_curr, d_begin, d_curr);
					d_curr = hn::NegMulAdd(c_scratch_curr, d_end, d_curr);

					hn::Store(d_curr, d, &d_bag[idx]);

					// #pragma omp critical
					// 						std::cout << "l: " << z_begin +z << " " << i << " " << x << " "
					// 								  << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << " " <<
					// a[state] << " " << c[state]
					// 								  << std::endl;
				}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_z_blocked_distributed(real_t** __restrict__ densities, real_t** __restrict__ a_data,
														real_t** __restrict__ c_data, const density_layout_t dens_l,
														const scratch_layout_t scratch_l,
														const thread_distribution_l dist_l, const index_t n,
														const index_t y_begin, const index_t y_end, const index_t tid,
														const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto sscratch_l = scratch_l ^ noarr::merge_blocks<'X', 'x'>();

	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = y_end - y_begin;
	const index_t x_simd_len = (x_len + simd_length - 1) / simd_length;

	const index_t block_size = n / coop_size;

	const index_t work_len = y_len * x_simd_len;
	const index_t block_size_y = work_len / coop_size;
	const index_t work_begin = tid * block_size_y + std::min(tid, work_len % coop_size);
	const index_t work_end = work_begin + block_size_y + ((tid < work_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		return std::make_tuple(block_idx, noarr::set_length<'z'>(actual_block_size) ^ noarr::fix<'z'>(offset));
	};

	for (index_t w_idx = work_begin; w_idx < work_end; w_idx++)
	{
		const index_t y = y_begin + w_idx / x_simd_len;
		const index_t x = (w_idx % x_simd_len) * simd_length;

		simd_t prev_c;
		simd_t prev_d;

		{
			const auto [prev_block_idx, fix_l] = get_i(0);
			const auto prev_c_bag =
				noarr::make_bag(sscratch_l ^ fix_l, dist_l | noarr::get_at<'z'>(c_data, prev_block_idx));
			const auto prev_d_bag =
				noarr::make_bag(dens_l ^ fix_l, dist_l | noarr::get_at<'z'>(densities, prev_block_idx));

			prev_c = hn::Load(t, &prev_c_bag.template at<'x', 'y'>(x, y - y_begin));
			prev_d = hn::Load(t, &prev_d_bag.template at<'x', 'y'>(x, y));
		}

		for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
		{
			const auto [block_idx, fix_l] = get_i(equation_idx);

			const auto a = noarr::make_bag(sscratch_l ^ fix_l, dist_l | noarr::get_at<'z'>(a_data, block_idx));
			const auto c = noarr::make_bag(sscratch_l ^ fix_l, dist_l | noarr::get_at<'z'>(c_data, block_idx));
			const auto d = noarr::make_bag(dens_l ^ fix_l, dist_l | noarr::get_at<'z'>(densities, block_idx));

			simd_t curr_a = hn::Load(t, &a.template at<'x', 'y'>(x, y - y_begin));
			simd_t curr_c = hn::Load(t, &c.template at<'x', 'y'>(x, y - y_begin));
			simd_t curr_d = hn::Load(t, &d.template at<'x', 'y'>(x, y));

			simd_t r = hn::Div(hn::Set(t, 1), hn::NegMulAdd(prev_c, curr_a, hn::Set(t, 1)));

			curr_d = hn::Mul(r, hn::NegMulAdd(prev_d, curr_a, curr_d));
			curr_c = hn::Mul(r, curr_c);

			hn::Store(curr_c, t, &c.template at<'x', 'y'>(x, y - y_begin));
			hn::Store(curr_d, t, &d.template at<'x', 'y'>(x, y));

			prev_c = curr_c;
			prev_d = curr_d;

			// #pragma omp critical
			// 			std::cout << "mf " << z << " " << i << " " << x << " rf: " << rf[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}

		for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto [block_idx, fix_l] = get_i(equation_idx);

			const auto c = noarr::make_bag(sscratch_l ^ fix_l, dist_l | noarr::get_at<'z'>(c_data, block_idx));
			const auto d = noarr::make_bag(dens_l ^ fix_l, dist_l | noarr::get_at<'z'>(densities, block_idx));

			simd_t curr_c = hn::Load(t, &c.template at<'x', 'y'>(x, y - y_begin));
			simd_t curr_d = hn::Load(t, &d.template at<'x', 'y'>(x, y));

			curr_d = hn::NegMulAdd(prev_d, curr_c, curr_d);

			hn::Store(curr_d, t, &d.template at<'x', 'y'>(x, y));

			prev_d = curr_d;

			// #pragma omp critical
			// 			std::cout << "mb " << z << " " << i << " " << x << " b: " << b[state] << " c: " << c[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	barrier.arrive_and_wait();
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename sync_func_t>
static void solve_block_z(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
						  const real_t* __restrict__ c, real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
						  const density_layout_t dens_l, const diagonal_layout_t diag_l,
						  const scratch_layout_t scratch_l, const index_t z_begin, const index_t z_end, const index_t s,
						  const index_t sync_step, const index_t x_tile_size, sync_func_t&& synchronize_blocked_z)
{
#define solve_block_z_vec

#ifdef solve_block_z_vec
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;
#endif

	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = z_end - z_begin;

	const index_t x_block_len = ((dens_l | noarr::get_length<'x'>()) + x_tile_size - 1) / x_tile_size;

	auto remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
	if (remainder == 0)
		remainder = x_tile_size;

	auto a_bag = noarr::make_bag(diag_l, a);
	auto b_bag = noarr::make_bag(diag_l, b);
	auto c_bag = noarr::make_bag(diag_l, c);
	auto d_bag = noarr::make_bag(dens_l ^ noarr::set_length<'z'>(z_end - z_begin), densities);
	auto a_scratch_bag = noarr::make_bag(scratch_l, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l, c_scratch);

	// Normalize the first and the second equation

	for (index_t y_begin = 0; y_begin < y_len; y_begin += sync_step)
	{
		const auto y_end = std::min(y_begin + sync_step, y_len);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t X = 0; X < x_block_len; X++)
			{
				const index_t tile_len = X == x_block_len - 1 ? remainder : x_tile_size;

				for (index_t i = 0; i < 2; i++)
				{
#ifdef solve_block_z_vec
					for (index_t x = 0; x < tile_len; x += simd_length)
#else
					for (index_t x = 0; x < tile_len; x++)
#endif
					{
						auto idx = noarr::idx<'s', 'z'>(s, i);

						auto scratch_idx = noarr::idx<'X', 'x', 'y'>(X, x, y - y_begin);
						auto diag_idx = noarr::idx<'X', 'x', 'y'>(X, x, y);
						auto dens_idx = noarr::idx<'x', 'y'>(X * x_tile_size + x, y);

#ifdef solve_block_z_vec
						simd_t a_curr = hn::Load(d, &(a_bag[idx & diag_idx]));
						simd_t b_curr = hn::Load(d, &(b_bag[idx & diag_idx]));
						simd_t c_curr = hn::Load(d, &(c_bag[idx & diag_idx]));
						simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);

						simd_t r = hn::Div(hn::Set(d, 1), b_curr);

						a_scratch_curr = hn::Mul(r, a_curr);
						c_scratch_curr = hn::Mul(r, c_curr);
						d_curr = hn::Mul(r, d_curr);

						hn::Store(a_scratch_curr, d, &a_scratch_bag[idx & scratch_idx]);
						hn::Store(c_scratch_curr, d, &c_scratch_bag[idx & scratch_idx]);
						hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else
						real_t r = 1 / b_bag[idx & diag_idx];

						a_scratch_bag[idx & scratch_idx] = r * a_bag[idx & diag_idx];
						c_scratch_bag[idx & scratch_idx] = r * c_bag[idx & diag_idx];
						d_bag[idx & dens_idx] = r * d_bag[idx & dens_idx];
#endif

						// #pragma omp critical
						// 					std::cout << "f0: " << z_begin + i << " " << blocked_y + y << " " << x << "
						// " << d_bag[idx] << " " << b_bag[idx]
						// 							  << std::endl;
					}
				}

				// Process the lower diagonal (forward)
				for (index_t i = 2; i < n; i++)
				{
#ifdef solve_block_z_vec
					for (index_t x = 0; x < tile_len; x += simd_length)
#else
					for (index_t x = 0; x < tile_len; x++)
#endif
					{
						auto idx = noarr::idx<'s', 'z'>(s, i);
						auto prev_idx = noarr::idx<'s', 'z'>(s, i - 1);

						auto scratch_idx = noarr::idx<'X', 'x', 'y'>(X, x, y - y_begin);
						auto diag_idx = noarr::idx<'X', 'x', 'y'>(X, x, y);
						auto dens_idx = noarr::idx<'x', 'y'>(X * x_tile_size + x, y);

#ifdef solve_block_z_vec
						simd_t a_curr = hn::Load(d, &(a_bag[idx & diag_idx]));
						simd_t b_curr = hn::Load(d, &(b_bag[idx & diag_idx]));
						simd_t c_curr = hn::Load(d, &(c_bag[idx & diag_idx]));
						simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
						simd_t d_prev = hn::Load(d, &(d_bag[prev_idx & dens_idx]));
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);
						simd_t a_scratch_prev = hn::Load(d, &a_scratch_bag[prev_idx & scratch_idx]);
						simd_t c_scratch_prev = hn::Load(d, &c_scratch_bag[prev_idx & scratch_idx]);

						simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

						a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
						c_scratch_curr = hn::Mul(r, c_curr);
						d_curr = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_curr));

						hn::Store(a_scratch_curr, d, &a_scratch_bag[idx & scratch_idx]);
						hn::Store(c_scratch_curr, d, &c_scratch_bag[idx & scratch_idx]);
						hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else
						real_t r =
							1 / (b_bag[idx & diag_idx] - a_bag[idx & diag_idx] * c_scratch_bag[prev_idx & scratch_idx]);

						a_scratch_bag[idx & scratch_idx] =
							r * (-a_bag[idx & diag_idx] * a_scratch_bag[prev_idx & scratch_idx]);
						c_scratch_bag[idx & scratch_idx] = r * c_bag[idx & diag_idx];
						d_bag[idx & dens_idx] =
							r * (d_bag[idx & dens_idx] - a_bag[idx & diag_idx] * d_bag[prev_idx & dens_idx]);
#endif
						// #pragma omp critical
						// 			std::cout << "f1: " << z << " " << i + y_begin << " " << x << " " << d_bag[idx]
						// 					  << " " << a_bag[idx]  << " " << b_bag[idx]  << " " <<
						// c_scratch_bag[idx]
						// << std::endl;
					}
				}

				// Process the upper diagonal (backward)
				for (index_t i = n - 3; i >= 1; i--)
				{
#ifdef solve_block_z_vec
					for (index_t x = 0; x < tile_len; x += simd_length)
#else
					for (index_t x = 0; x < tile_len; x++)
#endif
					{
						auto idx = noarr::idx<'s', 'z'>(s, i);
						auto next_idx = noarr::idx<'s', 'z'>(s, i + 1);

						auto scratch_idx = noarr::idx<'X', 'x', 'y'>(X, x, y - y_begin);
						auto dens_idx = noarr::idx<'x', 'y'>(X * x_tile_size + x, y);

#ifdef solve_block_z_vec
						simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
						simd_t d_prev = hn::Load(d, &(d_bag[next_idx & dens_idx]));
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);
						simd_t a_scratch_prev = hn::Load(d, &a_scratch_bag[next_idx & scratch_idx]);
						simd_t c_scratch_prev = hn::Load(d, &c_scratch_bag[next_idx & scratch_idx]);

						d_curr = hn::NegMulAdd(c_scratch_curr, d_prev, d_curr);
						a_scratch_curr = hn::NegMulAdd(c_scratch_curr, a_scratch_prev, a_scratch_curr);
						c_scratch_curr = hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0));

						hn::Store(a_scratch_curr, d, &a_scratch_bag[idx & scratch_idx]);
						hn::Store(c_scratch_curr, d, &c_scratch_bag[idx & scratch_idx]);
						hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else

						d_bag[idx & dens_idx] =
							d_bag[idx & dens_idx] - c_scratch_bag[idx & scratch_idx] * d_bag[next_idx & dens_idx];
						a_scratch_bag[idx & scratch_idx] =
							a_scratch_bag[idx & scratch_idx]
							- c_scratch_bag[idx & scratch_idx] * a_scratch_bag[next_idx & scratch_idx];
						c_scratch_bag[idx & scratch_idx] =
							-c_scratch_bag[idx & scratch_idx] * c_scratch_bag[next_idx & scratch_idx];
#endif
						// #pragma omp critical
						// 			std::cout << "b0: " << z << " " << i + y_begin << " " << x << " " << d_bag[idx] <<
						// std::endl;
					}
				}

				// Process the first row (backward)
				{
#ifdef solve_block_z_vec
					for (index_t x = 0; x < tile_len; x += simd_length)
#else
					for (index_t x = 0; x < tile_len; x++)
#endif
					{
						auto idx = noarr::idx<'s', 'z'>(s, 0);
						auto next_idx = noarr::idx<'s', 'z'>(s, 1);

						auto scratch_idx = noarr::idx<'X', 'x', 'y'>(X, x, y - y_begin);
						auto dens_idx = noarr::idx<'x', 'y'>(X * x_tile_size + x, y);

#ifdef solve_block_z_vec
						simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
						simd_t d_prev = hn::Load(d, &(d_bag[next_idx & dens_idx]));
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);
						simd_t a_scratch_prev = hn::Load(d, &a_scratch_bag[next_idx & scratch_idx]);
						simd_t c_scratch_prev = hn::Load(d, &c_scratch_bag[next_idx & scratch_idx]);

						simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_scratch_curr, a_scratch_prev, hn::Set(d, 1)));

						d_curr = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, d_prev, d_curr));
						a_scratch_curr = hn::Mul(r, a_scratch_curr);
						c_scratch_curr = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0)));

						hn::Store(a_scratch_curr, d, &a_scratch_bag[idx & scratch_idx]);
						hn::Store(c_scratch_curr, d, &c_scratch_bag[idx & scratch_idx]);
						hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else
						real_t r = 1 / (1 - c_scratch_bag[idx & scratch_idx] * a_scratch_bag[next_idx & scratch_idx]);

						d_bag[idx & dens_idx] =
							r * (d_bag[idx & dens_idx] - c_scratch_bag[idx & scratch_idx] * d_bag[next_idx & dens_idx]);
						a_scratch_bag[idx & scratch_idx] = r * a_scratch_bag[idx & scratch_idx];
						c_scratch_bag[idx & scratch_idx] =
							r * -c_scratch_bag[idx & scratch_idx] * c_scratch_bag[next_idx & scratch_idx];
#endif
						// #pragma omp critical
						// 			std::cout << "b1: " << z << " " << y_begin << " " << x << " " << d_bag[idx] <<
						// std::endl;
					}
				}
			}
		}

		synchronize_blocked_z(y_begin, y_end);

		for (index_t y = y_begin; y < y_end; y++)
		{
			// Final part of modified thomas algorithm
			// Solve the rest of the unknowns
			for (index_t X = 0; X < x_block_len; X++)
			{
				const index_t tile_len = X == x_block_len - 1 ? remainder : x_tile_size;

				for (index_t i = 1; i < n - 1; i++)
				{
#ifdef solve_block_z_vec
					for (index_t x = 0; x < tile_len; x += simd_length)
#else
					for (index_t x = 0; x < tile_len; x++)
#endif
					{
						auto idx_begin = noarr::idx<'s', 'z'>(s, 0);
						auto idx = noarr::idx<'s', 'z'>(s, i);
						auto idx_end = noarr::idx<'s', 'z'>(s, n - 1);

						auto scratch_idx = noarr::idx<'X', 'x', 'y'>(X, x, y - y_begin);
						auto dens_idx = noarr::idx<'x', 'y'>(X * x_tile_size + x, y);

#ifdef solve_block_z_vec
						simd_t d_curr = hn::Load(d, &(d_bag[idx & dens_idx]));
						simd_t d_begin = hn::Load(d, &(d_bag[idx_begin & dens_idx]));
						simd_t d_end = hn::Load(d, &(d_bag[idx_end & dens_idx]));
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx & scratch_idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx & scratch_idx]);

						d_curr = hn::NegMulAdd(a_scratch_curr, d_begin, d_curr);
						d_curr = hn::NegMulAdd(c_scratch_curr, d_end, d_curr);

						hn::Store(d_curr, d, &d_bag[idx & dens_idx]);
#else
						d_bag[idx & dens_idx] = d_bag[idx & dens_idx]
												- a_scratch_bag[idx & scratch_idx] * d_bag[idx_begin & dens_idx]
												- c_scratch_bag[idx & scratch_idx] * d_bag[idx_end & dens_idx];
#endif
						// #pragma omp critical
						// 						std::cout << "l: " << z << " " << i << " " << x << " "
						// 								  << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << " " <<
						// a[state] << " " << c[state]
						// 								  << std::endl;
					}
				}
			}
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename sync_func_t>
static void solve_block_z_alt(real_t* __restrict__ densities, const real_t* __restrict__ a,
							  const real_t* __restrict__ b, const real_t* __restrict__ c,
							  real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
							  const density_layout_t dens_l, const diagonal_layout_t diag_l,
							  const scratch_layout_t scratch_l, const index_t z_begin, const index_t z_end,
							  const index_t s, const index_t x_len, const index_t sync_step, const index_t x_tile_size,
							  sync_func_t&& synchronize_blocked_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s) ^ noarr::set_length<'z'>(z_end - z_begin)
						  ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size)
						  ^ noarr::fix<'b'>(noarr::lit<0>);

	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();
	const index_t y_len = blocked_dens_l | noarr::get_length<'y'>();
	const index_t z_len = blocked_dens_l | noarr::get_length<'z'>();

	auto remainder = ((x_len + simd_length - 1) / simd_length * simd_length) % x_tile_size;
	if (remainder == 0)
		remainder = x_tile_size;

	auto a_scratch_bag = noarr::make_bag(scratch_l, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l, c_scratch);

	// Normalize the first and the second equation

	for (index_t blocked_y = 0; blocked_y < y_len; blocked_y += sync_step)
	{
		const auto step_len = std::min(y_len - blocked_y, sync_step);

		auto a_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'y'>(blocked_y, step_len), a);
		auto b_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'y'>(blocked_y, step_len), b);
		auto c_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'y'>(blocked_y, step_len), c);
		auto d_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'y'>(blocked_y, step_len), densities);

		for (index_t y = 0; y < step_len; y++)
		{
			for (index_t X = 0; X < X_len; X++)
			{
				const index_t tile_len = X == X_len - 1 ? remainder : x_tile_size;

				for (index_t i = 0; i < 2; i++)
				{
					for (index_t x = 0; x < tile_len; x += simd_length)
					{
						const auto idx = noarr::idx<'X', 'x', 'y', 'z'>(X, x, y, i);

						simd_t a_curr = hn::Load(d, &(a_bag[idx]));
						simd_t b_curr = hn::Load(d, &(b_bag[idx]));
						simd_t c_curr = hn::Load(d, &(c_bag[idx]));
						simd_t d_curr = hn::Load(d, &(d_bag[idx]));
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

						simd_t r = hn::Div(hn::Set(d, 1), b_curr);

						a_scratch_curr = hn::Mul(r, a_curr);
						c_scratch_curr = hn::Mul(r, c_curr);
						d_curr = hn::Mul(r, d_curr);

						hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
						hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
						hn::Store(d_curr, d, &d_bag[idx]);

						if (i != 0 && i != z_len - 1)
						{
							const auto idx0 = noarr::idx<'X', 'x', 'y', 'z'>(X, x, y, noarr::lit<0>);

							simd_t a_scratch_0 = hn::Load(d, &a_scratch_bag[idx0]);
							simd_t c_scratch_0 = hn::Load(d, &c_scratch_bag[idx0]);
							simd_t r0 =
								hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_scratch_curr, c_scratch_0, hn::Set(d, 1)));

							simd_t d_0 = hn::Load(d, &d_bag[idx0]);

							d_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, d_curr, d_0));
							a_scratch_0 = hn::Mul(r0, a_scratch_0);
							c_scratch_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, c_scratch_curr, hn::Set(d, 0)));

							hn::Store(a_scratch_0, d, &a_scratch_bag[idx0]);
							hn::Store(c_scratch_0, d, &c_scratch_bag[idx0]);
							hn::Store(d_0, d, &d_bag[idx0]);
						}

						// #pragma omp critical
						// 					std::cout << "f0: " << z_begin + i << " " << blocked_y + y << " " << x << "
						// " << d_bag[idx] << " " << b_bag[idx]
						// 							  << std::endl;
					}
				}

				// Process the lower diagonal (forward)
				for (index_t i = 2; i < z_len; i++)
				{
					for (index_t x = 0; x < tile_len; x += simd_length)
					{
						const auto prev_idx = noarr::idx<'X', 'x', 'y', 'z'>(X, x, y, i - 1);
						const auto idx = noarr::idx<'X', 'x', 'y', 'z'>(X, x, y, i);

						simd_t a_curr = hn::Load(d, &(a_bag[idx]));
						simd_t b_curr = hn::Load(d, &(b_bag[idx]));
						simd_t c_curr = hn::Load(d, &(c_bag[idx]));
						simd_t d_curr = hn::Load(d, &(d_bag[idx]));
						simd_t d_prev = hn::Load(d, &(d_bag[prev_idx]));
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);
						simd_t a_scratch_prev = hn::Load(d, &a_scratch_bag[prev_idx]);
						simd_t c_scratch_prev = hn::Load(d, &c_scratch_bag[prev_idx]);

						simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

						a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
						c_scratch_curr = hn::Mul(r, c_curr);
						d_curr = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_curr));

						hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
						hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
						hn::Store(d_curr, d, &d_bag[idx]);

						if (i != 0 && i != z_len - 1)
						{
							const auto idx0 = noarr::idx<'X', 'x', 'y', 'z'>(X, x, y, noarr::lit<0>);

							simd_t a_scratch_0 = hn::Load(d, &a_scratch_bag[idx0]);
							simd_t c_scratch_0 = hn::Load(d, &c_scratch_bag[idx0]);
							simd_t r0 =
								hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_scratch_curr, c_scratch_0, hn::Set(d, 1)));

							simd_t d_0 = hn::Load(d, &d_bag[idx0]);

							d_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, d_curr, d_0));
							a_scratch_0 = hn::Mul(r0, a_scratch_0);
							c_scratch_0 = hn::Mul(r0, hn::NegMulAdd(c_scratch_0, c_scratch_curr, hn::Set(d, 0)));

							hn::Store(a_scratch_0, d, &a_scratch_bag[idx0]);
							hn::Store(c_scratch_0, d, &c_scratch_bag[idx0]);
							hn::Store(d_0, d, &d_bag[idx0]);
						}

						// #pragma omp critical
						// 			std::cout << "f1: " << z << " " << i + y_begin << " " << x << " " << d_bag[idx]
						// 					  << " " << a_bag[idx]  << " " << b_bag[idx]  << " " << c_scratch_bag[idx]
						// << std::endl;
					}
				}
			}
		}

		synchronize_blocked_z(blocked_y, blocked_y + step_len);

		for (index_t y = 0; y < step_len; y++)
		{
			// Final part of modified thomas algorithm
			// Solve the rest of the unknowns
			for (index_t X = 0; X < X_len; X++)
			{
				const index_t tile_len = X == X_len - 1 ? remainder : x_tile_size;

				for (index_t i = z_len - 2; i >= 1; i--)
				{
					for (index_t x = 0; x < tile_len; x += simd_length)
					{
						const auto idx_begin = noarr::idx<'X', 'x', 'y', 'z'>(X, x, y, 0);
						const auto idx = noarr::idx<'X', 'x', 'y', 'z'>(X, x, y, i);
						const auto idx_prev = noarr::idx<'X', 'x', 'y', 'z'>(X, x, y, i + 1);

						simd_t d_curr = hn::Load(d, &(d_bag[idx]));
						simd_t d_begin = hn::Load(d, &(d_bag[idx_begin]));
						simd_t d_end = hn::Load(d, &(d_bag[idx_prev]));
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

						d_curr = hn::NegMulAdd(a_scratch_curr, d_begin, d_curr);
						d_curr = hn::NegMulAdd(c_scratch_curr, d_end, d_curr);

						hn::Store(d_curr, d, &d_bag[idx]);

						// #pragma omp critical
						// 						std::cout << "l: " << z << " " << i << " " << x << " "
						// 								  << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << " " <<
						// a[state] << " " << c[state]
						// 								  << std::endl;
					}
				}
			}
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l,
							 const scratch_layout_t scratch_l, const index_t s_idx, const index_t z,
							 index_t x_tile_size)
{
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(diag_l, a);
	auto b_bag = noarr::make_bag(diag_l, b);
	auto c_bag = noarr::make_bag(diag_l, c);
	auto d = noarr::make_bag(dens_l, densities);
	auto scratch = noarr::make_bag(scratch_l, b_scratch);

	const index_t x_block_len = (x_len + x_tile_size - 1) / x_tile_size;

	for (index_t X = 0; X < x_block_len; X++)
	{
		const auto remainder = x_len % x_tile_size;
		const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
		const auto tile_size = X == x_block_len - 1 ? x_len_remainder : x_tile_size;

		for (index_t s = 0; s < tile_size; s++)
		{
			scratch[noarr::idx<'v', 'y'>(s, 0)] = 1 / b_bag[noarr::idx<'s', 'z', 'y', 'X', 'x'>(s_idx, z, 0, X, s)];
		}

		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, i);
				auto prev_idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, i - 1);

				auto scratch_idx = noarr::idx<'v'>(s);
				auto diag_idx = noarr::idx<'X', 'x'>(X, s);
				auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

				auto r = a_bag[idx & diag_idx] * scratch[prev_idx & scratch_idx];

				scratch[idx & scratch_idx] = 1 / (b_bag[idx & diag_idx] - c_bag[prev_idx & diag_idx] * r);

				d[idx & dens_idx] -= r * d[prev_idx & dens_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}

		for (index_t s = 0; s < tile_size; s++)
		{
			auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, n - 1);

			auto scratch_idx = noarr::idx<'v'>(s);
			auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

			d[idx & dens_idx] *= scratch[idx & scratch_idx];

			// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
		}

		for (index_t i = n - 2; i >= 0; i--)
			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, i);
				auto next_idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, i + 1);

				auto scratch_idx = noarr::idx<'v'>(s);
				auto diag_idx = noarr::idx<'X', 'x'>(X, s);
				auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

				d[idx & dens_idx] =
					(d[idx & dens_idx] - c_bag[idx & diag_idx] * d[next_idx & dens_idx]) * scratch[idx & scratch_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l,
							 const scratch_layout_t scratch_l, const index_t s_idx, index_t x_tile_size)
{
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto a_bag = noarr::make_bag(diag_l, a);
	auto b_bag = noarr::make_bag(diag_l, b);
	auto c_bag = noarr::make_bag(diag_l, c);
	auto d = noarr::make_bag(dens_l, densities);
	auto scratch = noarr::make_bag(scratch_l, b_scratch);

	const index_t x_block_len = (x_len + x_tile_size - 1) / x_tile_size;

	for (index_t y = 0; y < y_len; y++)
		for (index_t X = 0; X < x_block_len; X++)
		{
			const auto remainder = x_len % x_tile_size;
			const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
			const auto tile_size = X == x_block_len - 1 ? x_len_remainder : x_tile_size;

			for (index_t s = 0; s < tile_size; s++)
			{
				scratch[noarr::idx<'v', 'z'>(s, 0)] = 1 / b_bag[noarr::idx<'s', 'z', 'y', 'X', 'x'>(s_idx, 0, y, X, s)];
			}

			for (index_t i = 1; i < n; i++)
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, i, y);
					auto prev_idx = noarr::idx<'s', 'z', 'y'>(s_idx, i - 1, y);

					auto scratch_idx = noarr::idx<'v'>(s);
					auto diag_idx = noarr::idx<'X', 'x'>(X, s);
					auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

					auto r = a_bag[idx & diag_idx] * scratch[prev_idx & scratch_idx];

					scratch[idx & scratch_idx] = 1 / (b_bag[idx & diag_idx] - c_bag[prev_idx & diag_idx] * r);

					d[idx & dens_idx] -= r * d[prev_idx & dens_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, n - 1, y);

				auto scratch_idx = noarr::idx<'v'>(s);
				auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

				d[idx & dens_idx] *= scratch[idx & scratch_idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, i, y);
					auto next_idx = noarr::idx<'s', 'z', 'y'>(s_idx, i + 1, y);

					auto scratch_idx = noarr::idx<'v'>(s);
					auto diag_idx = noarr::idx<'X', 'x'>(X, s);
					auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

					d[idx & dens_idx] = (d[idx & dens_idx] - c_bag[idx & diag_idx] * d[next_idx & dens_idx])
										* scratch[idx & scratch_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
		}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::solve_x_nf()
{
	for (index_t i = 0; i < countersx_count_; i++)
	{
		countersx_[i]->value = 0;
	}

#pragma omp parallel
	{
		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_x_begin = group_block_offsetsx_[tid.x];
		const auto block_x_end = block_x_begin + group_block_lengthsx_[tid.x];

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_x = get_lane_id('x');

		barrier_t<true, index_t> barrier_x(cores_division_[0], countersx_[lane_id_x]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];


		const index_t group_size_x = cores_division_[0];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s++)
		{
			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s << " s_end: " << s + 1
			// 					  << " block_x_begin: " << block_x_begin << " block_x_end: " << block_x_end
			// 					  << " block_y_begin: " << group_block_offsetsy_[tid.y]
			// 					  << " block_y_end: " << group_block_offsetsy_[tid.y] - group_block_lengthsy_[tid.y]
			// 					  << " block_z_begin: " << block_z_begin << " block_z_end: " << block_z_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				auto scratch_x = get_scratch_layout<'x', true, false>(group_block_lengthsx_[tid.x],
																	  group_block_lengthsy_[tid.y], y_sync_step_);

				auto scratch_x_wo_x = get_scratch_layout<'x', false, false>(group_block_lengthsx_[tid.x],
																			group_block_lengthsy_[tid.y], y_sync_step_);

				auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout() ^ noarr::fix<'g'>(tid.group);

				auto current_a_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(a_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_c_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(c_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_densities =
					dist_l | noarr::get_at<'x', 'y', 'z'>(thread_substrate_array_.get(), tid.x, tid.y, tid.z);

				auto current_ax = dist_l | noarr::get_at<'x', 'y', 'z'>(ax_.get(), tid.x, tid.y, tid.z);
				auto current_bx = dist_l | noarr::get_at<'x', 'y', 'z'>(bx_.get(), tid.x, tid.y, tid.z);
				auto current_cx = dist_l | noarr::get_at<'x', 'y', 'z'>(cx_.get(), tid.x, tid.y, tid.z);

				auto dens_l =
					get_blocked_substrate_layout(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_x =
					get_blocked_substrate_layout<'x'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto diag_x = get_diag_layout<'x', false>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
														  group_block_lengthsz_[tid.z],
														  group_block_lengthss_[tid.group], y_sync_step_);

				for (index_t blocked_z = block_z_begin; blocked_z < block_z_end; blocked_z += y_sync_step_)
				{
					const index_t y_sync_step_len = std::min(y_sync_step_, block_z_end - blocked_z);


					auto sync_x = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_x ^ noarr::fix<'s'>(s),
								   scratch = scratch_x_wo_x, dist_l = dist_l ^ noarr::fix<'y', 'z'>(tid.y, tid.z),
								   n = this->problem_.nx,
								   n_alignment = (index_t)alignment_size_ / (index_t)sizeof(real_t), tid = tid.x,
								   group_size = group_size_x, &barrier = barrier_x](index_t z_begin, index_t z_end) {
						synchronize_x_blocked_distributed_nf(densities, a, c, dens_l, scratch, dist_l, n, n_alignment,
															 z_begin, z_end, tid, group_size, barrier);
					};

					if (cores_division_[0] != 1)
						solve_block_x_transpose_nf(
							current_densities, current_ax, current_bx, current_cx, current_a_scratch, current_c_scratch,
							dens_l, diag_x, scratch_x, block_x_begin, block_x_end, blocked_z - block_z_begin,
							blocked_z + y_sync_step_len - block_z_begin, s, x_tile_size_, std::move(sync_x));
					else
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_x_2d_and_3d_transpose_l_nf<index_t>(
								current_densities, current_ax, current_bx, current_cx, current_a_scratch, dens_l,
								diag_x,
								get_non_blocked_scratch_layout<'x'>(group_block_lengthsx_[tid.x],
																	alignment_size_ / sizeof(real_t)),
								s, z - block_z_begin, this->problem_.nx);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::solve_x()
{
	if (!fuse_z_)
	{
		solve_x_nf();
		return;
	}

	for (index_t i = 0; i < countersx_count_; i++)
	{
		countersx_[i]->value = 0;
	}

#pragma omp parallel
	{
		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_x_begin = group_block_offsetsx_[tid.x];
		const auto block_x_end = block_x_begin + group_block_lengthsx_[tid.x];

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_x = get_lane_id('x');

		barrier_t<true, index_t> barrier_x(cores_division_[0], countersx_[lane_id_x]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];


		const index_t group_size_x = cores_division_[0];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s++)
		{
			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s << " s_end: " << s + 1
			// 					  << " block_x_begin: " << block_x_begin << " block_x_end: " << block_x_end
			// 					  << " block_y_begin: " << group_block_offsetsy_[tid.y]
			// 					  << " block_y_end: " << group_block_offsetsy_[tid.y] - group_block_lengthsy_[tid.y]
			// 					  << " block_z_begin: " << block_z_begin << " block_z_end: " << block_z_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				auto scratch_x = get_scratch_layout<'x', true>(group_block_lengthsx_[tid.x],
															   group_block_lengthsy_[tid.y], y_sync_step_);

				auto scratch_x_wo_x = get_scratch_layout<'x', false>(group_block_lengthsx_[tid.x],
																	 group_block_lengthsy_[tid.y], y_sync_step_);

				auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout() ^ noarr::fix<'g'>(tid.group);

				auto current_a_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(a_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_c_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(c_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_densities =
					dist_l | noarr::get_at<'x', 'y', 'z'>(thread_substrate_array_.get(), tid.x, tid.y, tid.z);

				auto current_ax = dist_l | noarr::get_at<'x', 'y', 'z'>(ax_.get(), tid.x, tid.y, tid.z);
				auto current_bx = dist_l | noarr::get_at<'x', 'y', 'z'>(bx_.get(), tid.x, tid.y, tid.z);
				auto current_cx = dist_l | noarr::get_at<'x', 'y', 'z'>(cx_.get(), tid.x, tid.y, tid.z);

				auto dens_l =
					get_blocked_substrate_layout(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_x =
					get_blocked_substrate_layout<'x'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto diag_x =
					get_diag_layout<'x'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group], y_sync_step_);

				for (index_t blocked_z = block_z_begin; blocked_z < block_z_end; blocked_z += y_sync_step_)
				{
					const index_t y_sync_step_len = std::min(y_sync_step_, block_z_end - blocked_z);


					auto sync_x = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_x ^ noarr::fix<'s'>(s),
								   scratch = scratch_x_wo_x, dist_l = dist_l ^ noarr::fix<'y', 'z'>(tid.y, tid.z),
								   n = this->problem_.nx, sync_step = y_sync_step_,
								   n_alignment = (index_t)alignment_size_ / (index_t)sizeof(real_t), tid = tid.x,
								   z_begin = blocked_z - block_z_begin,
								   z_end = blocked_z + y_sync_step_len - block_z_begin, group_size = group_size_x,
								   &barrier = barrier_x]() {
						synchronize_x_blocked_distributed(densities, a, c, dens_l, scratch, dist_l, n, n_alignment,
														  z_begin, z_end, sync_step, tid, group_size, barrier);
					};

					if (cores_division_[0] != 1)
					{
						if (!alt_blocked_)
							solve_block_x_transpose(current_densities, current_ax, current_bx, current_cx,
													current_a_scratch, current_c_scratch, dens_l, diag_x, scratch_x,
													block_x_begin, block_x_end, blocked_z - block_z_begin,
													blocked_z + y_sync_step_len - block_z_begin, y_sync_step_, s,
													x_tile_size_, std::move(sync_x));
						else
							solve_block_x_transpose_alt(current_densities, current_ax, current_bx, current_cx,
														current_a_scratch, current_c_scratch, dens_l, diag_x, scratch_x,
														block_x_begin, block_x_end, blocked_z - block_z_begin,
														blocked_z + y_sync_step_len - block_z_begin, y_sync_step_, s,
														x_tile_size_, std::move(sync_x));
					}
					else
						solve_slice_x_2d_and_3d_transpose_l<index_t>(
							current_densities, current_ax, current_bx, current_cx, current_a_scratch, dens_l, diag_x,
							get_non_blocked_scratch_layout<'x'>(group_block_lengthsx_[tid.x],
																alignment_size_ / sizeof(real_t)),
							s, blocked_z - block_z_begin, blocked_z + y_sync_step_len - block_z_begin,
							this->problem_.nx);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::solve_y()
{
	for (index_t i = 0; i < countersy_count_; i++)
	{
		countersy_[i]->value = 0;
	}

#pragma omp parallel
	{
		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_y_begin = group_block_offsetsy_[tid.y];
		const auto block_y_end = block_y_begin + group_block_lengthsy_[tid.y];

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_y = get_lane_id('y');

		barrier_t<true, index_t> barrier_y(cores_division_[1], countersy_[lane_id_y]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];

		const index_t group_size_y = cores_division_[1];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s++)
		{
			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s << " s_end: " << s + 1
			// 					  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
			// 					  << " block_z_begin: " << block_z_begin << " block_z_end: " << block_z_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				auto scratch_y = get_scratch_layout<'y', true>(group_block_lengthsx_[tid.x],
															   group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_y_wo_y = get_scratch_layout<'y', false>(group_block_lengthsx_[tid.x],
																	 group_block_lengthsy_[tid.y], y_sync_step_);

				auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout() ^ noarr::fix<'g'>(tid.group);

				auto current_a_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(a_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_c_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(c_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_densities =
					dist_l | noarr::get_at<'x', 'y', 'z'>(thread_substrate_array_.get(), tid.x, tid.y, tid.z);

				auto current_ay = dist_l | noarr::get_at<'x', 'y', 'z'>(ay_.get(), tid.x, tid.y, tid.z);
				auto current_by = dist_l | noarr::get_at<'x', 'y', 'z'>(by_.get(), tid.x, tid.y, tid.z);
				auto current_cy = dist_l | noarr::get_at<'x', 'y', 'z'>(cy_.get(), tid.x, tid.y, tid.z);

				auto dens_l =
					get_blocked_substrate_layout(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_y =
					get_blocked_substrate_layout<'y'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto diag_y =
					get_diag_layout<'y'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group], y_sync_step_);


				for (index_t blocked_z = block_z_begin; blocked_z < block_z_end; blocked_z += y_sync_step_)
				{
					const index_t y_sync_step_len = std::min(y_sync_step_, block_z_end - blocked_z);

					auto sync_y = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_y ^ noarr::fix<'s'>(s),
								   scratch = scratch_y_wo_y, dist_l = dist_l ^ noarr::fix<'x', 'z'>(tid.x, tid.z),
								   n = this->problem_.ny, tid = tid.y, group_size = group_size_y,
								   &barrier = barrier_y](index_t z_begin, index_t z_end) {
						synchronize_y_blocked_distributed(densities, a, c, dens_l, scratch, dist_l, n, z_begin, z_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[1] != 1)
					{
						if (!alt_blocked_)
							solve_block_y(current_densities, current_ay, current_by, current_cy, current_a_scratch,
										  current_c_scratch, dens_l_wo_y, diag_y, scratch_y, block_y_begin, block_y_end,
										  blocked_z - block_z_begin, blocked_z + y_sync_step_len - block_z_begin, s,
										  x_tile_size_, std::move(sync_y));
						else
							solve_block_y_alt(current_densities, current_ay, current_by, current_cy, current_a_scratch,
											  current_c_scratch, dens_l_wo_y, diag_y, scratch_y, block_y_begin,
											  block_y_end, blocked_z - block_z_begin,
											  blocked_z + y_sync_step_len - block_z_begin, s,
											  group_block_lengthsx_[tid.x], x_tile_size_, std::move(sync_y));
					}
					else
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_y_3d<index_t>(
								current_densities, current_ay, current_by, current_cy, current_a_scratch, dens_l,
								diag_y,
								get_non_blocked_scratch_layout<'y'>(
									group_block_lengthsy_[tid.y], std::min(x_tile_size_, group_block_lengthsx_[tid.x])),
								s, z - block_z_begin, x_tile_size_);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::solve_z()
{
	for (index_t i = 0; i < countersz_count_; i++)
	{
		countersz_[i]->value = 0;
	}

#pragma omp parallel
	{
		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_z = get_lane_id('z');

		barrier_t<true, index_t> barrier_z(cores_division_[2], countersz_[lane_id_z]->value);
		// auto& barrier_z = *barriersz_[lane_id_z];

		const index_t group_size_z = cores_division_[2];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s++)
		{
			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s << " s_end: " << s + 1
			// 					  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
			// 					  << " block_z_begin: " << block_z_begin << " block_z_end: " << block_z_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				auto scratch_z = get_scratch_layout<'z', true>(group_block_lengthsx_[tid.x], z_sync_step_,
															   group_block_lengthsz_[tid.z]);
				auto scratch_z_wo_z = get_scratch_layout<'z', false>(group_block_lengthsx_[tid.x], z_sync_step_,
																	 group_block_lengthsz_[tid.z]);

				auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout() ^ noarr::fix<'g'>(tid.group);

				auto current_a_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(a_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_c_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(c_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_densities =
					dist_l | noarr::get_at<'x', 'y', 'z'>(thread_substrate_array_.get(), tid.x, tid.y, tid.z);


				auto current_az = dist_l | noarr::get_at<'x', 'y', 'z'>(az_.get(), tid.x, tid.y, tid.z);
				auto current_bz = dist_l | noarr::get_at<'x', 'y', 'z'>(bz_.get(), tid.x, tid.y, tid.z);
				auto current_cz = dist_l | noarr::get_at<'x', 'y', 'z'>(cz_.get(), tid.x, tid.y, tid.z);

				auto dens_l =
					get_blocked_substrate_layout(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_z =
					get_blocked_substrate_layout<'z'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto diag_z =
					get_diag_layout<'z'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group], y_sync_step_);

				if (this->problem_.dims == 3)
				{
					auto sync_z = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_z ^ noarr::fix<'s'>(s),
								   scratch = scratch_z_wo_z, dist_l = dist_l ^ noarr::fix<'x', 'y'>(tid.x, tid.y),
								   n = this->problem_.nz, tid = tid.z, group_size = group_size_z,
								   &barrier = barrier_z](index_t y_begin, index_t y_end) {
						synchronize_z_blocked_distributed(densities, a, c, dens_l, scratch, dist_l, n, y_begin, y_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[2] != 1)
					{
						if (!alt_blocked_)
							solve_block_z(current_densities, current_az, current_bz, current_cz, current_a_scratch,
										  current_c_scratch, dens_l_wo_z, diag_z, scratch_z, block_z_begin, block_z_end,
										  s, z_sync_step_, x_tile_size_, std::move(sync_z));
						else
							solve_block_z_alt(current_densities, current_az, current_bz, current_cz, current_a_scratch,
											  current_c_scratch, dens_l_wo_z, diag_z, scratch_z, block_z_begin,
											  block_z_end, s, group_block_lengthsx_[tid.x], z_sync_step_, x_tile_size_,
											  std::move(sync_z));
					}
					else
						solve_slice_z_3d<index_t>(
							current_densities, current_az, current_bz, current_cz, current_a_scratch, dens_l, diag_z,
							get_non_blocked_scratch_layout<'z'>(group_block_lengthsz_[tid.z],
																std::min(x_tile_size_, group_block_lengthsx_[tid.x])),
							s, x_tile_size_);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::solve()
{
	if (!fuse_z_)
	{
		solve_nf();
		return;
	}

	for (index_t i = 0; i < countersx_count_; i++)
	{
		countersx_[i]->value = 0;
	}
	for (index_t i = 0; i < countersy_count_; i++)
	{
		countersy_[i]->value = 0;
	}
	for (index_t i = 0; i < countersz_count_; i++)
	{
		countersz_[i]->value = 0;
	}

#pragma omp parallel
	{
		perf_counter counter("sdd-fb");

		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_x_begin = group_block_offsetsx_[tid.x];
		const auto block_x_end = block_x_begin + group_block_lengthsx_[tid.x];

		const auto block_y_begin = group_block_offsetsy_[tid.y];
		const auto block_y_end = block_y_begin + group_block_lengthsy_[tid.y];

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_x = get_lane_id('x');

		barrier_t<true, index_t> barrier_x(cores_division_[0], countersx_[lane_id_x]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];

		const auto lane_id_y = get_lane_id('y');

		barrier_t<true, index_t> barrier_y(cores_division_[1], countersy_[lane_id_y]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];

		const auto lane_id_z = get_lane_id('z');

		barrier_t<true, index_t> barrier_z(cores_division_[2], countersz_[lane_id_z]->value);
		// auto& barrier_z = *barriersz_[lane_id_z];

		const index_t group_size_x = cores_division_[0];
		const index_t group_size_y = cores_division_[1];
		const index_t group_size_z = cores_division_[2];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s++)
		{
			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s << " s_end: " << s + 1
			// 					  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
			// 					  << " block_z_begin: " << block_z_begin << " block_z_end: " << block_z_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				auto scratch_x = get_scratch_layout<'x', true>(group_block_lengthsx_[tid.x],
															   group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_y = get_scratch_layout<'y', true>(group_block_lengthsx_[tid.x],
															   group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_z = get_scratch_layout<'z', true>(group_block_lengthsx_[tid.x], z_sync_step_,
															   group_block_lengthsz_[tid.z]);
				auto scratch_x_wol = get_scratch_layout<'x', false>(group_block_lengthsx_[tid.x],
																	group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_y_wol = get_scratch_layout<'y', false>(group_block_lengthsx_[tid.x],
																	group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_z_wol = get_scratch_layout<'z', false>(group_block_lengthsx_[tid.x], z_sync_step_,
																	group_block_lengthsz_[tid.z]);

				auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout() ^ noarr::fix<'g'>(tid.group);

				auto current_a_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(a_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_c_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(c_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_densities =
					dist_l | noarr::get_at<'x', 'y', 'z'>(thread_substrate_array_.get(), tid.x, tid.y, tid.z);

				auto current_ax = dist_l | noarr::get_at<'x', 'y', 'z'>(ax_.get(), tid.x, tid.y, tid.z);
				auto current_bx = dist_l | noarr::get_at<'x', 'y', 'z'>(bx_.get(), tid.x, tid.y, tid.z);
				auto current_cx = dist_l | noarr::get_at<'x', 'y', 'z'>(cx_.get(), tid.x, tid.y, tid.z);

				auto current_ay = dist_l | noarr::get_at<'x', 'y', 'z'>(ay_.get(), tid.x, tid.y, tid.z);
				auto current_by = dist_l | noarr::get_at<'x', 'y', 'z'>(by_.get(), tid.x, tid.y, tid.z);
				auto current_cy = dist_l | noarr::get_at<'x', 'y', 'z'>(cy_.get(), tid.x, tid.y, tid.z);

				auto current_az = dist_l | noarr::get_at<'x', 'y', 'z'>(az_.get(), tid.x, tid.y, tid.z);
				auto current_bz = dist_l | noarr::get_at<'x', 'y', 'z'>(bz_.get(), tid.x, tid.y, tid.z);
				auto current_cz = dist_l | noarr::get_at<'x', 'y', 'z'>(cz_.get(), tid.x, tid.y, tid.z);

				auto dens_l =
					get_blocked_substrate_layout(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_x =
					get_blocked_substrate_layout<'x'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_y =
					get_blocked_substrate_layout<'y'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_z =
					get_blocked_substrate_layout<'z'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto diag_x =
					get_diag_layout<'x'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group], y_sync_step_);
				auto diag_y =
					get_diag_layout<'y'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group], y_sync_step_);
				auto diag_z =
					get_diag_layout<'z'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group], y_sync_step_);

				for (index_t blocked_z = block_z_begin; blocked_z < block_z_end; blocked_z += y_sync_step_)
				{
					const index_t y_sync_step_len = std::min(y_sync_step_, block_z_end - blocked_z);


					auto sync_x = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_x ^ noarr::fix<'s'>(s),
								   scratch = scratch_x_wol, dist_l = dist_l ^ noarr::fix<'y', 'z'>(tid.y, tid.z),
								   n = this->problem_.nx, sync_step = y_sync_step_,
								   n_alignment = (index_t)alignment_size_ / (index_t)sizeof(real_t), tid = tid.x,
								   z_begin = blocked_z - block_z_begin,
								   z_end = blocked_z + y_sync_step_len - block_z_begin, group_size = group_size_x,
								   &barrier = barrier_x]() {
						synchronize_x_blocked_distributed(densities, a, c, dens_l, scratch, dist_l, n, n_alignment,
														  z_begin, z_end, sync_step, tid, group_size, barrier);
					};

					{
						perf_counter counter("sdd-fb-x");

						if (cores_division_[0] != 1)
						{
							if (!alt_blocked_)
								solve_block_x_transpose(current_densities, current_ax, current_bx, current_cx,
														current_a_scratch, current_c_scratch, dens_l, diag_x, scratch_x,
														block_x_begin, block_x_end, blocked_z - block_z_begin,
														blocked_z + y_sync_step_len - block_z_begin, y_sync_step_, s,
														x_tile_size_, std::move(sync_x));
							else
								solve_block_x_transpose_alt(
									current_densities, current_ax, current_bx, current_cx, current_a_scratch,
									current_c_scratch, dens_l, diag_x, scratch_x, block_x_begin, block_x_end,
									blocked_z - block_z_begin, blocked_z + y_sync_step_len - block_z_begin,
									y_sync_step_, s, x_tile_size_, std::move(sync_x));
						}
						else
							solve_slice_x_2d_and_3d_transpose_l<index_t>(
								current_densities, current_ax, current_bx, current_cx, current_a_scratch, dens_l,
								diag_x,
								get_non_blocked_scratch_layout<'x'>(group_block_lengthsx_[tid.x],
																	alignment_size_ / sizeof(real_t)),
								s, blocked_z - block_z_begin, blocked_z + y_sync_step_len - block_z_begin,
								this->problem_.nx);
					}


					auto sync_y = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_y ^ noarr::fix<'s'>(s),
								   scratch = scratch_y_wol, dist_l = dist_l ^ noarr::fix<'x', 'z'>(tid.x, tid.z),
								   n = this->problem_.ny, tid = tid.y, group_size = group_size_y,
								   &barrier = barrier_y](index_t z_begin, index_t z_end) {
						synchronize_y_blocked_distributed(densities, a, c, dens_l, scratch, dist_l, n, z_begin, z_end,
														  tid, group_size, barrier);
					};

					{
						if (cores_division_[1] != 1)
						{
							if (!alt_blocked_)
								solve_block_y(current_densities, current_ay, current_by, current_cy, current_a_scratch,
											  current_c_scratch, dens_l_wo_y, diag_y, scratch_y, block_y_begin,
											  block_y_end, blocked_z - block_z_begin,
											  blocked_z + y_sync_step_len - block_z_begin, s, x_tile_size_,
											  std::move(sync_y));

							else
								solve_block_y_alt(current_densities, current_ay, current_by, current_cy,
												  current_a_scratch, current_c_scratch, dens_l_wo_y, diag_y, scratch_y,
												  block_y_begin, block_y_end, blocked_z - block_z_begin,
												  blocked_z + y_sync_step_len - block_z_begin, s,
												  group_block_lengthsx_[tid.x], x_tile_size_, std::move(sync_y));
						}
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_y_3d<index_t>(
								current_densities, current_ay, current_by, current_cy, current_a_scratch, dens_l,
								diag_y,
								get_non_blocked_scratch_layout<'y'>(
									group_block_lengthsy_[tid.y], std::min(x_tile_size_, group_block_lengthsx_[tid.x])),
								s, z - block_z_begin, x_tile_size_);
					}
				}

				if (this->problem_.dims == 3)
				{
					auto sync_z = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_z ^ noarr::fix<'s'>(s),
								   scratch = scratch_z_wol, dist_l = dist_l ^ noarr::fix<'x', 'y'>(tid.x, tid.y),
								   n = this->problem_.nz, tid = tid.z, group_size = group_size_z,
								   &barrier = barrier_z](index_t y_begin, index_t y_end) {
						synchronize_z_blocked_distributed(densities, a, c, dens_l, scratch, dist_l, n, y_begin, y_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[2] != 1)
					{
						if (!alt_blocked_)
							solve_block_z(current_densities, current_az, current_bz, current_cz, current_a_scratch,
										  current_c_scratch, dens_l_wo_z, diag_z, scratch_z, block_z_begin, block_z_end,
										  s, z_sync_step_, x_tile_size_, std::move(sync_z));
						else
							solve_block_z_alt(current_densities, current_az, current_bz, current_cz, current_a_scratch,
											  current_c_scratch, dens_l_wo_z, diag_z, scratch_z, block_z_begin,
											  block_z_end, s, group_block_lengthsx_[tid.x], z_sync_step_, x_tile_size_,
											  std::move(sync_z));
					}
					else
						solve_slice_z_3d<index_t>(
							current_densities, current_az, current_bz, current_cz, current_a_scratch, dens_l, diag_z,
							get_non_blocked_scratch_layout<'z'>(group_block_lengthsz_[tid.z],
																std::min(x_tile_size_, group_block_lengthsx_[tid.x])),
							s, x_tile_size_);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::solve_nf()
{
	for (index_t i = 0; i < countersx_count_; i++)
	{
		countersx_[i]->value = 0;
	}
	for (index_t i = 0; i < countersy_count_; i++)
	{
		countersy_[i]->value = 0;
	}
	for (index_t i = 0; i < countersz_count_; i++)
	{
		countersz_[i]->value = 0;
	}

#pragma omp parallel
	{
		perf_counter counter("sdd-fb");

		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_x_begin = group_block_offsetsx_[tid.x];
		const auto block_x_end = block_x_begin + group_block_lengthsx_[tid.x];

		const auto block_y_begin = group_block_offsetsy_[tid.y];
		const auto block_y_end = block_y_begin + group_block_lengthsy_[tid.y];

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_x = get_lane_id('x');

		barrier_t<true, index_t> barrier_x(cores_division_[0], countersx_[lane_id_x]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];

		const auto lane_id_y = get_lane_id('y');

		barrier_t<true, index_t> barrier_y(cores_division_[1], countersy_[lane_id_y]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];

		const auto lane_id_z = get_lane_id('z');

		barrier_t<true, index_t> barrier_z(cores_division_[2], countersz_[lane_id_z]->value);
		// auto& barrier_z = *barriersz_[lane_id_z];

		const index_t group_size_x = cores_division_[0];
		const index_t group_size_y = cores_division_[1];
		const index_t group_size_z = cores_division_[2];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s++)
		{
			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s << " s_end: " << s + 1
			// 					  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
			// 					  << " block_z_begin: " << block_z_begin << " block_z_end: " << block_z_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				auto scratch_x = get_scratch_layout<'x', true, false>(group_block_lengthsx_[tid.x],
																	  group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_y = get_scratch_layout<'y', true>(group_block_lengthsx_[tid.x],
															   group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_z = get_scratch_layout<'z', true>(group_block_lengthsx_[tid.x], z_sync_step_,
															   group_block_lengthsz_[tid.z]);
				auto scratch_x_wol = get_scratch_layout<'x', false, false>(group_block_lengthsx_[tid.x],
																		   group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_y_wol = get_scratch_layout<'y', false>(group_block_lengthsx_[tid.x],
																	group_block_lengthsy_[tid.y], y_sync_step_);
				auto scratch_z_wol = get_scratch_layout<'z', false>(group_block_lengthsx_[tid.x], z_sync_step_,
																	group_block_lengthsz_[tid.z]);

				auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout() ^ noarr::fix<'g'>(tid.group);

				auto current_a_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(a_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_c_scratch = dist_l | noarr::get_at<'x', 'y', 'z'>(c_scratch_.get(), tid.x, tid.y, tid.z);
				auto current_densities =
					dist_l | noarr::get_at<'x', 'y', 'z'>(thread_substrate_array_.get(), tid.x, tid.y, tid.z);

				auto current_ax = dist_l | noarr::get_at<'x', 'y', 'z'>(ax_.get(), tid.x, tid.y, tid.z);
				auto current_bx = dist_l | noarr::get_at<'x', 'y', 'z'>(bx_.get(), tid.x, tid.y, tid.z);
				auto current_cx = dist_l | noarr::get_at<'x', 'y', 'z'>(cx_.get(), tid.x, tid.y, tid.z);

				auto current_ay = dist_l | noarr::get_at<'x', 'y', 'z'>(ay_.get(), tid.x, tid.y, tid.z);
				auto current_by = dist_l | noarr::get_at<'x', 'y', 'z'>(by_.get(), tid.x, tid.y, tid.z);
				auto current_cy = dist_l | noarr::get_at<'x', 'y', 'z'>(cy_.get(), tid.x, tid.y, tid.z);

				auto current_az = dist_l | noarr::get_at<'x', 'y', 'z'>(az_.get(), tid.x, tid.y, tid.z);
				auto current_bz = dist_l | noarr::get_at<'x', 'y', 'z'>(bz_.get(), tid.x, tid.y, tid.z);
				auto current_cz = dist_l | noarr::get_at<'x', 'y', 'z'>(cz_.get(), tid.x, tid.y, tid.z);

				auto dens_l =
					get_blocked_substrate_layout(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_x =
					get_blocked_substrate_layout<'x'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_y =
					get_blocked_substrate_layout<'y'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto dens_l_wo_z =
					get_blocked_substrate_layout<'z'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													  group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				auto diag_x = get_diag_layout<'x', false>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
														  group_block_lengthsz_[tid.z],
														  group_block_lengthss_[tid.group], y_sync_step_);
				auto diag_y =
					get_diag_layout<'y'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group], y_sync_step_);
				auto diag_z =
					get_diag_layout<'z'>(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group], y_sync_step_);

				for (index_t blocked_z = block_z_begin; blocked_z < block_z_end; blocked_z += y_sync_step_)
				{
					const index_t y_sync_step_len = std::min(y_sync_step_, block_z_end - blocked_z);


					auto sync_x = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_x ^ noarr::fix<'s'>(s),
								   scratch = scratch_x_wol, dist_l = dist_l ^ noarr::fix<'y', 'z'>(tid.y, tid.z),
								   n = this->problem_.nx,
								   n_alignment = (index_t)alignment_size_ / (index_t)sizeof(real_t), tid = tid.x,
								   group_size = group_size_x, &barrier = barrier_x](index_t z_begin, index_t z_end) {
						synchronize_x_blocked_distributed_nf(densities, a, c, dens_l, scratch, dist_l, n, n_alignment,
															 z_begin, z_end, tid, group_size, barrier);
					};

					if (cores_division_[0] != 1)
						solve_block_x_transpose_nf(
							current_densities, current_ax, current_bx, current_cx, current_a_scratch, current_c_scratch,
							dens_l, diag_x, scratch_x, block_x_begin, block_x_end, blocked_z - block_z_begin,
							blocked_z + y_sync_step_len - block_z_begin, s, x_tile_size_, std::move(sync_x));
					else
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_x_2d_and_3d_transpose_l_nf<index_t>(
								current_densities, current_ax, current_bx, current_cx, current_a_scratch, dens_l,
								diag_x,
								get_non_blocked_scratch_layout<'x'>(group_block_lengthsx_[tid.x],
																	alignment_size_ / sizeof(real_t)),
								s, z - block_z_begin, this->problem_.nx);


					auto sync_y = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_y ^ noarr::fix<'s'>(s),
								   scratch = scratch_y_wol, dist_l = dist_l ^ noarr::fix<'x', 'z'>(tid.x, tid.z),
								   n = this->problem_.ny, tid = tid.y, group_size = group_size_y,
								   &barrier = barrier_y](index_t z_begin, index_t z_end) {
						synchronize_y_blocked_distributed(densities, a, c, dens_l, scratch, dist_l, n, z_begin, z_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[1] != 1)
						solve_block_y(current_densities, current_ay, current_by, current_cy, current_a_scratch,
									  current_c_scratch, dens_l_wo_y, diag_y, scratch_y, block_y_begin, block_y_end,
									  blocked_z - block_z_begin, blocked_z + y_sync_step_len - block_z_begin, s,
									  x_tile_size_, std::move(sync_y));
					else
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_y_3d<index_t>(
								current_densities, current_ay, current_by, current_cy, current_a_scratch, dens_l,
								diag_y,
								get_non_blocked_scratch_layout<'y'>(
									group_block_lengthsy_[tid.y], std::min(x_tile_size_, group_block_lengthsx_[tid.x])),
								s, z - block_z_begin, x_tile_size_);
				}

				if (this->problem_.dims == 3)
				{
					auto sync_z = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_z ^ noarr::fix<'s'>(s),
								   scratch = scratch_z_wol, dist_l = dist_l ^ noarr::fix<'x', 'y'>(tid.x, tid.y),
								   n = this->problem_.nz, tid = tid.z, group_size = group_size_z,
								   &barrier = barrier_z](index_t y_begin, index_t y_end) {
						synchronize_z_blocked_distributed(densities, a, c, dens_l, scratch, dist_l, n, y_begin, y_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[2] != 1)
						solve_block_z(current_densities, current_az, current_bz, current_cz, current_a_scratch,
									  current_c_scratch, dens_l_wo_z, diag_z, scratch_z, block_z_begin, block_z_end, s,
									  z_sync_step_, x_tile_size_, std::move(sync_z));
					else
						solve_slice_z_3d<index_t>(
							current_densities, current_az, current_bz, current_cz, current_a_scratch, dens_l, diag_z,
							get_non_blocked_scratch_layout<'z'>(group_block_lengthsz_[tid.z],
																std::min(x_tile_size_, group_block_lengthsx_[tid.x])),
							s, x_tile_size_);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
sdd_full_blocking<real_t, aligned_x>::sdd_full_blocking()
{}

template <typename real_t, bool aligned_x>
sdd_full_blocking<real_t, aligned_x>::~sdd_full_blocking()
{
	for (index_t i = 0; i < get_max_threads(); i++)
	{
		if (ax_)
		{
			std::free(ax_[i]);
			std::free(bx_[i]);
			std::free(cx_[i]);
			std::free(thread_substrate_array_[i]);
			std::free(a_scratch_[i]);
			std::free(c_scratch_[i]);
		}
		if (ay_)
		{
			std::free(ay_[i]);
			std::free(by_[i]);
			std::free(cy_[i]);
		}
		if (az_)
		{
			std::free(az_[i]);
			std::free(bz_[i]);
			std::free(cz_[i]);
		}
	}
}


template <typename real_t, bool aligned_x>
double sdd_full_blocking<real_t, aligned_x>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
	index_t block_idx_x = 0;
	while ((index_t)x >= group_block_offsetsx_[block_idx_x] + group_block_lengthsx_[block_idx_x])
	{
		block_idx_x++;
	}
	x -= group_block_offsetsx_[block_idx_x];

	index_t block_idx_y = 0;
	while ((index_t)y >= group_block_offsetsy_[block_idx_y] + group_block_lengthsy_[block_idx_y])
	{
		block_idx_y++;
	}
	y -= group_block_offsetsy_[block_idx_y];

	index_t block_idx_z = 0;
	while ((index_t)z >= group_block_offsetsz_[block_idx_z] + group_block_lengthsz_[block_idx_z])
	{
		block_idx_z++;
	}
	z -= group_block_offsetsz_[block_idx_z];

	index_t block_idx_s = 0;
	while ((index_t)s >= group_block_offsetss_[block_idx_s] + group_block_lengthss_[block_idx_s])
	{
		block_idx_s++;
	}
	s -= group_block_offsetss_[block_idx_s];

	auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout();

	auto density = dist_l
				   | noarr::get_at<'g', 'x', 'y', 'z'>(thread_substrate_array_.get(), block_idx_s, block_idx_x,
													   block_idx_y, block_idx_z);

	auto dens_l = get_blocked_substrate_layout(group_block_lengthsx_[block_idx_x], group_block_lengthsy_[block_idx_y],
											   group_block_lengthsz_[block_idx_z], group_block_lengthss_[block_idx_s]);

	return dens_l | noarr::get_at<'x', 'y', 'z', 's'>(density, x, y, z, s);
}

template class sdd_full_blocking<float, true>;
template class sdd_full_blocking<double, true>;
