#include "full_blocking.h"

#include <cstddef>
#include <iostream>

#include "../barrier.h"
#include "../perf_utils.h"
#include "../vector_transpose_helper.h"

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
template <bool dim_x>
void sdd_full_blocking<real_t, aligned_x>::precompute_values(std::unique_ptr<real_t*[]>& a,
															 std::unique_ptr<real_t*[]>& b,
															 std::unique_ptr<real_t*[]>& c, index_t shape, index_t n,
															 index_t dims, char dim)
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
			if constexpr (dim_x)
				return get_diag_layout_x(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
										 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);
			else
				return get_blocked_substrate_layout(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
													group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);
		};

		auto dens_t_l = get_layout();


		a_t = (real_t*)std::aligned_alloc(alignment_size_, (dens_t_l | noarr::get_size()));
		b_t = (real_t*)std::aligned_alloc(alignment_size_, (dens_t_l | noarr::get_size()));
		c_t = (real_t*)std::aligned_alloc(alignment_size_, (dens_t_l | noarr::get_size()));

		auto a_bag = noarr::make_bag(dens_t_l, a_t);
		auto b_bag = noarr::make_bag(dens_t_l, b_t);
		auto c_bag = noarr::make_bag(dens_t_l, c_t);

		auto get_diffusion_coefficients = [&](index_t, index_t, index_t, index_t s) {
			return this->problem_.diffusion_coefficients[s];
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
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;

	cores_division_ = params.contains("cores_division") ? (std::array<index_t, 3>)params["cores_division"]
														: std::array<index_t, 3> { 1, 2, 2 };

	y_sync_step_ = params.contains("y_sync_step") ? (index_t)params["y_sync_step"] : 1;
	z_sync_step_ = params.contains("z_sync_step") ? (index_t)params["z_sync_step"] : 1;

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
	alignment_size_ = std::max(alignment_size_, vector_length);
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::initialize()
{
	precompute_values<true>(ax_, bx_, cx_, this->problem_.dx, this->problem_.nx, this->problem_.dims, 'x');
	precompute_values<false>(ay_, by_, cy_, this->problem_.dy, this->problem_.ny, this->problem_.dims, 'y');
	precompute_values<false>(az_, bz_, cz_, this->problem_.dz, this->problem_.nz, this->problem_.dims, 'z');

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

	auto non_blocked_scratch_l = get_non_blocked_scratch_layout<'x'>();
	auto scratch_lx = get_scratch_layout<'x'>(group_blocks_[0], group_blocks_[1] + 1, y_sync_step_);
	auto scratch_ly = get_scratch_layout<'y'>(group_blocks_[0], group_blocks_[1] + 1, y_sync_step_);
	auto scratch_lz = get_scratch_layout<'z'>(group_blocks_[0], z_sync_step_, group_blocks_[2] + 1);

	auto max_size = std::max({ scratch_lx | noarr::get_size(), scratch_ly | noarr::get_size(),
							   scratch_lz | noarr::get_size(), non_blocked_scratch_l | noarr::get_size() });

	a_scratch_ = std::make_unique<real_t*[]>(get_max_threads());
	c_scratch_ = std::make_unique<real_t*[]>(get_max_threads());

#pragma omp parallel
	{
		auto tid = get_thread_num();
		a_scratch_[tid] = (real_t*)std::aligned_alloc(alignment_size_, max_size);
		c_scratch_[tid] = (real_t*)std::aligned_alloc(alignment_size_, max_size);
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


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_x_blocked_distributed(real_t** __restrict__ densities, real_t** __restrict__ a_data,
														real_t** __restrict__ c_data, const density_layout_t dens_l,
														const diagonal_layout_t diag_l,
														const thread_distribution_l dist_l, const index_t n,
														const index_t z_begin, const index_t z_end, const index_t tid,
														const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t simd_y_len = y_len / simd_length * simd_length;

	const index_t block_size = n / coop_size;

	const index_t z_len = z_end - z_begin;
	const index_t block_size_z = z_len / coop_size;
	const index_t t_z_begin = z_begin + tid * block_size_z + std::min(tid, z_len % coop_size);
	const index_t t_z_end = t_z_begin + block_size_z + ((tid < z_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_simd_begin << " block_end: " << x_simd_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size](index_t equation_idx, index_t y) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);

		const auto transposed_y_offset = offset % simd_length;
		const auto transposed_x = offset - transposed_y_offset;

		return std::make_tuple(block_idx,
							   noarr::set_length<'x'>(actual_block_size)
								   ^ noarr::fix<'x', 'y'>(transposed_x, y + transposed_y_offset),
							   noarr::fix<'x', 'y'>(offset, y));
	};

	for (index_t z = t_z_begin; z < t_z_end; z++)
	{
		for (index_t y = 0; y < simd_y_len; y += simd_length)
		{
			simd_t prev_c;
			simd_t prev_d;

			{
				const auto [prev_block_idx, fix_dens, fix_diag] = get_i(0, y);
				const auto prev_c_bag =
					noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, prev_block_idx));
				const auto prev_d_bag =
					noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, prev_block_idx));

				prev_c = hn::Load(t, &prev_c_bag.template at<'z'>(z - z_begin));
				prev_d = hn::Load(t, &prev_d_bag.template at<'z'>(z));
			}

			for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
			{
				const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx, y);

				const auto a = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(a_data, block_idx));
				const auto c = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
				const auto d = noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

				simd_t curr_a = hn::Load(t, &a.template at<'z'>(z - z_begin));
				simd_t curr_c = hn::Load(t, &c.template at<'z'>(z - z_begin));
				simd_t curr_d = hn::Load(t, &d.template at<'z'>(z));

				simd_t r = hn::Div(hn::Set(t, 1), hn::NegMulAdd(prev_c, curr_a, hn::Set(t, 1)));

				curr_d = hn::Mul(r, hn::NegMulAdd(prev_d, curr_a, curr_d));
				curr_c = hn::Mul(r, curr_c);

				hn::Store(curr_c, t, &c.template at<'z'>(z - z_begin));
				hn::Store(curr_d, t, &d.template at<'z'>(z));

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

				const auto c = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
				const auto d = noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

				simd_t curr_c = hn::Load(t, &c.template at<'z'>(z - z_begin));
				simd_t curr_d = hn::Load(t, &d.template at<'z'>(z));

				curr_d = hn::NegMulAdd(prev_d, curr_c, curr_d);

				hn::Store(curr_d, t, &d.template at<'z'>(z));

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

	barrier.arrive_and_wait();
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename sync_func_t>
static void solve_block_x_transpose(real_t* __restrict__ densities, const real_t* __restrict__ a,
									const real_t* __restrict__ b, const real_t* __restrict__ c,
									real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
									const density_layout_t dens_l, const diagonal_layout_t diag_l,
									const scratch_layout_t scratch_l, const index_t x_begin, const index_t x_end,
									const index_t z_begin, const index_t z_end, const index_t s, const index_t,
									sync_func_t&& synchronize_blocked_x)
{
	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s) ^ noarr::set_length<'x'>(x_end - x_begin);

	const index_t n = blocked_dens_l | noarr::get_length<'x'>();
	const index_t y_len = blocked_dens_l | noarr::get_length<'y'>();

	auto a_scratch_bag = noarr::make_bag(scratch_l, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l, c_scratch);

	const auto step_len = z_end - z_begin;

	auto a_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'z'>(z_begin, step_len), a);
	auto b_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'z'>(z_begin, step_len), b);
	auto c_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s) ^ noarr::slice<'z'>(z_begin, step_len), c);
	auto d_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), densities);

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	simd_t d_rows[simd_length];

	const index_t simd_y_len = y_len / simd_length * simd_length;

	for (index_t z = 0; z < step_len; z++)
	{
		const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

		for (index_t y = 0; y < simd_y_len; y += simd_length)
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
					d_rows[v] = hn::Load(d, &d_bag.template at<'z', 'y', 'x'>(z, y + v, i));

				// transposition to enable vectorization
				transpose(d_rows);

				for (index_t v = 0; v < simd_length; v++)
				{
					const index_t x = i + v;

					const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, x);

					simd_t a_curr = hn::Load(d, &(a_bag[idx]));
					simd_t b_curr = hn::Load(d, &(b_bag[idx]));
					simd_t c_curr = hn::Load(d, &(c_bag[idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);

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
						// 								std::cout << "f " << z_begin + z << " " << y + l << " " <<
						// x_begin + x << " "
						// 										  << hn::ExtractLane(a_curr, l) << " " <<
						// hn::ExtractLane(b_curr, l) << " "
						// 										  << hn::ExtractLane(r, l) << " " <<
						// hn::ExtractLane(d_rows[v], l) << std::endl;
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
					hn::Store(d_rows[v], d, &(d_bag.template at<'z', 'y', 'x'>(z, y + v, i)));
			}

			// we are aligned to the vector size, so we can safely continue
			// here we fuse the end of forward substitution and the beginning of backwards propagation
			{
				for (index_t v = 0; v < simd_length; v++)
					d_rows[v] = hn::Load(d, &(d_bag.template at<'z', 'y', 'x'>(z, y + v, full_n - simd_length)));

				// transposition to enable vectorization
				transpose(d_rows);

				index_t remainder_work = n % simd_length;
				remainder_work += remainder_work == 0 ? simd_length : 0;

				// the rest of forward part
				for (index_t v = 0; v < remainder_work; v++)
				{
					const index_t x = full_n - simd_length + v;

					const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, x);

					simd_t a_curr = hn::Load(d, &(a_bag[idx]));
					simd_t b_curr = hn::Load(d, &(b_bag[idx]));
					simd_t c_curr = hn::Load(d, &(c_bag[idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);

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
						// 								std::cout << "f " << z_begin + z << " " << y + l << " " <<
						// x_begin + x << " "
						// 										  << hn::ExtractLane(a_curr, l) << " " <<
						// hn::ExtractLane(b_curr, l) << " "
						// 										  << hn::ExtractLane(r, l) << " " <<
						// hn::ExtractLane(d_rows[v], l) << std::endl;
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

					const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, x);

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
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "  <<
						// hn::ExtractLane(c_scratch_curr, l) << " " << 1 << " "
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
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "  <<
						// hn::ExtractLane(c_scratch_curr, l) << " " << hn::ExtractLane(r, l) << " "
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
					hn::Store(d_rows[v], d, &(d_bag.template at<'z', 'y', 'x'>(z, y + v, full_n - simd_length)));
			}

			// we continue with backwards substitution
			for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
					d_rows[v] = hn::Load(d, &(d_bag.template at<'z', 'y', 'x'>(z, y + v, i)));

				// backward propagation
				for (index_t v = simd_length - 1; v >= 0; v--)
				{
					const index_t x = i + v;

					const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, x);

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
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "  <<
						// hn::ExtractLane(c_scratch_curr, l) << " " << 1<< " "
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
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "  <<
						// hn::ExtractLane(c_scratch_curr, l) << " " << hn::ExtractLane(r, l) << " "
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
					hn::Store(d_rows[v], d, &(d_bag.template at<'z', 'y', 'x'>(z, y + v, i)));
			}
		}
	}

	synchronize_blocked_x(z_begin, z_end);

	for (index_t z = 0; z < step_len; z++)
	{
		for (index_t y = 0; y < simd_y_len; y += simd_length)
		{
			const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

			const simd_t begin_unknowns = hn::Load(d, &(d_bag.template at<'y', 'z', 'x'>(y, z, 0)));

			const auto transposed_y_offset = (n - 1) % simd_length;
			const auto transposed_x = (n - 1) - transposed_y_offset;

			const simd_t end_unknowns =
				hn::Load(d, &(d_bag.template at<'y', 'z', 'x'>(y + transposed_y_offset, z, transposed_x)));

			for (index_t i = 0; i < full_n; i += simd_length)
			{
				for (index_t v = 0; v < simd_length; v++)
					d_rows[v] = hn::Load(d, &(d_bag.template at<'z', 'y', 'x'>(z, y + v, i)));

				for (index_t v = 0; v < simd_length; v++)
				{
					index_t x = i + v;

					const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, x);

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
					hn::Store(d_rows[v], d, &(d_bag.template at<'z', 'y', 'x'>(z, y + v, i)));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_x_2d_and_3d_transpose_l(real_t* __restrict__ densities, const real_t* __restrict__ a,
												const real_t* __restrict__ b, const real_t* __restrict__ c,
												real_t* __restrict__ b_scratch, const density_layout_t dens_l,
												const diagonal_layout_t diag_l, const scratch_layout_t scratch_l,
												const index_t s, const index_t z, index_t n)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	simd_t d_rows[simd_length];

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t simd_y_len = y_len / simd_length * simd_length;

	// vectorized body
	{
		const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

		for (index_t y = 0; y < simd_y_len; y += simd_length)
		{
			// vector registers that hold the to be transposed x*yz plane

			simd_t c_prev = hn::Zero(d);
			simd_t d_prev = hn::Zero(d);
			simd_t scratch_prev = hn::Zero(d);

			// forward substitution until last simd_length elements
			for (index_t i = 0; i < full_n - simd_length; i += simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
				{
					d_rows[v] = hn::Load(d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, i, s)));
				}

				// transposition to enable vectorization
				transpose(d_rows);

				for (index_t v = 0; v < simd_length; v++)
				{
					simd_t a_curr = hn::Load(d, &(diag_l | noarr::get_at<'z', 'y', 'x', 's'>(a, z, y, i + v, s)));
					simd_t b_curr = hn::Load(d, &(diag_l | noarr::get_at<'z', 'y', 'x', 's'>(b, z, y, i + v, s)));

					auto r = hn::Mul(a_curr, scratch_prev);

					scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
					hn::Store(scratch_prev, d, &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, i + v, 0)));

					d_rows[v] = hn::NegMulAdd(d_prev, r, d_rows[v]);

					d_prev = d_rows[v];
					c_prev = hn::Load(d, &(diag_l | noarr::get_at<'z', 'y', 'x', 's'>(c, z, y, i + v, s)));
				}

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(d_rows[v], d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, i, s)));
				}
			}

			// we are aligned to the vector size, so we can safely continue
			// here we fuse the end of forward substitution and the beginning of backwards propagation
			{
				for (index_t v = 0; v < simd_length; v++)
				{
					d_rows[v] = hn::Load(
						d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, full_n - simd_length, s)));
				}

				// transposition to enable vectorization
				transpose(d_rows);

				index_t remainder_work = n % simd_length;
				remainder_work += remainder_work == 0 ? simd_length : 0;

				// the rest of forward part
				{
					for (index_t v = 0; v < remainder_work; v++)
					{
						simd_t a_curr = hn::Load(
							d, &(diag_l | noarr::get_at<'z', 'y', 'x', 's'>(a, z, y, full_n - simd_length + v, s)));
						simd_t b_curr = hn::Load(
							d, &(diag_l | noarr::get_at<'z', 'y', 'x', 's'>(b, z, y, full_n - simd_length + v, s)));

						auto r = hn::Mul(a_curr, scratch_prev);

						scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
						hn::Store(scratch_prev, d,
								  &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, full_n - simd_length + v, 0)));

						d_rows[v] = hn::NegMulAdd(d_prev, r, d_rows[v]);

						d_prev = d_rows[v];
						c_prev = hn::Load(
							d, &(diag_l | noarr::get_at<'z', 'y', 'x', 's'>(c, z, y, full_n - simd_length + v, s)));
					}
				}

				{
					d_prev = hn::Mul(d_prev, scratch_prev);
					d_rows[remainder_work - 1] = d_prev;
				}

				// the begin of backward part
				{
					for (index_t v = remainder_work - 2; v >= 0; v--)
					{
						simd_t c_curr = hn::Load(
							d, &(diag_l | noarr::get_at<'z', 'y', 'x', 's'>(c, z, y, full_n - simd_length + v, s)));

						auto scratch =
							hn::Load(d, &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, full_n - simd_length + v, 0)));
						d_rows[v] = hn::Mul(hn::NegMulAdd(d_prev, c_curr, d_rows[v]), scratch);

						d_prev = d_rows[v];
					}
				}

				// transposition back to the original form
				transpose(d_rows);

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(
						d_rows[v], d,
						&(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, full_n - simd_length, s)));
				}
			}

			// we continue with backwards substitution
			for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
				{
					d_rows[v] = hn::Load(d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, i, s)));
				}

				// backward propagation
				{
					for (index_t v = simd_length - 1; v >= 0; v--)
					{
						simd_t c_curr = hn::Load(d, &(diag_l | noarr::get_at<'z', 'y', 'x', 's'>(c, z, y, i + v, s)));

						auto scratch = hn::Load(d, &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, i + v, 0)));
						d_rows[v] = hn::Mul(hn::NegMulAdd(d_prev, c_curr, d_rows[v]), scratch);

						d_prev = d_rows[v];
					}
				}

				// transposition back to the original form
				transpose(d_rows);

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(d_rows[v], d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, y + v, i, s)));
				}
			}
		}
	}

	// 	// yz remainder
	// 	{
	// 		auto d = noarr::make_bag(dens_l, densities);

	// 		auto scratch = noarr::make_bag(scratch_l, b_scratch);

	// #pragma omp for schedule(static) nowait
	// 		for (index_t yz = simd_yz_len; yz < m; yz++)
	// 		{
	// 			{
	// 				auto idx = noarr::idx<'s', 'v', 'x'>(s, yz, 0);
	// 				scratch[idx] = 1 / b_bag[idx];
	// 			}

	// 			for (index_t i = 1; i < n; i++)
	// 			{
	// 				auto idx = noarr::idx<'s', 'v', 'x'>(s, yz, i);
	// 				auto prev_idx = noarr::idx<'s', 'v', 'x'>(s, yz, i - 1);

	// 				auto r = a_bag[prev_idx] * scratch[prev_idx];

	// 				scratch[idx] = 1 / (b_bag[idx] - c_bag[idx] * r);

	// 				d[idx] -= r * d[prev_idx];

	// 				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
	// 			}

	// 			{
	// 				auto idx = noarr::idx<'s', 'v', 'x'>(s, yz, n - 1);
	// 				d[idx] *= scratch[idx];

	// 				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
	// 			}

	// 			for (index_t i = n - 2; i >= 0; i--)
	// 			{
	// 				auto idx = noarr::idx<'s', 'v', 'x'>(s, yz, i);
	// 				auto next_idx = noarr::idx<'s', 'v', 'x'>(s, yz, i + 1);

	// 				d[idx] = (d[idx] - c_bag[next_idx] * d[next_idx]) * scratch[idx];

	// 				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
	// 			}
	// 		}
	// 	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_y_blocked_distributed(real_t** __restrict__ densities, real_t** __restrict__ a_data,
														real_t** __restrict__ c_data, const density_layout_t dens_l,
														const diagonal_layout_t diag_l,
														const thread_distribution_l dist_l, const index_t n,
														const index_t z_begin, const index_t z_end, const index_t tid,
														const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t x_simd_len = (x_len + simd_length - 1) / simd_length * simd_length;

	const index_t block_size = n / coop_size;

	const index_t z_len = z_end - z_begin;
	const index_t block_size_z = z_len / coop_size;
	const index_t t_z_begin = z_begin + tid * block_size_z + std::min(tid, z_len % coop_size);
	const index_t t_z_end = t_z_begin + block_size_z + ((tid < z_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_simd_begin << " block_end: " << x_simd_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		return std::make_tuple(block_idx, noarr::set_length<'y'>(actual_block_size) ^ noarr::fix<'y'>(offset),
							   noarr::fix<'y'>(offset));
	};

	for (index_t z = t_z_begin; z < t_z_end; z++)
	{
		for (index_t x = 0; x < x_simd_len; x += simd_length)
		{
			simd_t prev_c;
			simd_t prev_d;

			{
				const auto [prev_block_idx, fix_dens, fix_diag] = get_i(0);
				const auto prev_c_bag =
					noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'y'>(c_data, prev_block_idx));
				const auto prev_d_bag =
					noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'y'>(densities, prev_block_idx));

				prev_c = hn::Load(t, &prev_c_bag.template at<'x', 'z'>(x, z - z_begin));
				prev_d = hn::Load(t, &prev_d_bag.template at<'x', 'z'>(x, z));
			}

			for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
			{
				const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx);

				const auto a = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'y'>(a_data, block_idx));
				const auto c = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'y'>(c_data, block_idx));
				const auto d = noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'y'>(densities, block_idx));

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
				const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx);

				const auto c = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'y'>(c_data, block_idx));
				const auto d = noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'y'>(densities, block_idx));

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
	}

	barrier.arrive_and_wait();
}


template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t, typename sync_func_t>
static void solve_block_y(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
						  const real_t* __restrict__ c, real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
						  const density_layout_t dens_l, const scratch_layout_t scratch_l, const index_t y_begin,
						  const index_t y_end, const index_t z_begin, const index_t z_end, const index_t s,
						  const index_t, sync_func_t&& synchronize_blocked_y)
{
	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s) ^ noarr::set_length<'y'>(y_end - y_begin);

	const index_t y_len = blocked_dens_l | noarr::get_length<'y'>();

	const index_t x_len = blocked_dens_l | noarr::get_length<'x'>();

	auto a_scratch_bag = noarr::make_bag(scratch_l, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l, c_scratch);

	const auto step_len = z_end - z_begin;

	auto a_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), a);
	auto b_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), b);
	auto c_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), c);
	auto d_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), densities);

	for (index_t z = 0; z < step_len; z++)
	{
		// Normalize the first and the second equation
		for (index_t i = 0; i < 2; i++)
			for (index_t x = 0; x < x_len; x++)
			{
				const auto idx = noarr::idx<'z', 'x', 'y'>(z, x, i);

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
		for (index_t i = 2; i < y_len; i++)
			for (index_t x = 0; x < x_len; x++)
			{
				const auto prev_idx = noarr::idx<'z', 'x', 'y'>(z, x, i - 1);
				const auto idx = noarr::idx<'z', 'x', 'y'>(z, x, i);

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
		for (index_t i = y_len - 3; i >= 1; i--)
			for (index_t x = 0; x < x_len; x++)
			{
				const auto idx = noarr::idx<'z', 'x', 'y'>(z, x, i);
				const auto next_idx = noarr::idx<'z', 'x', 'y'>(z, x, i + 1);

				d_bag[idx] = d_bag[idx] - c_scratch_bag[idx] * d_bag[next_idx];

				a_scratch_bag[idx] = a_scratch_bag[idx] - c_scratch_bag[idx] * a_scratch_bag[next_idx];
				c_scratch_bag[idx] = 0 - c_scratch_bag[idx] * c_scratch_bag[next_idx];


				// #pragma omp critical
				// 				std::cout << "b0: " << z_begin + z << " " << i + y_begin << " " << x << " " <<
				// d_bag[idx] << std::endl;
			}

		// Process the first row (backward)
		for (index_t x = 0; x < x_len; x++)
		{
			const auto idx = noarr::idx<'z', 'x', 'y'>(z, x, 0);
			const auto next_idx = noarr::idx<'z', 'x', 'y'>(z, x, 1);

			const auto r = 1 / (1 - c_scratch_bag[idx] * a_scratch_bag[next_idx]);

			d_bag[idx] = r * (d_bag[idx] - c_scratch_bag[idx] * d_bag[next_idx]);

			a_scratch_bag[idx] = r * a_scratch_bag[idx];
			c_scratch_bag[idx] = r * (0 - c_scratch_bag[idx] * c_scratch_bag[next_idx]);


			// #pragma omp critical
			// 			std::cout << "b1: " << z_begin + z << " " << y_begin << " " << x << " " << d_bag[idx] <<
			// std::endl;
		}
	}

	synchronize_blocked_y(z_begin, z_end);

	for (index_t z = 0; z < step_len; z++)
	{
		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		{
			for (index_t i = 1; i < y_len - 1; i++)
				for (index_t x = 0; x < x_len; x++)
				{
					const auto idx_begin = noarr::idx<'z', 'x', 'y'>(z, x, 0);
					const auto idx = noarr::idx<'z', 'x', 'y'>(z, x, i);
					const auto idx_end = noarr::idx<'z', 'x', 'y'>(z, x, y_len - 1);

					d_bag[idx] =
						d_bag[idx] - a_scratch_bag[idx] * d_bag[idx_begin] - c_scratch_bag[idx] * d_bag[idx_end];

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
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_z_blocked_distributed(real_t** __restrict__ densities, real_t** __restrict__ a_data,
														real_t** __restrict__ c_data, const density_layout_t dens_l,
														const diagonal_layout_t diag_l,
														const thread_distribution_l dist_l, const index_t n,
														const index_t y_begin, const index_t y_end, const index_t tid,
														const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t x_simd_len = (x_len + simd_length - 1) / simd_length * simd_length;

	const index_t block_size = n / coop_size;

	const index_t y_len = y_end - y_begin;
	const index_t block_size_y = y_len / coop_size;
	const index_t t_y_begin = y_begin + tid * block_size_y + std::min(tid, y_len % coop_size);
	const index_t t_y_end = t_y_begin + block_size_y + ((tid < y_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		return std::make_tuple(block_idx, noarr::set_length<'z'>(actual_block_size) ^ noarr::fix<'z'>(offset),
							   noarr::fix<'z'>(offset));
	};

	for (index_t y = t_y_begin; y < t_y_end; y++)
	{
		for (index_t x = 0; x < x_simd_len; x += simd_length)
		{
			simd_t prev_c;
			simd_t prev_d;

			{
				const auto [prev_block_idx, fix_dens, fix_diag] = get_i(0);
				const auto prev_c_bag =
					noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'z'>(c_data, prev_block_idx));
				const auto prev_d_bag =
					noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'z'>(densities, prev_block_idx));

				prev_c = hn::Load(t, &prev_c_bag.template at<'x', 'y'>(x, y - y_begin));
				prev_d = hn::Load(t, &prev_d_bag.template at<'x', 'y'>(x, y));
			}

			for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
			{
				const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx);

				const auto a = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'z'>(a_data, block_idx));
				const auto c = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'z'>(c_data, block_idx));
				const auto d = noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'z'>(densities, block_idx));

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
				const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx);

				const auto c = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'z'>(c_data, block_idx));
				const auto d = noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'z'>(densities, block_idx));

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
	}

	barrier.arrive_and_wait();
}


template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t, typename sync_func_t>
static void solve_block_z(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
						  const real_t* __restrict__ c, real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
						  const density_layout_t dens_l, const scratch_layout_t scratch_l, const index_t z_begin,
						  const index_t z_end, const index_t s, const index_t, const index_t sync_step,
						  sync_func_t&& synchronize_blocked_z)
{
	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s) ^ noarr::set_length<'z'>(z_end - z_begin);

	const index_t y_len = blocked_dens_l | noarr::get_length<'y'>();
	const index_t z_len = blocked_dens_l | noarr::get_length<'z'>();

	const index_t x_block_len = blocked_dens_l | noarr::get_length<'x'>();

	auto a_scratch_bag = noarr::make_bag(scratch_l, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l, c_scratch);

	// Normalize the first and the second equation

	for (index_t blocked_y = 0; blocked_y < y_len; blocked_y += sync_step)
	{
		const auto step_len = std::min(y_len - blocked_y, sync_step);

		auto a_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'y'>(blocked_y, step_len), a);
		auto b_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'y'>(blocked_y, step_len), b);
		auto c_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'y'>(blocked_y, step_len), c);
		auto d_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'y'>(blocked_y, step_len), densities);

		for (index_t y = 0; y < step_len; y++)
		{
			for (index_t i = 0; i < 2; i++)
			{
				for (index_t x = 0; x < x_block_len; x++)
				{
					const auto idx = noarr::idx<'x', 'y', 'z'>(x, y, i);

					const auto r = 1 / b_bag[idx];

					a_scratch_bag[idx] = a_bag[idx] * r;
					c_scratch_bag[idx] = c_bag[idx] * r;
					d_bag[idx] = d_bag[idx] * r;

					// #pragma omp critical
					// 					std::cout << "f0: " << z_begin + i << " " << blocked_y + y << " " << x << " " <<
					// d_bag[idx] << " " << b_bag[idx]
					// 							  << std::endl;
				}
			}

			// Process the lower diagonal (forward)
			for (index_t i = 2; i < z_len; i++)
			{
				for (index_t x = 0; x < x_block_len; x++)
				{
					const auto prev_idx = noarr::idx<'x', 'y', 'z'>(x, y, i - 1);
					const auto idx = noarr::idx<'x', 'y', 'z'>(x, y, i);

					const auto r = 1 / (b_bag[idx] - a_bag[idx] * c_scratch_bag[prev_idx]);

					a_scratch_bag[idx] = r * (0 - a_bag[idx] * a_scratch_bag[prev_idx]);
					c_scratch_bag[idx] = r * c_bag[idx];

					d_bag[idx] = r * (d_bag[idx] - a_bag[idx] * d_bag[prev_idx]);


					// #pragma omp critical
					// 			std::cout << "f1: " << z << " " << i + y_begin << " " << x << " " << d_bag[idx]
					// 					  << " " << a_bag[idx]  << " " << b_bag[idx]  << " " << c_scratch_bag[idx] <<
					// std::endl;
				}
			}

			// Process the upper diagonal (backward)
			for (index_t i = z_len - 3; i >= 1; i--)
			{
				for (index_t x = 0; x < x_block_len; x++)
				{
					const auto idx = noarr::idx<'x', 'y', 'z'>(x, y, i);
					const auto next_idx = noarr::idx<'x', 'y', 'z'>(x, y, i + 1);

					d_bag[idx] -= c_scratch_bag[idx] * d_bag[next_idx];

					a_scratch_bag[idx] = a_scratch_bag[idx] - c_scratch_bag[idx] * a_scratch_bag[next_idx];
					c_scratch_bag[idx] = 0 - c_scratch_bag[idx] * c_scratch_bag[next_idx];


					// #pragma omp critical
					// 			std::cout << "b0: " << z << " " << i + y_begin << " " << x << " " << d_bag[idx] <<
					// std::endl;
				}
			}

			// Process the first row (backward)
			{
				for (index_t x = 0; x < x_block_len; x++)
				{
					const auto idx = noarr::idx<'x', 'y', 'z'>(x, y, 0);
					const auto next_idx = noarr::idx<'x', 'y', 'z'>(x, y, 1);

					const auto r = 1 / (1 - c_scratch_bag[idx] * a_scratch_bag[next_idx]);

					d_bag[idx] = r * (d_bag[idx] - c_scratch_bag[idx] * d_bag[next_idx]);

					a_scratch_bag[idx] = r * a_scratch_bag[idx];
					c_scratch_bag[idx] = r * (0 - c_scratch_bag[idx] * c_scratch_bag[next_idx]);


					// #pragma omp critical
					// 			std::cout << "b1: " << z << " " << y_begin << " " << x << " " << d_bag[idx] <<
					// std::endl;
				}
			}
		}

		synchronize_blocked_z(blocked_y, blocked_y + step_len);

		for (index_t y = 0; y < step_len; y++)
		{
			// Final part of modified thomas algorithm
			// Solve the rest of the unknowns
			{
				for (index_t i = 1; i < z_len - 1; i++)
				{
					for (index_t x = 0; x < x_block_len; x++)
					{
						const auto idx_begin = noarr::idx<'x', 'y', 'z'>(x, y, 0);
						const auto idx = noarr::idx<'x', 'y', 'z'>(x, y, i);
						const auto idx_end = noarr::idx<'x', 'y', 'z'>(x, y, z_len - 1);

						d_bag[idx] =
							d_bag[idx] - a_scratch_bag[idx] * d_bag[idx_begin] - c_scratch_bag[idx] * d_bag[idx_end];

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


template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const scratch_layout_t scratch_l, const index_t s_idx,
							 const index_t z, index_t x_tile_size)
{
	const index_t n = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s_idx) ^ noarr::into_blocks<'x', 'x', 'v'>(x_tile_size);

	const index_t x_block_len = blocked_dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(blocked_dens_l, a);
	auto b_bag = noarr::make_bag(blocked_dens_l, b);
	auto c_bag = noarr::make_bag(blocked_dens_l, c);

	auto d = noarr::make_bag(blocked_dens_l, densities);

	auto scratch = noarr::make_bag(scratch_l, b_scratch);

	for (index_t x = 0; x < x_block_len; x++)
	{
		for (index_t s = 0; s < x_tile_size; s++)
		{
			auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, 0, x);
			scratch[idx] = 1 / b_bag[idx];
		}

		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < x_tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, i, x);
				auto prev_idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, i - 1, x);

				auto r = a_bag[idx] * scratch[prev_idx];

				scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

				d[idx] -= r * d[prev_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}

		for (index_t s = 0; s < x_tile_size; s++)
		{
			auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, n - 1, x);
			d[idx] *= scratch[idx];

			// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
		}

		for (index_t i = n - 2; i >= 0; i--)
			for (index_t s = 0; s < x_tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, i, x);
				auto next_idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, i + 1, x);

				d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const scratch_layout_t scratch_l, const index_t s_idx,
							 index_t x_tile_size)
{
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s_idx) ^ noarr::into_blocks<'x', 'x', 'v'>(x_tile_size);

	const index_t x_block_len = blocked_dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(blocked_dens_l, a);
	auto b_bag = noarr::make_bag(blocked_dens_l, b);
	auto c_bag = noarr::make_bag(blocked_dens_l, c);

	auto d = noarr::make_bag(blocked_dens_l, densities);

	auto scratch = noarr::make_bag(scratch_l, b_scratch);

	for (index_t y = 0; y < y_len; y++)
		for (index_t x = 0; x < x_block_len; x++)
		{
			for (index_t s = 0; s < x_tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, 0, y, x);
				scratch[idx] = 1 / b_bag[idx];
			}

			for (index_t i = 1; i < n; i++)
				for (index_t s = 0; s < x_tile_size; s++)
				{
					auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, i, y, x);
					auto prev_idx = noarr::idx<'v', 'z', 'y', 'x'>(s, i - 1, y, x);

					auto r = a_bag[idx] * scratch[prev_idx];

					scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

					d[idx] -= r * d[prev_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}

			for (index_t s = 0; s < x_tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, n - 1, y, x);
				d[idx] *= scratch[idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
				for (index_t s = 0; s < x_tile_size; s++)
				{
					auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, i, y, x);
					auto next_idx = noarr::idx<'v', 'z', 'y', 'x'>(s, i + 1, y, x);

					d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
		}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::solve_x()
{
	for (index_t i = 0; i < countersx_count_; i++)
	{
		countersx_[i]->value = 0;
	}

#pragma omp parallel
	{
		perf_counter counter("lstmfpai");

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
			// 					  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
			// 					  << " block_z_begin: " << block_z_begin << " block_z_end: " << block_z_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				auto scratch_x = get_scratch_layout<'x'>(group_blocks_[0], group_blocks_[1], y_sync_step_);

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

				auto diag_x = get_diag_layout_x(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				for (index_t blocked_z = block_z_begin; blocked_z < block_z_end; blocked_z += y_sync_step_)
				{
					const index_t y_sync_step_len = std::min(y_sync_step_, block_z_end - blocked_z);


					auto sync_x = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_x ^ noarr::fix<'s'>(s), scratch_x,
								   dist_l = dist_l ^ noarr::fix<'y', 'z'>(tid.y, tid.z), n = this->problem_.nx,
								   tid = tid.x, group_size = group_size_x,
								   &barrier = barrier_x](index_t z_begin, index_t z_end) {
						synchronize_x_blocked_distributed(densities, a, c, dens_l, scratch_x, dist_l, n, z_begin, z_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[0] != 1)
						solve_block_x_transpose(
							current_densities, current_ax, current_bx, current_cx, current_a_scratch, current_c_scratch,
							dens_l_wo_x, diag_x, scratch_x, block_x_begin, block_x_end, blocked_z - block_z_begin,
							blocked_z + y_sync_step_len - block_z_begin, s, x_tile_size_, std::move(sync_x));
					else
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_x_2d_and_3d_transpose_l<index_t>(
								current_densities, current_ax, current_bx, current_cx, current_a_scratch, dens_l,
								diag_x, get_non_blocked_scratch_layout<'x'>(), s, z - block_z_begin, this->problem_.nx);
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
		perf_counter counter("lstmfpai");

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
				auto scratch_y = get_scratch_layout<'y'>(group_blocks_[0], group_blocks_[1] + 1, y_sync_step_);

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

				for (index_t blocked_z = block_z_begin; blocked_z < block_z_end; blocked_z += y_sync_step_)
				{
					const index_t y_sync_step_len = std::min(y_sync_step_, block_z_end - blocked_z);

					auto sync_y = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_y ^ noarr::fix<'s'>(s), scratch_y,
								   dist_l = dist_l ^ noarr::fix<'x', 'z'>(tid.x, tid.z), n = this->problem_.ny,
								   tid = tid.y, group_size = group_size_y,
								   &barrier = barrier_y](index_t z_begin, index_t z_end) {
						synchronize_y_blocked_distributed(densities, a, c, dens_l, scratch_y, dist_l, n, z_begin, z_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[1] != 1)
						solve_block_y(current_densities, current_ay, current_by, current_cy, current_a_scratch,
									  current_c_scratch, dens_l_wo_y, scratch_y, block_y_begin, block_y_end,
									  blocked_z - block_z_begin, blocked_z + y_sync_step_len - block_z_begin, s,
									  x_tile_size_, std::move(sync_y));
					else
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_y_3d<index_t>(current_densities, current_ay, current_by, current_cy,
													  current_a_scratch, dens_l, get_non_blocked_scratch_layout<'y'>(),
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
		perf_counter counter("lstmfpai");

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
				auto scratch_z = get_scratch_layout<'z'>(group_blocks_[0], z_sync_step_, group_blocks_[2] + 1);

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

				if (this->problem_.dims == 3)
				{
					auto sync_z = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_z ^ noarr::fix<'s'>(s), scratch_z,
								   dist_l = dist_l ^ noarr::fix<'x', 'y'>(tid.x, tid.y), n = this->problem_.nz,
								   tid = tid.z, group_size = group_size_z,
								   &barrier = barrier_z](index_t y_begin, index_t y_end) {
						synchronize_z_blocked_distributed(densities, a, c, dens_l, scratch_z, dist_l, n, y_begin, y_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[2] != 1)
						solve_block_z(current_densities, current_az, current_bz, current_cz, current_a_scratch,
									  current_c_scratch, dens_l_wo_z, scratch_z, block_z_begin, block_z_end, s,
									  x_tile_size_, z_sync_step_, std::move(sync_z));
					else
						solve_slice_z_3d<index_t>(current_densities, current_az, current_bz, current_cz,
												  current_a_scratch, dens_l, get_non_blocked_scratch_layout<'z'>(), s,
												  x_tile_size_);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void sdd_full_blocking<real_t, aligned_x>::solve()
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
		perf_counter counter("lstmfpai");

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
				auto scratch_x = get_scratch_layout<'x'>(group_blocks_[0], group_blocks_[1], y_sync_step_);
				auto scratch_y = get_scratch_layout<'y'>(group_blocks_[0], group_blocks_[1] + 1, y_sync_step_);
				auto scratch_z = get_scratch_layout<'z'>(group_blocks_[0], z_sync_step_, group_blocks_[2] + 1);

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

				auto diag_x = get_diag_layout_x(group_block_lengthsx_[tid.x], group_block_lengthsy_[tid.y],
												group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				for (index_t blocked_z = block_z_begin; blocked_z < block_z_end; blocked_z += y_sync_step_)
				{
					const index_t y_sync_step_len = std::min(y_sync_step_, block_z_end - blocked_z);


					auto sync_x = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_x ^ noarr::fix<'s'>(s), scratch_x,
								   dist_l = dist_l ^ noarr::fix<'y', 'z'>(tid.y, tid.z), n = this->problem_.nx,
								   tid = tid.x, group_size = group_size_x,
								   &barrier = barrier_x](index_t z_begin, index_t z_end) {
						synchronize_x_blocked_distributed(densities, a, c, dens_l, scratch_x, dist_l, n, z_begin, z_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[0] != 1)
						solve_block_x_transpose(
							current_densities, current_ax, current_bx, current_cx, current_a_scratch, current_c_scratch,
							dens_l_wo_x, diag_x, scratch_x, block_x_begin, block_x_end, blocked_z - block_z_begin,
							blocked_z + y_sync_step_len - block_z_begin, s, x_tile_size_, std::move(sync_x));
					else
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_x_2d_and_3d_transpose_l<index_t>(
								current_densities, current_ax, current_bx, current_cx, current_a_scratch, dens_l,
								diag_x, get_non_blocked_scratch_layout<'x'>(), s, z - block_z_begin, this->problem_.nx);


					auto sync_y = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_y ^ noarr::fix<'s'>(s), scratch_y,
								   dist_l = dist_l ^ noarr::fix<'x', 'z'>(tid.x, tid.z), n = this->problem_.ny,
								   tid = tid.y, group_size = group_size_y,
								   &barrier = barrier_y](index_t z_begin, index_t z_end) {
						synchronize_y_blocked_distributed(densities, a, c, dens_l, scratch_y, dist_l, n, z_begin, z_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[1] != 1)
						solve_block_y(current_densities, current_ay, current_by, current_cy, current_a_scratch,
									  current_c_scratch, dens_l_wo_y, scratch_y, block_y_begin, block_y_end,
									  blocked_z - block_z_begin, blocked_z + y_sync_step_len - block_z_begin, s,
									  x_tile_size_, std::move(sync_y));
					else
						for (index_t z = blocked_z; z < blocked_z + y_sync_step_len; z++)
							solve_slice_y_3d<index_t>(current_densities, current_ay, current_by, current_cy,
													  current_a_scratch, dens_l, get_non_blocked_scratch_layout<'y'>(),
													  s, z - block_z_begin, x_tile_size_);
				}

				if (this->problem_.dims == 3)
				{
					auto sync_z = [densities = thread_substrate_array_.get(), a = a_scratch_.get(),
								   c = c_scratch_.get(), dens_l = dens_l_wo_z ^ noarr::fix<'s'>(s), scratch_z,
								   dist_l = dist_l ^ noarr::fix<'x', 'y'>(tid.x, tid.y), n = this->problem_.nz,
								   tid = tid.z, group_size = group_size_z,
								   &barrier = barrier_z](index_t y_begin, index_t y_end) {
						synchronize_z_blocked_distributed(densities, a, c, dens_l, scratch_z, dist_l, n, y_begin, y_end,
														  tid, group_size, barrier);
					};

					if (cores_division_[2] != 1)
						solve_block_z(current_densities, current_az, current_bz, current_cz, current_a_scratch,
									  current_c_scratch, dens_l_wo_z, scratch_z, block_z_begin, block_z_end, s,
									  x_tile_size_, z_sync_step_, std::move(sync_z));
					else
						solve_slice_z_3d<index_t>(current_densities, current_az, current_bz, current_cz,
												  current_a_scratch, dens_l, get_non_blocked_scratch_layout<'z'>(), s,
												  x_tile_size_);
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
