#include "least_memory_thomas_solver_d_f_p.h"

#include <cstddef>
#include <iostream>

#include "barrier.h"
#include "noarr/structures/extra/funcs.hpp"
#include "omp_helper.h"
#include "perf_utils.h"
#include "vector_transpose_helper.h"


template <typename real_t, bool aligned_x>
thread_id_t<typename least_memory_thomas_solver_d_f_p<real_t, aligned_x>::index_t> least_memory_thomas_solver_d_f_p<
	real_t, aligned_x>::get_thread_id() const
{
	thread_id_t<typename least_memory_thomas_solver_d_f_p<real_t, aligned_x>::index_t> id;

	const index_t tid = get_thread_num();
	const index_t group_size = cores_division_[1] * cores_division_[2];

	const index_t substrate_group_tid = tid % group_size;

	id.group = tid / group_size;

	id.x = substrate_group_tid % cores_division_[0];
	id.y = (substrate_group_tid / cores_division_[0]) % cores_division_[1];
	id.z = substrate_group_tid / (cores_division_[0] * cores_division_[1]);

	return id;
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::set_block_bounds(index_t n, index_t group_size,
																		   index_t& block_size,
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
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::precompute_values(
	std::unique_ptr<real_t*[]>& a, std::unique_ptr<real_t*[]>& b, std::unique_ptr<real_t*[]>& c,
	std::unique_ptr<real_t*[]>& rf, std::unique_ptr<real_t*[]>& rb, index_t n, index_t shape, index_t dims,
	index_t counters_count, std::unique_ptr<std::unique_ptr<aligned_atomic<index_t>>[]>& counters,
	std::unique_ptr<std::unique_ptr<std::barrier<>>[]>& barriers, index_t group_size,
	const std::vector<index_t> group_block_lengths, const std::vector<index_t> group_block_offsets, char dim)
{
	a = std::make_unique<real_t*[]>(get_max_threads());
	b = std::make_unique<real_t*[]>(get_max_threads());
	c = std::make_unique<real_t*[]>(get_max_threads());
	rf = std::make_unique<real_t*[]>(get_max_threads());
	rb = std::make_unique<real_t*[]>(get_max_threads());

	counters = std::make_unique<std::unique_ptr<aligned_atomic<index_t>>[]>(counters_count);
	barriers = std::make_unique<std::unique_ptr<std::barrier<>>[]>(counters_count);

#pragma omp parallel
	{
		auto tid = get_thread_id();

		auto arrays_layout = noarr::scalar<real_t*>() ^ get_thread_distribution_layout();

		real_t*& a_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(a.get(), tid.y, tid.z, tid.group);
		real_t*& b_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(b.get(), tid.y, tid.z, tid.group);
		real_t*& c_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(c.get(), tid.y, tid.z, tid.group);
		real_t*& rf_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(rf.get(), tid.y, tid.z, tid.group);
		real_t*& rb_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(rb.get(), tid.y, tid.z, tid.group);

		auto diag_l = get_diagonal_layout(this->problem_, n);

		if (aligned_x)
		{
			a_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
			b_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
			c_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
			rf_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
			rb_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
		}
		else
		{
			a_t = (real_t*)std::malloc(diag_l | noarr::get_size());
			b_t = (real_t*)std::malloc(diag_l | noarr::get_size());
			c_t = (real_t*)std::malloc(diag_l | noarr::get_size());
			rf_t = (real_t*)std::malloc(diag_l | noarr::get_size());
			rb_t = (real_t*)std::malloc(diag_l | noarr::get_size());
		}

		auto a_diag = noarr::make_bag(diag_l, a_t);
		auto b_diag = noarr::make_bag(diag_l, b_t);
		auto c_diag = noarr::make_bag(diag_l, c_t);
		auto rf_diag = noarr::make_bag(diag_l, rf_t);
		auto rb_diag = noarr::make_bag(diag_l, rb_t);

		for (index_t s = 0; s < this->problem_.substrates_count; s++)
		{
			for (std::size_t block_idx = 0; block_idx < group_block_lengths.size(); block_idx++)
			{
				index_t i_begin = group_block_offsets[block_idx];
				index_t i_end = i_begin + group_block_lengths[block_idx];

				real_t a = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);
				real_t b = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims - 2 * a;
				real_t b0 = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims - a;

				for (index_t i = i_begin; i < i_begin + 2; i++)
				{
					index_t global_i = i;
					real_t a_tmp = global_i == 0 ? 0 : a;
					real_t b_tmp = (global_i == 0 || global_i == n - 1) ? b0 : b;
					real_t c_tmp = global_i == n - 1 ? 0 : a;

					auto idx = noarr::idx<'s', 'i'>(s, i);

					a_diag[idx] = a_tmp;
					b_diag[idx] = b_tmp;
					c_diag[idx] = c_tmp;
					rf_diag[idx] = 0;

					// if (get_thread_num() == 0)
					// 	std::cout << dim << " " << i << " a: " << a_diag[idx] << " b: " << b_diag[idx]
					// 			  << " c: " << c_diag[idx] << " rf: " << rf_diag[idx] << std::endl;
				}

				for (index_t i = i_begin + 2; i < i_end; i++)
				{
					index_t global_i = i;
					real_t a_tmp = global_i == 0 ? 0 : a;
					real_t b_tmp = (global_i == 0 || global_i == n - 1) ? b0 : b;
					real_t c_tmp = global_i == n - 1 ? 0 : a;

					auto idx = noarr::idx<'s', 'i'>(s, i);
					auto prev_idx = noarr::idx<'s', 'i'>(s, i - 1);

					a_diag[idx] = a_tmp;
					b_diag[idx] = b_tmp;
					c_diag[idx] = c_tmp;

					rf_diag[idx] = a_diag[idx] / b_diag[prev_idx];

					// if (get_thread_num() == 0)
					// 	std::cout << dim << " " << i << " a: " << a_diag[idx] << " b: " << b_diag[idx]
					// 			  << " c: " << c_diag[idx] << " rf: " << rf_diag[idx];

					a_diag[idx] = -a_diag[prev_idx] * rf_diag[idx];
					b_diag[idx] = b_diag[idx] - c_diag[prev_idx] * rf_diag[idx];

					// if (get_thread_num() == 0)
					// 	std::cout << " a': " << a_diag[idx] << " b': " << b_diag[idx] << " c: " << c_diag[idx]
					// 			  << std::endl;
				}

				for (index_t i = i_end - 1; i >= i_end - 2; i--)
				{
					auto idx = noarr::idx<'s', 'i'>(s, i);
					rb_diag[idx] = 0;
				}

				for (index_t i = i_end - 3; i >= i_begin; i--)
				{
					auto idx = noarr::idx<'s', 'i'>(s, i);
					auto prev_idx = noarr::idx<'s', 'i'>(s, i + 1);

					rb_diag[idx] = c_diag[idx] / b_diag[prev_idx];

					if (i != i_begin)
						a_diag[idx] = a_diag[idx] - a_diag[prev_idx] * rb_diag[idx];
					else
						b_diag[idx] = b_diag[idx] - a_diag[prev_idx] * rb_diag[idx];

					c_diag[idx] = -c_diag[prev_idx] * rb_diag[idx];

					// if (get_thread_num() == 0)
					// 	std::cout << dim << " " << i << " a'': " << a_diag[idx] << " b': " << b_diag[idx]
					// 			  << " c': " << c_diag[idx] << " rb: " << rb_diag[idx] << std::endl;
				}

				for (index_t i = i_begin + 1; i < i_end - 1; i++)
				{
					auto idx = noarr::idx<'s', 'i'>(s, i);

					b_diag[idx] = 1 / b_diag[idx];

					// if (get_thread_num() == 0)
					// 	std::cout << dim << " " << i << " a'': " << a_diag[idx] << " b'': " << b_diag[idx]
					// 			  << " c': " << c_diag[idx] << std::endl;
				}
			}

			auto get_idx = [&](index_t equation_idx) {
				auto idx = equation_idx / 2;
				return group_block_offsets[idx] + (group_block_lengths[idx] - 1) * (equation_idx % 2);
			};

			index_t equations = group_block_lengths.size();

			for (index_t equation_idx = 1; equation_idx < equations * 2; equation_idx++)
			{
				index_t i = get_idx(equation_idx);
				index_t prev_i = get_idx(equation_idx - 1);

				auto idx = noarr::idx<'s', 'i'>(s, i);
				auto prev_idx = noarr::idx<'s', 'i'>(s, prev_i);

				a_diag[idx] /= b_diag[prev_idx];
				b_diag[idx] -= c_diag[prev_idx] * a_diag[idx];

				// if (get_thread_num() == 0)
				// 	std::cout << dim << " m " << i << " rf: " << a_diag[idx] << " b: " << b_diag[idx]
				// 			  << " c: " << c_diag[idx] << std::endl;
			}

			for (index_t equation_idx = 0; equation_idx < equations * 2; equation_idx++)
			{
				index_t i = get_idx(equation_idx);
				auto idx = noarr::idx<'s', 'i'>(s, i);

				b_diag[idx] = 1 / b_diag[idx];

				// if (get_thread_num() == 0)
				// 	std::cout << dim << " m " << i << " rf: " << a_diag[idx] << " b': " << b_diag[idx]
				// 			  << " c: " << c_diag[idx] << std::endl;
			}

			// for (index_t i = 0; i < n; i++)
			// {
			// 	auto idx = noarr::idx<'s', 'i'>(s, i);
			// 	if (get_thread_num() == 0)
			// 		std::cout << dim << " f " << i << " a: " << a_diag[idx] << " b: " << b_diag[idx]
			// 				  << " c: " << c_diag[idx] << " rf: " << rf_diag[idx] << " rb: " << rb_diag[idx]
			// 				  << std::endl;
			// }
		}

		index_t lane_id;
		if (dim == 'y')
			lane_id = tid.z + tid.group * cores_division_[2];
		else
			lane_id = tid.y + tid.group * cores_division_[1];

		index_t dim_id = dim == 'y' ? tid.y : tid.z;

		if (dim_id == 0)
		{
			counters[lane_id] = std::make_unique<aligned_atomic<index_t>>(0);
			barriers[lane_id] = std::make_unique<std::barrier<>>(group_size);
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::precompute_values(real_t*& rf, real_t*& b, real_t*& c,
																			index_t shape, index_t dims, index_t n)
{
	auto layout = get_diagonal_layout(this->problem_, n);

	if (aligned_x)
	{
		rf = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
		b = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
		c = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
	}
	else
	{
		rf = (real_t*)std::malloc((layout | noarr::get_size()));
		b = (real_t*)std::malloc((layout | noarr::get_size()));
		c = (real_t*)std::malloc((layout | noarr::get_size()));
	}

	auto rf_diag = noarr::make_bag(layout, rf);
	auto b_diag = noarr::make_bag(layout, b);
	auto c_diag = noarr::make_bag(layout, c);

	for (index_t s = 0; s < this->problem_.substrates_count; s++)
	{
		real_t a = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

		rf_diag.template at<'i', 's'>(0, s) = 0;
		b_diag.template at<'i', 's'>(0, s) = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims - a;
		c_diag.template at<'i', 's'>(0, s) = a;

		for (index_t i = 1; i < n; i++)
		{
			real_t b = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims - 2 * a;
			if (i == n - 1)
				b += a;

			rf_diag.template at<'i', 's'>(i, s) = a / b_diag.template at<'i', 's'>(i - 1, s);
			b_diag.template at<'i', 's'>(i, s) = b - a * rf_diag.template at<'i', 's'>(i, s);
			c_diag.template at<'i', 's'>(i, s) = a;
		}


		for (index_t i = 0; i < n; i++)
		{
			b_diag.template at<'i', 's'>(i, s) = 1 / b_diag.template at<'i', 's'>(i, s);
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::precompute_values(std::unique_ptr<real_t*[]>& rf,
																			std::unique_ptr<real_t*[]>& b,
																			std::unique_ptr<real_t*[]>& c,
																			index_t shape, index_t dims, index_t n)
{
	rf = std::make_unique<real_t*[]>(get_max_threads());
	b = std::make_unique<real_t*[]>(get_max_threads());
	c = std::make_unique<real_t*[]>(get_max_threads());

#pragma omp parallel
	{
		real_t*& rf_t = rf[get_thread_num()];
		real_t*& b_t = b[get_thread_num()];
		real_t*& c_t = c[get_thread_num()];

		precompute_values(rf_t, b_t, c_t, shape, dims, n);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::prepare(const max_problem_t& problem)
{
	this->problem_ = problems::cast<std::int32_t, real_t>(problem);

	if (this->problem_.dims == 2)
		cores_division_[2] = 1;

	cores_division_[0] = 1;

	set_block_bounds(this->problem_.ny, cores_division_[1], group_blocks_[1], group_block_lengthsy_,
					 group_block_offsetsy_);

	set_block_bounds(this->problem_.nz, cores_division_[2], group_blocks_[2], group_block_lengthsz_,
					 group_block_offsetsz_);

	{
		substrate_groups_ = get_max_threads() / (cores_division_[0] * cores_division_[1] * cores_division_[2]);

		auto ss_len = (this->problem_.substrates_count + substrate_step_ - 1) / substrate_step_;

		for (index_t group_id = 0; group_id < substrate_groups_; group_id++)
		{
			const auto [ss_begin, ss_end] = evened_work_distribution(ss_len, substrate_groups_, group_id);

			const auto s_begin = ss_begin * substrate_step_;
			const auto s_end = std::min(this->problem_.substrates_count, ss_end * substrate_step_);

			group_block_lengthss_.push_back(std::max(s_end - s_begin, 0));
			group_block_offsetss_.push_back(s_begin);
		}
	}

	if (use_thread_distributed_allocation_)
	{
		thread_substrate_array_ = std::make_unique<real_t*[]>(get_max_threads());

#pragma omp parallel
		{
			const auto tid = get_thread_id();

			if (group_block_lengthss_[tid.group] != 0)
			{
				auto arrays_layout = noarr::scalar<real_t*>() ^ get_thread_distribution_layout();

				real_t*& substrates_t =
					arrays_layout
					| noarr::get_at<'y', 'z', 'g'>(thread_substrate_array_.get(), tid.y, tid.z, tid.group);

				auto dens_t_l =
					get_blocked_substrate_layout(this->problem_.nx, group_block_lengthsy_[tid.y],
												 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				substrates_t = (real_t*)std::aligned_alloc(alignment_size_, (dens_t_l | noarr::get_size()));

				if (problem.gaussian_pulse)
				{
					omp_trav_for_each(noarr::traverser(dens_t_l), [&](auto state) {
						index_t s = noarr::get_index<'s'>(state) + group_block_offsetss_[tid.group];
						index_t x = noarr::get_index<'x'>(state);
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
	else
	{
		auto substrates_layout = get_substrates_layout<3>();

		if (aligned_x)
			this->substrates_ = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
		else
			this->substrates_ = (real_t*)std::malloc((substrates_layout | noarr::get_size()));

		// Initialize substrates
		solver_utils::initialize_substrate(substrates_layout, this->substrates_, this->problem_);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	substrate_step_ =
		params.contains("substrate_step") ? (index_t)params["substrate_step"] : this->problem_.substrates_count;

	cores_division_ = params.contains("cores_division") ? (std::array<index_t, 3>)params["cores_division"]
														: std::array<index_t, 3> { 1, 2, 2 };

	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
		alignment_size_ = std::max(alignment_size_, vector_length);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
	{
		precompute_values(thread_rf_x_, thread_bx_, thread_cx_, this->problem_.dx, this->problem_.dims,
						  this->problem_.nx);
		// else
		// 	precompute_values(ax_, b1x_, cx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	}
	if (this->problem_.dims >= 2)
	{
		if (cores_division_[1] == 1)
		{
			precompute_values(thread_rf_y_, thread_by_, thread_cy_, this->problem_.dy, this->problem_.dims,
							  this->problem_.ny);
			// else
			// 	precompute_values(ay_, b1y_, cy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
		}
		else
		{
			countersy_count_ = cores_division_[0] * cores_division_[2] * substrate_groups_;

			precompute_values(thread_ay_, thread_by_, thread_cy_, thread_rf_y_, thread_rb_y_, this->problem_.ny,
							  this->problem_.dy, this->problem_.dims, countersy_count_, countersy_, barriersy_,
							  cores_division_[1], group_block_lengthsy_, group_block_offsetsy_, 'y');
			// else
			// 	precompute_values(ay_, b1y_, a_scratchy_, c_scratchy_, this->problem_.dy, this->problem_.dims,
			// 					  this->problem_.ny, countersy_count_, countersy_, barriersy_, cores_division_[1]);
		}
	}
	if (this->problem_.dims >= 3)
	{
		if (cores_division_[2] == 1)
		{
			precompute_values(thread_rf_z_, thread_bz_, thread_cz_, this->problem_.dz, this->problem_.dims,
							  this->problem_.nz);
			// else
			// 	precompute_values(az_, b1z_, cz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
		}
		else
		{
			countersz_count_ = cores_division_[0] * cores_division_[1] * substrate_groups_;

			precompute_values(thread_az_, thread_bz_, thread_cz_, thread_rf_z_, thread_rb_z_, this->problem_.nz,
							  this->problem_.dz, this->problem_.dims, countersz_count_, countersz_, barriersz_,
							  cores_division_[2], group_block_lengthsz_, group_block_offsetsz_, 'z');
			// else
			// 	precompute_values(az_, b1z_, a_scratchz_, c_scratchz_, this->problem_.dz, this->problem_.dims,
			// 					  this->problem_.nz, countersz_count_, countersz_, barriersz_, cores_division_[2]);
		}
	}
}

template <typename real_t, bool aligned_x>
auto least_memory_thomas_solver_d_f_p<real_t, aligned_x>::get_thread_distribution_layout() const
{
	return noarr::vectors<'y', 'z', 'g'>(cores_division_[1], cores_division_[2], substrate_groups_);
}

template <typename real_t, bool aligned_x>
auto least_memory_thomas_solver_d_f_p<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
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

template <typename real_t>
struct inside_data
{
	real_t& c_tmp;
};

template <typename index_t, typename real_t, typename density_bag_t>
constexpr static real_t z_forward_inside_y(const density_bag_t d, const index_t s, const index_t z, const index_t y,
										   const index_t x, const real_t a_s, const real_t b1_s, real_t c_tmp,
										   real_t data)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	real_t r;

	if (z == 0)
	{
		r = 1 / (b1_s + a_s);

		data *= r;

		d.template at<'s', 'z', 'x', 'y'>(s, z, x, y) = data;
	}
	else
	{
		const real_t b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);
		r = 1 / (b_tmp - a_s * c_tmp);

		data = r * (data - a_s * d.template at<'s', 'z', 'x', 'y'>(s, z - 1, x, y));
		d.template at<'s', 'z', 'x', 'y'>(s, z, x, y) = data;
	}

	return r;
}

template <typename index_t, typename scratch_bag_t>
struct inside_data_blocked
{
	const index_t begin;
	scratch_bag_t c;
};

template <typename index_t, typename scratch_bag_t>
struct inside_data_blocked_alt
{
	const index_t begin;
	const index_t end;
	scratch_bag_t a;
	scratch_bag_t c;
};

template <typename index_t, typename scratch_bag_t>
struct inside_data_blocked_alt_numa
{
	const index_t begin;
	const index_t end;
	const index_t length;
	scratch_bag_t a;
	scratch_bag_t c;
};

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked(const density_bag_t d, const index_t s, const index_t z,
												 const index_t y, const index_t x, real_t data, const real_t a_s,
												 const real_t b1_s, const index_t z_begin, scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	if (z < z_begin + 2)
	{
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data /= b_tmp;

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = r * (data - a_tmp * d.template at<'s', 'x', 'y', 'z'>(s, x, y, z - 1));

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}
}

template <typename index_t, typename vec_t, typename simd_tag, typename density_bag_t, typename diag_bag_t>
constexpr static vec_t z_forward_inside_y_blocked_vectorized(const density_bag_t d, vec_t data, simd_tag t,
															 const index_t y, const index_t z, const diag_bag_t rf)

{
	if (z < 2)
		return data;

	vec_t prev = hn::Load(t, &(d.template at<'y', 'z'>(y, z - 1)));

	auto idx = noarr::idx<'i'>(z);

	return hn::MulAdd(hn::Set(t, -rf[idx]), prev, data);
}

template <typename vec_t, typename tag, typename index_t, typename real_t, typename density_bag_t,
		  typename scratch_bag_t>
constexpr static void z_forward_inside_y_vectorized_blocked(const density_bag_t d, tag t, const index_t z,
															const index_t y, vec_t data, const real_t a_s,
															const real_t b1_s, const index_t z_begin, scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	if (z < z_begin + 2)
	{
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data = hn::Mul(data, hn::Set(t, 1 / b_tmp));

		hn::Store(data, t, &d.template at<'y', 'z'>(y, z));

		// #pragma omp critical
		// 		for (std::size_t v = 0; v < hn::Lanes(t); v++)
		// 		{
		// 			std::cout << "f0: " << z << " " << y << " " << x + v << " " << hn::ExtractLane(data, v) << " " <<
		// b_tmp
		// 					  << std::endl;
		// 		}
	}
	else
	{
		const auto prev_state = noarr::idx<'i'>(z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = hn::MulAdd(hn::Set(t, -a_tmp), hn::Load(t, &d.template at<'y', 'z'>(y, z - 1)), data);
		data = hn::Mul(data, hn::Set(t, r));

		hn::Store(data, t, &d.template at<'y', 'z'>(y, z));

		// #pragma omp critical
		// 		for (std::size_t v = 0; v < hn::Lanes(t); v++)
		// 		{
		// 			std::cout << "f1: " << z << " " << y << " " << x + v << " " << hn::ExtractLane(data, v) << " " <<
		// a_tmp << " "
		// 					  << b_tmp << " " << c[prev_state] << std::endl;
		// 		}
	}
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked_next(const index_t z, const real_t a_s, const real_t b1_s,
													  const index_t z_len, const index_t z_begin, scratch_bag_t a,
													  scratch_bag_t c)

{
	if (z < z_begin + 2)
	{
		const auto state = noarr::idx<'i'>(z);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);
		const auto c_tmp = a_s * (z == z_len - 1 ? 0 : 1);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'i'>(z);
		const auto prev_state = noarr::idx<'i'>(z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);
		const auto c_tmp = a_s * (z == z_len - 1 ? 0 : 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void z_backward_blocked_vectorized(const dens_bag_t d, const diag_bag_t rb, simd_tag t,
													const index_t simd_length, const index_t y_len, const index_t z_len)

{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const auto d_blocked =
				noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

			vec_t prev = hn::Load(t, &d_blocked.template at<'z', 'y'>(z_len - 2, y));

			for (index_t i = z_len - 3; i >= 0; i--)
			{
				auto idx = noarr::idx<'i'>(i);

				vec_t curr = hn::Load(t, &d_blocked.template at<'z', 'y'>(i, y));
				curr = hn::MulAdd(prev, hn::Set(t, -rb[idx]), curr);

				hn::Store(curr, t, &d_blocked.template at<'z', 'y'>(i, y));

				prev = curr;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_backward_blocked(const density_bag_t d, const real_t y_begin, const real_t y_end,
										 const real_t z_begin, const real_t z_end, const scratch_bag_t a,
										 const scratch_bag_t c, const index_t s)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	// Process the upper diagonal (backward)
	index_t i;
	for (i = z_end - 3; i >= z_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		for (index_t y = y_begin; y < y_end; y++)
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', 'z'>(s, x, y, i) -=
					c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y, i + 1);

				// #pragma omp critical
				// 				std::cout << "b0: " << i << " " << y << " " << x << " " << d.template at<'s', 'x', 'y',
				// 'z'>(s, x, y, i)
				// 						  << std::endl;
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
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', 'z'>(s, x, y, i) =
					r
					* (d.template at<'s', 'x', 'y', 'z'>(s, x, y, i)
					   - c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y, i + 1));

				// #pragma omp critical
				// 				std::cout << "b1: " << i << " " << y << " " << x << " " << d.template at<'s', 'x', 'y',
				// 'z'>(s, x, y, i)
				// 						  << " " << a[next_state] << " " << c[state] << std::endl;
			}

		a[state] = r * a[state];
		c[state] = r * (0 - c[state] * c[next_state]);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t, typename barrier_t>
constexpr static void synchronize_z_blocked(real_t* __restrict__ densities, const real_t* __restrict__ rf_data,
											const real_t* __restrict__ b_data, const real_t* __restrict__ c_data,
											const density_layout_t dens_l, const diagonal_layout_t diag_l,
											const index_t tid, const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	const auto d = noarr::make_bag(dens_l, densities);
	const auto rf = noarr::make_bag(diag_l, rf_data);
	const auto b = noarr::make_bag(diag_l, b_data);
	const auto c = noarr::make_bag(diag_l, c_data);

	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();
	const index_t n = d | noarr::get_length<'z'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_y = y_len / coop_size;
	const index_t y_begin = tid * block_size_y + std::min(tid, y_len % coop_size);
	const index_t y_end = y_begin + block_size_y + ((tid < y_len % coop_size) ? 1 : 0);

	barrier.wait();

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " s_begin: " << s << " block_y_begin: " << y_begin << " block_y_end: " << y_end
	// 			  << " block_size: " << block_size_y << std::endl;

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
		return i;
	};

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) =
					d.template at<'x', 'y', 'z'>(x, y, i) - rf[state] * d.template at<'x', 'y', 'z'>(x, y, prev_i);
			}
		}
	}

	for (index_t y = y_begin; y < y_end; y++)
	{
		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'x', 'y', 'z'>(x, y, n - 1) *= b.template at<'i'>(n - 1);
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) =
					b[state]
					* (d.template at<'x', 'y', 'z'>(x, y, i) - c[state] * d.template at<'x', 'y', 'z'>(x, y, next_i));
			}
		}
	}

	barrier.arrive_and_wait();
}


template <typename index_t, typename density_bag_t, typename diag_bag_t>
constexpr static void z_blocked_middle(density_bag_t d, const diag_bag_t rf, const diag_bag_t b, const diag_bag_t c,
									   const index_t tid, const index_t coop_size, const index_t y_len)
{
	constexpr char dim = 'z';
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t n = d | noarr::get_length<dim>();

	const index_t block_size = n / coop_size;

	const index_t block_size_y = y_len / coop_size;
	const index_t y_begin = tid * block_size_y + std::min(tid, y_len % coop_size);
	const index_t y_end = y_begin + block_size_y + ((tid < y_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " s_begin: " << s << " block_y_begin: " << y_begin << " block_y_end: " << y_end
	// 			  << " block_size: " << block_size_y << std::endl;

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
		return i;
	};

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) =
					d.template at<'x', 'y', 'z'>(x, y, i) - rf[state] * d.template at<'x', 'y', 'z'>(x, y, prev_i);
			}
		}
	}

	for (index_t y = y_begin; y < y_end; y++)
	{
		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'x', 'y', 'z'>(x, y, n - 1) *= b.template at<'i'>(n - 1);
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) =
					b[state]
					* (d.template at<'x', 'y', 'z'>(x, y, i) - c[state] * d.template at<'x', 'y', 'z'>(x, y, next_i));
			}
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t,
		  typename thread_distribution_l, typename dim_scratch_bag_t>
constexpr static void z_blocked_middle(density_layout_t thread_density_l, scratch_layout_t thread_scratch_l,
									   thread_distribution_l thread_dist, real_t** __restrict__ densities_array,
									   real_t** __restrict__ as_array, real_t** __restrict__ cs_array, const index_t n,
									   dim_scratch_bag_t c_scratch, const index_t tid, const index_t coop_size,
									   index_t y_begin, index_t y_end)
{
	const index_t x_len = thread_density_l | noarr::get_length<'x'>();

	const index_t block_size = n / coop_size;

	const index_t y_len = y_end - y_begin;
	const index_t block_size_y = y_len / coop_size;
	y_begin = y_begin + tid * block_size_y + std::min(tid, y_len % coop_size);
	y_end = y_begin + block_size_y + ((tid < y_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " s_begin: " << s << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = (equation_idx % 2) * (actual_block_size - 1);
		return std::make_pair(actual_block_size, i);
	};

	const auto [z_len0, i0] = get_i(0);

	const auto c0 = noarr::make_bag(thread_scratch_l ^ noarr::set_length<'i'>(z_len0),
									thread_dist | noarr::get_at<'z'>(cs_array, 0));
	auto c_tmp = c0.template at<'i'>(0);
	c_scratch.template at<'i'>(0) = c_tmp;

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const auto [z_len, i] = get_i(equation_idx);
		const auto [prev_z_len, prev_i] = get_i(equation_idx - 1);

		const index_t block_idx = equation_idx / 2;
		const index_t prev_block_idx = (equation_idx - 1) / 2;

		const auto a = noarr::make_bag(thread_scratch_l ^ noarr::set_length<'i'>(z_len),
									   thread_dist | noarr::get_at<'z'>(as_array, block_idx));
		const auto c = noarr::make_bag(thread_scratch_l ^ noarr::set_length<'i'>(z_len),
									   thread_dist | noarr::get_at<'z'>(cs_array, block_idx));

		const auto d = noarr::make_bag(thread_density_l ^ noarr::set_length<'z'>(z_len),
									   thread_dist | noarr::get_at<'z'>(densities_array, block_idx));
		const auto prev_d = noarr::make_bag(thread_density_l ^ noarr::set_length<'z'>(prev_z_len),
											thread_dist | noarr::get_at<'z'>(densities_array, prev_block_idx));

		const auto state = noarr::idx<'i'>(i);

		const auto r = 1 / (1 - a[state] * c_tmp);

		c_tmp = c[state] * r;
		c_scratch.template at<'i'>(equation_idx) = c_tmp;

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) =
					r
					* (d.template at<'x', 'y', 'z'>(x, y, i)
					   - a[state] * prev_d.template at<'x', 'y', 'z'>(x, y, prev_i));
			}
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const auto [z_len, i] = get_i(equation_idx);
		const auto [next_z_len, next_i] = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(equation_idx);

		const index_t block_idx = equation_idx / 2;
		const index_t next_block_idx = (equation_idx + 1) / 2;

		const auto d = noarr::make_bag(thread_density_l ^ noarr::set_length<'z'>(z_len),
									   thread_dist | noarr::get_at<'z'>(densities_array, block_idx));
		const auto next_d = noarr::make_bag(thread_density_l ^ noarr::set_length<'z'>(next_z_len),
											thread_dist | noarr::get_at<'z'>(densities_array, next_block_idx));

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) =
					d.template at<'x', 'y', 'z'>(x, y, i)
					- c_scratch[state] * next_d.template at<'x', 'y', 'z'>(x, y, next_i);
			}
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_z_blocked_distributed(
	real_t** __restrict__ densities, const real_t* __restrict__ rf_data, const real_t* __restrict__ b_data,
	const real_t* __restrict__ c_data, const density_layout_t dens_l, const diagonal_layout_t diag_l,
	const thread_distribution_l dist_l, const index_t n, const index_t tid, const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	const auto rf = noarr::make_bag(diag_l, rf_data);
	const auto b = noarr::make_bag(diag_l, b_data);
	const auto c = noarr::make_bag(diag_l, c_data);

	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_y = y_len / coop_size;
	const index_t y_begin = tid * block_size_y + std::min(tid, y_len % coop_size);
	const index_t y_end = y_begin + block_size_y + ((tid < y_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid  << " block_begin: " << y_begin << " block_end: " << y_end
	// 			  << " block_size: " << block_size_y << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size, dens_l](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		const auto i = block_start + offset;
		return std::make_tuple(i, block_idx,
							   dens_l ^ noarr::set_length<'z'>(actual_block_size) ^ noarr::fix<'z'>(offset));
	};

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const auto [i, block_idx, curr_dens_l] = get_i(equation_idx);
		const auto [prev_i, prev_block_idx, prev_dens_l] = get_i(equation_idx - 1);
		const auto state = noarr::idx<'i'>(i);

		const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'z'>(densities, block_idx));
		const auto prev_d = noarr::make_bag(prev_dens_l, dist_l | noarr::get_at<'z'>(densities, prev_block_idx));

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y'>(x, y) =
					d.template at<'x', 'y'>(x, y) - rf[state] * prev_d.template at<'x', 'y'>(x, y);
			}
		}
	}

	{
		const auto [i, block_idx, curr_dens_l] = get_i(coop_size * 2 - 1);

		const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'z'>(densities, block_idx));

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y'>(x, y) *= b.template at<'i'>(i);
			}
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const auto [i, block_idx, curr_dens_l] = get_i(equation_idx);
		const auto [next_i, next_block_idx, next_dens_l] = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(i);

		const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'z'>(densities, block_idx));
		const auto next_d = noarr::make_bag(next_dens_l, dist_l | noarr::get_at<'z'>(densities, next_block_idx));

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y'>(x, y) =
					b[state] * (d.template at<'x', 'y'>(x, y) - c[state] * next_d.template at<'x', 'y'>(x, y));
			}
		}
	}

	barrier.arrive_and_wait();
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void z_blocked_end_vectorized(const dens_bag_t d, const diag_bag_t a, const diag_bag_t b,
											   const diag_bag_t c, simd_tag t, const index_t simd_length,
											   const index_t y_len, const index_t z_len)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const auto d_blocked =
				noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

			vec_t begins = hn::Load(t, &d_blocked.template at<'z', 'y'>(0, y));
			vec_t ends = hn::Load(t, &d_blocked.template at<'z', 'y'>(z_len - 1, y));

			for (index_t i = 1; i < z_len - 1; i++)
			{
				const auto state = noarr::idx<'i'>(i);

				vec_t data = hn::Load(t, &d_blocked.template at<'z', 'y'>(i, y));
				data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
				data = hn::MulAdd(ends, hn::Set(t, -c[state]), data);
				data = hn::Mul(data, hn::Set(t, b[state]));

				hn::Store(data, t, &d_blocked.template at<'z', 'y'>(i, y));
			}
		}
	}
}

template <typename index_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_blocked_end_alt(density_bag_t d, scratch_bag_t a, scratch_bag_t c, const index_t y_begin,
										const index_t y_end, const index_t z_begin, const index_t z_end)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = z_end - 2; i >= z_begin + 1; i--)
	{
		const auto state = noarr::idx<'i'>(i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) = d.template at<'x', 'y', 'z'>(x, y, i)
														- a[state] * d.template at<'x', 'y', 'z'>(x, y, z_begin)
														- c[state] * d.template at<'x', 'y', 'z'>(x, y, i + 1);


				// #pragma omp critical
				// 			std::cout << "0b z: " << z << " y: " << i << " x: " << x << " a: " << a[state] << " c: " <<
				// c[state]
				// 					  << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				// 					  << " d0: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) << std::endl;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_backward_blocked(const density_bag_t d, const index_t z, const real_t y_begin,
										 const real_t y_end, const scratch_bag_t a, const scratch_bag_t c,
										 const index_t s)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	// Process the upper diagonal (backward)
	index_t i;
	for (i = y_end - 3; i >= y_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) -=
				c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z);


			// #pragma omp critical
			// 			std::cout << "b0: " << i << " " << i << " " << x << " " << d.template at<'s', 'x', 'y', 'z'>(s,
			// x, i, z)
			// 					  << std::endl;
		}

		a[state] = a[state] - c[state] * a[next_state];
		c[state] = 0 - c[state] * c[next_state];
	}

	// Process the first row (backward)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		const auto r = 1 / (1 - c[state] * a[next_state]);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				r
				* (d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				   - c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z));

			// std::cout << "b1: " << z << " " << i << " " << x << " " << d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
			// 		  << std::endl;
		}

		a[state] = r * a[state];
		c[state] = r * (0 - c[state] * c[next_state]);
	}
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void y_backward_blocked_vectorized(const dens_bag_t d, const diag_bag_t rb, simd_tag t,
													const index_t z, const index_t simd_length, const index_t y_len)

{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d_blocked =
			noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

		vec_t prev = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, y_len - 2));

		for (index_t i = y_len - 3; i >= 0; i--)
		{
			auto idx = noarr::idx<'i'>(i);

			vec_t curr = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, i));
			curr = hn::MulAdd(prev, hn::Set(t, -rb[idx]), curr);

			hn::Store(curr, t, &d_blocked.template at<'z', 'y'>(z, i));

			prev = curr;

			// #pragma omp critical
			// 			std::cout << "b " << z << " " << i << " " << X * simd_length << " rb: " << rb[idx] << std::endl;
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t, typename barrier_t>
constexpr static void synchronize_y_blocked(real_t* __restrict__ densities, const real_t* __restrict__ rf_data,
											const real_t* __restrict__ b_data, const real_t* __restrict__ c_data,
											const density_layout_t dens_l, const diagonal_layout_t diag_l,
											const index_t z, const index_t tid, const index_t coop_size,
											barrier_t& barrier)
{
	barrier.arrive();

	const auto d = noarr::make_bag(dens_l, densities);
	const auto rf = noarr::make_bag(diag_l, rf_data);
	const auto b = noarr::make_bag(diag_l, b_data);
	const auto c = noarr::make_bag(diag_l, c_data);

	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t n = d | noarr::get_length<'y'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_x = x_len / coop_size;
	const auto x_begin = tid * block_size_x + std::min(tid, x_len % coop_size);
	const auto x_end = x_begin + block_size_x + ((tid < x_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
		return i;
	};

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) =
				d.template at<'x', 'y', 'z'>(x, i, z) - rf[state] * d.template at<'x', 'y', 'z'>(x, prev_i, z);

			// #pragma omp critical
			// 			std::cout << "mf " << z << " " << i << " " << x << " rf: " << rf[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	for (index_t x = x_begin; x < x_end; x++)
	{
		d.template at<'x', 'y', 'z'>(x, n - 1, z) *= b.template at<'i'>(n - 1);
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) =
				b[state]
				* (d.template at<'x', 'y', 'z'>(x, i, z) - c[state] * d.template at<'x', 'y', 'z'>(x, next_i, z));

			// #pragma omp critical
			// 			std::cout << "mb " << z << " " << i << " " << x << " b: " << b[state] << " c: " << c[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	barrier.arrive_and_wait();
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_y_blocked_distributed(
	real_t** __restrict__ densities, const real_t* __restrict__ rf_data, const real_t* __restrict__ b_data,
	const real_t* __restrict__ c_data, const density_layout_t dens_l, const diagonal_layout_t diag_l,
	const thread_distribution_l dist_l, const index_t n, const index_t z, const index_t tid, const index_t coop_size,
	barrier_t& barrier)
{
	barrier.arrive();

	const auto rf = noarr::make_bag(diag_l, rf_data);
	const auto b = noarr::make_bag(diag_l, b_data);
	const auto c = noarr::make_bag(diag_l, c_data);

	const index_t x_len = dens_l | noarr::get_length<'x'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_x = x_len / coop_size;
	const auto x_begin = tid * block_size_x + std::min(tid, x_len % coop_size);
	const auto x_end = x_begin + block_size_x + ((tid < x_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size, dens_l](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		const auto i = block_start + offset;
		return std::make_tuple(i, block_idx,
							   dens_l ^ noarr::set_length<'y'>(actual_block_size) ^ noarr::fix<'y'>(offset));
	};

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const auto [i, block_idx, curr_dens_l] = get_i(equation_idx);
		const auto [prev_i, prev_block_idx, prev_dens_l] = get_i(equation_idx - 1);
		const auto state = noarr::idx<'i'>(i);

		const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'y'>(densities, block_idx));
		const auto prev_d = noarr::make_bag(prev_dens_l, dist_l | noarr::get_at<'y'>(densities, prev_block_idx));

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'z'>(x, z) =
				d.template at<'x', 'z'>(x, z) - rf[state] * prev_d.template at<'x', 'z'>(x, z);

			// #pragma omp critical
			// 			std::cout << "mf " << z << " " << i << " " << x << " rf: " << rf[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	{
		const auto [i, block_idx, curr_dens_l] = get_i(coop_size * 2 - 1);

		const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'y'>(densities, block_idx));

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'z'>(x, z) *= b.template at<'i'>(i);
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const auto [i, block_idx, curr_dens_l] = get_i(equation_idx);
		const auto [next_i, next_block_idx, next_dens_l] = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(i);

		const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'y'>(densities, block_idx));
		const auto next_d = noarr::make_bag(next_dens_l, dist_l | noarr::get_at<'y'>(densities, next_block_idx));

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'z'>(x, z) =
				b[state] * (d.template at<'x', 'z'>(x, z) - c[state] * next_d.template at<'x', 'z'>(x, z));

			// #pragma omp critical
			// 			std::cout << "mb " << z << " " << i << " " << x << " b: " << b[state] << " c: " << c[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	barrier.arrive_and_wait();
}

template <typename index_t, typename density_bag_t, typename diag_bag_t>
constexpr static void y_blocked_middle(density_bag_t d, const diag_bag_t rf, const diag_bag_t b, const diag_bag_t c,
									   const index_t z, const index_t tid, const index_t coop_size)
{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t n = d | noarr::get_length<'y'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_x = x_len / coop_size;
	const auto x_begin = tid * block_size_x + std::min(tid, x_len % coop_size);
	const auto x_end = x_begin + block_size_x + ((tid < x_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
		return i;
	};

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) =
				d.template at<'x', 'y', 'z'>(x, i, z) - rf[state] * d.template at<'x', 'y', 'z'>(x, prev_i, z);

			// #pragma omp critical
			// 			std::cout << "mf " << z << " " << i << " " << x << " rf: " << rf[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	for (index_t x = x_begin; x < x_end; x++)
	{
		d.template at<'x', 'y', 'z'>(x, n - 1, z) *= b.template at<'i'>(n - 1);
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) =
				b[state]
				* (d.template at<'x', 'y', 'z'>(x, i, z) - c[state] * d.template at<'x', 'y', 'z'>(x, next_i, z));

			// #pragma omp critical
			// 			std::cout << "mb " << z << " " << i << " " << x << " b: " << b[state] << " c: " << c[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t,
		  typename thread_distribution_l, typename dim_scratch_bag_t>
constexpr static void y_blocked_middle(density_layout_t thread_density_l, scratch_layout_t thread_scratch_l,
									   thread_distribution_l thread_dist, real_t** __restrict__ densities_array,
									   real_t** __restrict__ as_array, real_t** __restrict__ cs_array, const index_t n,
									   const index_t z, dim_scratch_bag_t c_scratch, const index_t tid,
									   const index_t coop_size)
{
	const index_t x_len = thread_density_l | noarr::get_length<'x'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_x = x_len / coop_size;
	const auto x_begin = tid * block_size_x + std::min(tid, x_len % coop_size);
	const auto x_end = x_begin + block_size_x + ((tid < x_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " s_begin: " << s << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = (equation_idx % 2) * (actual_block_size - 1);
		return std::make_pair(actual_block_size, i);
	};

	const auto [y_len0, i0] = get_i(0);

	const auto c0 = noarr::make_bag(thread_scratch_l ^ noarr::set_length<'i'>(y_len0),
									thread_dist | noarr::get_at<'y'>(cs_array, 0));
	auto c_tmp = c0.template at<'i'>(0);
	c_scratch.template at<'i'>(0) = c_tmp;

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const auto [y_len, i] = get_i(equation_idx);
		const auto [prev_y_len, prev_i] = get_i(equation_idx - 1);

		const index_t block_idx = equation_idx / 2;
		const index_t prev_block_idx = (equation_idx - 1) / 2;

		const auto a = noarr::make_bag(thread_scratch_l ^ noarr::set_length<'i'>(y_len),
									   thread_dist | noarr::get_at<'y'>(as_array, block_idx));
		const auto c = noarr::make_bag(thread_scratch_l ^ noarr::set_length<'i'>(y_len),
									   thread_dist | noarr::get_at<'y'>(cs_array, block_idx));

		const auto d = noarr::make_bag(thread_density_l ^ noarr::set_length<'y'>(y_len),
									   thread_dist | noarr::get_at<'y'>(densities_array, block_idx));
		const auto prev_d = noarr::make_bag(thread_density_l ^ noarr::set_length<'y'>(prev_y_len),
											thread_dist | noarr::get_at<'y'>(densities_array, prev_block_idx));

		const auto state = noarr::idx<'i'>(i);

		const auto r = 1 / (1 - a[state] * c_tmp);

		c_tmp = c[state] * r;
		c_scratch.template at<'i'>(equation_idx) = c_tmp;

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) =
				r
				* (d.template at<'x', 'y', 'z'>(x, i, z) - a[state] * prev_d.template at<'x', 'y', 'z'>(x, prev_i, z));

			// #pragma omp critical
			// 			std::cout << "mf z: " << z << " y: " << i << " x: " << x << " a: " << a[state] << " cp: " <<
			// c[prev_state]
			// 					  << " d: " << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << std::endl;
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const auto [y_len, i] = get_i(equation_idx);
		const auto [next_y_len, next_i] = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(equation_idx);

		const index_t block_idx = equation_idx / 2;
		const index_t next_block_idx = (equation_idx + 1) / 2;

		const auto d = noarr::make_bag(thread_density_l ^ noarr::set_length<'y'>(y_len),
									   thread_dist | noarr::get_at<'y'>(densities_array, block_idx));
		const auto next_d = noarr::make_bag(thread_density_l ^ noarr::set_length<'y'>(next_y_len),
											thread_dist | noarr::get_at<'y'>(densities_array, next_block_idx));

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) =
				d.template at<'x', 'y', 'z'>(x, i, z)
				- c_scratch[state] * next_d.template at<'x', 'y', 'z'>(x, next_i, z);

			// #pragma omp critical
			// 			std::cout << "mb z: " << z << " y: " << i << " x: " << x << " c: " << c[state]
			// 					  << " d: " << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << std::endl;
		}
	}
}

template <typename index_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_blocked_end(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
									const index_t y_begin, const index_t y_end, const index_t s)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = y_begin + 1; i < y_end - 1; i++)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				- a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z)
				- c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_end - 1, z);
		}
	}
}


template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t,
		  typename z_func_t>
constexpr static void y_blocked_end_vectorized(const dens_bag_t d, const diag_bag_t a, const diag_bag_t b,
											   const diag_bag_t c, simd_tag t, const index_t z,
											   const index_t simd_length, const index_t y_len, z_func_t&& z_forward)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d_blocked =
			noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

		vec_t begins = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, 0));
		vec_t ends = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, y_len - 1));

		for (index_t i = 1; i < y_len - 1; i++)
		{
			const auto state = noarr::idx<'i'>(i);

			vec_t data = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(ends, hn::Set(t, -c[state]), data);
			data = hn::Mul(data, hn::Set(t, b[state]));

			data = z_forward(data, d_blocked, i);

			hn::Store(data, t, &d_blocked.template at<'z', 'y'>(z, i));

			// #pragma omp critical
			// 			std::cout << "f " << z << " " << i << " " << X * simd_length << " a: " << a[state] << " b: " <<
			// b[state]
			// 					  << " c: " << c[state] << std::endl;
		}

		begins = z_forward(begins, d_blocked, 0);
		hn::Store(begins, t, &d_blocked.template at<'z', 'y'>(z, 0));

		ends = z_forward(ends, d_blocked, y_len - 1);
		hn::Store(ends, t, &d_blocked.template at<'z', 'y'>(z, y_len - 1));
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked_alt(const density_bag_t d, const index_t s, const index_t y,
													 const index_t z, const index_t x, real_t data, const real_t a_s,
													 const real_t b1_s, const index_t z_begin, const index_t z_end,
													 scratch_bag_t a, scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	real_t a_state;

	if (z < z_begin + 2)
	{
		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data /= b_tmp;

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		a_state = a_tmp / b_tmp;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = r * (data - a_tmp * d.template at<'s', 'x', 'y', 'z'>(s, x, y, z - 1));

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		a_state = r * (0 - a_tmp * a[prev_state]);

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}

	if (z != z_begin && z != z_end - 1)
	{
		const auto state0 = noarr::idx<'s', 'i'>(s, z_begin);

		const auto r0 = 1 / (1 - a_state * c[state0]);

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z_begin) =
			r0 * (d.template at<'s', 'x', 'y', 'z'>(s, x, y, z_begin) - c[state0] * data);


		// #pragma omp critical
		// 		std::cout << "0 z: " << z << " y: " << y << " x: " << x << " a0: " << a[state0] << " c0: " << c[state0]
		// 				  << " r0: " << r0 << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) <<
		// std::endl;
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t,
		  typename simd_tag>
constexpr static void z_forward_inside_y_blocked_alt_vectorized(const density_bag_t d, simd_tag t, const index_t y,
																const index_t z, const index_t X, vec_t data,
																const real_t a_s, const real_t b1_s,
																const index_t z_begin, const index_t z_end,
																scratch_bag_t a, scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	real_t a_state;

	if (z < z_begin + 2) [[unlikely]]
	{
		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data = hn::Mul(data, hn::Set(t, 1 / b_tmp));

		hn::Store(data, t, &d.template at<'X', 'z', 'y'>(X, z, y));

		a_state = a_tmp / b_tmp;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else [[likely]]
	{
		const auto prev_state = noarr::idx<'i'>(z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		vec_t prev = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z - 1, y));

		data = hn::MulAdd(hn::Set(t, -a_tmp), prev, data);
		data = hn::Mul(data, hn::Set(t, r));

		hn::Store(data, t, &d.template at<'X', 'z', 'y'>(X, z, y));

		a_state = r * (0 - a_tmp * a[prev_state]);

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}

	if (z != z_begin && z != z_end - 1) [[likely]]
	{
		const auto state0 = noarr::idx<'i'>(z_begin);

		const auto r0 = 1 / (1 - a_state * c[state0]);

		vec_t z0 = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z_begin, y));

		z0 = hn::MulAdd(hn::Set(t, -c[state0]), data, z0);
		z0 = hn::Mul(z0, hn::Set(t, r0));

		hn::Store(z0, t, &d.template at<'X', 'z', 'y'>(X, z_begin, y));


		// #pragma omp critical
		// 		std::cout << "0 z: " << z << " y: " << y << " x: " << x << " a0: " << a[state0] << " c0: " << c[state0]
		// 				  << " r0: " << r0 << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) <<
		// std::endl;
	}
}


template <typename vec_t, typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t,
		  typename simd_tag>
constexpr static void z_forward_inside_y_blocked_alt_numa_vectorized(
	const density_bag_t d, simd_tag t, const index_t y, const index_t z, const index_t X, vec_t data, const real_t a_s,
	const real_t b1_s, const index_t z_global_begin, const index_t z_global_end, const index_t z_global_len,
	scratch_bag_t a, scratch_bag_t c)

{
	real_t a_state;

	const index_t z_local_len = z_global_end - z_global_begin;
	const auto global_z = z + z_global_begin;

	const auto a_tmp = a_s * (global_z == 0 ? 0 : 1);
	const auto b_tmp = b1_s + ((global_z == 0) || (global_z == z_global_len - 1) ? a_s : 0);

	if (z < 2) [[unlikely]]
	{
		data = hn::Mul(data, hn::Set(t, 1 / b_tmp));

		hn::Store(data, t, &d.template at<'X', 'z', 'y'>(X, z, y));

		a_state = a_tmp / b_tmp;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else [[likely]]
	{
		const auto prev_state = noarr::idx<'i'>(z - 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		vec_t prev = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z - 1, y));

		data = hn::MulAdd(hn::Set(t, -a_tmp), prev, data);
		data = hn::Mul(data, hn::Set(t, r));

		hn::Store(data, t, &d.template at<'X', 'z', 'y'>(X, z, y));

		a_state = r * (0 - a_tmp * a[prev_state]);

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}

	if (z != 0 && z != z_local_len - 1) [[likely]]
	{
		const auto state0 = noarr::idx<'i'>(0);

		const auto r0 = 1 / (1 - a_state * c[state0]);

		vec_t z0 = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, 0, y));

		z0 = hn::MulAdd(hn::Set(t, -c[state0]), data, z0);
		z0 = hn::Mul(z0, hn::Set(t, r0));

		hn::Store(z0, t, &d.template at<'X', 'z', 'y'>(X, 0, y));


		// #pragma omp critical
		// 		std::cout << "0 z: " << z << " y: " << y << " x: " << x << " a0: " << a[state0] << " c0: " << c[state0]
		// 				  << " r0: " << r0 << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) <<
		// std::endl;
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t, typename z_data_t>
constexpr static void y_blocked_end(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
									const index_t y_begin, const index_t y_end, const index_t s, const real_t az_s,
									const real_t b1z_s, z_data_t z_data)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = y_begin + 1; i < y_end - 1; i++)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			real_t data = d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
						  - a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z)
						  - c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_end - 1, z);

			// #pragma omp critical
			// 			std::cout << "l: " << z << " " << i << " " << x << " " << data << " " << a[state] << " " <<
			// c[state]
			// 					  << std::endl;

			z_forward_inside_y_blocked(d, s, z, i, x, data, az_s, b1z_s, z_data.begin, z_data.c);
		}
	}
}

template <typename index_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_blocked_end_alt(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
										const index_t y_begin, const index_t y_end)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = y_end - 2; i >= y_begin + 1; i--)
	{
		const auto state = noarr::idx<'i'>(i);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) = d.template at<'x', 'y', 'z'>(x, i, z)
													- a[state] * d.template at<'x', 'y', 'z'>(x, y_begin, z)
													- c[state] * d.template at<'x', 'y', 'z'>(x, i + 1, z);


			// #pragma omp critical
			// 			std::cout << "0b z: " << z << " y: " << i << " x: " << x << " a: " << a[state] << " c: " <<
			// c[state]
			// 					  << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
			// 					  << " d0: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) << std::endl;
		}
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_layout_t, typename scratch_bag_t,
		  typename simd_tag>
constexpr static void y_blocked_end_alt_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
												   simd_tag t, const index_t simd_length, const index_t z,
												   scratch_bag_t a, scratch_bag_t c, const index_t y_begin,
												   const index_t y_end, const index_t s)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'x'>(noarr::lit<0>, noarr::lit<0>), densities);

	for (index_t X = 0; X < X_len; X++)
	{
		vec_t begins = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_begin));
		vec_t prev = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_end - 1));

		for (index_t i = y_end - 2; i >= y_begin + 1; i--)
		{
			const auto state = noarr::idx<'s', 'i'>(s, i);

			vec_t data = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);
			prev = data;

			hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
		}
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t, typename z_data_t>
constexpr static void y_blocked_end_alt(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
										const index_t y_begin, const index_t y_end, const index_t s, const real_t az_s,
										const real_t b1z_s, z_data_t z_data)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = y_end - 2; i >= y_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				- a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z)
				- c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z);


			// #pragma omp critical
			// 			std::cout << "0b z: " << z << " y: " << i << " x: " << x << " a: " << a[state] << " c: " <<
			// c[state]
			// 					  << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
			// 					  << " d0: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) << std::endl;

			z_forward_inside_y_blocked_alt(d, s, i + 1, z, x, d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z), az_s,
										   b1z_s, z_data.begin, z_data.end, z_data.a, z_data.c);
		}
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_layout_t, typename scratch_bag_t,
		  typename z_data_t, typename simd_tag>
constexpr static void y_blocked_end_alt_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
												   simd_tag t, const index_t simd_length, const index_t z,
												   scratch_bag_t a, scratch_bag_t c, const index_t y_begin,
												   const index_t y_end, const real_t az_s, const real_t b1z_s,
												   z_data_t z_data)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'x'>(noarr::lit<0>, noarr::lit<0>), densities);

	for (index_t X = 0; X < X_len; X++)
	{
		vec_t begins = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z, y_begin));
		vec_t prev = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z, y_end - 1));

		for (index_t i = y_end - 2; i >= y_begin + 1; i--)
		{
			const auto state = noarr::idx<'i'>(i);

			vec_t data = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);

			// hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));

			z_forward_inside_y_blocked_alt_vectorized(d, t, i + 1, z, X, prev, az_s, b1z_s, z_data.begin, z_data.end,
													  z_data.a, z_data.c);
			prev = data;
		}

		z_forward_inside_y_blocked_alt_vectorized(d, t, y_begin + 1, z, X, prev, az_s, b1z_s, z_data.begin, z_data.end,
												  z_data.a, z_data.c);

		z_forward_inside_y_blocked_alt_vectorized(d, t, y_begin, z, X, begins, az_s, b1z_s, z_data.begin, z_data.end,
												  z_data.a, z_data.c);
	}
}


template <typename vec_t, typename index_t, typename real_t, typename density_layout_t, typename scratch_bag_t,
		  typename z_data_t, typename simd_tag>
constexpr static void y_blocked_end_alt_numa_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
														simd_tag t, const index_t simd_length, const index_t z,
														scratch_bag_t a, scratch_bag_t c, const index_t y_begin,
														const index_t y_end, const real_t az_s, const real_t b1z_s,
														z_data_t z_data)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'x'>(noarr::lit<0>, noarr::lit<0>), densities);

	for (index_t X = 0; X < X_len; X++)
	{
		vec_t begins = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z, y_begin));
		vec_t prev = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z, y_end - 1));

		for (index_t i = y_end - 2; i >= y_begin + 1; i--)
		{
			const auto state = noarr::idx<'i'>(i);

			vec_t data = hn::Load(t, &d.template at<'X', 'z', 'y'>(X, z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);

			// hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));

			z_forward_inside_y_blocked_alt_numa_vectorized(d, t, i + 1, z, X, prev, az_s, b1z_s, z_data.begin,
														   z_data.end, z_data.length, z_data.a, z_data.c);
			prev = data;
		}

		z_forward_inside_y_blocked_alt_numa_vectorized(d, t, y_begin + 1, z, X, prev, az_s, b1z_s, z_data.begin,
													   z_data.end, z_data.length, z_data.a, z_data.c);

		z_forward_inside_y_blocked_alt_numa_vectorized(d, t, y_begin, z, X, begins, az_s, b1z_s, z_data.begin,
													   z_data.end, z_data.length, z_data.a, z_data.c);
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t>
constexpr static real_t y_forward_inside_x(const density_bag_t d, const index_t z, const index_t y, const index_t x,
										   real_t data, const diag_bag_t rf)

{
	real_t prev;

	if (y == 0)
		prev = 0;
	else
		prev = d.template at<'z', 'x', 'y'>(z, x, y - 1);

	return data - rf.template at<'i'>(y) * prev;
}

template <typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t>
constexpr static real_t y_forward_inside_x_blocked(const density_bag_t d, const index_t z, const index_t y,
												   const index_t x, real_t data, const diag_bag_t rf)

{
	if (y < 2)
		return data;

	real_t prev = d.template at<'z', 'x', 'y'>(z, x, y - 1);

	// #pragma omp critical
	// 	std::cout << "f " << z << " " << y << " " << x << " rf: " << rf.template at<'i'>(y) << std::endl;

	return data - rf.template at<'i'>(y) * prev;
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked_alt(const density_bag_t d, const index_t z, const index_t y,
													 const index_t x, real_t data, const real_t a_s, const real_t b1_s,
													 const index_t y_begin, const index_t y_end, scratch_bag_t a,
													 scratch_bag_t c)

{
	const index_t y_len = d | noarr::get_length<'y'>();

	real_t a_state;

	if (y < y_begin + 2)
	{
		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((y == 0) || (y == y_len - 1) ? a_s : 0);

		data /= b_tmp;

		d.template at<'x', 'y', 'z'>(x, y, z) = data;

		a_state = a_tmp / b_tmp;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else
	{
		const auto prev_state = noarr::idx<'i'>(y - 1);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = r * (data - a_tmp * d.template at<'x', 'y', 'z'>(x, y - 1, z));

		d.template at<'x', 'y', 'z'>(x, y, z) = data;

		a_state = r * (0 - a_tmp * a[prev_state]);

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}

	if (y != y_begin && y != y_end - 1)
	{
		const auto state0 = noarr::idx<'i'>(y_begin);

		const auto r0 = 1 / (1 - a_state * c[state0]);

		d.template at<'x', 'y', 'z'>(x, y_begin, z) =
			r0 * (d.template at<'x', 'y', 'z'>(x, y_begin, z) - c[state0] * data);

		// #pragma omp critical
		// 		std::cout << "0 z: " << z << " y: " << y_begin << " x: " << x << " a0: " << a[state0] << " c0: " <<
		// c[state0]
		// 				  << " r0: " << r0 << " d: " << d.template at<'x', 'y', 'z'>(x, y_begin, z) << std::endl;
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked_alt_numa(const density_bag_t d, const index_t z, const index_t y,
														  const index_t x, real_t data, const real_t a_s,
														  const real_t b1_s, const index_t y_global_begin,
														  const index_t y_global_end, const index_t y_global_len,
														  scratch_bag_t a, scratch_bag_t c)

{
	const index_t y_local_len = y_global_end - y_global_begin;
	const index_t global_y = y_global_begin + y;

	real_t a_state;

	const auto a_tmp = a_s * (global_y == 0 ? 0 : 1);
	const auto b_tmp = b1_s + ((global_y == 0) || (global_y == y_global_len - 1) ? a_s : 0);

	if (y < 2) [[unlikely]]
	{
		data /= b_tmp;

		d.template at<'x', 'y', 'z'>(x, y, z) = data;

		a_state = a_tmp / b_tmp;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else [[likely]]
	{
		const auto prev_state = noarr::idx<'i'>(y - 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = r * (data - a_tmp * d.template at<'x', 'y', 'z'>(x, y - 1, z));

		d.template at<'x', 'y', 'z'>(x, y, z) = data;

		a_state = r * (0 - a_tmp * a[prev_state]);

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}

	if (y != 0 && y != y_local_len - 1)
	{
		const auto state0 = noarr::idx<'i'>(0);

		const auto r0 = 1 / (1 - a_state * c[state0]);

		d.template at<'x', 'y', 'z'>(x, 0, z) = r0 * (d.template at<'x', 'y', 'z'>(x, 0, z) - c[state0] * data);


		// #pragma omp critical
		// 		std::cout << "0 z: " << z << " y: " << y << " x: " << x << " a0: " << a[state0] << " c0: " << c[state0]
		// 				  << " r0: " << r0 << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) <<
		// std::endl;
	}
}

template <bool z_blocked, typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t,
		  typename z_data_t>
constexpr static void y_backward(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z,
								 const real_t az_s, const real_t b1z_s, z_data_t z_data)

{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();

	for (index_t i = y_len - 2; i >= 0; i--)
	{
		const auto back_c = c.template at<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			real_t prev_data = d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x);
			d.template at<'s', 'z', 'y', 'x'>(s, z, i, x) -= back_c * prev_data;

			if constexpr (!z_blocked)
				z_forward_inside_y(d, s, z, i + 1, x, az_s, b1z_s, z_data.c_tmp, prev_data);
			else
				z_forward_inside_y_blocked(d, s, z, i + 1, x, prev_data, az_s, b1z_s, z_data.begin, z_data.c);
		}
	}
}

// template <typename index_t, typename density_bag_t, typename diag_bag_t, typename z_data_t>
// constexpr static void y_backward_blocked(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z,
// 							   z_data_t z_data)

// {
// 	const index_t x_len = d | noarr::get_length<'x'>();
// 	const index_t y_len = d | noarr::get_length<'y'>();

// 	for (index_t i = y_len - 2; i >= 0; i--)
// 	{
// 		const auto back_c = c.template at<'s', 'i'>(s, i);

// 		for (index_t x = 0; x < x_len; x++)
// 		{
// 			d.template at<'s', 'z', 'y', 'x'>(s, z, i, x) -= back_c * d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x);

// 			z_forward_inside_y_blocked(d, s, z, i + 1, x, z_data.begin, z_data.a_s, z_data.b1_s, z_data.c,
// 									   d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x));
// 		}
// 	}
// }

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t>
constexpr static void x_forward(const density_bag_t d, const index_t z, const index_t y, const diag_bag_t rf)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	real_t prev = d.template at<'z', 'y', 'x'>(z, y, 0);

	for (index_t i = 1; i < x_len; i++)
	{
		real_t curr = d.template at<'z', 'y', 'x'>(z, y, i);
		curr = curr - rf.template at<'i'>(i) * prev;
		d.template at<'z', 'y', 'x'>(z, y, i) = curr;

		prev = curr;
	}
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t>
constexpr static void x_backward(const density_bag_t d, const diag_bag_t b, const diag_bag_t c, const index_t z,
								 const index_t y, y_func_t&& y_forward)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	real_t prev = d.template at<'z', 'x', 'y'>(z, x_len - 1, y);

	prev *= b.template at<'i'>(x_len - 1);

	for (index_t i = x_len - 2; i >= 0; i--)
	{
		auto idx = noarr::idx<'i'>(i);
		real_t curr = d.template at<'z', 'x', 'y'>(z, i, y);
		curr = (curr - c[idx] * prev) * b[idx];

		prev = y_forward(i + 1, prev);

		d.template at<'z', 'x', 'y'>(z, i + 1, y) = prev;

		prev = curr;
	}

	prev = y_forward(0, prev);

	d.template at<'z', 'x', 'y'>(z, 0, y) = prev;
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_xyz_fused(real_t* __restrict__ densities, const real_t* __restrict__ ax,
								  const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
								  const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
								  const real_t* __restrict__ back_cy, const real_t* __restrict__ az,
								  const real_t* __restrict__ b1z, const real_t* __restrict__ back_cz,
								  const density_layout_t dens_l, const diagonal_layout_t diagx_l,
								  const diagonal_layout_t diagy_l, const diagonal_layout_t diagz_l,
								  const index_t s_begin, const index_t s_end)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

#pragma omp for schedule(static) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);
		const auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		const auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);
		const auto cz = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s), back_cz);

		real_t c_tmp_z = az_s;

		for (index_t z = 0; z < z_len; z++)
		{
			real_t c_tmp_y = ay_s;

			for (index_t y = 0; y < y_len; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward<false, false, false>(d, cx, z, y, prev, ay_s, b1y_s, inside_data { c_tmp_y });

				y_forward_inside_x<true>(d, z, y, 0, prev, ay_s, b1y_s, c_tmp_y);
			}

			y_backward<false>(d, cy, s, z, ay_s, b1y_s, inside_data { c_tmp_z });


			{
				real_t r_z = 0;

				for (index_t x = 0; x < x_len; x++)
				{
					r_z = z_forward_inside_y(d, s, z, 0, x, az_s, b1z_s, c_tmp_z,
											 d.template at<'s', 'z', 'y', 'x'>(s, z, 0, x));
				}

				c_tmp_z = az_s * r_z;
			}
		}

		z_backward(d, cz, s);
	}
}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t>
constexpr static void x_forward_vectorized(vec_t* rows, simd_tag t, const index_t length, const index_t x,
										   const diagonal_bag_t rf, vec_t& prev)

{
	for (index_t v = 0; v < length; v++)
	{
		rows[v] = hn::MulAdd(prev, hn::Set(t, -rf.template at<'i'>(x + v)), rows[v]);
		prev = rows[v];
	};
}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t>
constexpr static void x_backward_vectorized(const diagonal_bag_t b, const diagonal_bag_t c, vec_t* rows, simd_tag t,
											const index_t length, const index_t x, vec_t& prev)
{
	for (index_t v = length - 1; v >= 0; v--)
	{
		rows[v] = hn::Mul(hn::MulAdd(prev, hn::Set(t, -c.template at<'i'>(x + v)), rows[v]),
						  hn::Set(t, b.template at<'i'>(x + v)));
		prev = rows[v];
	}
}

template <typename index_t, typename vec_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static vec_t z_forward_inside_y_vectorized(const dens_bag_t d, simd_tag t, const index_t y, const index_t z,
													 vec_t data, const diag_bag_t rf)
{
	vec_t prev;

	if (z == 0)
		prev = hn::Zero(t);
	else
		prev = hn::Load(t, &(d.template at<'y', 'z'>(y, z - 1)));

	auto idx = noarr::idx<'i'>(z);

	return hn::MulAdd(hn::Set(t, -rf[idx]), prev, data);
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void z_backward_vectorized(const dens_bag_t d, const diag_bag_t b, const diag_bag_t c, simd_tag t,
											const index_t simd_length, const index_t y_len, const index_t z_len)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const auto d_blocked = noarr::make_bag(
				blocked_dens_l ^ noarr::fix<'y', 'b', 'X', 'x'>(y, noarr::lit<0>, X, noarr::lit<0>), d.data());

			vec_t prev = hn::Load(t, &d_blocked.template at<'z'>(z_len - 1));

			prev = hn::Mul(prev, hn::Set(t, b.template at<'i'>(z_len - 1)));
			hn::Store(prev, t, &d_blocked.template at<'z'>(z_len - 1));

			for (index_t i = z_len - 2; i >= 0; i--)
			{
				auto idx = noarr::idx<'i'>(i);

				vec_t curr = hn::Load(t, &d_blocked.template at<'z'>(i));
				curr = hn::Mul(hn::MulAdd(prev, hn::Set(t, -c[idx]), curr), hn::Set(t, b[idx]));
				hn::Store(curr, t, &d_blocked.template at<'z'>(i));

				prev = curr;
			}
		}
	}
}

template <typename index_t, typename vec_t, typename simd_tag, typename density_bag_t, typename diag_bag_t>
constexpr static void y_forward_inside_x_vectorized(const density_bag_t d, vec_t* rows, simd_tag t, const index_t z,
													const index_t y_offset, const index_t x, const index_t length,
													const diag_bag_t rf)
{
	vec_t prev;

	if (y_offset == 0)
		prev = hn::Zero(t);
	else
		prev = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y_offset - 1, x)));

	for (index_t v = 0; v < length; v++)
	{
		const index_t y = y_offset + v;
		auto idx = noarr::idx<'i'>(y);

		rows[v] = hn::MulAdd(hn::Set(t, -rf[idx]), prev, rows[v]);
		prev = rows[v];
	}
}

template <typename index_t, typename vec_t, typename simd_tag, typename density_bag_t, typename diag_bag_t>
constexpr static void y_forward_inside_x_blocked_vectorized(const density_bag_t d, vec_t* rows, simd_tag t,
															const index_t z, const index_t y_offset, const index_t x,
															const index_t length, const diag_bag_t rf)
{
	const index_t y_begin = std::max(y_offset, 2);

	vec_t prev;
	if (y_begin == y_offset)
		prev = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y_begin - 1, x)));
	else
		prev = rows[1];

	for (index_t y = y_begin; y < y_offset + length; y++)
	{
		const index_t v = y - y_offset;
		auto idx = noarr::idx<'i'>(y);

		rows[v] = hn::MulAdd(hn::Set(t, -rf[idx]), prev, rows[v]);
		prev = rows[v];

		// #pragma omp critical
		// 		std::cout << "f " << z << " " << y << " " << x << " rf: " << rf[idx] << std::endl;
	}
}

template <typename index_t, typename real_t, typename vec_t, typename simd_tag, typename density_bag_t>
constexpr static void y_forward_inside_x_vectorized_blocked_alt(
	const density_bag_t d, vec_t* rows, simd_tag t, const index_t z, const index_t y_offset, const index_t x,
	const index_t length, const index_t y_len, const real_t ay_s, const real_t b1y_s, const index_t y_begin,
	const index_t y_end, real_t ay_tmp, real_t cy_tmp, real_t cy0_tmp)
{
	vec_t prev;
	vec_t y0;

	if (y_offset == y_begin)
		prev = hn::Zero(t);
	else
	{
		prev = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y_offset - 1, x)));
		y0 = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y_begin, x)));
	}

	for (index_t v = 0; v < length; v++)
	{
		const index_t y = y_offset + v;

		if (y < y_begin + 2)
		{
			const auto a_tmp = ay_s * (y == 0 ? 0 : 1);
			const auto b_tmp = b1y_s + ((y == 0) || (y == y_len - 1) ? ay_s : 0);
			const auto c_tmp = ay_s * (y == y_len - 1 ? 0 : 1);

			rows[v] = hn::Mul(rows[v], hn::Set(t, 1 / b_tmp));
			prev = rows[v];

			ay_tmp = a_tmp / b_tmp;
			cy_tmp = c_tmp / b_tmp;

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "f0: " << z << " " << y << " " << x + l << " " << hn::ExtractLane(rows[v], l)
			// << " "
			// 						  << b_tmp << std::endl;
			// 			}
		}
		else
		{
			const auto a_tmp = ay_s * (y == 0 ? 0 : 1);
			const auto b_tmp = b1y_s + (y == y_len - 1 ? ay_s : 0);
			const auto c_tmp = ay_s * (y == y_len - 1 ? 0 : 1);

			const auto r = 1 / (b_tmp - a_tmp * cy_tmp);

			ay_tmp = r * (0 - a_tmp * ay_tmp);
			cy_tmp = r * c_tmp;

			rows[v] = hn::MulAdd(hn::Set(t, -a_tmp), prev, rows[v]);
			rows[v] = hn::Mul(rows[v], hn::Set(t, r));
			prev = rows[v];

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "f1: " << z << " " << y << " " << x + l << " " << hn::ExtractLane(rows[v], l)
			// << " "
			// 						  << a_tmp << " " << b_tmp << " " << cy_tmp << std::endl;
			// 			}
		}

		if (y != y_begin && y != y_end - 1)
		{
			// const auto state0 = noarr::idx<'s', 'i'>(s, y_begin);
			const auto r0 = 1 / (1 - ay_tmp * cy0_tmp);

			if (y == y_begin + 1)
				y0 = rows[0];

			y0 = hn::MulAdd(hn::Set(t, -cy0_tmp), rows[v], y0);
			y0 = hn::Mul(y0, hn::Set(t, r0));

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "0 z: " << z << " y: " << y << " x: " << x + l << " a0: " << ay0_tmp << " c0: "
			// << cy0_tmp
			// 						  << " r0: " << r0 << " d: " << hn::ExtractLane(y0, l) << std::endl;
			// 			}

			cy0_tmp = r0 * -cy_tmp * cy0_tmp;
		}
	}

	if (y_offset == y_begin)
		rows[0] = y0;
	else
		hn::Store(y0, t, &(d.template at<'z', 'y', 'x'>(z, y_begin, x)));
}

template <typename index_t, typename real_t, typename vec_t, typename simd_tag, typename density_bag_t>
constexpr static void y_forward_inside_x_vectorized_blocked_alt_numa(
	const density_bag_t d, vec_t* rows, simd_tag t, const index_t z, const index_t y_offset, const index_t x,
	const index_t length, const index_t y_global_len, const real_t ay_s, const real_t b1y_s,
	const index_t y_global_begin, const index_t y_global_end, real_t ay_tmp, real_t cy_tmp, real_t cy0_tmp)
{
	vec_t prev;
	vec_t y0;

	const index_t y_local_len = y_global_end - y_global_begin;

	if (y_offset == 0)
		prev = hn::Zero(t);
	else
	{
		prev = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y_offset - 1, x)));
		y0 = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, 0, x)));
	}

	for (index_t v = 0; v < length; v++)
	{
		const index_t y = y_offset + v;
		const index_t y_global = y + y_global_begin;

		const auto a_tmp = ay_s * (y_global == 0 ? 0 : 1);
		const auto b_tmp = b1y_s + ((y_global == 0) || (y_global == y_global_len - 1) ? ay_s : 0);
		const auto c_tmp = ay_s * (y_global == y_global_len - 1 ? 0 : 1);

		if (y < 2) [[unlikely]]
		{
			rows[v] = hn::Mul(rows[v], hn::Set(t, 1 / b_tmp));
			prev = rows[v];

			ay_tmp = a_tmp / b_tmp;
			cy_tmp = c_tmp / b_tmp;

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "f0: " << z << " " << y << " " << x + l << " " << hn::ExtractLane(rows[v], l)
			// << " "
			// 						  << b_tmp << std::endl;
			// 			}
		}
		else [[likely]]
		{
			const auto r = 1 / (b_tmp - a_tmp * cy_tmp);

			ay_tmp = r * (0 - a_tmp * ay_tmp);
			cy_tmp = r * c_tmp;

			rows[v] = hn::MulAdd(hn::Set(t, -a_tmp), prev, rows[v]);
			rows[v] = hn::Mul(rows[v], hn::Set(t, r));
			prev = rows[v];

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "f1: " << z << " " << y << " " << x + l << " " << hn::ExtractLane(rows[v], l)
			// << " "
			// 						  << a_tmp << " " << b_tmp << " " << cy_tmp << std::endl;
			// 			}
		}

		if (y != 0 && y != y_local_len - 1) [[likely]]
		{
			// const auto state0 = noarr::idx<'s', 'i'>(s, y_begin);
			const auto r0 = 1 / (1 - ay_tmp * cy0_tmp);

			if (y == 1)
				y0 = rows[0];

			y0 = hn::MulAdd(hn::Set(t, -cy0_tmp), rows[v], y0);
			y0 = hn::Mul(y0, hn::Set(t, r0));

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "0 z: " << z << " y: " << y << " x: " << x + l << " a0: " << ay0_tmp << " c0: "
			// << cy0_tmp
			// 						  << " r0: " << r0 << " d: " << hn::ExtractLane(y0, l) << std::endl;
			// 			}

			cy0_tmp = r0 * -cy_tmp * cy0_tmp;
		}
	}

	if (y_offset == 0)
		rows[0] = y0;
	else
		hn::Store(y0, t, &(d.template at<'z', 'y', 'x'>(z, 0, x)));
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked_next(const index_t y, const real_t a_s, const real_t b1_s,
													  const index_t y_len, const index_t y_begin, scratch_bag_t a,
													  scratch_bag_t c)

{
	if (y < y_begin + 2)
	{
		const auto state = noarr::idx<'i'>(y);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((y == 0) || (y == y_len - 1) ? a_s : 0);
		const auto c_tmp = a_s * (y == y_len - 1 ? 0 : 1);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'i'>(y);
		const auto prev_state = noarr::idx<'i'>(y - 1);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);
		const auto c_tmp = a_s * (y == y_len - 1 ? 0 : 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked_next_alt(const index_t y, const real_t a_s, const real_t b1_s,
														  const index_t y_len, const index_t y_begin,
														  const index_t y_end, scratch_bag_t a, scratch_bag_t c)

{
	if (y < y_begin + 2)
	{
		const auto state = noarr::idx<'i'>(y);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((y == 0) || (y == y_len - 1) ? a_s : 0);
		const auto c_tmp = a_s * (y == y_len - 1 ? 0 : 1);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'i'>(y);
		const auto prev_state = noarr::idx<'i'>(y - 1);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);
		const auto c_tmp = a_s * (y == y_len - 1 ? 0 : 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}

	if (y != y_begin && y != y_end - 1)
	{
		const auto state = noarr::idx<'i'>(y);
		const auto state0 = noarr::idx<'i'>(y_begin);
		const auto r0 = 1 / (1 - a[state] * c[state0]);

		a[state0] *= r0;
		c[state0] = r0 * -c[state] * c[state0];
	}
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked_next_alt_numa(const index_t y, const real_t a_s, const real_t b1_s,
															   const index_t y_global_len, const index_t y_global_begin,
															   const index_t y_global_end, scratch_bag_t a,
															   scratch_bag_t c)

{
	const index_t y_local_len = y_global_end - y_global_begin;

	const index_t global_y = y + y_global_begin;

	const auto a_tmp = a_s * (global_y == 0 ? 0 : 1);
	const auto b_tmp = b1_s + ((global_y == 0) || (global_y == y_global_len - 1) ? a_s : 0);
	const auto c_tmp = a_s * (global_y == y_global_len - 1 ? 0 : 1);

	if (y < 2)
	{
		const auto state = noarr::idx<'i'>(y);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'i'>(y);
		const auto prev_state = noarr::idx<'i'>(y - 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}

	if (y != 0 && y != y_local_len - 1)
	{
		const auto state = noarr::idx<'i'>(y);
		const auto state0 = noarr::idx<'i'>(0);
		const auto r0 = 1 / (1 - a[state] * c[state0]);

		a[state0] *= r0;
		c[state0] = r0 * -c[state] * c[state0];
	}
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked_next_alt(const index_t z, const real_t a_s, const real_t b1_s,
														  const index_t z_len, const index_t z_begin,
														  const index_t z_end, scratch_bag_t a, scratch_bag_t c)

{
	if (z < z_begin + 2)
	{
		const auto state = noarr::idx<'i'>(z);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);
		const auto c_tmp = a_s * (z == z_len - 1 ? 0 : 1);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'i'>(z);
		const auto prev_state = noarr::idx<'i'>(z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);
		const auto c_tmp = a_s * (z == z_len - 1 ? 0 : 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}

	if (z != z_begin && z != z_end - 1)
	{
		const auto state = noarr::idx<'i'>(z);
		const auto state0 = noarr::idx<'i'>(z_begin);
		const auto r0 = 1 / (1 - a[state] * c[state0]);

		a[state0] *= r0;
		c[state0] = r0 * -c[state] * c[state0];
	}
}


template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked_next_alt_numa(const index_t z, const real_t a_s, const real_t b1_s,
															   const index_t z_global_len, const index_t z_global_begin,
															   const index_t z_global_end, scratch_bag_t a,
															   scratch_bag_t c)

{
	const index_t z_local_len = z_global_end - z_global_begin;

	const index_t global_z = z + z_global_begin;

	const auto a_tmp = a_s * (global_z == 0 ? 0 : 1);
	const auto b_tmp = b1_s + ((global_z == 0) || (global_z == z_global_len - 1) ? a_s : 0);
	const auto c_tmp = a_s * (global_z == z_global_len - 1 ? 0 : 1);

	if (z < 2)
	{
		const auto state = noarr::idx<'i'>(z);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'i'>(z);
		const auto prev_state = noarr::idx<'i'>(z - 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}

	if (z != 0 && z != z_local_len - 1)
	{
		const auto state = noarr::idx<'i'>(z);
		const auto state0 = noarr::idx<'i'>(0);
		const auto r0 = 1 / (1 - a[state] * c[state0]);

		a[state0] *= r0;
		c[state0] = r0 * -c[state] * c[state0];
	}
}

template <typename index_t>
struct inner_z_data
{
	index_t& cz_tmp;
};

template <typename index_t, typename scratch_bag_t>
struct inner_z_data_blocked
{
	index_t z_begin;
	scratch_bag_t az_scratch;
	scratch_bag_t cz_scratch;
};

template <typename index_t>
struct inner_y_data
{
	index_t& cy_tmp;
};

template <typename index_t, typename real_t, typename scratch_bag_t>
struct inner_y_data_blocked
{
	index_t y_begin;
	real_t& cy_tmp;
	scratch_bag_t ay_scratch;
	scratch_bag_t cy_scratch;
};

template <typename index_t, typename real_t, typename scratch_bag_t>
struct inner_y_data_blocked_alt
{
	index_t y_begin;
	index_t y_end;
	real_t& ay_tmp;
	real_t& cy_tmp;
	real_t& ay0_tmp;
	real_t& cy0_tmp;
	scratch_bag_t ay_scratch;
	scratch_bag_t cy_scratch;
};

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t,
		  typename z_func_t>
constexpr static void y_backward_vectorized(const dens_bag_t d, const diag_bag_t b, const diag_bag_t c, simd_tag t,
											const index_t z, const index_t simd_length, const index_t y_len,
											z_func_t&& z_forward)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d_blocked =
			noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

		vec_t prev = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, y_len - 1));

		prev = hn::Mul(prev, hn::Set(t, b.template at<'i'>(y_len - 1)));

		for (index_t i = y_len - 2; i >= 0; i--)
		{
			auto idx = noarr::idx<'i'>(i);

			vec_t curr = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, i));
			curr = hn::Mul(hn::MulAdd(prev, hn::Set(t, -c[idx]), curr), hn::Set(t, b[idx]));

			prev = z_forward(prev, d_blocked, i + 1);

			hn::Store(prev, t, &d_blocked.template at<'z', 'y'>(z, i + 1));

			prev = curr;
		}

		prev = z_forward(prev, d_blocked, 0);

		hn::Store(prev, t, &d_blocked.template at<'z', 'y'>(z, 0));
	}
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename y_func_t>
constexpr static void xy_fused_transpose_part(const density_bag_t d, simd_tag t, const index_t simd_length,
											  const index_t y_len, const index_t z, const diag_bag_t b,
											  const diag_bag_t c, const diag_bag_t rf, y_func_t&& y_forward)
{
	const index_t n = d | noarr::get_length<'x'>();

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	// vector registers that hold the to be transposed x*y plane
	simd_t* rows = new simd_t[simd_length];

	for (index_t y = 0; y < y_len; y += simd_length)
	{
		simd_t prev = hn::Zero(t);

		// forward substitution until last simd_length elements
		for (index_t i = 0; i < full_n - simd_length; i += simd_length)
		{
			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				rows[v] = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y + v, i)));

			// transposition to enable vectorization
			transpose(rows);

			// actual forward substitution (vectorized)
			{
				x_forward_vectorized(rows, t, simd_length, i, rf, prev);
			}

			// transposition back to the original form
			transpose(rows);

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(rows[v], t, &(d.template at<'z', 'y', 'x'>(z, y + v, i)));
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				rows[v] = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y + v, full_n - simd_length)));

			// transposition to enable vectorization
			transpose(rows);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			{
				x_forward_vectorized(rows, t, remainder_work, full_n - simd_length, rf, prev);
			}

			prev = hn::Zero(t);

			// the begin of backward part
			{
				x_backward_vectorized(b, c, rows, t, remainder_work, full_n - simd_length, prev);
			}

			// transposition back to the original form
			transpose(rows);

			y_forward(rows, y, full_n - simd_length);

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(rows[v], t, &(d.template at<'z', 'y', 'x'>(z, y + v, full_n - simd_length)));
		}

		// we continue with backwards substitution
		for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
		{
			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				rows[v] = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y + v, i)));

			// transposition to enable vectorization
			transpose(rows);

			// backward propagation
			{
				x_backward_vectorized(b, c, rows, t, simd_length, i, prev);
			}

			// transposition back to the original form
			transpose(rows);

			y_forward(rows, y, i);

			// if constexpr (!y_blocked)
			// 	y_forward_inside_x_vectorized(d, rows, t, z, y, i, simd_length, y_len, ay_s, b1y_s, ay_tmp, by_tmp,
			// 								  y_data.c_tmp);
			// else if constexpr (!blocked_alt)
			// 	y_forward_inside_x_vectorized_blocked(d, rows, t, z, y, i, simd_length, y_len, ay_s, b1y_s,
			// 										  y_data.y_begin, y_data.cy_tmp);
			// else if constexpr (!numa)
			// 	y_forward_inside_x_vectorized_blocked_alt(d, rows, t, z, y, i, simd_length, y_len, ay_s, b1y_s,
			// 											  y_data.y_begin, y_data.y_end, y_data.ay_tmp, y_data.cy_tmp,
			// 											  y_data.cy0_tmp);
			// else
			// 	y_forward_inside_x_vectorized_blocked_alt_numa(d, rows, t, z, y, i, simd_length, y_len, ay_s, b1y_s,
			// 												   y_data.y_begin, y_data.y_end, y_data.ay_tmp,
			// 												   y_data.cy_tmp, y_data.cy0_tmp);


			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(rows[v], t, &(d.template at<'z', 'y', 'x'>(z, y + v, i)));
		}
	}

	delete[] rows;
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
constexpr static void solve_slice_xyz_fused_transpose(real_t* __restrict__ densities, const real_t* __restrict__ bx,
													  const real_t* __restrict__ cx, const real_t* __restrict__ rfx,
													  const real_t* __restrict__ by, const real_t* __restrict__ cy,
													  const real_t* __restrict__ rfy, const real_t* __restrict__ bz,
													  const real_t* __restrict__ cz, const real_t* __restrict__ rfz,
													  const density_layout_t dens_l, const diagonal_layout_t diagx_l,
													  const diagonal_layout_t diagy_l, const diagonal_layout_t diagz_l,
													  const index_t s_begin, const index_t s_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), rfy);

		auto bz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s), bz);
		auto cz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s), cz);
		auto rfz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s), rfz);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		for (index_t z = 0; z < z_len; z++)
		{
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;

				auto y_forward = [t, d, z, rfy_bag](simd_t* rows, index_t y_offset, index_t x) {
					y_forward_inside_x_vectorized(d, rows, t, z, y_offset, x, simd_length, rfy_bag);
				};

				xy_fused_transpose_part<simd_t>(d, t, simd_length, y_len_body, z, bx_bag, cx_bag, rfx_bag,
												std::move(y_forward));
			}

			// y remainder
			{
				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len - y_len_remainder; y < y_len; y++)
				{
					x_forward<real_t>(d, z, y, rfx_bag);

					auto y_forward = [d, z, y, rfy_bag](index_t x, real_t data) {
						return y_forward_inside_x(d, z, y, x, data, rfy_bag);
					};

					x_backward<real_t>(d, bx_bag, cx_bag, z, y, std::move(y_forward));
				}
			}

			auto z_forward = [t, z, rfz_bag](simd_t data, auto d, index_t y) {
				return z_forward_inside_y_vectorized(d, t, y, z, data, rfz_bag);
			};

			y_backward_vectorized<simd_t>(d, by_bag, cy_bag, t, z, simd_length, y_len, std::move(z_forward));
		}

		z_backward_vectorized<simd_t>(d, bz_bag, cz_bag, t, simd_length, y_len, z_len);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
constexpr static void solve_slice_xy_fused_transpose(real_t* __restrict__ densities, const real_t* __restrict__ bx,
													 const real_t* __restrict__ cx, const real_t* __restrict__ rfx,
													 const real_t* __restrict__ by, const real_t* __restrict__ cy,
													 const real_t* __restrict__ rfy, const density_layout_t dens_l,
													 const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
													 const index_t s_begin, const index_t s_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), rfy);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t y_len = dens_l | noarr::get_length<'y'>();

		auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

		{
			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;

			auto y_forward = [t, d, rfy_bag](simd_t* rows, index_t y_offset, index_t x) {
				y_forward_inside_x_vectorized(d, rows, t, 0, y_offset, x, simd_length, rfy_bag);
			};

			xy_fused_transpose_part<simd_t>(d, t, simd_length, y_len_body, 0, bx_bag, cx_bag, rfx_bag,
											std::move(y_forward));
		}

		// y remainder
		{
			auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

			for (index_t y = y_len - y_len_remainder; y < y_len; y++)
			{
				x_forward<real_t>(d, 0, y, rfx_bag);

				auto y_forward = [d, y, rfy_bag](index_t x, real_t data) {
					return y_forward_inside_x(d, 0, y, x, data, rfy_bag);
				};

				x_backward<real_t>(d, bx_bag, cx_bag, 0, y, std::move(y_forward));
			}
		}

		auto empty_f = [](auto data, auto, auto) { return data; };
		y_backward_vectorized<simd_t>(d, by_bag, cy_bag, t, 0, simd_length, y_len, std::move(empty_f));
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename sync_func_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ bx, const real_t* __restrict__ cx,
	const real_t* __restrict__ rfx, const real_t* __restrict__ by, const real_t* __restrict__ cy,
	const real_t* __restrict__ rfy, const real_t* __restrict__ az, const real_t* __restrict__ bz,
	const real_t* __restrict__ cz, const real_t* __restrict__ rfz, const real_t* __restrict__ rbz,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
	const diagonal_layout_t diagz_l, const index_t s_begin, const index_t s_end, const index_t z_begin,
	const index_t z_end, sync_func_t&& synchronize_blocked_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), rfy);

		auto slicei = noarr::slice<'i'>(z_begin, z_end - z_begin);
		auto az_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, az);
		auto bz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, bz);
		auto cz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, cz);
		auto rfz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, rfz);
		auto rbz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, rbz);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t y_len = dens_l | noarr::get_length<'y'>();
		const index_t z_len = z_end - z_begin;

		auto blocked_dens_l = d.structure() ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

		for (index_t z = 0; z < z_len; z++)
		{
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;

				auto y_forward = [t, d, z, rfy_bag](simd_t* rows, index_t y_offset, index_t x) {
					y_forward_inside_x_vectorized(d, rows, t, z, y_offset, x, simd_length, rfy_bag);
				};

				xy_fused_transpose_part<simd_t>(d, t, simd_length, y_len_body, z, bx_bag, cx_bag, rfx_bag,
												std::move(y_forward));
			}

			// y remainder
			{
				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len - y_len_remainder; y < y_len; y++)
				{
					x_forward<real_t>(d, z, y, rfx_bag);

					auto y_forward = [d, z, y, rfy_bag](index_t x, real_t data) {
						return y_forward_inside_x(d, z, y, x, data, rfy_bag);
					};

					x_backward<real_t>(d, bx_bag, cx_bag, z, y, std::move(y_forward));
				}
			}

			auto z_forward = [t, z, rfz_bag](simd_t data, auto d, index_t y) {
				return z_forward_inside_y_blocked_vectorized(d, data, t, y, z, rfz_bag);
			};

			y_backward_vectorized<simd_t>(d, by_bag, cy_bag, t, z, simd_length, y_len, std::move(z_forward));
		}

		z_backward_blocked_vectorized<simd_t>(d, rbz_bag, t, simd_length, y_len, z_len);

		synchronize_blocked_z(s);

		z_blocked_end_vectorized<simd_t>(d, az_bag, bz_bag, cz_bag, t, simd_length, y_len, z_len);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t, typename barrier_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked_alt(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ back_cy, const real_t* __restrict__ az, const real_t* __restrict__ b1z,
	real_t* __restrict__ a_data, real_t* __restrict__ c_data, real_t* __restrict__ z_data,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
	const scratch_layout_t scratch_l, const dim_scratch_layout_t dim_scratch_l, const index_t s_begin,
	const index_t s_end, const index_t z_begin, const index_t z_end, const index_t tid, const index_t coop_size,
	barrier_t& barrier)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

		const auto denss_l = dens_l ^ noarr::fix<'s'>(s);

		const auto d = noarr::make_bag(denss_l, densities);

		const auto a_scratch = noarr::make_bag(scratch_l ^ noarr::fix<'s'>(s), a_data);
		const auto c_scratch = noarr::make_bag(scratch_l ^ noarr::fix<'s'>(s), c_data);

		const auto z_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), z_data);

		auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

		for (index_t z = z_begin; z < z_end; z++)
		{
			real_t cy_tmp = ay_s;

			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;

				xy_fused_transpose_part<false, false, false, simd_t>(d, t, simd_length, 0, y_len_body, y_len, z, ax_s,
																	 b1x_s, ay_s, b1y_s, cx, inside_data { cy_tmp });
			}

			// y remainder
			{
				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len - y_len_remainder; y < y_len; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<false, false, false>(d, cx, z, y, prev, ay_s, b1y_s, inside_data { cy_tmp });

					y_forward_inside_x<true>(d, z, y, 0, prev, ay_s, b1y_s, cy_tmp);
				}
			}

			y_backward_vectorized<true, true, false, simd_t>(
				denss_l, densities, cy, t, z, simd_length, y_len, az_s, b1z_s,
				inside_data_blocked_alt { z_begin, z_end, a_scratch, c_scratch });
		}

		barrier.arrive_and_wait();

		z_blocked_middle(d, a_scratch, c_scratch, z_scratch, tid, coop_size, 0, y_len, s);

		barrier.arrive_and_wait();

		z_blocked_end_alt(d, a_scratch, c_scratch, 0, y_len, z_begin, z_end);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t, typename thread_distribution_layout,
		  typename barrier_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked_alt(
	thread_distribution_layout dist_l, real_t** __restrict__ densities, const real_t* __restrict__ ax,
	const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx, const real_t* __restrict__ ay,
	const real_t* __restrict__ b1y, const real_t* __restrict__ back_cy, const real_t* __restrict__ az,
	const real_t* __restrict__ b1z, real_t** __restrict__ a_data, real_t** __restrict__ c_data,
	real_t** __restrict__ z_data, const density_layout_t dens_l, const diagonal_layout_t diagx_l,
	const diagonal_layout_t diagy_l, const scratch_layout_t scratch_l, const dim_scratch_layout_t dim_scratch_l,
	const index_t s_global_offset, const index_t s_begin, const index_t s_end, const index_t z_global_begin,
	const index_t z_global_end, const index_t z_global_len, const index_t tid, const index_t coop_size,
	barrier_t& barrier)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_local_len = z_global_end - z_global_begin;

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		const auto denss_l = dens_l ^ noarr::fix<'s'>(s - s_global_offset);
		const auto scratchs_l = scratch_l ^ noarr::fix<'s'>(s - s_global_offset);

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

		real_t* __restrict__ densitites_t = dist_l | noarr::get_at<'z'>(densities, tid);

		const auto d = noarr::make_bag(denss_l ^ noarr::set_length<'z'>(z_local_len), densitites_t);

		const auto a_scratch =
			noarr::make_bag(scratchs_l ^ noarr::set_length<'i'>(z_local_len), dist_l | noarr::get_at<'z'>(a_data, tid));
		const auto c_scratch =
			noarr::make_bag(scratchs_l ^ noarr::set_length<'i'>(z_local_len), dist_l | noarr::get_at<'z'>(c_data, tid));


		const auto z_scratch = noarr::make_bag(dim_scratch_l, z_data[get_thread_num()]);

		auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

		for (index_t z = 0; z < z_local_len; z++)
		{
			real_t cy_tmp = ay_s;

			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;

				xy_fused_transpose_part<false, false, false, simd_t>(d, t, simd_length, 0, y_len_body, y_len, z, ax_s,
																	 b1x_s, ay_s, b1y_s, cx, inside_data { cy_tmp });
			}

			// y remainder
			{
				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len - y_len_remainder; y < y_len; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<false, false, false>(d, cx, z, y, prev, ay_s, b1y_s, inside_data { cy_tmp });

					y_forward_inside_x<true>(d, z, y, 0, prev, ay_s, b1y_s, cy_tmp);
				}
			}

			y_backward_vectorized<true, true, true, simd_t>(
				d.structure(), densitites_t, cy, t, z, simd_length, y_len, az_s, b1z_s,
				inside_data_blocked_alt_numa { z_global_begin, z_global_end, z_global_len, a_scratch, c_scratch });
		}

		barrier.arrive_and_wait();

		z_blocked_middle(denss_l, scratchs_l, dist_l, densities, a_data, c_data, z_global_len, z_scratch, tid,
						 coop_size, 0, y_len);

		barrier.arrive_and_wait();

		z_blocked_end_alt(d, a_scratch, c_scratch, 0, y_len, 0, z_local_len);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename sync_func_t>
constexpr static void solve_slice_xy_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ bx, const real_t* __restrict__ cx,
	const real_t* __restrict__ rfx, const real_t* __restrict__ ay, const real_t* __restrict__ by,
	const real_t* __restrict__ cy, const real_t* __restrict__ rfy, const real_t* __restrict__ rby,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
	const index_t s_begin, const index_t s_end, const index_t y_begin, const index_t y_end,
	sync_func_t&& synchronize_blocked_y)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto sliceyi = noarr::slice<'i'>(y_begin, y_end - y_begin);
		auto ay_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, ay);
		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, rfy);
		auto rby_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, rby);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t y_len = y_end - y_begin;

		auto blocked_dens_l = d.structure() ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

		{
			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;

			auto y_forward = [t, d, rfy_bag](simd_t* rows, index_t y_offset, index_t x) {
				y_forward_inside_x_blocked_vectorized(d, rows, t, 0, y_offset, x, simd_length, rfy_bag);
			};

			xy_fused_transpose_part<simd_t>(d, t, simd_length, y_len_body, 0, bx_bag, cx_bag, rfx_bag,
											std::move(y_forward));
		}

		// y remainder
		{
			auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

			for (index_t y = y_len - y_len_remainder; y < y_len; y++)
			{
				x_forward<real_t>(d, 0, y, rfx_bag);

				auto y_forward = [d, y, rfy_bag](index_t x, real_t data) {
					return y_forward_inside_x_blocked(d, 0, y, x, data, rfy_bag);
				};

				x_backward<real_t>(d, bx_bag, cx_bag, 0, y, std::move(y_forward));
			}
		}

		y_backward_blocked_vectorized<simd_t>(d, rby_bag, t, 0, simd_length, y_len);

		synchronize_blocked_y(0, s);

		auto empty_f = [](auto data, auto, auto) { return data; };
		y_blocked_end_vectorized<simd_t>(d, ay_bag, by_bag, cy_bag, t, 0, simd_length, y_len, std::move(empty_f));
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t, typename barrier_t>
constexpr static void solve_slice_xy_fused_transpose_blocked_alt(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	real_t* __restrict__ a_data, real_t* __restrict__ c_data, real_t* __restrict__ y_data,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const scratch_layout_t scratch_l,
	const dim_scratch_layout_t dim_scratch_l, const index_t s_begin, const index_t s_end, const index_t y_begin,
	const index_t y_end, const index_t tid, const index_t coop_size, barrier_t& barrier)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);

		const auto a_scratch = noarr::make_bag(scratch_l ^ noarr::fix<'s'>(s), a_data);
		const auto c_scratch = noarr::make_bag(scratch_l ^ noarr::fix<'s'>(s), c_data);

		const auto y_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), y_data);

		{
			real_t ay_tmp = ay_s;
			real_t cy_tmp = ay_s;

			const auto a_tmp = ay_s * (y_begin == 0 ? 0 : 1);
			const auto b_tmp = b1y_s + ((y_begin == 0) || (y_begin == y_len - 1) ? ay_s : 0);
			const auto c_tmp = ay_s * (y_begin == y_len - 1 ? 0 : 1);

			real_t ay0_tmp = a_tmp / b_tmp;
			real_t cy0_tmp = c_tmp / b_tmp;

			auto y_remainder = (y_end - y_begin) % simd_length;

			xy_fused_transpose_part<true, true, false, simd_t>(
				d, t, simd_length, y_begin, y_end - y_remainder, y_len, 0, ax_s, b1x_s, ay_s, b1y_s, cx,
				inner_y_data_blocked_alt { y_begin, y_end, ay_tmp, cy_tmp, ay0_tmp, cy0_tmp, a_scratch, c_scratch });
		}

		// y remainder
		{
			auto y_remainder = (y_end - y_begin) % simd_length;

			for (index_t y = y_end - y_remainder; y < y_end; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, 0, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward<true, true, false>(d, cx, 0, y, prev, ay_s, b1y_s,
											  inside_data_blocked_alt { y_begin, y_end, a_scratch, c_scratch });

				y_forward_inside_x_blocked_alt(d, 0, y, 0, prev, ay_s, b1y_s, y_begin, y_end, a_scratch, c_scratch);

				y_forward_inside_x_blocked_next_alt(y, ay_s, b1y_s, y_len, y_begin, y_end, a_scratch, c_scratch);
			}
		}

		barrier.arrive_and_wait();

		y_blocked_middle(d, 0, a_scratch, c_scratch, y_scratch, tid, coop_size);

		barrier.arrive_and_wait();

		y_blocked_end_alt(d, 0, a_scratch, c_scratch, y_begin, y_end);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t, typename thread_distribution_layout,
		  typename barrier_t>
constexpr static void solve_slice_xy_fused_transpose_blocked_alt(
	thread_distribution_layout dist_l, real_t** __restrict__ densities, const real_t* __restrict__ ax,
	const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx, const real_t* __restrict__ ay,
	const real_t* __restrict__ b1y, real_t** __restrict__ a_data, real_t** __restrict__ c_data,
	real_t** __restrict__ y_data, const density_layout_t dens_l, const diagonal_layout_t diagx_l,
	const scratch_layout_t scratch_l, const dim_scratch_layout_t dim_scratch_l, const index_t s_global_offset,
	const index_t s_begin, const index_t s_end, const index_t y_global_begin, const index_t y_global_end,
	const index_t y_global_len, const index_t tid, const index_t coop_size, barrier_t& barrier)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_local_len = y_global_end - y_global_begin;

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const auto denss_l = dens_l ^ noarr::fix<'s'>(s - s_global_offset);
		const auto scratchs_l = scratch_l ^ noarr::fix<'s'>(s - s_global_offset);

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);

		const auto d =
			noarr::make_bag(denss_l ^ noarr::set_length<'y'>(y_local_len), dist_l | noarr::get_at<'y'>(densities, tid));

		const auto a_scratch =
			noarr::make_bag(scratchs_l ^ noarr::set_length<'i'>(y_local_len), dist_l | noarr::get_at<'y'>(a_data, tid));
		const auto c_scratch =
			noarr::make_bag(scratchs_l ^ noarr::set_length<'i'>(y_local_len), dist_l | noarr::get_at<'y'>(c_data, tid));

		const auto y_scratch = noarr::make_bag(dim_scratch_l, y_data[get_thread_num()]);

		{
			real_t ay_tmp = ay_s;
			real_t cy_tmp = ay_s;

			const auto a_tmp = ay_s * (y_global_begin == 0 ? 0 : 1);
			const auto b_tmp = b1y_s + ((y_global_begin == 0) || (y_global_begin == y_global_len - 1) ? ay_s : 0);
			const auto c_tmp = ay_s * (y_global_begin == y_global_len - 1 ? 0 : 1);

			real_t ay0_tmp = a_tmp / b_tmp;
			real_t cy0_tmp = c_tmp / b_tmp;

			auto y_remainder = y_local_len % simd_length;

			xy_fused_transpose_part<true, true, true, simd_t>(
				d, t, simd_length, 0, y_local_len - y_remainder, y_global_len, 0, ax_s, b1x_s, ay_s, b1y_s, cx,
				inner_y_data_blocked_alt { y_global_begin, y_global_end, ay_tmp, cy_tmp, ay0_tmp, cy0_tmp, a_scratch,
										   c_scratch });
		}

		// y remainder
		{
			auto y_remainder = y_local_len % simd_length;

			for (index_t y = y_local_len - y_remainder; y < y_local_len; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, 0, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward<true, true, true>(
					d, cx, 0, y, prev, ay_s, b1y_s,
					inside_data_blocked_alt_numa { y_global_begin, y_global_end, y_global_len, a_scratch, c_scratch });

				y_forward_inside_x_blocked_alt_numa(d, 0, y, 0, prev, ay_s, b1y_s, y_global_begin, y_global_end,
													y_global_len, a_scratch, c_scratch);

				y_forward_inside_x_blocked_next_alt_numa(y, ay_s, b1y_s, y_global_len, y_global_begin, y_global_end,
														 a_scratch, c_scratch);
			}
		}

		barrier.arrive_and_wait();

		y_blocked_middle(denss_l, scratchs_l, dist_l, densities, a_data, c_data, y_global_len, 0, y_scratch, tid,
						 coop_size);

		barrier.arrive_and_wait();

		y_blocked_end_alt(d, 0, a_scratch, c_scratch, 0, y_local_len);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename sync_func_y_t, typename sync_func_z_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ bx, const real_t* __restrict__ cx,
	const real_t* __restrict__ rfx, const real_t* __restrict__ ay, const real_t* __restrict__ by,
	const real_t* __restrict__ cy, const real_t* __restrict__ rfy, const real_t* __restrict__ rby,
	const real_t* __restrict__ az, const real_t* __restrict__ bz, const real_t* __restrict__ cz,
	const real_t* __restrict__ rfz, const real_t* __restrict__ rbz, const density_layout_t dens_l,
	const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l, const diagonal_layout_t diagz_l,
	const index_t s_begin, const index_t s_end, const index_t y_begin, const index_t y_end, const index_t z_begin,
	const index_t z_end, sync_func_y_t&& synchronize_blocked_y, sync_func_z_t&& synchronize_blocked_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto sliceyi = noarr::slice<'i'>(y_begin, y_end - y_begin);
		auto ay_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, ay);
		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, rfy);
		auto rby_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, rby);

		auto slicezi = noarr::slice<'i'>(z_begin, z_end - z_begin);
		auto az_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, az);
		auto bz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, bz);
		auto cz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, cz);
		auto rfz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, rfz);
		auto rbz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, rbz);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t y_len = y_end - y_begin;
		const index_t z_len = z_end - z_begin;

		auto blocked_dens_l = d.structure() ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

		for (index_t z = 0; z < z_len; z++)
		{
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;

				auto y_forward = [t, z, d, rfy_bag](simd_t* rows, index_t y_offset, index_t x) {
					y_forward_inside_x_blocked_vectorized(d, rows, t, z, y_offset, x, simd_length, rfy_bag);
				};

				xy_fused_transpose_part<simd_t>(d, t, simd_length, y_len_body, z, bx_bag, cx_bag, rfx_bag,
												std::move(y_forward));
			}

			// y remainder
			{
				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len - y_len_remainder; y < y_len; y++)
				{
					x_forward<real_t>(d, z, y, rfx_bag);

					auto y_forward = [d, y, z, rfy_bag](index_t x, real_t data) {
						return y_forward_inside_x_blocked(d, z, y, x, data, rfy_bag);
					};

					x_backward<real_t>(d, bx_bag, cx_bag, z, y, std::move(y_forward));
				}
			}

			y_backward_blocked_vectorized<simd_t>(d, rby_bag, t, z, simd_length, y_len);

			synchronize_blocked_y(z, s);

			auto z_forward = [t, z, rfz_bag](simd_t data, auto d, index_t y) {
				return z_forward_inside_y_blocked_vectorized(d, data, t, y, z, rfz_bag);
			};

			y_blocked_end_vectorized<simd_t>(d, ay_bag, by_bag, cy_bag, t, z, simd_length, y_len, std::move(z_forward));
		}

		z_backward_blocked_vectorized<simd_t>(d, rbz_bag, t, simd_length, y_len, z_len);

		synchronize_blocked_z(s);

		z_blocked_end_vectorized<simd_t>(d, az_bag, bz_bag, cz_bag, t, simd_length, y_len, z_len);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t, typename barrier_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked_alt(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ az, const real_t* __restrict__ b1z, real_t* __restrict__ ay_data,
	real_t* __restrict__ cy_data, real_t* __restrict__ az_data, real_t* __restrict__ cz_data,
	real_t* __restrict__ dim_data, const density_layout_t dens_l, const diagonal_layout_t diagx_l,
	const scratch_layout_t scratchy_l, const scratch_layout_t scratchz_l, const dim_scratch_layout_t dim_scratch_l,
	const index_t s_begin, const index_t s_end, const index_t y_begin, const index_t y_end, const index_t z_begin,
	const index_t z_end, const index_t tid_y, const index_t coop_size_y, const index_t tid_z, const index_t coop_size_z,
	barrier_t& barrier_y, barrier_t& barrier_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t z_len = dens_l | noarr::get_length<'z'>();
		const index_t y_len = dens_l | noarr::get_length<'y'>();

		const auto ay_scratch = noarr::make_bag(scratchy_l ^ noarr::fix<'s'>(s), ay_data);
		const auto cy_scratch = noarr::make_bag(scratchy_l ^ noarr::fix<'s'>(s), cy_data);

		const auto az_scratch = noarr::make_bag(scratchz_l ^ noarr::fix<'s'>(s), az_data);
		const auto cz_scratch = noarr::make_bag(scratchz_l ^ noarr::fix<'s'>(s), cz_data);

		const auto dim_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), dim_data);

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);

		for (index_t z = z_begin; z < z_end; z++)
		{
			{
				real_t ay_tmp = ay_s;
				real_t cy_tmp = ay_s;

				const auto a_tmp = ay_s * (y_begin == 0 ? 0 : 1);
				const auto b_tmp = b1y_s + ((y_begin == 0) || (y_begin == y_len - 1) ? ay_s : 0);
				const auto c_tmp = ay_s * (y_begin == y_len - 1 ? 0 : 1);

				real_t ay0_tmp = a_tmp / b_tmp;
				real_t cy0_tmp = c_tmp / b_tmp;

				auto y_remainder = (y_end - y_begin) % simd_length;

				xy_fused_transpose_part<true, true, false, simd_t>(
					d, t, simd_length, y_begin, y_end - y_remainder, y_len, z, ax_s, b1x_s, ay_s, b1y_s, cx,
					inner_y_data_blocked_alt { y_begin, y_end, ay_tmp, cy_tmp, ay0_tmp, cy0_tmp, ay_scratch,
											   cy_scratch });
			}

			// y remainder
			{
				auto y_remainder = (y_end - y_begin) % simd_length;

				for (index_t y = y_end - y_remainder; y < y_end; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<true, true, false>(d, cx, z, y, prev, ay_s, b1y_s,
												  inside_data_blocked_alt { y_begin, y_end, ay_scratch, cy_scratch });

					y_forward_inside_x_blocked_alt(d, z, y, 0, prev, ay_s, b1y_s, y_begin, y_end, ay_scratch,
												   cy_scratch);

					y_forward_inside_x_blocked_next_alt(y, ay_s, b1y_s, y_len, y_begin, y_end, ay_scratch, cy_scratch);
				}
			}

			barrier_y.arrive_and_wait();

			y_blocked_middle(d, z, ay_scratch, cy_scratch, dim_scratch, tid_y, coop_size_y);

			barrier_y.arrive_and_wait();

			y_blocked_end_alt_vectorized<simd_t>(d.structure(), densities, t, simd_length, z, ay_scratch, cy_scratch,
												 y_begin, y_end, az_s, b1z_s,
												 inside_data_blocked_alt { z_begin, z_end, az_scratch, cz_scratch });

			z_forward_inside_y_blocked_next_alt(z, az_s, b1z_s, z_len, z_begin, z_end, az_scratch, cz_scratch);
		}

		barrier_z.arrive_and_wait();

		z_blocked_middle(d, az_scratch, cz_scratch, dim_scratch, tid_z, coop_size_z, y_begin, y_end, s);

		barrier_z.arrive_and_wait();

		z_blocked_end_alt(d, az_scratch, cz_scratch, y_begin, y_end, z_begin, z_end);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t, typename thread_distribution_layout,
		  typename barrier_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked_alt(
	thread_distribution_layout dist_l, real_t** __restrict__ densities, const real_t* __restrict__ ax,
	const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx, const real_t* __restrict__ ay,
	const real_t* __restrict__ b1y, const real_t* __restrict__ az, const real_t* __restrict__ b1z,
	real_t** __restrict__ ay_data, real_t** __restrict__ cy_data, real_t** __restrict__ az_data,
	real_t** __restrict__ cz_data, real_t** __restrict__ dim_data, const density_layout_t dens_l,
	const diagonal_layout_t diagx_l, const scratch_layout_t scratch_l, const dim_scratch_layout_t dim_scratch_l,
	const index_t s_global_offset, const index_t s_begin, const index_t s_end, const index_t y_global_begin,
	const index_t y_global_end, const index_t y_global_len, const index_t z_global_begin, const index_t z_global_end,
	const index_t z_global_len, const index_t tid_y, const index_t coop_size_y, const index_t tid_z,
	const index_t coop_size_z, barrier_t& barrier_y, barrier_t& barrier_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_local_len = y_global_end - y_global_begin;
	const index_t z_local_len = z_global_end - z_global_begin;

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		const auto denss_l = dens_l ^ noarr::fix<'s'>(s - s_global_offset);
		const auto scratchs_l = scratch_l ^ noarr::fix<'s'>(s - s_global_offset);

		real_t* __restrict__ densities_t = dist_l | noarr::get_at<'y', 'z'>(densities, tid_y, tid_z);

		const auto d = noarr::make_bag(
			denss_l ^ noarr::set_length<'y'>(y_local_len) ^ noarr::set_length<'z'>(z_local_len), densities_t);


		const auto ay_scratch = noarr::make_bag(scratchs_l ^ noarr::set_length<'i'>(y_local_len),
												dist_l | noarr::get_at<'y', 'z'>(ay_data, tid_y, tid_z));
		const auto cy_scratch = noarr::make_bag(scratchs_l ^ noarr::set_length<'i'>(y_local_len),
												dist_l | noarr::get_at<'y', 'z'>(cy_data, tid_y, tid_z));


		const auto az_scratch = noarr::make_bag(scratchs_l ^ noarr::set_length<'i'>(z_local_len),
												dist_l | noarr::get_at<'y', 'z'>(az_data, tid_y, tid_z));
		const auto cz_scratch = noarr::make_bag(scratchs_l ^ noarr::set_length<'i'>(z_local_len),
												dist_l | noarr::get_at<'y', 'z'>(cz_data, tid_y, tid_z));

		const auto dim_scratch = noarr::make_bag(dim_scratch_l, dim_data[get_thread_num()]);

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);

		for (index_t z = 0; z < z_local_len; z++)
		{
			{
				real_t ay_tmp = ay_s;
				real_t cy_tmp = ay_s;

				const auto a_tmp = ay_s * (y_global_begin == 0 ? 0 : 1);
				const auto b_tmp = b1y_s + ((y_global_begin == 0) || (y_global_begin == y_global_len - 1) ? ay_s : 0);
				const auto c_tmp = ay_s * (y_global_begin == y_global_len - 1 ? 0 : 1);

				real_t ay0_tmp = a_tmp / b_tmp;
				real_t cy0_tmp = c_tmp / b_tmp;

				auto y_remainder = y_local_len % simd_length;

				xy_fused_transpose_part<true, true, true, simd_t>(
					d, t, simd_length, 0, y_local_len - y_remainder, y_global_len, z, ax_s, b1x_s, ay_s, b1y_s, cx,
					inner_y_data_blocked_alt { y_global_begin, y_global_end, ay_tmp, cy_tmp, ay0_tmp, cy0_tmp,
											   ay_scratch, cy_scratch });
			}

			// y remainder
			{
				auto y_remainder = y_local_len % simd_length;

				for (index_t y = y_local_len - y_remainder; y < y_local_len; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<true, true, true>(d, cx, z, y, prev, ay_s, b1y_s,
												 inside_data_blocked_alt_numa { y_global_begin, y_global_end,
																				y_global_len, ay_scratch, cy_scratch });

					y_forward_inside_x_blocked_alt_numa(d, z, y, 0, prev, ay_s, b1y_s, y_global_begin, y_global_end,
														y_global_len, ay_scratch, cy_scratch);

					y_forward_inside_x_blocked_next_alt_numa(y, ay_s, b1y_s, y_global_len, y_global_begin, y_global_end,
															 ay_scratch, cy_scratch);
				}
			}

			barrier_y.arrive_and_wait();

			y_blocked_middle(denss_l ^ noarr::set_length<'z'>(z_local_len),
							 scratchs_l ^ noarr::set_length<'z'>(z_local_len), dist_l ^ noarr::fix<'z'>(tid_z),
							 densities, ay_data, cy_data, y_global_len, z, dim_scratch, tid_y, coop_size_y);

			barrier_y.arrive_and_wait();

			y_blocked_end_alt_numa_vectorized<simd_t>(
				d.structure(), densities_t, t, simd_length, z, ay_scratch, cy_scratch, 0, y_local_len, az_s, b1z_s,
				inside_data_blocked_alt_numa { z_global_begin, z_global_end, z_global_len, az_scratch, cz_scratch });

			z_forward_inside_y_blocked_next_alt_numa(z, az_s, b1z_s, z_global_len, z_global_begin, z_global_end,
													 az_scratch, cz_scratch);
		}

		barrier_z.arrive_and_wait();

		z_blocked_middle(denss_l ^ noarr::set_length<'y'>(y_local_len),
						 scratchs_l ^ noarr::set_length<'y'>(y_local_len), dist_l ^ noarr::fix<'y'>(tid_y), densities,
						 az_data, cz_data, z_global_len, dim_scratch, tid_z, coop_size_z, 0, y_local_len);

		barrier_z.arrive_and_wait();

		z_blocked_end_alt(d, az_scratch, cz_scratch, 0, y_local_len, 0, z_local_len);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_x()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_y()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_z()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve()
{
	if (!(cores_division_[0] == 1 && cores_division_[1] == 1 && cores_division_[2] == 1))
	{
		if (this->problem_.dims == 2)
		{
			solve_blocked_2d();
		}
		else if (this->problem_.dims == 3)
		{
			if (cores_division_[1] == 1)
				solve_blocked_3d_z();
			else
				solve_blocked_3d_yz();
		}

		return;
	}

#pragma omp parallel
	{
		perf_counter counter("lstmfpai");

		auto tid = get_thread_id();
		auto work_id = tid.group;

		auto s_global_begin = group_block_offsetss_[work_id];
		auto s_global_end = s_global_begin + group_block_lengthss_[work_id];

		for (index_t s = 0; s < group_block_lengthss_[work_id]; s += substrate_step_)
		{
			auto s_end = std::min(s + substrate_step_, group_block_lengthss_[work_id]);

			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s_global_begin + s
			// 					  << " s_end: " << s_global_begin + s_end << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (use_thread_distributed_allocation_)
				{
					auto s_slice = noarr::slice<'s'>(s_global_begin, s_global_end - s_global_begin);

					if (this->problem_.dims == 3)
						solve_slice_xyz_fused_transpose(
							thread_substrate_array_[work_id], thread_bx_[work_id], thread_cx_[work_id],
							thread_rf_x_[work_id], thread_by_[work_id], thread_cy_[work_id], thread_rf_y_[work_id],
							thread_bz_[work_id], thread_cz_[work_id], thread_rf_z_[work_id], get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice,
							get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice,
							get_diagonal_layout(this->problem_, this->problem_.nz) ^ s_slice, s, s_end);
					else if (this->problem_.dims == 2)
						solve_slice_xy_fused_transpose(
							thread_substrate_array_[work_id], thread_bx_[work_id], thread_cx_[work_id],
							thread_rf_x_[work_id], thread_by_[work_id], thread_cy_[work_id], thread_rf_y_[work_id],
							get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice,
							get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice, s, s_end);
				}
				else
				{
					if (this->problem_.dims == 3)
						solve_slice_xyz_fused_transpose(
							this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_by_[0],
							thread_cy_[0], thread_rf_y_[0], thread_bz_[0], thread_cz_[0], thread_rf_z_[0],
							get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nx),
							get_diagonal_layout(this->problem_, this->problem_.ny),
							get_diagonal_layout(this->problem_, this->problem_.nz), s_global_begin + s,
							s_global_begin + s_end);
					else if (this->problem_.dims == 2)
						solve_slice_xy_fused_transpose(this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0],
													   thread_by_[0], thread_cy_[0], thread_rf_y_[0],
													   get_substrates_layout<3>(),
													   get_diagonal_layout(this->problem_, this->problem_.nx),
													   get_diagonal_layout(this->problem_, this->problem_.ny),
													   s_global_begin + s, s_global_begin + s_end);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_blocked_2d()
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

		barrier_t<true, index_t> barrier(cores_division_[1], countersy_[tid.group]->value);

		const auto block_s_begin = group_block_offsetss_[tid.group];
		const auto block_s_end = block_s_begin + group_block_lengthss_[tid.group];

		const index_t group_size = cores_division_[1];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s += substrate_step_)
		{
			auto s_end = std::min(s + substrate_step_, group_block_lengthss_[tid.group]);

			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s << " s_end: " << s_end
			// 					  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (use_thread_distributed_allocation_)
				{
					auto s_slice = noarr::slice<'s'>(block_s_begin, block_s_end - block_s_begin);

					const auto thread_num = get_thread_num();

					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice;
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice;

					auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout()
								  ^ noarr::fix<'z', 'g'>(tid.z, tid.group);

					auto dens_l = get_blocked_substrate_layout<'y'>(this->problem_.nx, group_block_lengthsy_[tid.y],
																	group_block_lengthsz_[tid.z],
																	group_block_lengthss_[tid.group]);

					auto sync_y = [densities = thread_substrate_array_.get(), rf_data = thread_ay_[thread_num],
								   b_data = thread_by_[thread_num], c_data = thread_cy_[thread_num], dens_l, diag_y,
								   dist_l, n = this->problem_.ny, tid = tid.y, group_size,
								   &barrier](index_t z, index_t s) {
						synchronize_y_blocked_distributed(densities, rf_data, b_data, c_data,
														  dens_l ^ noarr::fix<'s'>(s), diag_y ^ noarr::fix<'s'>(s),
														  dist_l, n, z, tid, group_size, barrier);
					};

					auto current_densities = dist_l | noarr::get_at<'y'>(thread_substrate_array_.get(), tid.y);

					solve_slice_xy_fused_transpose_blocked(
						current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
						thread_ay_[thread_num], thread_by_[thread_num], thread_cy_[thread_num],
						thread_rf_y_[thread_num], thread_rb_y_[thread_num],
						dens_l ^ noarr::set_length<'y'>(block_y_end - block_y_begin), diag_x, diag_y, s, s_end,
						block_y_begin, block_y_end, std::move(sync_y));
				}
				else
				{
					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx);
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny);

					auto dens_l = get_substrates_layout<3>();

					auto sync_y = [densities = this->substrates_, rf_data = thread_ay_[0], b_data = thread_by_[0],
								   c_data = thread_cy_[0], dens_l, diag_y, tid = tid.y, group_size,
								   &barrier](index_t z, index_t s) {
						synchronize_y_blocked(densities, rf_data, b_data, c_data, dens_l ^ noarr::fix<'s'>(s),
											  diag_y ^ noarr::fix<'s'>(s), z, tid, group_size, barrier);
					};

					solve_slice_xy_fused_transpose_blocked(
						this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_ay_[0], thread_by_[0],
						thread_cy_[0], thread_rf_y_[0], thread_rb_y_[0],
						dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_end - block_y_begin), diag_x, diag_y,
						block_s_begin + s, block_s_begin + s_end, block_y_begin, block_y_end, std::move(sync_y));
				}
				// else
				// solve_slice_xy_fused_transpose_blocked_alt<index_t>(
				// 	this->substrates_, ax_, b1x_, cx_, ay_, b1y_, a_scratchy_, c_scratchy_, dim_scratch_,
				// 	get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nx),
				// 	get_scratch_layout(this->problem_.ny, substrate_groups_) ^ noarr::fix<'l'>(tid.group),
				// 	get_dim_scratch_layout(), s, s_end, block_y_begin, block_y_end, tid.y, group_size, barrier);
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_blocked_3d_z()
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

		const auto lane_id_z = tid.y + tid.group * cores_division_[1];

		barrier_t<true, index_t> barrier_z(cores_division_[2], countersz_[lane_id_z]->value);

		const auto block_s_begin = group_block_offsetss_[tid.group];
		const auto block_s_end = block_s_begin + group_block_lengthss_[tid.group];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s += substrate_step_)
		{
			auto s_end = std::min(s + substrate_step_, group_block_lengthss_[tid.group]);

			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " [0, " << tid.y << ", " << tid.z << "]
			// s_begin: "
			// << s
			// 					  << " s_end: " << s + s_step_length << " block_y_begin: " << block_y_begin
			// 					  << " block_y_end: " << block_y_end << " block_z_begin: " << block_z_begin
			// 					  << " block_z_end: " << block_z_end << " group: " << tid.group << " lane_y: " <<
			// lane_id_y
			// 					  << " lane_z: " << lane_id_z << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (use_thread_distributed_allocation_)
				{
					auto s_slice = noarr::slice<'s'>(block_s_begin, block_s_end - block_s_begin);

					const auto thread_num = get_thread_num();

					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice;
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice;
					auto diag_z = get_diagonal_layout(this->problem_, this->problem_.nz) ^ s_slice;

					auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout()
								  ^ noarr::fix<'y', 'g'>(tid.y, tid.group);

					auto dens_l = get_blocked_substrate_layout<'z'>(this->problem_.nx, group_block_lengthsy_[tid.y],
																	group_block_lengthsz_[tid.z],
																	group_block_lengthss_[tid.group]);

					auto sync_z = [densities = thread_substrate_array_.get(), rf_data = thread_az_[thread_num],
								   b_data = thread_bz_[thread_num], c_data = thread_cz_[thread_num], dens_l,
								   diag_l = diag_z, dist_l, n = this->problem_.nz, tid = tid.z,
								   group_size = cores_division_[2], &barrier = barrier_z](index_t s) {
						synchronize_z_blocked_distributed(densities, rf_data, b_data, c_data,
														  dens_l ^ noarr::fix<'s'>(s), diag_l ^ noarr::fix<'s'>(s),
														  dist_l, n, tid, group_size, barrier);
					};

					auto current_densities = dist_l | noarr::get_at<'z'>(thread_substrate_array_.get(), tid.z);

					solve_slice_xyz_fused_transpose_blocked(
						current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
						thread_by_[thread_num], thread_cy_[thread_num], thread_rf_y_[thread_num],
						thread_az_[thread_num], thread_bz_[thread_num], thread_cz_[thread_num],
						thread_rf_z_[thread_num], thread_rb_z_[thread_num],
						dens_l ^ noarr::set_length<'z'>(block_z_end - block_z_begin), diag_x, diag_y, diag_z, s, s_end,
						block_z_begin, block_z_end, std::move(sync_z));
				}
				else
				{
					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx);
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny);
					auto diag_z = get_diagonal_layout(this->problem_, this->problem_.nz);

					auto dens_l = get_substrates_layout<3>();

					auto sync_z = [densities = this->substrates_, rf_data = thread_az_[0], b_data = thread_bz_[0],
								   c_data = thread_cz_[0], dens_l, diag_l = diag_z, tid = tid.z,
								   group_size = cores_division_[2], &barrier = barrier_z](index_t s) {
						synchronize_z_blocked(densities, rf_data, b_data, c_data, dens_l ^ noarr::fix<'s'>(s),
											  diag_l ^ noarr::fix<'s'>(s), tid, group_size, barrier);
					};

					solve_slice_xyz_fused_transpose_blocked(
						this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_by_[0], thread_cy_[0],
						thread_rf_y_[0], thread_az_[0], thread_bz_[0], thread_cz_[0], thread_rf_z_[0], thread_rb_z_[0],
						dens_l ^ noarr::slice<'z'>(block_z_begin, block_z_end - block_z_begin), diag_x, diag_y, diag_z,
						block_s_begin + s, block_s_begin + s_end, block_z_begin, block_z_end, std::move(sync_z));
				}
				// 	else
				// 		solve_slice_xyz_fused_transpose_blocked_alt<index_t>(
				// 			this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, az_, b1z_, a_scratchz_, c_scratchz_,
				// 			dim_scratch_, get_substrates_layout<3>(),
				// 			get_diagonal_layout(this->problem_, this->problem_.nx),
				// 			get_diagonal_layout(this->problem_, this->problem_.ny), lane_scratchz_l,
				// 			get_dim_scratch_layout(), s, s_end, block_z_begin, block_z_end, tid.z, cores_division_[2],
				// 			barrier_z);
				// }
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_blocked_3d_yz()
{
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

		const auto block_y_begin = group_block_offsetsy_[tid.y];
		const auto block_y_end = block_y_begin + group_block_lengthsy_[tid.y];

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_y = tid.z + tid.group * cores_division_[2];

		barrier_t<true, index_t> barrier_y(cores_division_[1], countersy_[lane_id_y]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];

		const auto lane_id_z = tid.y + tid.group * cores_division_[1];

		barrier_t<true, index_t> barrier_z(cores_division_[2], countersz_[lane_id_z]->value);
		// auto& barrier_z = *barriersz_[lane_id_z];

		const auto block_s_begin = group_block_offsetss_[tid.group];
		const auto block_s_end = block_s_begin + group_block_lengthss_[tid.group];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s += substrate_step_)
		{
			auto s_end = std::min(s + substrate_step_, group_block_lengthss_[tid.group]);

			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " [0, " << tid.y << ", " << tid.z << "]
			// s_begin:
			// 	"
			// << s
			// 					  << " s_end: " << s + s_step_length << " block_y_begin: " << block_y_begin
			// 					  << " block_y_end: " << block_y_end << " block_z_begin: " << block_z_begin
			// 					  << " block_z_end: " << block_z_end << " group: " << tid.group << " lane_y: " <<
			// lane_id_y
			// 					  << " lane_z: " << lane_id_z << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (use_thread_distributed_allocation_)
				{
					auto s_slice = noarr::slice<'s'>(block_s_begin, block_s_end - block_s_begin);

					const auto thread_num = get_thread_num();

					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice;
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice;
					auto diag_z = get_diagonal_layout(this->problem_, this->problem_.nz) ^ s_slice;

					auto dist_l =
						noarr::scalar<real_t*>() ^ get_thread_distribution_layout() ^ noarr::fix<'g'>(tid.group);

					auto dens_l = get_blocked_substrate_layout<'*'>(this->problem_.nx, group_block_lengthsy_[tid.y],
																	group_block_lengthsz_[tid.z],
																	group_block_lengthss_[tid.group]);

					auto sync_y = [densities = thread_substrate_array_.get(), rf_data = thread_ay_[thread_num],
								   b_data = thread_by_[thread_num], c_data = thread_cy_[thread_num],
								   dens_l = dens_l ^ noarr::set_length<'z'>(block_z_end - block_z_begin), diag_y,
								   dist_l = dist_l ^ noarr::fix<'z'>(tid.z), n = this->problem_.ny, tid = tid.y,
								   group_size = cores_division_[1], &barrier = barrier_y](index_t z, index_t s) {
						synchronize_y_blocked_distributed(densities, rf_data, b_data, c_data,
														  dens_l ^ noarr::fix<'s'>(s), diag_y ^ noarr::fix<'s'>(s),
														  dist_l, n, z, tid, group_size, barrier);
					};

					auto sync_z = [densities = thread_substrate_array_.get(), rf_data = thread_az_[thread_num],
								   b_data = thread_bz_[thread_num], c_data = thread_cz_[thread_num],
								   dens_l = dens_l ^ noarr::set_length<'y'>(block_y_end - block_y_begin),
								   diag_l = diag_z, dist_l = dist_l ^ noarr::fix<'y'>(tid.y), n = this->problem_.nz,
								   tid = tid.z, group_size = cores_division_[2], &barrier = barrier_z](index_t s) {
						synchronize_z_blocked_distributed(densities, rf_data, b_data, c_data,
														  dens_l ^ noarr::fix<'s'>(s), diag_l ^ noarr::fix<'s'>(s),
														  dist_l, n, tid, group_size, barrier);
					};

					auto current_densities =
						dist_l | noarr::get_at<'y', 'z'>(thread_substrate_array_.get(), tid.y, tid.z);

					solve_slice_xyz_fused_transpose_blocked(
						current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
						thread_ay_[thread_num], thread_by_[thread_num], thread_cy_[thread_num],
						thread_rf_y_[thread_num], thread_rb_y_[thread_num], thread_az_[thread_num],
						thread_bz_[thread_num], thread_cz_[thread_num], thread_rf_z_[thread_num],
						thread_rb_z_[thread_num],
						dens_l ^ noarr::set_length<'y'>(block_y_end - block_y_begin)
							^ noarr::set_length<'z'>(block_z_end - block_z_begin),
						diag_x, diag_y, diag_z, s, s_end, block_y_begin, block_y_end, block_z_begin, block_z_end,
						std::move(sync_y), std::move(sync_z));
				}
				else
				{
					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx);
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny);
					auto diag_z = get_diagonal_layout(this->problem_, this->problem_.nz);

					auto dens_l = get_substrates_layout<3>();

					auto sync_y = [densities = this->substrates_, rf_data = thread_ay_[0], b_data = thread_by_[0],
								   c_data = thread_cy_[0],
								   dens_l = dens_l ^ noarr::slice<'z'>(block_z_begin, block_z_end - block_z_begin),
								   diag_y, tid = tid.y, group_size = cores_division_[1],
								   &barrier = barrier_y](index_t z, index_t s) {
						synchronize_y_blocked(densities, rf_data, b_data, c_data, dens_l ^ noarr::fix<'s'>(s),
											  diag_y ^ noarr::fix<'s'>(s), z, tid, group_size, barrier);
					};

					auto sync_z = [densities = this->substrates_, rf_data = thread_az_[0], b_data = thread_bz_[0],
								   c_data = thread_cz_[0],
								   dens_l = dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_end - block_y_begin),
								   diag_l = diag_z, tid = tid.z, group_size = cores_division_[2],
								   &barrier = barrier_z](index_t s) {
						synchronize_z_blocked(densities, rf_data, b_data, c_data, dens_l ^ noarr::fix<'s'>(s),
											  diag_l ^ noarr::fix<'s'>(s), tid, group_size, barrier);
					};

					solve_slice_xyz_fused_transpose_blocked<index_t>(
						this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_ay_[0], thread_by_[0],
						thread_cy_[0], thread_rf_y_[0], thread_rb_y_[0], thread_az_[0], thread_bz_[0], thread_cz_[0],
						thread_rf_z_[0], thread_rb_z_[0],
						dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_end - block_y_begin)
							^ noarr::slice<'z'>(block_z_begin, block_z_end - block_z_begin),
						diag_x, diag_y, diag_z, block_s_begin + s, block_s_begin + s_end, block_y_begin, block_y_end,
						block_z_begin, block_z_end, std::move(sync_y), std::move(sync_z));
				}
				// 	else
				// 		solve_slice_xyz_fused_transpose_blocked_alt<index_t>(
				// 			this->substrates_, ax_, b1x_, cx_, ay_, b1y_, az_, b1z_, a_scratchy_, c_scratchy_,
				// 			a_scratchz_, c_scratchz_, dim_scratch_, get_substrates_layout<3>(),
				// 			get_diagonal_layout(this->problem_, this->problem_.nx), lane_scratchy_l, lane_scratchz_l,
				// 			get_dim_scratch_layout(), s, s_end, block_y_begin, block_y_end, block_z_begin, block_z_end,
				// 			tid.y, cores_division_[1], tid.z, cores_division_[2], barrier_y, barrier_z);
				// }
			}
		}
	}
}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_f_p<real_t, aligned_x>::least_memory_thomas_solver_d_f_p(
	bool use_alt_blocked, bool use_thread_distributed_allocation)
	: countersy_count_(0),
	  countersz_count_(0),
	  use_alt_blocked_(use_alt_blocked),
	  use_thread_distributed_allocation_(use_thread_distributed_allocation)
{}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_f_p<real_t, aligned_x>::~least_memory_thomas_solver_d_f_p()
{
	for (index_t i = 0; i < get_max_threads(); i++)
	{
		if (thread_cx_)
		{
			std::free(thread_rf_x_[i]);
			std::free(thread_bx_[i]);
			std::free(thread_cx_[i]);
		}

		if (thread_cy_)
		{
			std::free(thread_rf_y_[i]);
			std::free(thread_by_[i]);
			std::free(thread_cy_[i]);
		}

		if (thread_cz_)
		{
			std::free(thread_rf_z_[i]);
			std::free(thread_bz_[i]);
			std::free(thread_cz_[i]);
		}

		if (thread_ax_)
		{
			std::free(thread_ax_[i]);
			std::free(thread_rb_x_[i]);
		}

		if (thread_ay_)
		{
			std::free(thread_ay_[i]);
			std::free(thread_rb_y_[i]);
		}

		if (thread_az_)
		{
			std::free(thread_az_[i]);
			std::free(thread_rb_z_[i]);
		}

		if (thread_substrate_array_)
		{
			std::free(thread_substrate_array_[i]);
		}
	}
}

template <typename real_t, bool aligned_x>
double least_memory_thomas_solver_d_f_p<real_t, aligned_x>::access(std::size_t s, std::size_t x, std::size_t y,
																   std::size_t z) const
{
	if (!use_thread_distributed_allocation_)
		return base_solver<real_t, least_memory_thomas_solver_d_f_p<real_t, aligned_x>>::access(s, x, y, z);

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

	auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout()
				  ^ noarr::fix<'s', 'y', 'z'>(block_idx_s, block_idx_y, block_idx_z);

	auto density =
		dist_l | noarr::get_at<'g', 'y', 'z'>(thread_substrate_array_.get(), block_idx_s, block_idx_y, block_idx_z);

	auto dens_l = get_blocked_substrate_layout(this->problem_.nx, group_block_lengthsy_[block_idx_y],
											   group_block_lengthsz_[block_idx_z], group_block_lengthss_[block_idx_s]);

	return dens_l | noarr::get_at<'x', 'y', 'z', 's'>(density, x, y, z, s);
}

template class least_memory_thomas_solver_d_f_p<float, true>;
template class least_memory_thomas_solver_d_f_p<double, true>;
