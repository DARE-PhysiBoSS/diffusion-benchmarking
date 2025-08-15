#pragma once

#include <iostream>

#include "../base_solver.h"
#include "../least_memory_thomas_solver_d_f.h"
#include "../substrate_layouts.h"
#include "../tridiagonal_solver.h"


template <typename real_t, bool aligned_x>
class sdd_full_blocking : public locally_onedimensional_solver,
						  public base_solver<real_t, sdd_full_blocking<real_t, aligned_x>>

{
	using index_t = std::int32_t;

	std::unique_ptr<std::unique_ptr<aligned_atomic<index_t>>[]> countersy_, countersz_;
	std::unique_ptr<std::unique_ptr<std::barrier<>>[]> barriersy_, barriersz_;
	index_t countersy_count_, countersz_count_;

	std::unique_ptr<real_t*[]> a_, b_, c_, thread_substrate_array_;
	std::unique_ptr<real_t*[]> a_scratch_, c_scratch_;

	std::size_t x_tile_size_;
	std::size_t alignment_size_;

	std::array<index_t, 3> cores_division_;

	std::array<index_t, 3> group_blocks_;
	std::vector<index_t> group_block_lengthsy_;
	std::vector<index_t> group_block_lengthsz_;
	std::vector<index_t> group_block_lengthss_;

	std::vector<index_t> group_block_offsetsy_;
	std::vector<index_t> group_block_offsetsz_;
	std::vector<index_t> group_block_offsetss_;

	index_t substrate_groups_;

	template <char dim_to_skip = ' '>
	auto get_blocked_substrate_layout(index_t nx, index_t ny, index_t nz, index_t substrates_count) const
	{
		std::size_t x_size = nx * sizeof(real_t);
		std::size_t x_size_padded = (x_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		x_size_padded /= sizeof(real_t);

		auto layout = noarr::scalar<real_t>() ^ noarr::vector<'x'>(x_size_padded) ^ noarr::vector<'y'>()
					  ^ noarr::vector<'z'>() ^ noarr::vector<'s'>() ^ noarr::slice<'x'>(nx);

		if constexpr (dim_to_skip == 'y')
			return layout ^ noarr::set_length<'z', 's'>(nz, substrates_count);
		else if constexpr (dim_to_skip == 'z')
			return layout ^ noarr::set_length<'y', 's'>(ny, substrates_count);
		else if constexpr (dim_to_skip == '*')
			return layout ^ noarr::set_length<'s'>(substrates_count);
		else
			return layout ^ noarr::set_length<'y', 'z', 's'>(ny, nz, substrates_count);
	}

	template <char dim>
	auto get_diagonal_layout()
	{
		const auto n = std::max({ this->problem_.nx, group_blocks_[1], group_blocks_[2] });
		const auto s = std::max(alignment_size_ / sizeof(real_t), x_tile_size_);

		std::size_t size = n * sizeof(real_t);
		std::size_t size_padded = (size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		size_padded /= sizeof(real_t);


		std::size_t ssize = s * sizeof(real_t);
		std::size_t ssize_padded = (ssize + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		ssize_padded /= sizeof(real_t);

		return noarr::scalar<real_t>() ^ noarr::vectors<'v', dim>(ssize_padded, size_padded);
	}

	auto get_thread_distribution_layout() const
	{
		return noarr::vectors<'y', 'z', 'g'>(cores_division_[1], cores_division_[2], substrate_groups_);
	}

	void precompute_values(std::unique_ptr<real_t*[]>& a, std::unique_ptr<real_t*[]>& b, std::unique_ptr<real_t*[]>& c,
						   index_t shape, index_t dims);

	void precompute_values(index_t counters_count,
						   std::unique_ptr<std::unique_ptr<aligned_atomic<index_t>>[]>& counters,
						   std::unique_ptr<std::unique_ptr<std::barrier<>>[]>& barriers, index_t group_size, char dim);

	void set_block_bounds(index_t n, index_t group_size, index_t& block_size, std::vector<index_t>& group_block_lengths,
						  std::vector<index_t>& group_block_offsets);

	thread_id_t<index_t> get_thread_id() const;

public:
	sdd_full_blocking();

	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		std::size_t x_size = this->problem_.nx * sizeof(real_t);
		std::size_t x_size_padded = (x_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		x_size_padded /= sizeof(real_t);

		if constexpr (dims == 1)
			return noarr::scalar<real_t>() ^ noarr::vectors<'x', 's'>(x_size_padded, this->problem_.substrates_count);
		else if constexpr (dims == 2)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 's'>(x_size_padded, this->problem_.ny, this->problem_.substrates_count);
		else if constexpr (dims == 3)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 'z', 's'>(x_size_padded, this->problem_.ny, this->problem_.nz,
														this->problem_.substrates_count);
	}

	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;
	void solve_blocked_2d();

	~sdd_full_blocking();
};
