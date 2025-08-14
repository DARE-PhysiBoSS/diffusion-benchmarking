#pragma once

#include <iostream>

#include "../base_solver.h"
#include "../substrate_layouts.h"
#include "../tridiagonal_solver.h"


template <typename real_t, bool aligned_x>
class sdd_least_memory_thomas_solver_t : public locally_onedimensional_solver,
										 public base_solver<real_t, sdd_least_memory_thomas_solver_t<real_t, aligned_x>>

{
	using index_t = std::int32_t;

	real_t *a_, *b_, *c_;
	std::vector<real_t*> b_scratch_;

	std::size_t x_tile_size_;
	std::size_t alignment_size_;

	template <char dim>
	auto get_diagonal_layout()
	{
		const auto n = std::max({ this->problem_.nx, this->problem_.ny, this->problem_.nz });
		const auto s = std::max(alignment_size_ / sizeof(real_t), x_tile_size_);

		std::size_t size = n * sizeof(real_t);
		std::size_t size_padded = (size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		size_padded /= sizeof(real_t);


		std::size_t ssize = s * sizeof(real_t);
		std::size_t ssize_padded = (ssize + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		ssize_padded /= sizeof(real_t);

		return noarr::scalar<real_t>() ^ noarr::vectors<'v', dim>(ssize_padded, size_padded);
	}

	void precompute_values(real_t*& a, real_t*& b, real_t*& c, index_t shape, index_t dims);

public:
	sdd_least_memory_thomas_solver_t();

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

	~sdd_least_memory_thomas_solver_t();
};
