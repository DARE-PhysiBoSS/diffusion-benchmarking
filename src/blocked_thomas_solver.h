#pragma once

#include <barrier>
#include <functional>

#include "base_solver.h"
#include "omp_helper.h"
#include "substrate_layouts.h"
#include "tridiagonal_solver.h"

/*

Restrictions:
- dimension sizes must be divisible by block size
- dimension sizes must all be the same
*/

template <typename real_t, bool aligned_x>
class blocked_thomas_solver : public locally_onedimensional_solver,
							  public base_solver<real_t, blocked_thomas_solver<real_t, aligned_x>>
{
protected:
	using index_t = std::int32_t;

	real_t *ax_, *b1x_;
	real_t *ay_, *b1y_;
	real_t *az_, *b1z_;

	real_t *a_scratch_, *c_scratch_;

	std::size_t block_size_;
	std::size_t alignment_size_;

	using sync_func_t = std::function<void>;

	std::unique_ptr<std::unique_ptr<std::barrier<>>[]> barriers_;

	void precompute_values(real_t*& a, real_t*& b1, index_t shape, index_t dims);

	auto get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n);

public:
	static constexpr index_t min_block_size = 2;

	blocked_thomas_solver();

	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		if constexpr (aligned_x)
			return substrate_layouts::get_xyzs_aligned_layout<dims>(this->problem_, alignment_size_);
		else
			return substrate_layouts::get_xyzs_layout<dims>(this->problem_);
	}

	auto get_scratch_layout() const
	{
		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 't'>(block_size_, get_max_threads());
	}

	std::function<void> get_synchronization_function();

	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;

	~blocked_thomas_solver();
};
