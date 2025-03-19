#pragma once

#include "base_solver.h"
#include "substrate_layouts.h"
#include "tridiagonal_solver.h"

/*
The same as least_compute_thomas_solver, but substrate dimension is the outermost one.
*/

template <typename real_t>
class least_compute_thomas_solver_s : public locally_onedimensional_solver,
									  public base_solver<real_t, least_compute_thomas_solver_s<real_t>>
{
	using index_t = std::int32_t;

	std::unique_ptr<real_t[]> bx_, cx_, ex_;
	std::unique_ptr<real_t[]> by_, cy_, ey_;
	std::unique_ptr<real_t[]> bz_, cz_, ez_;

	std::size_t work_items_;

	void precompute_values(std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& c, std::unique_ptr<real_t[]>& e,
						   index_t shape, index_t dims, index_t n);

	static auto get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n);

public:
	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		return substrate_layouts::get_xyzs_layout<dims>(this->problem_);
	}

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;
};
