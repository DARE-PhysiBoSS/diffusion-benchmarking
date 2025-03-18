#pragma once

#include <memory>

#include <noarr/structures_extended.hpp>

#include "tridiagonal_solver.h"

/*
The same as least_compute_thomas_solver_s_t, but x is aligned.
*/

template <typename real_t, bool aligned_x>
class least_compute_thomas_solver_s_t : public locally_onedimensional_solver
{
	using index_t = std::int32_t;

	problem_t<index_t, real_t> problem_;

	std::unique_ptr<real_t[]> substrates_;

	std::unique_ptr<real_t[]> bx_, cx_, ex_;
	std::unique_ptr<real_t[]> by_, cy_, ey_;
	std::unique_ptr<real_t[]> bz_, cz_, ez_;

	std::size_t work_items_;
	std::size_t x_tile_size_;
	std::size_t alignment_size_;

	void precompute_values(std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& c, std::unique_ptr<real_t[]>& e,
						   index_t shape, index_t dims, index_t n);

	template <std::size_t dims>
	auto get_substrates_layout(const problem_t<index_t, real_t>& problem) const;

	static auto get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n);

public:
	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;

	void save(const std::string& file) const override;

	double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override;
};
