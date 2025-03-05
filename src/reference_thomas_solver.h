#pragma once

#include <memory>

#include "tridiagonal_solver.h"

template <typename real_t>
class reference_thomas_solver : public tridiagonal_solver
{
	using index_t = std::int32_t;

	problem_t<index_t, real_t> problem_;

	std::unique_ptr<real_t[]> substrates_;

	std::unique_ptr<real_t[]> ax_, b0x_, bx_;
	std::unique_ptr<real_t[]> ay_, b0y_, by_;
	std::unique_ptr<real_t[]> az_, b0z_, bz_;

	static auto get_substrates_layout(const problem_t<index_t, real_t>& problem);

	void precompute_values(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& b0,
						   index_t shape, index_t dims, index_t n);

public:
	void prepare(const max_problem_t& problem) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void save(const std::string& file) const override;

	double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override;
};
