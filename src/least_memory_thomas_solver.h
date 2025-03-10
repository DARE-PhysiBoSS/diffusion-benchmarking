#pragma once

#include <memory>

#include <noarr/structures_extended.hpp>

#include "tridiagonal_solver.h"

/*
The diffusion is the problem of solving tridiagonal matrix system with these coeficients:
For dimension x:
a_i  == -dt*diffusion_coefs/dx^2                              1 <= i <= n
b_1  == 1 + dt*decay_rates/dims + dt*diffusion_coefs/dx^2
b_i  == 1 + dt*decay_rates/dims + 2*dt*diffusion_coefs/dx^2   1 <  i <  n
b_n  == 1 + dt*decay_rates/dims + dt*diffusion_coefs/dx^2
c_i  == -dt*diffusion_coefs/dx^2                              1 <= i <= n
d_i  == current diffusion rates
For dimension y/z (if they exist):
substitute dx accordingly to dy/dz

Since the matrix is constant for multiple right hand sides, we precompute its values in the following way:
b_1'  == 1/b_1
b_i'  == 1/(b_i - a_i*c_i*b_(i-1)')                           1 <  i <= n
e_i   == a_i*b_(i-1)'                                         1 <  i <= n

Then, the forward substitution is as follows (n multiplications + n subtractions):
d_i'  == d_i - e_i*d_(i-1)                                    1 <  i <= n
The backpropagation (2n multiplications + n subtractions):
d_n'' == d_n'/b_n'
d_i'' == (d_i' - c_i*d_(i+1)'')*b_i'                          n >  i >= 1
*/

template <typename real_t>
class least_memory_thomas_solver : public locally_onedimensional_solver
{
	using index_t = std::int32_t;

	problem_t<index_t, real_t> problem_;

	std::unique_ptr<real_t[]> substrates_;

	std::unique_ptr<real_t[]> ax_, b0x_, scratchpadx_;
	std::unique_ptr<real_t[]> ay_, b0y_, scratchpady_;
	std::unique_ptr<real_t[]> az_, b0z_, scratchpadz_;

	std::unique_ptr<index_t[]> threshold_indexx_, threshold_indexy_, threshold_indexz_;

	static real_t limit_threshold_;

	std::size_t work_items_;

	template <std::size_t dims>
	static auto get_substrates_layout(const problem_t<index_t, real_t>& problem);

	void precompute_values(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b0,
						   std::unique_ptr<index_t[]>& threshold_index, index_t shape, index_t dims, index_t n);

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
