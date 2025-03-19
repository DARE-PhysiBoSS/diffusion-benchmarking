#pragma once

#include "least_compute_thomas_solver.h"
#include "substrate_layouts.h"

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
class least_compute_thomas_solver_trav : public least_compute_thomas_solver<real_t>
{
	using index_t = std::int32_t;

	template <char dim>
	static auto get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n);

public:
	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		return substrate_layouts::get_sxyz_layout<dims>(this->problem_);
	}

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;
};
