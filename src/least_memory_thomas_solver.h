#pragma once

#include "base_solver.h"
#include "least_memory_thomas_solver_data.h"
#include "substrate_layouts.h"
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

Optimizations:
- Substrates are the outermost dimension
- Precomputed a_i, b_1, b_i'
- Minimized memory accesses by computing e_i on the fly
*/

template <typename real_t>
class least_memory_thomas_solver : public locally_onedimensional_solver,
								   public base_solver<real_t, least_memory_thomas_solver<real_t>>

{
	using index_t = std::int32_t;
	using data_t = least_memory_thomas_solver_data<real_t>;

	data_t x_data_, y_data_, z_data_;

	static auto get_diagonal_layout(const problem_t<index_t, real_t>& problem_, index_t n);

public:
	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		return substrate_layouts::get_xyzs_layout<dims>(this->problem_);
	}

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;
};
