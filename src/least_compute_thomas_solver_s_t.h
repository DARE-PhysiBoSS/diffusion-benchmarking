#pragma once

#include "base_solver.h"
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
a_i' = c_i' = -a_i
b_1'  == 1/b_1
b_i'  == 1/(b_i - a_i'*c_i'*b_(i-1)')                         1 <  i <= n
e_i   == a_i'*b_(i-1)'                                        1 <  i <= n

Then, the forward substitution is as follows (n FMAs):
d_i'  == d_i + e_i*d_(i-1)                                    1 <  i <= n
The backpropagation (n multiplications + n FMAs):
d_n'' == d_n'/b_n'
d_i'' == (d_i' + c_i*d_(i+1)'')*b_i'                          n >  i >= 1

Optimizations:
- Precomputed a_i, b_i', e_i
- Substrate dimension is now outermost, so longer systems can fit into the same cache size
- Aligned memory for x dimension (tunable by 'alignment_size')
- Better temporal locality of memory accesses - x dimension is divided into smaller tiles (tunable by 'x_tile_size') and
y/z dimensions are solved alongside tiled x dimension
*/

template <typename real_t, bool aligned_x>
class least_compute_thomas_solver_s_t : public locally_onedimensional_solver,
										public base_solver<real_t, least_compute_thomas_solver_s_t<real_t, aligned_x>>
{
	using index_t = std::int32_t;

	std::unique_ptr<real_t[]> bx_, cx_, ex_;
	std::unique_ptr<real_t[]> by_, cy_, ey_;
	std::unique_ptr<real_t[]> bz_, cz_, ez_;

	bool vectorized_x_;
	std::size_t x_tile_size_;
	std::size_t alignment_size_;

	void precompute_values(std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& c, std::unique_ptr<real_t[]>& e,
						   index_t shape, index_t dims, index_t n);

	static auto get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n);

public:
	least_compute_thomas_solver_s_t(bool vectorized_x);

	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		if constexpr (aligned_x)
			return substrate_layouts::get_xyzs_aligned_layout<dims>(this->problem_, alignment_size_);
		else
			return substrate_layouts::get_xyzs_layout<dims>(this->problem_);
	}

	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;
};
