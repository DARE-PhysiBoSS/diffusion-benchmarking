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

This algorithm uses LAPACK pttrf+pttrs routines to solve the tridiagonal matrix system.
NOTE: Currently, only x dimension is implemented. The y and z dimensions require transposition of the matrix, because
LAPACK uses column-major order.
*/

template <typename real_t>
class lapack_thomas_solver : public locally_onedimensional_solver,
							 public base_solver<real_t, lapack_thomas_solver<real_t>>
{
	using index_t = std::int32_t;

	std::vector<std::unique_ptr<real_t[]>> ax_, bx_;
	std::vector<std::unique_ptr<real_t[]>> ay_, by_;
	std::vector<std::unique_ptr<real_t[]>> az_, bz_;

	std::size_t work_items_;

	void precompute_values(std::vector<std::unique_ptr<real_t[]>>& a, std::vector<std::unique_ptr<real_t[]>>& b,
						   index_t shape, index_t dims, index_t n);

	static void pttrf(const int* n, real_t* d, real_t* e, int* info);
	static void pttrs(const int* n, const int* nrhs, const real_t* d, const real_t* e, real_t* b, const int* ldb,
					  int* info);
	static void ptsv(const int* n, const int* nrhs, real_t* d, real_t* e, real_t* b, const int* ldb, int* info);

public:
	auto get_substrates_layout() const { return substrate_layouts::get_xyzs_layout<3>(this->problem_); }

	void initialize() override;

	void tune(const nlohmann::json& params) override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;
};
