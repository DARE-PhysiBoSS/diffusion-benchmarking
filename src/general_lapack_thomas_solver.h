#pragma once

#include "base_solver.h"
#include "substrate_layouts.h"
#include "tridiagonal_solver.h"

template <typename real_t>
class general_lapack_thomas_solver : public locally_onedimensional_solver,
									 public base_solver<real_t, general_lapack_thomas_solver<real_t>>
{
	using index_t = std::int32_t;

	std::vector<std::unique_ptr<real_t[]>> dlx_, dx_, dux_, du2x_;
	std::vector<std::unique_ptr<real_t[]>> dly_, dy_, duy_, du2y_;
	std::vector<std::unique_ptr<real_t[]>> dlz_, dz_, duz_, du2z_;

	std::vector<std::unique_ptr<int[]>> ipivx_, ipivy_, ipivz_;

	std::size_t work_items_;

	void precompute_values(std::vector<std::unique_ptr<real_t[]>>& dls, std::vector<std::unique_ptr<real_t[]>>& ds,
						   std::vector<std::unique_ptr<real_t[]>>& dus, std::vector<std::unique_ptr<real_t[]>>& du2s,
						   std::vector<std::unique_ptr<int[]>>& ipivs, index_t shape, index_t dims, index_t n);

	void gttrf(const int* n, real_t* dl, real_t* d, real_t* du, real_t* du2, int* ipiv, int* info);
	void gttrs(const char* trans, const int* n, const int* nrhs, const real_t* dl, const real_t* d, const real_t* du,
			   const real_t* du2, const int* ipiv, real_t* b, const int* ldb, int* info);

public:
	auto get_substrates_layout() const { return substrate_layouts::get_xyzs_layout<3>(this->problem_); }

	void initialize() override;

	void tune(const nlohmann::json& params) override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;
};
