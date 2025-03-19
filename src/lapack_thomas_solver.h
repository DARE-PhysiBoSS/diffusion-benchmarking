#pragma once

#include "base_solver.h"
#include "substrate_layouts.h"
#include "tridiagonal_solver.h"

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
