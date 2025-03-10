#pragma once

#include <memory>

#include "tridiagonal_solver.h"

template <typename real_t>
class lapack_thomas_solver : public tridiagonal_solver
{
	using index_t = std::int32_t;

	problem_t<index_t, real_t> problem_;

	std::unique_ptr<real_t[]> substrates_;

	std::vector<std::unique_ptr<real_t[]>> ax_, bx_;
	std::vector<std::unique_ptr<real_t[]>> ay_, by_;
	std::vector<std::unique_ptr<real_t[]>> az_, bz_;

	std::size_t work_items_;

	static auto get_substrates_layout(const problem_t<index_t, real_t>& problem);

	void precompute_values(std::vector<std::unique_ptr<real_t[]>>& a, std::vector<std::unique_ptr<real_t[]>>& b,
						   index_t shape, index_t dims, index_t n);

	static void pttrf(const int* n, real_t* d, real_t* e, int* info);
	static void pttrs(const int* n, const int* nrhs, const real_t* d, const real_t* e, real_t* b, const int* ldb,
					  int* info);
	static void ptsv(const int* n, const int* nrhs, real_t* d, real_t* e, real_t* b, const int* ldb, int* info);

public:
	void prepare(const max_problem_t& problem) override;

	void initialize() override;

	void tune(const nlohmann::json& params) override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;

	void save(const std::string& file) const override;

	double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override;
};
