#pragma once

#include <memory>

#include "diffusion_solver.h"

template <typename real_t>
class full_lapack_solver : public diffusion_solver
{
	using index_t = std::int32_t;

	problem_t<index_t, real_t> problem_;

	std::unique_ptr<real_t[]> substrates_;

	std::vector<std::unique_ptr<real_t[]>> ab_;

	std::size_t work_items_;

	static auto get_substrates_layout(const problem_t<index_t, real_t>& problem);

	void precompute_values();

	static void pbtrf(const char* uplo, const int* n, const int* kd, real_t* ab, const int* ldab, int* info);
	static void pbtrs(const char* uplo, const int* n, const int* kd, const int* nrhs, const real_t* ab, const int* ldab,
					  real_t* b, const int* ldb, int* info);

public:
	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve() override;

	void save(const std::string& file) const override;

	double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override;
};
