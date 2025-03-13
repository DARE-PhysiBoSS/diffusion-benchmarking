#pragma once

#include <memory>

#include "tridiagonal_solver.h"
#include "solver_utils.h"


template <typename real_t>
class simd : public locally_onedimensional_solver
{
	using index_t = std::int32_t;

	problem_t<index_t, real_t> problem_;

	std::unique_ptr<real_t[]> substrates_;

	std::unique_ptr<real_t[]> bx_, cx_; // bx = denomx | cx = cx
	std::unique_ptr<real_t[]> by_, cy_;
	std::unique_ptr<real_t[]> bz_, cz_;
    std::unique_ptr<real_t[]> constant1;
    index_t thomas_i_jump;
    index_t thomas_j_jump;
    index_t thomas_k_jump;
	//SIMD variables
	index_t vl_; //vector length elements in a vector register (SIMD)
	index_t gvec_size; //help to iterate over substrates and vector length
	
	std::unique_ptr<real_t[]> vby_, vcy_;
	std::unique_ptr<real_t[]> vbz_, vcz_;
    std::unique_ptr<real_t[]> vconstant1;

	std::size_t work_items_;

	void precompute_values();
	void precompute_values_vec(std::int32_t vl);

	template <std::size_t dims>
	static auto get_substrates_layout(const problem_t<index_t, real_t>& problem);

public:
	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override; //done

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override; //done

	void save(const std::string& file) const override;

	double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override;
};
