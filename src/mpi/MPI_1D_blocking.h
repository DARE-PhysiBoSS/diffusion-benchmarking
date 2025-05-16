#pragma once

#include <memory>
#include <mpi.h>
#include <vector>

#include "../tridiagonal_solver.h"
#include "../solver_utils.h"


template <typename real_t>
class MPI_1D_blocking : public virtual locally_onedimensional_solver
{
	using index_t = std::int32_t;

	index_t mpi_size, mpi_rank;
	index_t mpi_blocks;
	//MPI_Comm mpi_comm;
	MPI_Datatype mpi_type;

	index_t snd_data_size;
	index_t snd_data_size_last;
	index_t rcv_data_size;
	index_t rcv_data_size_last;

	index_t local_x_nodes;
	index_t local_y_nodes;
	index_t local_z_nodes;
	index_t init_z_node;  // Initial z node for each rank 

	problem_t<index_t, real_t> problem_;

	std::unique_ptr<real_t[]> substrates_;

	std::vector<std::vector<real_t>> bx_, cx_; // bx = denomx | cx = cx
	std::vector<std::vector<real_t>> by_, cy_;
	std::vector<std::vector<real_t>> bz_, cz_;
    std::vector<real_t> constant1;
    index_t thomas_i_jump;
    index_t thomas_j_jump;
    index_t thomas_k_jump;
	//SIMD variables
	index_t vl_; //vector length elements in a vector register (SIMD)
	index_t gvec_size; //help to iterate over substrates and vector length
	
	std::vector<std::vector<real_t>> vby_, vcy_;
	std::vector<std::vector<real_t>> vbz_, vcz_;
    std::vector<real_t> vconstant1;

	std::size_t work_items_;

	void precompute_values();
	void precompute_values_vec(std::int32_t vl);

	static auto get_substrates_layout(const problem_t<index_t, real_t>& problem, const index_t x_nodes, const index_t y_nodes, const index_t z_nodes);

public:
	void prepare(const max_problem_t& problem) override;

	void initialize() override; //done

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override; //done

	void save(std::ostream& out) const override; //done

	double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override; //

	void tune(const nlohmann::json& params) override;


	virtual ~MPI_1D_blocking() override {  }
};
