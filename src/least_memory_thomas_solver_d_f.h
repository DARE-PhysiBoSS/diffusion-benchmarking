#pragma once

#include "base_solver.h"
#include "blocked_thomas_solver.h"
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
- Precomputed a_i, b_1, b_i'
- Minimized memory accesses by computing e_i on the fly
- Aligned memory for x dimension (tunable by 'alignment_size')
- Better temporal locality of memory accesses - x dimension is divided into smaller tiles (tunable by 'x_tile_size') and
y/z dimensions are solved alongside tiled x dimension
- X dimension is vectorized manually - squares of x*yz plane are loaded into vector registers, then the data is
transposed so the vectorization can be utilized
*/

template <typename real_t, bool aligned_x>
class least_memory_thomas_solver_d_f : public locally_onedimensional_solver,
									   public base_solver<real_t, least_memory_thomas_solver_d_f<real_t, aligned_x>>

{
	using index_t = std::int32_t;

	real_t *ax_, *b1x_, *cx_;
	real_t *ay_, *b1y_, *cy_;
	real_t *az_, *b1z_, *cz_;

	std::unique_ptr<aligned_atomic<long>[]> countersy_, countersz_;
	index_t countersy_count_, countersz_count_;

	real_t *a_scratchy_, *c_scratchy_;
	real_t *a_scratchz_, *c_scratchz_;

	real_t* dim_scratch_;

	bool use_alt_blocked_;
	std::size_t alignment_size_;
	index_t substrate_step_;

	std::array<index_t, 3> cores_division_;

	std::array<index_t, 3> group_blocks_;
	std::vector<index_t> group_block_lengthsy_;
	std::vector<index_t> group_block_lengthsz_;

	std::vector<index_t> group_block_offsetsy_;
	std::vector<index_t> group_block_offsetsz_;

	index_t substrate_groups_;

	auto get_diagonal_layout(const problem_t<index_t, real_t>& problem_, index_t n);

	void precompute_values(real_t*& a, real_t*& b1, real_t*& a_data, real_t*& c_data, index_t shape, index_t dims,
						   index_t n, index_t counters_count, std::unique_ptr<aligned_atomic<long>[]>& counters,
						   index_t group_size, index_t& block_size, std::vector<index_t>& group_block_lengths,
						   std::vector<index_t>& group_block_offsets);

	void precompute_values(real_t*& a, real_t*& b1, real_t*& b, index_t shape, index_t dims, index_t n);

public:
	least_memory_thomas_solver_d_f(bool use_alt_blocked);

	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		if constexpr (aligned_x)
			return substrate_layouts::get_xyzs_aligned_layout<dims>(this->problem_, alignment_size_);
		else
			return substrate_layouts::get_xyzs_layout<dims>(this->problem_);
	}

	auto get_scratch_layout(const index_t n, const index_t groups) const
	{
		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 'l', 's'>(n, groups, this->problem_.substrates_count);
	}

	auto get_dim_scratch_layout() const
	{
		index_t max_n = std::max({ this->problem_.nx, this->problem_.ny, this->problem_.nz });

		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 't'>(max_n, get_max_threads());
	}

	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;

	void solve_blocked_2d();
	void solve_blocked_3d();

	~least_memory_thomas_solver_d_f();
};
