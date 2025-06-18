#include "least_memory_thomas_solver_d_f.h"

#include <atomic>
#include <cstddef>
#include <iostream>

#include "noarr/structures/extra/funcs.hpp"
#include "omp_helper.h"
#include "vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::precompute_values(
	real_t*& a, real_t*& b1, real_t*& a_data, real_t*& c_data, index_t shape, index_t dims, index_t n,
	index_t counters_count, std::unique_ptr<aligned_atomic<long>[]>& counters, index_t group_size, index_t& block_size,
	std::vector<index_t>& group_block_lengths, std::vector<index_t>& group_block_offsets)
{
	// allocate memory for a and b1
	a = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));
	b1 = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));

	// compute a
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		a[s] = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b1
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		b1[s] = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
				+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	counters = std::make_unique<aligned_atomic<long>[]>(counters_count);

	block_size = n / group_size;

	group_block_lengths.clear();
	group_block_offsets.clear();

	for (index_t i = 0; i < group_size; i++)
	{
		if (i < n % group_size)
			group_block_lengths.push_back(block_size + 1);
		else
			group_block_lengths.push_back(block_size);
	}

	group_block_offsets.resize(group_size);

	for (index_t i = 0; i < group_size; i++)
	{
		if (i == 0)
			group_block_offsets[i] = 0;
		else
			group_block_offsets[i] = group_block_offsets[i - 1] + group_block_lengths[i - 1];
	}

	auto scratch_layout = get_scratch_layout(n, counters_count);

	if (aligned_x)
	{
		a_data = (real_t*)std::aligned_alloc(alignment_size_, (scratch_layout | noarr::get_size()));
		c_data = (real_t*)std::aligned_alloc(alignment_size_, (scratch_layout | noarr::get_size()));
	}
	else
	{
		a_data = (real_t*)std::malloc((scratch_layout | noarr::get_size()));
		c_data = (real_t*)std::malloc((scratch_layout | noarr::get_size()));
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, real_t*& c,
																		  index_t shape, index_t dims, index_t n)
{
	auto layout = get_diagonal_layout(this->problem_, n);

	if (aligned_x)
	{
		c = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
	}
	else
	{
		c = (real_t*)std::malloc((layout | noarr::get_size()));
	}

	a = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));
	b1 = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));

	auto c_diag = noarr::make_bag(layout, c);

	// compute a
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		a[s] = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b1
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		b1[s] = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
				+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute c_i'
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
	{
		c_diag.template at<'i', 's'>(0, s) = a[s] / (b1[s] + a[s]);

		for (index_t i = 1; i < n - 1; i++)
		{
			const real_t r = 1 / (b1[s] - a[s] * c_diag.template at<'i', 's'>(i - 1, s));
			c_diag.template at<'i', 's'>(i, s) = a[s] * r;
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::prepare(const max_problem_t& problem)
{
	this->problem_ = problems::cast<std::int32_t, real_t>(problem);

	auto substrates_layout = get_substrates_layout<3>();

	if (aligned_x)
		this->substrates_ = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
	else
		this->substrates_ = (real_t*)std::malloc((substrates_layout | noarr::get_size()));

	// Initialize substrates
	solver_utils::initialize_substrate(substrates_layout, this->substrates_, this->problem_);
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	substrate_step_ =
		params.contains("substrate_step") ? (index_t)params["substrate_step"] : this->problem_.substrates_count;

	cores_division_ = params.contains("cores_division") ? (std::array<index_t, 3>)params["cores_division"]
														: std::array<index_t, 3> { 1, 2, 2 };

	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
		alignment_size_ = std::max(alignment_size_, vector_length);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims == 2)
		cores_division_[2] = 1;

	cores_division_[0] = 1;

	const index_t substrate_group_size = cores_division_[0] * cores_division_[1] * cores_division_[2];

	substrate_groups_ = (get_max_threads() + substrate_group_size - 1) / substrate_group_size;

	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, cx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
	{
		if (cores_division_[1] == 1)
		{
			countersy_count_ = 0;
			group_block_offsetsy_ = { 0 };
			group_block_lengthsy_ = { this->problem_.ny };
			precompute_values(ay_, b1y_, cy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
		}
		else
		{
			countersy_count_ = cores_division_[0] * cores_division_[2] * substrate_groups_;

			precompute_values(ay_, b1y_, a_scratchy_, c_scratchy_, this->problem_.dy, this->problem_.dims,
							  this->problem_.ny, countersy_count_, countersy_, cores_division_[1], group_blocks_[1],
							  group_block_lengthsy_, group_block_offsetsy_);
		}
	}
	if (this->problem_.dims >= 3)
	{
		if (cores_division_[2] == 1)
		{
			countersz_count_ = 0;
			group_block_offsetsz_ = { 0 };
			group_block_lengthsz_ = { this->problem_.nz };

			precompute_values(az_, b1z_, cz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
		}
		else
		{
			countersz_count_ = cores_division_[0] * cores_division_[1] * substrate_groups_;

			precompute_values(az_, b1z_, a_scratchz_, c_scratchz_, this->problem_.dz, this->problem_.dims,
							  this->problem_.nz, countersz_count_, countersz_, cores_division_[2], group_blocks_[2],
							  group_block_lengthsz_, group_block_offsetsz_);
		}
	}

	auto dim_scratch_layout = get_dim_scratch_layout();
	if (aligned_x)
	{
		dim_scratch_ = (real_t*)std::aligned_alloc(alignment_size_, (dim_scratch_layout | noarr::get_size()));
	}
	else
	{
		dim_scratch_ = (real_t*)std::malloc((dim_scratch_layout | noarr::get_size()));
	}
}

template <typename real_t, bool aligned_x>
auto least_memory_thomas_solver_d_f<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
																			index_t n)
{
	if constexpr (aligned_x)
	{
		std::size_t size = n * sizeof(real_t);
		std::size_t size_padded = (size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		size_padded /= sizeof(real_t);

		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(size_padded, problem.substrates_count)
			   ^ noarr::slice<'i'>(n);
	}
	else
	{
		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(n, problem.substrates_count);
	}
}

template <typename real_t>
struct inside_data
{
	real_t& c_tmp;
};

template <typename index_t, typename real_t, typename density_bag_t>
static real_t z_forward_inside_y(const density_bag_t d, const index_t s, const index_t z, const index_t y,
								 const index_t x, const real_t a_s, const real_t b1_s, real_t c_tmp, real_t data)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	real_t r;

	if (z == 0)
	{
		r = 1 / (b1_s + a_s);

		data *= r;

		d.template at<'s', 'z', 'x', 'y'>(s, z, x, y) = data;
	}
	else
	{
		const real_t b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);
		r = 1 / (b_tmp - a_s * c_tmp);

		data = r * (data - a_s * d.template at<'s', 'z', 'x', 'y'>(s, z - 1, x, y));
		d.template at<'s', 'z', 'x', 'y'>(s, z, x, y) = data;
	}

	return r;
}

template <typename index_t, typename density_bag_t, typename diag_bag_t>
constexpr static void z_backward(const density_bag_t d, const diag_bag_t c, const index_t s)

{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();
	const index_t z_len = d | noarr::get_length<'z'>();

	for (index_t i = z_len - 2; i >= 0; i--)
	{
		const auto back_c = c.template at<'s', 'i'>(s, i);

		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
				d.template at<'s', 'z', 'y', 'x'>(s, i, y, x) -=
					back_c * d.template at<'s', 'z', 'y', 'x'>(s, i + 1, y, x);
		}
	}
}

template <typename index_t, typename scratch_bag_t>
struct inside_data_blocked
{
	const index_t begin;
	scratch_bag_t c;
};

template <typename index_t, typename scratch_bag_t>
struct inside_data_blocked_alt
{
	const index_t begin;
	const index_t end;
	scratch_bag_t a;
	scratch_bag_t c;
};

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked(const density_bag_t d, const index_t s, const index_t z,
												 const index_t y, const index_t x, real_t data, const real_t a_s,
												 const real_t b1_s, const index_t z_begin, scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	if (z < z_begin + 2)
	{
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data /= b_tmp;

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = r * (data - a_tmp * d.template at<'s', 'x', 'y', 'z'>(s, x, y, z - 1));

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t,
		  typename simd_tag>
constexpr static void z_forward_inside_y_blocked_vectorized(const density_bag_t d, simd_tag t, const index_t s,
															const index_t z, const index_t y, const index_t X,
															vec_t data, const real_t a_s, const real_t b1_s,
															const index_t z_begin, scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	if (z < z_begin + 2)
	{
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data = hn::Mul(data, hn::Set(t, 1 / b_tmp));

		hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y));
	}
	else
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		vec_t prev = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z - 1, y));

		data = hn::MulAdd(hn::Set(t, -a_tmp), prev, data);
		data = hn::Mul(data, hn::Set(t, r));

		hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y));
	}
}

template <typename vec_t, typename tag, typename index_t, typename real_t, typename density_bag_t,
		  typename scratch_bag_t>
constexpr static void z_forward_inside_y_vectorized_blocked(const density_bag_t d, tag t, const index_t s,
															const index_t z, const index_t y, vec_t data,
															const real_t a_s, const real_t b1_s, const index_t z_begin,
															scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	if (z < z_begin + 2)
	{
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data = hn::Mul(data, hn::Set(t, 1 / b_tmp));

		hn::Store(data, t, &d.template at<'y', 'z'>(y, z));

		// #pragma omp critical
		// 		for (std::size_t v = 0; v < hn::Lanes(t); v++)
		// 		{
		// 			std::cout << "f0: " << z << " " << y << " " << x + v << " " << hn::ExtractLane(data, v) << " " <<
		// b_tmp
		// 					  << std::endl;
		// 		}
	}
	else
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = hn::MulAdd(hn::Set(t, -a_tmp), hn::Load(t, &d.template at<'y', 'z'>(y, z - 1)), data);
		data = hn::Mul(data, hn::Set(t, r));

		hn::Store(data, t, &d.template at<'y', 'z'>(y, z));

		// #pragma omp critical
		// 		for (std::size_t v = 0; v < hn::Lanes(t); v++)
		// 		{
		// 			std::cout << "f1: " << z << " " << y << " " << x + v << " " << hn::ExtractLane(data, v) << " " <<
		// a_tmp << " "
		// 					  << b_tmp << " " << c[prev_state] << std::endl;
		// 		}
	}
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked_next(const index_t s, const index_t z, const real_t a_s,
													  const real_t b1_s, const index_t z_len, const index_t z_begin,
													  scratch_bag_t a, scratch_bag_t c)

{
	if (z < z_begin + 2)
	{
		const auto state = noarr::idx<'s', 'i'>(s, z);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);
		const auto c_tmp = a_s * (z == z_len - 1 ? 0 : 1);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'s', 'i'>(s, z);
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);
		const auto c_tmp = a_s * (z == z_len - 1 ? 0 : 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_backward_blocked(const density_bag_t d, const real_t y_begin, const real_t y_end,
										 const real_t z_begin, const real_t z_end, const scratch_bag_t a,
										 const scratch_bag_t c, const index_t s)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	// Process the upper diagonal (backward)
	index_t i;
	for (i = z_end - 3; i >= z_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		for (index_t y = y_begin; y < y_end; y++)
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', 'z'>(s, x, y, i) -=
					c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y, i + 1);


				// #pragma omp critical
				// 				std::cout << "b0: " << i << " " << y << " " << x << " " << d.template at<'s', 'x', 'y',
				// 'z'>(s, x, y, i)
				// 						  << std::endl;
			}

		a[state] = a[state] - c[state] * a[next_state];
		c[state] = 0 - c[state] * c[next_state];
	}

	// Process the first row (backward)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		const auto r = 1 / (1 - c[state] * a[next_state]);

		for (index_t y = y_begin; y < y_end; y++)
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', 'z'>(s, x, y, i) =
					r
					* (d.template at<'s', 'x', 'y', 'z'>(s, x, y, i)
					   - c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y, i + 1));

				// #pragma omp critical
				// 				std::cout << "b1: " << i << " " << y << " " << x << " " << d.template at<'s', 'x', 'y',
				// 'z'>(s, x, y, i)
				// 						  << " " << a[next_state] << " " << c[state] << std::endl;
			}

		a[state] = r * a[state];
		c[state] = r * (0 - c[state] * c[next_state]);
	}
}


template <typename index_t, typename density_bag_t, typename scratch_bag_t, typename dim_scratch_bag_t>
constexpr static void z_blocked_middle(density_bag_t d, scratch_bag_t a, scratch_bag_t c, dim_scratch_bag_t c_scratch,
									   const index_t tid, const index_t coop_size, index_t y_begin, index_t y_end,
									   const index_t s)
{
	constexpr char dim = 'z';
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t n = d | noarr::get_length<dim>();

	const index_t block_size = n / coop_size;

	const index_t y_len = y_end - y_begin;
	const index_t block_size_y = y_len / coop_size;
	y_begin = y_begin + tid * block_size_y + std::min(tid, y_len % coop_size);
	y_end = y_begin + block_size_y + ((tid < y_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " s_begin: " << s << " block_y_begin: " << y_begin << " block_y_end: " << y_end
	// 			  << " block_size: " << block_size_y << std::endl;

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
		return i;
	};

	auto c_tmp = c.template at<'s', 'i'>(s, get_i(0));
	c_scratch.template at<'i'>(0) = c_tmp;

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'s', 'i'>(s, i);

		const auto r = 1 / (1 - a[state] * c_tmp);

		c_tmp = c[state] * r;
		c_scratch.template at<'i'>(equation_idx) = c_tmp;

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
					r
					* (d.template at<'s', 'x', 'y', dim>(s, x, y, i)
					   - a[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, prev_i));
			}
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(equation_idx);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
					d.template at<'s', 'x', 'y', dim>(s, x, y, i)
					- c_scratch[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, next_i);
			}
		}
	}
}

template <typename index_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_blocked_end(density_bag_t d, scratch_bag_t a, scratch_bag_t c, const index_t y_begin,
									const index_t y_end, const index_t z_begin, const index_t z_end, const index_t s)
{
	constexpr char dim = 'z';
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = z_begin + 1; i < z_end - 1; i++)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
					d.template at<'s', 'x', 'y', dim>(s, x, y, i)
					- a[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, z_begin)
					- c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, z_end - 1);
			}
		}
	}
}

template <typename index_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_blocked_end_alt(density_bag_t d, scratch_bag_t a, scratch_bag_t c, const index_t y_begin,
										const index_t y_end, const index_t z_begin, const index_t z_end,
										const index_t s)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = z_end - 2; i >= z_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', 'z'>(s, x, y, i) =
					d.template at<'s', 'x', 'y', 'z'>(s, x, y, i)
					- a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y, z_begin)
					- c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y, i + 1);


				// #pragma omp critical
				// 			std::cout << "0b z: " << z << " y: " << i << " x: " << x << " a: " << a[state] << " c: " <<
				// c[state]
				// 					  << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				// 					  << " d0: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) << std::endl;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_backward_blocked(const density_bag_t d, const index_t z, const real_t y_begin,
										 const real_t y_end, const scratch_bag_t a, const scratch_bag_t c,
										 const index_t s)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	// Process the upper diagonal (backward)
	index_t i;
	for (i = y_end - 3; i >= y_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) -=
				c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z);


			// #pragma omp critical
			// 			std::cout << "b0: " << i << " " << i << " " << x << " " << d.template at<'s', 'x', 'y', 'z'>(s,
			// x, i, z)
			// 					  << std::endl;
		}

		a[state] = a[state] - c[state] * a[next_state];
		c[state] = 0 - c[state] * c[next_state];
	}

	// Process the first row (backward)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		const auto r = 1 / (1 - c[state] * a[next_state]);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				r
				* (d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				   - c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z));

			// std::cout << "b1: " << i << " " << y << " " << x << " " << d.template at<'s', 'x', 'y', 'z'>(s, x, y,
			// i)
			// 		  << std::endl;
		}

		a[state] = r * a[state];
		c[state] = r * (0 - c[state] * c[next_state]);
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_layout_t, typename scratch_bag_t,
		  typename simd_tag>
constexpr static void y_backward_blocked_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
													simd_tag t, const index_t simd_length, const index_t z,
													const real_t y_begin, const real_t y_end, const scratch_bag_t a,
													const scratch_bag_t c, const index_t s)

{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'x'>(noarr::lit<0>, noarr::lit<0>), densities);

	for (index_t X = 0; X < X_len; X++)
	{
		vec_t prev = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_end - 2));

		// Process the upper diagonal (backward)
		index_t i;
		for (i = y_end - 3; i >= y_begin + 1; i--)
		{
			const auto state = noarr::idx<'s', 'i'>(s, i);
			const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

			vec_t data = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
			data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);
			prev = data;

			hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));

			a[state] = a[state] - c[state] * a[next_state];
			c[state] = 0 - c[state] * c[next_state];
		}

		// Process the first row (backward)
		{
			const auto state = noarr::idx<'s', 'i'>(s, i);
			const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

			const auto r = 1 / (1 - c[state] * a[next_state]);

			vec_t data = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
			data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);
			data = hn::Mul(data, hn::Set(t, r));

			a[state] = r * a[state];
			c[state] = r * (0 - c[state] * c[next_state]);
		}
	}
}


template <typename index_t, typename density_bag_t, typename scratch_bag_t, typename dim_scratch_bag_t>
constexpr static void y_blocked_middle(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
									   dim_scratch_bag_t c_scratch, const index_t tid, const index_t coop_size,
									   const index_t s)
{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t n = d | noarr::get_length<'y'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_x = x_len / coop_size;
	const auto x_begin = tid * block_size_x + std::min(tid, x_len % coop_size);
	const auto x_end = x_begin + block_size_x + ((tid < x_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " s_begin: " << s << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
		return i;
	};

	auto c_tmp = c.template at<'s', 'i'>(s, get_i(0));
	c_scratch.template at<'i'>(0) = c_tmp;

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'s', 'i'>(s, i);

		const auto r = 1 / (1 - a[state] * c_tmp);

		c_tmp = c[state] * r;
		c_scratch.template at<'i'>(equation_idx) = c_tmp;

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				r
				* (d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				   - a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, prev_i, z));

			// #pragma omp critical
			// 			std::cout << "mf z: " << z << " y: " << i << " x: " << x << " a: " << a[state] << " cp: " <<
			// c[prev_state]
			// 					  << " d: " << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << std::endl;
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(equation_idx);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				- c_scratch[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, next_i, z);

			// #pragma omp critical
			// 			std::cout << "mb z: " << z << " y: " << i << " x: " << x << " c: " << c[state]
			// 					  << " d: " << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << std::endl;
		}
	}
}

template <typename index_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_blocked_end(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
									const index_t y_begin, const index_t y_end, const index_t s)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = y_begin + 1; i < y_end - 1; i++)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				- a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z)
				- c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_end - 1, z);
		}
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_layout_t, typename scratch_bag_t,
		  typename simd_tag>
constexpr static void y_blocked_end_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
											   simd_tag t, const index_t simd_length, const index_t z, scratch_bag_t a,
											   scratch_bag_t c, const index_t y_begin, const index_t y_end,
											   const index_t s)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'x'>(noarr::lit<0>, noarr::lit<0>), densities);

	for (index_t X = 0; X < X_len; X++)
	{
		vec_t begins = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_begin));
		vec_t ends = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_end - 1));

		for (index_t i = y_begin + 1; i < y_end - 1; i++)
		{
			const auto state = noarr::idx<'s', 'i'>(s, i);

			vec_t data = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(ends, hn::Set(t, -c[state]), data);

			hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
		}
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked_alt(const density_bag_t d, const index_t s, const index_t y,
													 const index_t z, const index_t x, real_t data, const real_t a_s,
													 const real_t b1_s, const index_t z_begin, const index_t z_end,
													 scratch_bag_t a, scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	real_t a_state;

	if (z < z_begin + 2)
	{
		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data /= b_tmp;

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		a_state = a_tmp / b_tmp;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = r * (data - a_tmp * d.template at<'s', 'x', 'y', 'z'>(s, x, y, z - 1));

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		a_state = r * (0 - a_tmp * a[prev_state]);

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}

	if (z != z_begin && z != z_end - 1)
	{
		const auto state0 = noarr::idx<'s', 'i'>(s, z_begin);

		const auto r0 = 1 / (1 - a_state * c[state0]);

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z_begin) =
			r0 * (d.template at<'s', 'x', 'y', 'z'>(s, x, y, z_begin) - c[state0] * data);


		// #pragma omp critical
		// 		std::cout << "0 z: " << z << " y: " << y << " x: " << x << " a0: " << a[state0] << " c0: " << c[state0]
		// 				  << " r0: " << r0 << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) <<
		// std::endl;
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t,
		  typename simd_tag>
constexpr static void z_forward_inside_y_blocked_alt_vectorized(const density_bag_t d, simd_tag t, const index_t s,
																const index_t y, const index_t z, const index_t X,
																vec_t data, const real_t a_s, const real_t b1_s,
																const index_t z_begin, const index_t z_end,
																scratch_bag_t a, scratch_bag_t c)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	real_t a_state;

	if (z < z_begin + 2) [[unlikely]]
	{
		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);

		data = hn::Mul(data, hn::Set(t, 1 / b_tmp));

		hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y));

		a_state = a_tmp / b_tmp;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else [[likely]]
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		vec_t prev = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z - 1, y));

		data = hn::MulAdd(hn::Set(t, -a_tmp), prev, data);
		data = hn::Mul(data, hn::Set(t, r));

		hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y));

		a_state = r * (0 - a_tmp * a[prev_state]);

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}

	if (z != z_begin && z != z_end - 1) [[likely]]
	{
		const auto state0 = noarr::idx<'s', 'i'>(s, z_begin);

		const auto r0 = 1 / (1 - a_state * c[state0]);

		vec_t z0 = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z_begin, y));

		z0 = hn::MulAdd(hn::Set(t, -c[state0]), data, z0);
		z0 = hn::Mul(z0, hn::Set(t, r0));

		hn::Store(z0, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z_begin, y));


		// #pragma omp critical
		// 		std::cout << "0 z: " << z << " y: " << y << " x: " << x << " a0: " << a[state0] << " c0: " << c[state0]
		// 				  << " r0: " << r0 << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) <<
		// std::endl;
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t, typename z_data_t>
constexpr static void y_blocked_end(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
									const index_t y_begin, const index_t y_end, const index_t s, const real_t az_s,
									const real_t b1z_s, z_data_t z_data)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = y_begin + 1; i < y_end - 1; i++)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			real_t data = d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
						  - a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z)
						  - c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_end - 1, z);

			// #pragma omp critical
			// 			std::cout << "l: " << z << " " << i << " " << x << " " << data << " " << a[state] << " " <<
			// c[state]
			// 					  << std::endl;

			z_forward_inside_y_blocked(d, s, z, i, x, data, az_s, b1z_s, z_data.begin, z_data.c);
		}
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_layout_t, typename scratch_bag_t,
		  typename z_data_t, typename simd_tag>
constexpr static void y_blocked_end_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
											   simd_tag t, const index_t simd_length, const index_t z, scratch_bag_t a,
											   scratch_bag_t c, const index_t y_begin, const index_t y_end,
											   const index_t s, const real_t az_s, const real_t b1z_s, z_data_t z_data)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'x'>(noarr::lit<0>, noarr::lit<0>), densities);

	for (index_t X = 0; X < X_len; X++)
	{
		vec_t begins = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_begin));
		vec_t ends = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_end - 1));

		for (index_t i = y_begin + 1; i < y_end - 1; i++)
		{
			const auto state = noarr::idx<'s', 'i'>(s, i);

			vec_t data = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(ends, hn::Set(t, -c[state]), data);

			// hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));

			z_forward_inside_y_blocked_vectorized(d, t, s, z, i, X, data, az_s, b1z_s, z_data.begin, z_data.c);
		}
	}
}

template <typename index_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_blocked_end_alt(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
										const index_t y_begin, const index_t y_end, const index_t s)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = y_end - 2; i >= y_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				- a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z)
				- c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z);


			// #pragma omp critical
			// 			std::cout << "0b z: " << z << " y: " << i << " x: " << x << " a: " << a[state] << " c: " <<
			// c[state]
			// 					  << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
			// 					  << " d0: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) << std::endl;
		}
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_layout_t, typename scratch_bag_t,
		  typename simd_tag>
constexpr static void y_blocked_end_alt_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
												   simd_tag t, const index_t simd_length, const index_t z,
												   scratch_bag_t a, scratch_bag_t c, const index_t y_begin,
												   const index_t y_end, const index_t s)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'x'>(noarr::lit<0>, noarr::lit<0>), densities);

	for (index_t X = 0; X < X_len; X++)
	{
		vec_t begins = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_begin));
		vec_t prev = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_end - 1));

		for (index_t i = y_end - 2; i >= y_begin + 1; i--)
		{
			const auto state = noarr::idx<'s', 'i'>(s, i);

			vec_t data = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);
			prev = data;

			hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
		}
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t, typename z_data_t>
constexpr static void y_blocked_end_alt(density_bag_t d, const index_t z, scratch_bag_t a, scratch_bag_t c,
										const index_t y_begin, const index_t y_end, const index_t s, const real_t az_s,
										const real_t b1z_s, z_data_t z_data)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = y_end - 2; i >= y_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'x', 'y', 'z'>(s, x, i, z) =
				d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
				- a[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z)
				- c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z);


			// #pragma omp critical
			// 			std::cout << "0b z: " << z << " y: " << i << " x: " << x << " a: " << a[state] << " c: " <<
			// c[state]
			// 					  << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, i, z)
			// 					  << " d0: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) << std::endl;

			z_forward_inside_y_blocked_alt(d, s, i + 1, z, x, d.template at<'s', 'x', 'y', 'z'>(s, x, i + 1, z), az_s,
										   b1z_s, z_data.begin, z_data.end, z_data.a, z_data.c);
		}
	}
}

template <typename vec_t, typename index_t, typename real_t, typename density_layout_t, typename scratch_bag_t,
		  typename z_data_t, typename simd_tag>
constexpr static void y_blocked_end_alt_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
												   simd_tag t, const index_t simd_length, const index_t z,
												   scratch_bag_t a, scratch_bag_t c, const index_t y_begin,
												   const index_t y_end, const index_t s, const real_t az_s,
												   const real_t b1z_s, z_data_t z_data)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'x'>(noarr::lit<0>, noarr::lit<0>), densities);

	for (index_t X = 0; X < X_len; X++)
	{
		vec_t begins = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_begin));
		vec_t prev = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, y_end - 1));

		for (index_t i = y_end - 2; i >= y_begin + 1; i--)
		{
			const auto state = noarr::idx<'s', 'i'>(s, i);

			vec_t data = hn::Load(t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);

			// hn::Store(data, t, &d.template at<'s', 'X', 'z', 'y'>(s, X, z, i));

			z_forward_inside_y_blocked_alt_vectorized(d, t, s, i + 1, z, X, prev, az_s, b1z_s, z_data.begin, z_data.end,
													  z_data.a, z_data.c);
			prev = data;
		}

		z_forward_inside_y_blocked_alt_vectorized(d, t, s, y_begin + 1, z, X, prev, az_s, b1z_s, z_data.begin,
												  z_data.end, z_data.a, z_data.c);

		z_forward_inside_y_blocked_alt_vectorized(d, t, s, y_begin, z, X, begins, az_s, b1z_s, z_data.begin, z_data.end,
												  z_data.a, z_data.c);
	}
}

template <bool update_c, typename index_t, typename real_t, typename density_bag_t>
constexpr static void y_forward_inside_x(const density_bag_t d, const index_t s, const index_t z, const index_t y,
										 const index_t x, real_t data, const real_t a_s, const real_t b1_s,
										 real_t& c_tmp)

{
	const index_t y_len = d | noarr::get_length<'y'>();

	real_t r;

	if (y == 0)
	{
		r = 1 / (b1_s + a_s);

		data *= r;

		d.template at<'s', 'z', 'x', 'y'>(s, z, x, 0) = data;
	}
	else
	{
		const real_t b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);
		r = 1 / (b_tmp - a_s * c_tmp);

		data = r * (data - a_s * d.template at<'s', 'z', 'x', 'y'>(s, z, x, y - 1));
		d.template at<'s', 'z', 'x', 'y'>(s, z, x, y) = data;
	}

	if constexpr (update_c)
	{
		c_tmp = a_s * r;
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked(const density_bag_t d, const index_t s, const index_t z,
												 const index_t y, const index_t x, real_t data, const real_t a_s,
												 const real_t b1_s, const index_t y_begin, scratch_bag_t c)

{
	const index_t y_len = d | noarr::get_length<'y'>();

	if (y < y_begin + 2)
	{
		const auto b_tmp = b1_s + ((y == 0) || (y == y_len - 1) ? a_s : 0);

		data /= b_tmp;

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, y - 1);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = r * (data - a_tmp * d.template at<'s', 'x', 'y', 'z'>(s, x, y - 1, z));

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked_alt(const density_bag_t d, const index_t s, const index_t z,
													 const index_t y, const index_t x, real_t data, const real_t a_s,
													 const real_t b1_s, const index_t y_begin, const index_t y_end,
													 scratch_bag_t a, scratch_bag_t c)

{
	const index_t y_len = d | noarr::get_length<'y'>();

	real_t a_state;

	if (y < y_begin + 2)
	{
		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((y == 0) || (y == y_len - 1) ? a_s : 0);

		data /= b_tmp;

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		a_state = a_tmp / b_tmp;

		// #pragma omp critical
		// 		std::cout << "f0: " << z << " " << y << " " << x << " " << data << " " << b_tmp << std::endl;
	}
	else
	{
		const auto prev_state = noarr::idx<'s', 'i'>(s, y - 1);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		data = r * (data - a_tmp * d.template at<'s', 'x', 'y', 'z'>(s, x, y - 1, z));

		d.template at<'s', 'x', 'y', 'z'>(s, x, y, z) = data;

		a_state = r * (0 - a_tmp * a[prev_state]);

		// #pragma omp critical
		// 		std::cout << "f1: " << z << " " << y << " " << x << " " << data << " " << a_tmp << " " << b_tmp << " "
		// 				  << c[prev_state] << std::endl;
	}

	if (y != y_begin && y != y_end - 1)
	{
		const auto state0 = noarr::idx<'s', 'i'>(s, y_begin);

		const auto r0 = 1 / (1 - a_state * c[state0]);

		d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) =
			r0 * (d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) - c[state0] * data);


		// #pragma omp critical
		// 		std::cout << "0 z: " << z << " y: " << y << " x: " << x << " a0: " << a[state0] << " c0: " << c[state0]
		// 				  << " r0: " << r0 << " d: " << d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z) <<
		// std::endl;
	}
}

template <bool z_blocked, typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t,
		  typename z_data_t>
constexpr static void y_backward(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z,
								 const real_t az_s, const real_t b1z_s, z_data_t z_data)

{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();

	for (index_t i = y_len - 2; i >= 0; i--)
	{
		const auto back_c = c.template at<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			real_t prev_data = d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x);
			d.template at<'s', 'z', 'y', 'x'>(s, z, i, x) -= back_c * prev_data;

			if constexpr (!z_blocked)
				z_forward_inside_y(d, s, z, i + 1, x, az_s, b1z_s, z_data.c_tmp, prev_data);
			else
				z_forward_inside_y_blocked(d, s, z, i + 1, x, prev_data, az_s, b1z_s, z_data.begin, z_data.c);
		}
	}
}

// template <typename index_t, typename density_bag_t, typename diag_bag_t, typename z_data_t>
// constexpr static void y_backward_blocked(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z,
// 							   z_data_t z_data)

// {
// 	const index_t x_len = d | noarr::get_length<'x'>();
// 	const index_t y_len = d | noarr::get_length<'y'>();

// 	for (index_t i = y_len - 2; i >= 0; i--)
// 	{
// 		const auto back_c = c.template at<'s', 'i'>(s, i);

// 		for (index_t x = 0; x < x_len; x++)
// 		{
// 			d.template at<'s', 'z', 'y', 'x'>(s, z, i, x) -= back_c * d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x);

// 			z_forward_inside_y_blocked(d, s, z, i + 1, x, z_data.begin, z_data.a_s, z_data.b1_s, z_data.c,
// 									   d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x));
// 		}
// 	}
// }

template <typename index_t, typename real_t, typename density_bag_t>
constexpr static void x_forward(const density_bag_t d, const index_t s, const index_t z, const index_t y,
								const real_t a_s, const real_t b1_s, real_t& a_tmp, real_t& b_tmp, real_t& c_tmp,
								real_t& prev)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = 0; i < x_len; i++)
	{
		const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

		real_t curr = d.template at<'s', 'z', 'y', 'x'>(s, z, y, i);
		curr = r * (curr - a_tmp * prev);
		d.template at<'s', 'z', 'y', 'x'>(s, z, y, i) = curr;

		a_tmp = a_s;
		b_tmp = b1_s + (i == x_len - 2 ? a_s : 0);
		c_tmp = a_s * r;
		prev = curr;
	}
}

template <bool y_blocked, bool blocked_alt, typename index_t, typename real_t, typename density_bag_t,
		  typename diag_bag_t, typename y_data_t>
constexpr static void x_backward(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z,
								 const index_t y, real_t& prev, const real_t ay_s, const real_t b1y_s, y_data_t y_data)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = x_len - 2; i >= 0; i--)
	{
		real_t curr = d.template at<'s', 'z', 'x', 'y'>(s, z, i, y);
		curr = curr - c.template at<'s', 'i'>(s, i) * prev;
		// d.template at<'s', 'z', 'x', 'y'>(s, z, i, y) = curr;

		if constexpr (!y_blocked)
			y_forward_inside_x<false>(d, s, z, y, i + 1, prev, ay_s, b1y_s, y_data.c_tmp);
		else if constexpr (!blocked_alt)
			y_forward_inside_x_blocked(d, s, z, y, i + 1, prev, ay_s, b1y_s, y_data.begin, y_data.c);
		else
			y_forward_inside_x_blocked_alt(d, s, z, y, i + 1, prev, ay_s, b1y_s, y_data.begin, y_data.end, y_data.a,
										   y_data.c);

		prev = curr;
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_xyz_fused(real_t* __restrict__ densities, const real_t* __restrict__ ax,
								  const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
								  const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
								  const real_t* __restrict__ back_cy, const real_t* __restrict__ az,
								  const real_t* __restrict__ b1z, const real_t* __restrict__ back_cz,
								  const density_layout_t dens_l, const diagonal_layout_t diagx_l,
								  const diagonal_layout_t diagy_l, const diagonal_layout_t diagz_l,
								  const index_t s_begin, const index_t s_end)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const auto d = noarr::make_bag(dens_l, densities);
	const auto cx = noarr::make_bag(diagx_l, back_cx);
	const auto cy = noarr::make_bag(diagy_l, back_cy);
	const auto cz = noarr::make_bag(diagz_l, back_cz);

#pragma omp for schedule(static) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		real_t c_tmp_z = az_s;

		for (index_t z = 0; z < z_len; z++)
		{
			real_t c_tmp_y = ay_s;

			for (index_t y = 0; y < y_len; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward<false, false>(d, cx, s, z, y, prev, ay_s, b1y_s, inside_data { c_tmp_y });

				y_forward_inside_x<true>(d, s, z, y, 0, prev, ay_s, b1y_s, c_tmp_y);
			}

			y_backward<false>(d, cy, s, z, ay_s, b1y_s, inside_data { c_tmp_z });


			{
				real_t r_z = 0;

				for (index_t x = 0; x < x_len; x++)
				{
					r_z = z_forward_inside_y(d, s, z, 0, x, az_s, b1z_s, c_tmp_z,
											 d.template at<'s', 'z', 'y', 'x'>(s, z, 0, x));
				}

				c_tmp_z = az_s * r_z;
			}
		}

		z_backward(d, cz, s);
	}
}

// template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
// 		  typename scratch_layout_t>
// constexpr static void solve_slice_xyz_fused_blocked(
// 	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
// 	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
// 	const real_t* __restrict__ back_cy, const real_t* __restrict__ az, const real_t* __restrict__ b1z,
// 	real_t* __restrict__ a_data, real_t* __restrict__ c_data, const density_layout_t dens_l,
// 	const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l, const scratch_layout_t scratch_l, const index_t s,
// 	const index_t z_begin, const index_t z_end, const index_t z_block_size, index_t& epoch, std::atomic<long>& counter)
// {
// 	const index_t x_len = dens_l | noarr::get_length<'x'>();
// 	const index_t y_len = dens_l | noarr::get_length<'y'>();
// 	const index_t z_len = dens_l | noarr::get_length<'z'>();

// 	const auto d = noarr::make_bag(dens_l, densities);
// 	const auto cx = noarr::make_bag(diagx_l, back_cx);
// 	const auto cy = noarr::make_bag(diagy_l, back_cy);

// 	const auto a = noarr::make_bag(scratch_l, a_data);
// 	const auto c = noarr::make_bag(scratch_l, c_data);

// 	const real_t ax_s = ax[s];
// 	const real_t b1x_s = b1x[s];

// 	const real_t ay_s = ay[s];
// 	const real_t b1y_s = b1y[s];

// 	const real_t az_s = az[s];
// 	const real_t b1z_s = b1z[s];

// 	for (index_t z = z_begin; z < z_end; z++)
// 	{
// 		real_t c_tmp_y = ay_s;

// 		for (index_t y = 0; y < y_len; y++)
// 		{
// 			real_t a_tmp = 0;
// 			real_t b_tmp = b1x_s + ax_s;
// 			real_t c_tmp = ax_s;
// 			real_t prev = 0;

// 			x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

// 			x_backward<false>(d, cx, s, z, y, prev, ay_s, b1y_s, inside_data<real_t> { c_tmp_y });

// 			y_forward_inside_x<true>(d, s, z, y, 0, prev, ay_s, b1y_s, c_tmp_y);
// 		}

// 		y_backward<true>(d, cy, s, z, az_s, b1z_s, inside_data_blocked { z_begin, c });

// 		{
// 			for (index_t x = 0; x < x_len; x++)
// 			{
// 				z_forward_inside_y_blocked(d, s, z, 0, x, d.template at<'s', 'z', 'y', 'x'>(s, z, 0, x), az_s, b1z_s,
// 										   z_begin, c);
// 			}

// 			z_forward_inside_y_blocked_next(s, z, az_s, b1z_s, z_len, z_begin, a, c);
// 		}
// 	}

// 	z_backward_blocked(d, 0, y_len, z_begin, z_end, a, c, s);

// 	index_t cooperating_threads = (z_len + z_block_size - 1) / z_block_size;

// 	{
// 		auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

// 		while (val < epoch * cooperating_threads)
// 		{
// 			val = counter.load(std::memory_order_acquire);
// 		}
// 	}

// 	// #pragma omp barrier

// 	if (z_begin == 0)
// 		z_blocked_middle(d, a, c, z_block_size, 0, y_len, s);

// 	// #pragma omp barrier

// 	epoch++;

// 	{
// 		auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

// 		while (val < epoch * cooperating_threads)
// 		{
// 			val = counter.load(std::memory_order_acquire);
// 		}
// 	}

// 	z_blocked_end(d, a, c, 0, y_len, z_begin, z_end, s);
// }

template <bool is_end, typename index_t, typename real_t, typename vec_t, typename simd_tag>
constexpr static void x_forward_vectorized(vec_t* rows, simd_tag t, const index_t length, const real_t a_s,
										   const real_t b1_s, real_t& a_tmp, real_t& b_tmp, real_t& c_tmp)

{
	for (index_t v = 0; v < length; v++)
	{
		const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

		rows[v] = hn::Mul(hn::MulAdd(rows[v - 1], hn::Set(t, -a_tmp), rows[v]), hn::Set(t, r));

		a_tmp = a_s;

		if constexpr (is_end)
			b_tmp = b1_s + (v == length - 2 ? a_s : 0);
		else
			b_tmp = b1_s;

		c_tmp = a_s * r;
	};
}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t>
constexpr static void x_backward_vectorized(const diagonal_bag_t c, vec_t* rows, simd_tag t, const index_t length,
											const index_t x, const index_t s)

{
	auto c_vec = hn::Load(t, &(c.template at<'i', 's'>(x, s)));

	for (index_t v = length - 1; v >= 0; v--)
	{
		rows[v] = hn::NegMulAdd(rows[v + 1], hn::Set(t, hn::ExtractLane(c_vec, v)), rows[v]);
	}
}

template <typename index_t, typename real_t, typename vec_t, typename simd_tag, typename density_layout_t>
static real_t z_forward_inside_y_vectorized(const density_layout_t d, simd_tag t, const index_t z, const index_t y,
											vec_t data, const real_t az_s, const real_t b1z_s, const index_t z_len,
											const real_t cz_tmp)
{
	real_t r;

	if (z == 0)
	{
		r = 1 / (b1z_s + az_s);

		data = hn::Mul(data, hn::Set(t, r));

		hn::Store(data, t, &d.template at<'z', 'y'>(z, y));
	}
	else
	{
		const real_t b_tmp = b1z_s + (z == z_len - 1 ? az_s : 0);
		r = 1 / (b_tmp - az_s * cz_tmp);

		vec_t tmp = hn::Load(t, &d.template at<'z', 'y'>(z - 1, y));

		data = hn::Mul(hn::Set(t, r), hn::MulAdd(hn::Set(t, -az_s), tmp, data));

		hn::Store(data, t, &d.template at<'z', 'y'>(z, y));
	}

	return r;
}

template <typename vec_t, typename index_t, typename real_t, typename simd_tag, typename density_layout_t,
		  typename diag_bag_t>
constexpr static void z_backward(const density_layout_t dens_l, real_t* __restrict__ densities, const diag_bag_t c,
								 simd_tag t, const index_t s, const index_t simd_length, const index_t y_len,
								 const index_t z_len)
{
	constexpr index_t x_tile_multiple = 1;

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length * x_tile_multiple);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const auto d =
				noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'y', 'b', 'X'>(s, y, noarr::lit<0>, X), densities);

			vec_t prev[x_tile_multiple];

			for (index_t x = 0; x < x_tile_multiple; x++)
				prev[x] = hn::Load(t, &d.template at<'z', 'y', 'x'>(z_len - 1, y, x * simd_length));

			for (index_t i = z_len - 2; i >= 0; i--)
			{
				for (index_t x = 0; x < x_tile_multiple; x++)
				{
					vec_t curr = hn::Load(t, &d.template at<'z', 'x'>(i, x * simd_length));
					curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev[x], curr);
					hn::Store(curr, t, &d.template at<'z', 'x'>(i, x * simd_length));

					prev[x] = curr;
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename vec_t, typename simd_tag, typename density_bag_t>
constexpr static void y_forward_inside_x_vectorized(const density_bag_t d, vec_t* rows, simd_tag t, const index_t z,
													const index_t y_offset, const index_t x, const index_t s,
													const index_t length, const index_t y_len, const real_t ay_s,
													const real_t b1y_s, real_t ay_bl_tmp, real_t by_bl_tmp,
													real_t cy_bl_tmp)
{
	vec_t prev;

	if (y_offset == 0)
		prev = hn::Zero(t);
	else
		prev = hn::Load(t, &(d.template at<'z', 'y', 'x', 's'>(z, y_offset - 1, x, s)));

	for (index_t v = 0; v < length; v++)
	{
		const index_t y = y_offset + v;
		const real_t r = 1 / (by_bl_tmp - ay_bl_tmp * cy_bl_tmp);

		rows[v] = hn::Mul(hn::MulAdd(hn::Set(t, -ay_bl_tmp), prev, rows[v]), hn::Set(t, r));
		prev = rows[v];

		ay_bl_tmp = ay_s;
		by_bl_tmp = b1y_s + (y == y_len - 2 ? ay_s : 0);
		cy_bl_tmp = ay_s * r;
	}
}

template <typename index_t, typename real_t, typename vec_t, typename simd_tag, typename density_bag_t>
constexpr static void y_forward_inside_x_vectorized_blocked(const density_bag_t d, vec_t* rows, simd_tag t,
															const index_t z, const index_t y_offset, const index_t x,
															const index_t s, const index_t length, const index_t y_len,
															const real_t ay_s, const real_t b1y_s,
															const index_t y_begin, real_t cy_tmp)
{
	vec_t prev;

	if (y_offset == y_begin)
		prev = hn::Zero(t);
	else
		prev = hn::Load(t, &(d.template at<'z', 'y', 'x', 's'>(z, y_offset - 1, x, s)));

	for (index_t v = 0; v < length; v++)
	{
		const index_t y = y_offset + v;

		if (y < y_begin + 2)
		{
			const auto b_tmp = b1y_s + ((y == 0) || (y == y_len - 1) ? ay_s : 0);

			rows[v] = hn::Mul(rows[v], hn::Set(t, 1 / b_tmp));
			prev = rows[v];

			cy_tmp = ay_s / b_tmp;

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "f0: " << z << " " << y << " " << x + l << " " << hn::ExtractLane(rows[v], l)
			// << " "
			// 						  << b_tmp << std::endl;
			// 			}
		}
		else
		{
			const auto a_tmp = ay_s * (y == 0 ? 0 : 1);
			const auto b_tmp = b1y_s + (y == y_len - 1 ? ay_s : 0);
			const auto c_tmp = ay_s * (y == y_len - 1 ? 0 : 1);

			const auto r = 1 / (b_tmp - a_tmp * cy_tmp);

			rows[v] = hn::MulAdd(hn::Set(t, -a_tmp), prev, rows[v]);
			rows[v] = hn::Mul(rows[v], hn::Set(t, r));
			prev = rows[v];

			cy_tmp = r * c_tmp;

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "f1: " << z << " " << y << " " << x + l << " " << hn::ExtractLane(rows[v], l)
			// << " "
			// 						  << a_tmp << " " << b_tmp << " " << cy_tmp << std::endl;
			// 			}
		}
	}
}

template <typename index_t, typename real_t, typename vec_t, typename simd_tag, typename density_bag_t>
constexpr static void y_forward_inside_x_vectorized_blocked_alt(
	const density_bag_t d, vec_t* rows, simd_tag t, const index_t z, const index_t y_offset, const index_t x,
	const index_t s, const index_t length, const index_t y_len, const real_t ay_s, const real_t b1y_s,
	const index_t y_begin, const index_t y_end, real_t ay_tmp, real_t cy_tmp, real_t cy0_tmp)
{
	vec_t prev;
	vec_t y0;

	if (y_offset == y_begin)
		prev = hn::Zero(t);
	else
	{
		prev = hn::Load(t, &(d.template at<'z', 'y', 'x', 's'>(z, y_offset - 1, x, s)));
		y0 = hn::Load(t, &(d.template at<'z', 'y', 'x', 's'>(z, y_begin, x, s)));
	}

	for (index_t v = 0; v < length; v++)
	{
		const index_t y = y_offset + v;

		if (y < y_begin + 2)
		{
			const auto a_tmp = ay_s * (y == 0 ? 0 : 1);
			const auto b_tmp = b1y_s + ((y == 0) || (y == y_len - 1) ? ay_s : 0);
			const auto c_tmp = ay_s * (y == y_len - 1 ? 0 : 1);

			rows[v] = hn::Mul(rows[v], hn::Set(t, 1 / b_tmp));
			prev = rows[v];

			ay_tmp = a_tmp / b_tmp;
			cy_tmp = c_tmp / b_tmp;

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "f0: " << z << " " << y << " " << x + l << " " << hn::ExtractLane(rows[v], l)
			// << " "
			// 						  << b_tmp << std::endl;
			// 			}
		}
		else
		{
			const auto a_tmp = ay_s * (y == 0 ? 0 : 1);
			const auto b_tmp = b1y_s + (y == y_len - 1 ? ay_s : 0);
			const auto c_tmp = ay_s * (y == y_len - 1 ? 0 : 1);

			const auto r = 1 / (b_tmp - a_tmp * cy_tmp);

			ay_tmp = r * (0 - a_tmp * ay_tmp);
			cy_tmp = r * c_tmp;

			rows[v] = hn::MulAdd(hn::Set(t, -a_tmp), prev, rows[v]);
			rows[v] = hn::Mul(rows[v], hn::Set(t, r));
			prev = rows[v];

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "f1: " << z << " " << y << " " << x + l << " " << hn::ExtractLane(rows[v], l)
			// << " "
			// 						  << a_tmp << " " << b_tmp << " " << cy_tmp << std::endl;
			// 			}
		}

		if (y != y_begin && y != y_end - 1)
		{
			// const auto state0 = noarr::idx<'s', 'i'>(s, y_begin);
			const auto r0 = 1 / (1 - ay_tmp * cy0_tmp);

			if (y == y_begin + 1)
				y0 = rows[0];

			y0 = hn::MulAdd(hn::Set(t, -cy0_tmp), rows[v], y0);
			y0 = hn::Mul(y0, hn::Set(t, r0));

			// #pragma omp critical
			// 			for (std::size_t l = 0; l < hn::Lanes(t); l++)
			// 			{
			// 				std::cout << "0 z: " << z << " y: " << y << " x: " << x + l << " a0: " << ay0_tmp << " c0: "
			// << cy0_tmp
			// 						  << " r0: " << r0 << " d: " << hn::ExtractLane(y0, l) << std::endl;
			// 			}

			cy0_tmp = r0 * -cy_tmp * cy0_tmp;
		}
	}

	if (y_offset == y_begin)
		rows[0] = y0;
	else
		hn::Store(y0, t, &(d.template at<'z', 'y', 'x', 's'>(z, y_begin, x, s)));
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked_next(const index_t s, const index_t y, const real_t a_s,
													  const real_t b1_s, const index_t y_len, const index_t y_begin,
													  scratch_bag_t a, scratch_bag_t c)

{
	if (y < y_begin + 2)
	{
		const auto state = noarr::idx<'s', 'i'>(s, y);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((y == 0) || (y == y_len - 1) ? a_s : 0);
		const auto c_tmp = a_s * (y == y_len - 1 ? 0 : 1);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'s', 'i'>(s, y);
		const auto prev_state = noarr::idx<'s', 'i'>(s, y - 1);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);
		const auto c_tmp = a_s * (y == y_len - 1 ? 0 : 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void y_forward_inside_x_blocked_next_alt(const index_t s, const index_t y, const real_t a_s,
														  const real_t b1_s, const index_t y_len, const index_t y_begin,
														  const index_t y_end, scratch_bag_t a, scratch_bag_t c)

{
	if (y < y_begin + 2)
	{
		const auto state = noarr::idx<'s', 'i'>(s, y);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((y == 0) || (y == y_len - 1) ? a_s : 0);
		const auto c_tmp = a_s * (y == y_len - 1 ? 0 : 1);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'s', 'i'>(s, y);
		const auto prev_state = noarr::idx<'s', 'i'>(s, y - 1);

		const auto a_tmp = a_s * (y == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);
		const auto c_tmp = a_s * (y == y_len - 1 ? 0 : 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}

	if (y != y_begin && y != y_end - 1)
	{
		const auto state = noarr::idx<'s', 'i'>(s, y);
		const auto state0 = noarr::idx<'s', 'i'>(s, y_begin);
		const auto r0 = 1 / (1 - a[state] * c[state0]);

		a[state0] *= r0;
		c[state0] = r0 * -c[state] * c[state0];
	}
}

template <typename index_t, typename real_t, typename scratch_bag_t>
constexpr static void z_forward_inside_y_blocked_next_alt(const index_t s, const index_t z, const real_t a_s,
														  const real_t b1_s, const index_t z_len, const index_t z_begin,
														  const index_t z_end, scratch_bag_t a, scratch_bag_t c)

{
	if (z < z_begin + 2)
	{
		const auto state = noarr::idx<'s', 'i'>(s, z);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + ((z == 0) || (z == z_len - 1) ? a_s : 0);
		const auto c_tmp = a_s * (z == z_len - 1 ? 0 : 1);

		a[state] = a_tmp / b_tmp;
		c[state] = c_tmp / b_tmp;
	}
	else
	{
		const auto state = noarr::idx<'s', 'i'>(s, z);
		const auto prev_state = noarr::idx<'s', 'i'>(s, z - 1);

		const auto a_tmp = a_s * (z == 0 ? 0 : 1);
		const auto b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);
		const auto c_tmp = a_s * (z == z_len - 1 ? 0 : 1);

		const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

		a[state] = r * (0 - a_tmp * a[prev_state]);
		c[state] = r * c_tmp;
	}

	if (z != z_begin && z != z_end - 1)
	{
		const auto state = noarr::idx<'s', 'i'>(s, z);
		const auto state0 = noarr::idx<'s', 'i'>(s, z_begin);
		const auto r0 = 1 / (1 - a[state] * c[state0]);

		a[state0] *= r0;
		c[state0] = r0 * -c[state] * c[state0];
	}
}

template <typename index_t>
struct inner_z_data
{
	index_t& cz_tmp;
};

template <typename index_t, typename scratch_bag_t>
struct inner_z_data_blocked
{
	index_t z_begin;
	scratch_bag_t az_scratch;
	scratch_bag_t cz_scratch;
};

template <typename index_t>
struct inner_y_data
{
	index_t& cy_tmp;
};

template <typename index_t, typename real_t, typename scratch_bag_t>
struct inner_y_data_blocked
{
	index_t y_begin;
	real_t& cy_tmp;
	scratch_bag_t ay_scratch;
	scratch_bag_t cy_scratch;
};

template <typename index_t, typename real_t, typename scratch_bag_t>
struct inner_y_data_blocked_alt
{
	index_t y_begin;
	index_t y_end;
	real_t& ay_tmp;
	real_t& cy_tmp;
	real_t& ay0_tmp;
	real_t& cy0_tmp;
	scratch_bag_t ay_scratch;
	scratch_bag_t cy_scratch;
};

template <bool z_blocked, bool alt_blocked, typename vec_t, typename index_t, typename real_t, typename simd_tag,
		  typename density_layout_t, typename diag_bag_t, typename z_data_t>
constexpr static void y_backward_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
											const diag_bag_t c, simd_tag t, const index_t z, const index_t s,
											const index_t simd_length, const index_t y_len, const real_t az_s,
											const real_t b1z_s, z_data_t z_data)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();
	const index_t z_len = blocked_dens_l | noarr::get_length<'z'>();

	real_t r_z = 0;

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d = noarr::make_bag(
			blocked_dens_l ^ noarr::fix<'s', 'b', 'X', 'x'>(s, noarr::lit<0>, X, noarr::lit<0>), densities);

		vec_t prev = hn::Load(t, &d.template at<'z', 'y'>(z, y_len - 1));

		for (index_t i = y_len - 2; i >= 0; i--)
		{
			vec_t curr = hn::Load(t, &d.template at<'z', 'y'>(z, i));
			curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev, curr);
			// hn::Store(curr, t, &d.template at<'z', 'y'>(z, i));

			if constexpr (!z_blocked)
				z_forward_inside_y_vectorized(d, t, z, i + 1, prev, az_s, b1z_s, z_len, z_data.cz_tmp);
			else if constexpr (!alt_blocked)
				z_forward_inside_y_vectorized_blocked(d, t, s, z, i + 1, prev, az_s, b1z_s, z_data.z_begin,
													  z_data.cz_scratch);
			else
				z_forward_inside_y_blocked_alt_vectorized(d, t, s, i + 1, z, X, prev, az_s, b1z_s, z_data.begin,
														  z_data.end, z_data.a, z_data.c);

			prev = curr;
		}

		if constexpr (!z_blocked)
			r_z = z_forward_inside_y_vectorized(d, t, z, 0, prev, az_s, b1z_s, z_len, z_data.cz_tmp);
		else if constexpr (!alt_blocked)
			z_forward_inside_y_vectorized_blocked(d, t, s, z, 0, prev, az_s, b1z_s, z_data.z_begin, z_data.cz_scratch);
		else
			z_forward_inside_y_blocked_alt_vectorized(d, t, s, 0, z, X, prev, az_s, b1z_s, z_data.begin, z_data.end,
													  z_data.a, z_data.c);
	}

	if constexpr (!z_blocked)
		z_data.cz_tmp = az_s * r_z;
	else if constexpr (!alt_blocked)
		z_forward_inside_y_blocked_next(s, z, az_s, b1z_s, z_len, z_data.z_begin, z_data.az_scratch, z_data.cz_scratch);
	else
		z_forward_inside_y_blocked_next_alt(s, z, az_s, b1z_s, z_len, z_data.begin, z_data.end, z_data.a, z_data.c);
}

template <typename vec_t, typename index_t, typename real_t, typename simd_tag, typename density_layout_t,
		  typename diag_bag_t>
constexpr static void y_backward_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
											const diag_bag_t c, simd_tag t, const index_t z, const index_t s,
											const index_t simd_length, const index_t y_len)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d = noarr::make_bag(
			blocked_dens_l ^ noarr::fix<'s', 'b', 'X', 'x'>(s, noarr::lit<0>, X, noarr::lit<0>), densities);

		vec_t prev = hn::Load(t, &d.template at<'z', 'y'>(z, y_len - 1));

		for (index_t i = y_len - 2; i >= 0; i--)
		{
			vec_t curr = hn::Load(t, &d.template at<'z', 'y'>(z, i));
			curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev, curr);
			hn::Store(curr, t, &d.template at<'z', 'y'>(z, i));

			prev = curr;
		}
	}
}

template <bool y_blocked, bool blocked_alt, typename simd_t, typename simd_tag, typename index_t, typename real_t,
		  typename density_bag_t, typename diag_bag_t, typename y_data_t>
constexpr static void xy_fused_transpose_part(const density_bag_t d, simd_tag t, const index_t simd_length,
											  const index_t y_begin, const index_t y_end, const index_t z,
											  const index_t s, const real_t ax_s, const real_t b1x_s, const real_t ay_s,
											  const real_t b1y_s, const diag_bag_t cx, y_data_t y_data)
{
	real_t ay_tmp = 0;
	real_t by_tmp = b1y_s + ay_s;

	const index_t n = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	// vector registers that hold the to be transposed x*y plane
	simd_t* rows = new simd_t[simd_length + 1];

	for (index_t y = y_begin; y < y_end; y += simd_length)
	{
		real_t ax_tmp = 0;
		real_t bx_tmp = b1x_s + ax_s;
		real_t cx_tmp = ax_s;
		simd_t prev_x = hn::Zero(t);

		// forward substitution until last simd_length elements
		for (index_t i = 0; i < full_n - simd_length; i += simd_length)
		{
			rows[0] = prev_x;

			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				rows[v + 1] = hn::Load(t, &(d.template at<'z', 'y', 'x', 's'>(z, y + v, i, s)));

			// transposition to enable vectorization
			transpose(rows + 1);

			// actual forward substitution (vectorized)
			{
				x_forward_vectorized<false>(rows + 1, t, simd_length, ax_s, b1x_s, ax_tmp, bx_tmp, cx_tmp);

				prev_x = rows[simd_length];
			}

			// transposition back to the original form
			transpose(rows + 1);

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(rows[v + 1], t, &(d.template at<'z', 'y', 'x', 's'>(z, y + v, i, s)));
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			rows[0] = prev_x;

			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				rows[v + 1] = hn::Load(t, &(d.template at<'z', 'y', 'x', 's'>(z, y + v, full_n - simd_length, s)));

			// transposition to enable vectorization
			transpose(rows + 1);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			{
				x_forward_vectorized<true>(rows + 1, t, remainder_work, ax_s, b1x_s, ax_tmp, bx_tmp, cx_tmp);
			}

			// the begin of backward part
			{
				x_backward_vectorized(cx, rows + 1, t, remainder_work - 1, full_n - simd_length, s);

				prev_x = rows[1];
			}

			// transposition back to the original form
			transpose(rows + 1);

			if constexpr (!y_blocked)
				y_forward_inside_x_vectorized(d, rows + 1, t, z, y, full_n - simd_length, s, simd_length, y_len, ay_s,
											  b1y_s, ay_tmp, by_tmp, y_data.c_tmp);
			else if constexpr (!blocked_alt)
				y_forward_inside_x_vectorized_blocked(d, rows + 1, t, z, y, full_n - simd_length, s, simd_length, y_len,
													  ay_s, b1y_s, y_data.y_begin, y_data.cy_tmp);
			else
				y_forward_inside_x_vectorized_blocked_alt(d, rows + 1, t, z, y, full_n - simd_length, s, simd_length,
														  y_len, ay_s, b1y_s, y_data.y_begin, y_data.y_end,
														  y_data.ay_tmp, y_data.cy_tmp, y_data.cy0_tmp);

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(rows[v + 1], t, &(d.template at<'z', 'y', 'x', 's'>(z, y + v, full_n - simd_length, s)));
		}

		// we continue with backwards substitution
		for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
		{
			rows[simd_length] = prev_x;

			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				rows[v] = hn::Load(t, &(d.template at<'z', 'y', 'x', 's'>(z, y + v, i, s)));

			// transposition to enable vectorization
			transpose(rows);

			// backward propagation
			{
				x_backward_vectorized(cx, rows, t, simd_length, i, s);

				prev_x = rows[0];
			}

			// transposition back to the original form
			transpose(rows);

			if constexpr (!y_blocked)
				y_forward_inside_x_vectorized(d, rows, t, z, y, i, s, simd_length, y_len, ay_s, b1y_s, ay_tmp, by_tmp,
											  y_data.c_tmp);
			else if constexpr (!blocked_alt)
				y_forward_inside_x_vectorized_blocked(d, rows, t, z, y, i, s, simd_length, y_len, ay_s, b1y_s,
													  y_data.y_begin, y_data.cy_tmp);
			else
				y_forward_inside_x_vectorized_blocked_alt(d, rows, t, z, y, i, s, simd_length, y_len, ay_s, b1y_s,
														  y_data.y_begin, y_data.y_end, y_data.ay_tmp, y_data.cy_tmp,
														  y_data.cy0_tmp);

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(rows[v], t, &(d.template at<'z', 'y', 'x', 's'>(z, y + v, i, s)));
		}


		if constexpr (!y_blocked)
			for (index_t v = 0; v < simd_length; v++)
			{
				const real_t r = 1 / (by_tmp - ay_tmp * y_data.c_tmp);

				ay_tmp = ay_s;
				by_tmp = b1y_s + (y + v == y_len - 2 ? ay_s : 0);
				y_data.c_tmp = ay_s * r;
			}
		else if constexpr (!blocked_alt)
		{
			for (index_t v = 0; v < simd_length; v++)
			{
				y_forward_inside_x_blocked_next(s, y + v, ay_s, b1y_s, y_len, y_data.y_begin, y_data.ay_scratch,
												y_data.cy_scratch);
			}
			y_data.cy_tmp = y_data.cy_scratch.template at<'s', 'i'>(s, y + simd_length - 1);
		}
		else
		{
			for (index_t v = 0; v < simd_length; v++)
			{
				y_forward_inside_x_blocked_next_alt(s, y + v, ay_s, b1y_s, y_len, y_data.y_begin, y_data.y_end,
													y_data.ay_scratch, y_data.cy_scratch);
			}

			y_data.ay_tmp = y_data.ay_scratch.template at<'s', 'i'>(s, y + simd_length - 1);
			y_data.cy_tmp = y_data.cy_scratch.template at<'s', 'i'>(s, y + simd_length - 1);

			y_data.cy0_tmp = y_data.cy_scratch.template at<'s', 'i'>(s, y_begin);
		}
	}

	delete[] rows;
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
constexpr static void solve_slice_xyz_fused_transpose(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ back_cy, const real_t* __restrict__ az, const real_t* __restrict__ b1z,
	const real_t* __restrict__ back_cz, const density_layout_t dens_l, const diagonal_layout_t diagx_l,
	const diagonal_layout_t diagy_l, const diagonal_layout_t diagz_l, const index_t s_begin, const index_t s_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);
		auto cz = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s), back_cz);

		real_t cz_tmp = az_s;

		for (index_t z = 0; z < z_len; z++)
		{
			real_t cy_tmp = ay_s;

			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;
				const auto d = noarr::make_bag(dens_l, densities);

				xy_fused_transpose_part<false, false, simd_t>(d, t, simd_length, 0, y_len_body, z, s, ax_s, b1x_s, ay_s,
															  b1y_s, cx, inside_data { cy_tmp });
			}

			// y remainder
			{
				const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z), densities);

				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len - y_len_remainder; y < y_len; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<false, false>(d, cx, s, z, y, prev, ay_s, b1y_s, inside_data<real_t> { cy_tmp });

					y_forward_inside_x<true>(d, s, z, y, 0, prev, ay_s, b1y_s, cy_tmp);
				}
			}

			y_backward_vectorized<false, false, simd_t>(dens_l, densities, cy, t, z, s, simd_length, y_len, az_s, b1z_s,
														inner_z_data { cz_tmp });
		}

		z_backward<simd_t>(dens_l, densities, cz, t, s, simd_length, y_len, z_len);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
constexpr static void solve_slice_xy_fused_transpose(real_t* __restrict__ densities, const real_t* __restrict__ ax,
													 const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
													 const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
													 const real_t* __restrict__ back_cy, const density_layout_t dens_l,
													 const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
													 const index_t s_begin, const index_t s_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto d = noarr::make_bag(dens_l, densities);

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);


		real_t cy_tmp = ay_s;

		{
			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;

			xy_fused_transpose_part<false, false, simd_t>(d, t, simd_length, 0, y_len_body, 0, s, ax_s, b1x_s, ay_s,
														  b1y_s, cx, inside_data { cy_tmp });
		}

		// y remainder
		{
			auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

			for (index_t y = y_len - y_len_remainder; y < y_len; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, s, 0, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward<false, false>(d, cx, s, 0, y, prev, ay_s, b1y_s, inside_data<real_t> { cy_tmp });

				y_forward_inside_x<true>(d, s, 0, y, 0, prev, ay_s, b1y_s, cy_tmp);
			}
		}

		y_backward_vectorized<simd_t>(dens_l, densities, cy, t, 0, s, simd_length, y_len);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ back_cy, const real_t* __restrict__ az, const real_t* __restrict__ b1z,
	real_t* __restrict__ a_data, real_t* __restrict__ c_data, real_t* __restrict__ z_data,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
	const scratch_layout_t scratch_l, const dim_scratch_layout_t dim_scratch_l, const index_t s_begin,
	const index_t s_end, const index_t z_begin, const index_t z_end, const index_t tid, const index_t coop_size,
	index_t& epoch, std::atomic<long>& counter)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto d = noarr::make_bag(dens_l, densities);

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const auto a_scratch = noarr::make_bag(scratch_l, a_data);
	const auto c_scratch = noarr::make_bag(scratch_l, c_data);

	const auto z_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), z_data);

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

		for (index_t z = z_begin; z < z_end; z++)
		{
			real_t cy_tmp = ay_s;

			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;
				const auto d = noarr::make_bag(dens_l, densities);

				xy_fused_transpose_part<false, false, simd_t>(d, t, simd_length, 0, y_len_body, z, s, ax_s, b1x_s, ay_s,
															  b1y_s, cx, inside_data { cy_tmp });
			}

			// y remainder
			{
				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len - y_len_remainder; y < y_len; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<false, false>(d, cx, s, z, y, prev, ay_s, b1y_s, inside_data { cy_tmp });

					y_forward_inside_x<true>(d, s, z, y, 0, prev, ay_s, b1y_s, cy_tmp);
				}
			}

			y_backward_vectorized<true, false, simd_t>(dens_l, densities, cy, t, z, s, simd_length, y_len, az_s, b1z_s,
													   inner_z_data_blocked { z_begin, a_scratch, c_scratch });
		}

		z_backward_blocked(d, 0, y_len, z_begin, z_end, a_scratch, c_scratch, s);


		{
			epoch++;

			auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

			while (val < epoch * coop_size)
			{
				val = counter.load(std::memory_order_acquire);
			}
		}

		// #pragma omp barrier

		z_blocked_middle(d, a_scratch, c_scratch, z_scratch, tid, coop_size, 0, y_len, s);

		// #pragma omp barrier


		{
			epoch++;

			auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

			while (val < epoch * coop_size)
			{
				val = counter.load(std::memory_order_acquire);
			}
		}

		z_blocked_end(d, a_scratch, c_scratch, 0, y_len, z_begin, z_end, s);
	}
}

long now;

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked_alt(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ back_cy, const real_t* __restrict__ az, const real_t* __restrict__ b1z,
	real_t* __restrict__ a_data, real_t* __restrict__ c_data, real_t* __restrict__ z_data,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
	const scratch_layout_t scratch_l, const dim_scratch_layout_t dim_scratch_l, const index_t s_begin,
	const index_t s_end, const index_t z_begin, const index_t z_end, const index_t tid, const index_t coop_size,
	index_t& epoch, std::atomic<long>& counter)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto d = noarr::make_bag(dens_l, densities);

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const auto a_scratch = noarr::make_bag(scratch_l, a_data);
	const auto c_scratch = noarr::make_bag(scratch_l, c_data);
	const auto z_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), z_data);

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

		for (index_t z = z_begin; z < z_end; z++)
		{
			real_t cy_tmp = ay_s;

			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len_body = (body_dens_l | noarr::get_length<'y'>()) * simd_length;
				const auto d = noarr::make_bag(dens_l, densities);

				xy_fused_transpose_part<false, false, simd_t>(d, t, simd_length, 0, y_len_body, z, s, ax_s, b1x_s, ay_s,
															  b1y_s, cx, inside_data { cy_tmp });
			}

			// y remainder
			{
				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t y_len_remainder = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len - y_len_remainder; y < y_len; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<false, false>(d, cx, s, z, y, prev, ay_s, b1y_s, inside_data { cy_tmp });

					y_forward_inside_x<true>(d, s, z, y, 0, prev, ay_s, b1y_s, cy_tmp);
				}
			}

			y_backward_vectorized<true, true, simd_t>(dens_l, densities, cy, t, z, s, simd_length, y_len, az_s, b1z_s,
													  inside_data_blocked_alt { z_begin, z_end, a_scratch, c_scratch });
		}

		{
			epoch++;

			auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;
			counter.notify_all();

			while (val < epoch * coop_size)
			{
				counter.wait(val, std::memory_order_acquire);
				val = counter.load(std::memory_order_acquire);
			}
		}

		z_blocked_middle(d, a_scratch, c_scratch, z_scratch, tid, coop_size, 0, y_len, s);

		{
			epoch++;

			auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;
			counter.notify_all();

			while (val < epoch * coop_size)
			{
				counter.wait(val, std::memory_order_acquire);
				val = counter.load(std::memory_order_acquire);
			}
		}

		z_blocked_end_alt(d, a_scratch, c_scratch, 0, y_len, z_begin, z_end, s);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t>
constexpr static void solve_slice_xy_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	real_t* __restrict__ a_data, real_t* __restrict__ c_data, real_t* __restrict__ y_data,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const scratch_layout_t scratch_l,
	const dim_scratch_layout_t dim_scratch_l, const index_t s_begin, const index_t s_end, const index_t y_begin,
	const index_t y_end, const index_t tid, const index_t coop_size, index_t& epoch, std::atomic<long>& counter)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		// const real_t az_s = az[s];
		// const real_t b1z_s = b1z[s];

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);

		const auto a_scratch = noarr::make_bag(scratch_l, a_data);
		const auto c_scratch = noarr::make_bag(scratch_l, c_data);

		const auto y_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), y_data);

		real_t cy_tmp = ay_s;

		{
			const auto d = noarr::make_bag(dens_l, densities);

			auto y_remainder = (y_end - y_begin) % simd_length;

			xy_fused_transpose_part<true, false, simd_t>(
				d, t, simd_length, y_begin, y_end - y_remainder, 0, s, ax_s, b1x_s, ay_s, b1y_s, cx,
				inner_y_data_blocked { y_begin, cy_tmp, a_scratch, c_scratch });
		}

		// y remainder
		{
			const auto d = noarr::make_bag(dens_l, densities);

			auto y_remainder = (y_end - y_begin) % simd_length;

			for (index_t y = y_end - y_remainder; y < y_end; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, s, 0, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward<true, false>(d, cx, s, 0, y, prev, ay_s, b1y_s, inside_data_blocked { y_begin, c_scratch });

				y_forward_inside_x_blocked(d, s, 0, y, 0, prev, ay_s, b1y_s, y_begin, c_scratch);

				y_forward_inside_x_blocked_next(s, y, ay_s, b1y_s, y_len, y_begin, a_scratch, c_scratch);
			}
		}

		const auto d = noarr::make_bag(dens_l, densities);

		y_backward_blocked(d, 0, y_begin, y_end, a_scratch, c_scratch, s);

		{
			epoch++;

			auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

			while (val < epoch * coop_size)
			{
				val = counter.load(std::memory_order_acquire);
			}
		}

		// #pragma omp barrier

		y_blocked_middle(d, 0, a_scratch, c_scratch, y_scratch, tid, coop_size, s);

		// #pragma omp barrier

		{
			epoch++;

			auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

			while (val < epoch * coop_size)
			{
				val = counter.load(std::memory_order_acquire);
			}
		}

		y_blocked_end(d, 0, a_scratch, c_scratch, y_begin, y_end, s);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t>
constexpr static void solve_slice_xy_fused_transpose_blocked_alt(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	real_t* __restrict__ a_data, real_t* __restrict__ c_data, real_t* __restrict__ y_data,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const scratch_layout_t scratch_l,
	const dim_scratch_layout_t dim_scratch_l, const index_t s_begin, const index_t s_end, const index_t y_begin,
	const index_t y_end, const index_t tid, const index_t coop_size, index_t& epoch, std::atomic<long>& counter)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto d = noarr::make_bag(dens_l, densities);

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);

		const auto a_scratch = noarr::make_bag(scratch_l, a_data);
		const auto c_scratch = noarr::make_bag(scratch_l, c_data);

		const auto y_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), y_data);

		{
			real_t ay_tmp = ay_s;
			real_t cy_tmp = ay_s;

			const auto a_tmp = ay_s * (y_begin == 0 ? 0 : 1);
			const auto b_tmp = b1y_s + ((y_begin == 0) || (y_begin == y_len - 1) ? ay_s : 0);
			const auto c_tmp = ay_s * (y_begin == y_len - 1 ? 0 : 1);

			real_t ay0_tmp = a_tmp / b_tmp;
			real_t cy0_tmp = c_tmp / b_tmp;

			auto y_remainder = (y_end - y_begin) % simd_length;

			xy_fused_transpose_part<true, true, simd_t>(
				d, t, simd_length, y_begin, y_end - y_remainder, 0, s, ax_s, b1x_s, ay_s, b1y_s, cx,
				inner_y_data_blocked_alt { y_begin, y_end, ay_tmp, cy_tmp, ay0_tmp, cy0_tmp, a_scratch, c_scratch });
		}

		// y remainder
		{
			const auto d = noarr::make_bag(dens_l, densities);

			auto y_remainder = (y_end - y_begin) % simd_length;

			for (index_t y = y_end - y_remainder; y < y_end; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, s, 0, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward<true, true>(d, cx, s, 0, y, prev, ay_s, b1y_s,
									   inside_data_blocked_alt { y_begin, y_end, a_scratch, c_scratch });

				y_forward_inside_x_blocked_alt(d, s, 0, y, 0, prev, ay_s, b1y_s, y_begin, y_end, a_scratch, c_scratch);

				y_forward_inside_x_blocked_next_alt(s, y, ay_s, b1y_s, y_len, y_begin, y_end, a_scratch, c_scratch);
			}
		}

		{
			epoch++;

			auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

			while (val < epoch * coop_size)
			{
				val = counter.load(std::memory_order_acquire);
			}
		}

		y_blocked_middle(d, 0, a_scratch, c_scratch, y_scratch, tid, coop_size, s);

		{
			epoch++;

			auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

			while (val < epoch * coop_size)
			{
				val = counter.load(std::memory_order_acquire);
			}
		}

		y_blocked_end_alt(d, 0, a_scratch, c_scratch, y_begin, y_end, s);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ az, const real_t* __restrict__ b1z, real_t* __restrict__ ay_data,
	real_t* __restrict__ cy_data, real_t* __restrict__ az_data, real_t* __restrict__ cz_data,
	real_t* __restrict__ dim_data, const density_layout_t dens_l, const diagonal_layout_t diagx_l,
	const scratch_layout_t scratchy_l, const scratch_layout_t scratchz_l, const dim_scratch_layout_t dim_scratch_l,
	const index_t s_begin, const index_t s_end, const index_t y_begin, const index_t y_end, const index_t z_begin,
	const index_t z_end, const index_t tid_y, const index_t coop_size_y, const index_t tid_z, const index_t coop_size_z,
	index_t& epoch_y, std::atomic<long>& counter_y, index_t& epoch_z, std::atomic<long>& counter_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto d = noarr::make_bag(dens_l, densities);

	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	const auto ay_scratch = noarr::make_bag(scratchy_l, ay_data);
	const auto cy_scratch = noarr::make_bag(scratchy_l, cy_data);

	const auto az_scratch = noarr::make_bag(scratchz_l, az_data);
	const auto cz_scratch = noarr::make_bag(scratchz_l, cz_data);

	const auto dim_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), dim_data);

	auto cx = noarr::make_bag(diagx_l, back_cx);

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		for (index_t z = z_begin; z < z_end; z++)
		{
			real_t cy_tmp = ay_s;

			{
				const auto d = noarr::make_bag(dens_l, densities);

				auto y_remainder = (y_end - y_begin) % simd_length;

				xy_fused_transpose_part<true, false, simd_t>(
					d, t, simd_length, y_begin, y_end - y_remainder, z, s, ax_s, b1x_s, ay_s, b1y_s, cx,
					inner_y_data_blocked { y_begin, cy_tmp, ay_scratch, cy_scratch });
			}

			// y remainder
			{
				const auto d = noarr::make_bag(dens_l, densities);

				auto y_remainder = (y_end - y_begin) % simd_length;

				for (index_t y = y_end - y_remainder; y < y_end; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<true, false>(d, cx, s, z, y, prev, ay_s, b1y_s,
											inside_data_blocked { y_begin, cy_scratch });

					y_forward_inside_x_blocked(d, s, z, y, 0, prev, ay_s, b1y_s, y_begin, cy_scratch);

					y_forward_inside_x_blocked_next(s, y, ay_s, b1y_s, y_len, y_begin, ay_scratch, cy_scratch);
				}
			}

			const auto d = noarr::make_bag(dens_l, densities);

			y_backward_blocked(d, z, y_begin, y_end, ay_scratch, cy_scratch, s);

			{
				epoch_y++;

				auto val = counter_y.fetch_add(1, std::memory_order_acq_rel) + 1;

				while (val < epoch_y * coop_size_y)
				{
					val = counter_y.load(std::memory_order_acquire);
				}
			}

			y_blocked_middle(d, z, ay_scratch, cy_scratch, dim_scratch, tid_y, coop_size_y, s);

			{
				epoch_y++;

				auto val = counter_y.fetch_add(1, std::memory_order_acq_rel) + 1;

				while (val < epoch_y * coop_size_y)
				{
					val = counter_y.load(std::memory_order_acquire);
				}
			}

			y_blocked_end(d, z, ay_scratch, cy_scratch, y_begin, y_end, s, az_s, b1z_s,
						  inside_data_blocked { z_begin, cz_scratch });

			for (index_t x = 0; x < x_len; x++)
			{
				z_forward_inside_y_blocked(d, s, z, y_begin, x, d.template at<'s', 'x', 'y', 'z'>(s, x, y_begin, z),
										   az_s, b1z_s, z_begin, cz_scratch);

				z_forward_inside_y_blocked(d, s, z, y_end - 1, x, d.template at<'s', 'x', 'y', 'z'>(s, x, y_end - 1, z),
										   az_s, b1z_s, z_begin, cz_scratch);
			}

			z_forward_inside_y_blocked_next(s, z, az_s, b1z_s, z_len, z_begin, az_scratch, cz_scratch);
		}

		z_backward_blocked(d, y_begin, y_end, z_begin, z_end, az_scratch, cz_scratch, s);

		{
			epoch_z++;

			auto val = counter_z.fetch_add(1, std::memory_order_acq_rel) + 1;

			while (val < epoch_z * coop_size_z)
			{
				val = counter_z.load(std::memory_order_acquire);
			}
		}

		// #pragma omp barrier

		z_blocked_middle(d, az_scratch, cz_scratch, dim_scratch, tid_z, coop_size_z, y_begin, y_end, s);

		// #pragma omp barrier


		{
			epoch_z++;

			auto val = counter_z.fetch_add(1, std::memory_order_acq_rel) + 1;

			while (val < epoch_z * coop_size_z)
			{
				val = counter_z.load(std::memory_order_acquire);
			}
		}

		z_blocked_end(d, az_scratch, cz_scratch, y_begin, y_end, z_begin, z_end, s);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename dim_scratch_layout_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked_alt(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ az, const real_t* __restrict__ b1z, real_t* __restrict__ ay_data,
	real_t* __restrict__ cy_data, real_t* __restrict__ az_data, real_t* __restrict__ cz_data,
	real_t* __restrict__ dim_data, const density_layout_t dens_l, const diagonal_layout_t diagx_l,
	const scratch_layout_t scratchy_l, const scratch_layout_t scratchz_l, const dim_scratch_layout_t dim_scratch_l,
	const index_t s_begin, const index_t s_end, const index_t y_begin, const index_t y_end, const index_t z_begin,
	const index_t z_end, const index_t tid_y, const index_t coop_size_y, const index_t tid_z, const index_t coop_size_z,
	index_t& epoch_y, std::atomic<long>& counter_y, index_t& epoch_z, std::atomic<long>& counter_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto d = noarr::make_bag(dens_l, densities);

	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const auto ay_scratch = noarr::make_bag(scratchy_l, ay_data);
	const auto cy_scratch = noarr::make_bag(scratchy_l, cy_data);

	const auto az_scratch = noarr::make_bag(scratchz_l, az_data);
	const auto cz_scratch = noarr::make_bag(scratchz_l, cz_data);

	const auto dim_scratch = noarr::make_bag(dim_scratch_l ^ noarr::fix<'t'>(get_thread_num()), dim_data);

	auto cx = noarr::make_bag(diagx_l, back_cx);

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const real_t az_s = az[s];
		const real_t b1z_s = b1z[s];

		for (index_t z = z_begin; z < z_end; z++)
		{
			{
				real_t ay_tmp = ay_s;
				real_t cy_tmp = ay_s;

				const auto a_tmp = ay_s * (y_begin == 0 ? 0 : 1);
				const auto b_tmp = b1y_s + ((y_begin == 0) || (y_begin == y_len - 1) ? ay_s : 0);
				const auto c_tmp = ay_s * (y_begin == y_len - 1 ? 0 : 1);

				real_t ay0_tmp = a_tmp / b_tmp;
				real_t cy0_tmp = c_tmp / b_tmp;

				auto y_remainder = (y_end - y_begin) % simd_length;

				xy_fused_transpose_part<true, true, simd_t>(
					d, t, simd_length, y_begin, y_end - y_remainder, z, s, ax_s, b1x_s, ay_s, b1y_s, cx,
					inner_y_data_blocked_alt { y_begin, y_end, ay_tmp, cy_tmp, ay0_tmp, cy0_tmp, ay_scratch,
											   cy_scratch });
			}

			// y remainder
			{
				const auto d = noarr::make_bag(dens_l, densities);

				auto y_remainder = (y_end - y_begin) % simd_length;

				for (index_t y = y_end - y_remainder; y < y_end; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward<true, true>(d, cx, s, z, y, prev, ay_s, b1y_s,
										   inside_data_blocked_alt { y_begin, y_end, ay_scratch, cy_scratch });

					y_forward_inside_x_blocked_alt(d, s, z, y, 0, prev, ay_s, b1y_s, y_begin, y_end, ay_scratch,
												   cy_scratch);

					y_forward_inside_x_blocked_next_alt(s, y, ay_s, b1y_s, y_len, y_begin, y_end, ay_scratch,
														cy_scratch);
				}
			}

			{
				epoch_y++;

				auto val = counter_y.fetch_add(1, std::memory_order_acq_rel) + 1;
				counter_y.notify_all();

				while (val < epoch_y * coop_size_y)
				{
					counter_y.wait(val, std::memory_order_acquire);
					val = counter_y.load(std::memory_order_acquire);
				}
			}

			y_blocked_middle(d, z, ay_scratch, cy_scratch, dim_scratch, tid_y, coop_size_y, s);

			{
				epoch_y++;

				auto val = counter_y.fetch_add(1, std::memory_order_acq_rel) + 1;
				counter_y.notify_all();

				while (val < epoch_y * coop_size_y)
				{
					counter_y.wait(val, std::memory_order_acquire);
					val = counter_y.load(std::memory_order_acquire);
				}
			}

			y_blocked_end_alt_vectorized<simd_t>(dens_l, densities, t, simd_length, z, ay_scratch, cy_scratch, y_begin,
												 y_end, s, az_s, b1z_s,
												 inside_data_blocked_alt { z_begin, z_end, az_scratch, cz_scratch });

			z_forward_inside_y_blocked_next_alt(s, z, az_s, b1z_s, z_len, z_begin, z_end, az_scratch, cz_scratch);
		}

		{
			epoch_z++;

			auto val = counter_z.fetch_add(1, std::memory_order_acq_rel) + 1;
			counter_z.notify_all();

			while (val < epoch_z * coop_size_z)
			{
				counter_z.wait(val, std::memory_order_acquire);
				val = counter_z.load(std::memory_order_acquire);
			}
		}

		z_blocked_middle(d, az_scratch, cz_scratch, dim_scratch, tid_z, coop_size_z, y_begin, y_end, s);

		{
			epoch_z++;

			auto val = counter_z.fetch_add(1, std::memory_order_acq_rel) + 1;
			counter_z.notify_all();

			while (val < epoch_z * coop_size_z)
			{
				counter_z.wait(val, std::memory_order_acquire);
				val = counter_z.load(std::memory_order_acquire);
			}
		}

		z_blocked_end_alt(d, az_scratch, cz_scratch, y_begin, y_end, z_begin, z_end, s);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::solve_x()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::solve_y()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::solve_z()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::solve()
{
	if (!(cores_division_[0] == 1 && cores_division_[1] == 1 && cores_division_[2] == 1))
	{
		if (this->problem_.dims == 2)
		{
			solve_blocked_2d();
		}
		else if (this->problem_.dims == 3)
		{
			solve_blocked_3d();
		}

		return;
	}

#pragma omp parallel for schedule(static)
	for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
	{
		auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

		for (index_t i = 0; i < this->problem_.iterations; i++)
		{
			if (this->problem_.dims == 3)
				solve_slice_xyz_fused_transpose<index_t>(
					this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, az_, b1z_, cz_, get_substrates_layout<3>(),
					get_diagonal_layout(this->problem_, this->problem_.nx),
					get_diagonal_layout(this->problem_, this->problem_.ny),
					get_diagonal_layout(this->problem_, this->problem_.nz), s, s + s_step_length);
			else if (this->problem_.dims == 2)
				solve_slice_xy_fused_transpose<index_t>(
					this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<3>(),
					get_diagonal_layout(this->problem_, this->problem_.nx),
					get_diagonal_layout(this->problem_, this->problem_.ny), s, s + s_step_length);
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::solve_blocked_2d()
{
	for (index_t i = 0; i < countersy_count_; i++)
	{
		countersy_[i].value = 0;
	}

#pragma omp parallel
	{
		const index_t s_len = this->problem_.substrates_count;

		index_t epoch = 0;

		const index_t tid = get_thread_num();

		const index_t substrate_group_tid = tid % cores_division_[1];
		const index_t substrate_group = tid / cores_division_[1];

		const auto block_y_begin = group_block_offsetsy_[substrate_group_tid];
		const auto block_y_end = block_y_begin + group_block_lengthsy_[substrate_group_tid];


		const index_t group_size = cores_division_[1];

		for (index_t s = substrate_step_ * substrate_group; s < s_len; s += substrate_step_ * substrate_groups_)
		{
			auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

			// #pragma omp critical
			// 			std::cout << "Thread " << tid << " s_begin: " << s << " s_end: " << s + s_step_length
			// 					  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
			// 					  << " group: " << substrate_group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (!use_alt_blocked_)
					solve_slice_xy_fused_transpose_blocked<index_t>(
						this->substrates_, ax_, b1x_, cx_, ay_, b1y_, a_scratchy_, c_scratchy_, dim_scratch_,
						get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nx),
						get_scratch_layout(this->problem_.ny, substrate_groups_) ^ noarr::fix<'l'>(substrate_group),
						get_dim_scratch_layout(), s, s + s_step_length, block_y_begin, block_y_end, substrate_group_tid,
						group_size, epoch, countersy_[substrate_group].value);
				else
					solve_slice_xy_fused_transpose_blocked_alt<index_t>(
						this->substrates_, ax_, b1x_, cx_, ay_, b1y_, a_scratchy_, c_scratchy_, dim_scratch_,
						get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nx),
						get_scratch_layout(this->problem_.ny, substrate_groups_) ^ noarr::fix<'l'>(substrate_group),
						get_dim_scratch_layout(), s, s + s_step_length, block_y_begin, block_y_end, substrate_group_tid,
						group_size, epoch, countersy_[substrate_group].value);
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f<real_t, aligned_x>::solve_blocked_3d()
{
	for (index_t i = 0; i < countersy_count_; i++)
	{
		countersy_[i].value = 0;
	}

	for (index_t i = 0; i < countersz_count_; i++)
	{
		countersz_[i].value = 0;
	}

#pragma omp parallel
	{
		const index_t s_len = this->problem_.substrates_count;

		index_t epoch_y = 0;
		index_t epoch_z = 0;

		const index_t tid = get_thread_num();
		const index_t group_size = cores_division_[1] * cores_division_[2];

		const index_t substrate_group_tid = tid % group_size;
		const index_t substrate_group = tid / group_size;

		const auto tid_y = (substrate_group_tid / cores_division_[0]) % cores_division_[1];
		const auto tid_z = substrate_group_tid / (cores_division_[0] * cores_division_[1]);

		const auto block_y_begin = group_block_offsetsy_[tid_y];
		const auto block_y_end = block_y_begin + group_block_lengthsy_[tid_y];

		const auto block_z_begin = group_block_offsetsz_[tid_z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid_z];

		const auto lane_id_y = tid_z + substrate_group * cores_division_[2];
		const auto lane_scratchy_l =
			get_scratch_layout(this->problem_.ny, substrate_groups_ * cores_division_[2]) ^ noarr::fix<'l'>(lane_id_y);

		const auto lane_id_z = tid_y + substrate_group * cores_division_[1];
		const auto lane_scratchz_l =
			get_scratch_layout(this->problem_.nz, substrate_groups_ * cores_division_[1]) ^ noarr::fix<'l'>(lane_id_z);

		for (index_t s = substrate_step_ * substrate_group; s < s_len; s += substrate_step_ * substrate_groups_)
		{
			auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

			// #pragma omp critical
			// 			std::cout << "Thread " << tid << " [0, " << tid_y << ", " << tid_z << "] s_begin: " << s
			// 					  << " s_end: " << s + s_step_length << " block_y_begin: " << block_y_begin
			// 					  << " block_y_end: " << block_y_end << " block_z_begin: " << block_z_begin
			// 					  << " block_z_end: " << block_z_end << " group: " << substrate_group << " lane_y: " <<
			// lane_id_y
			// 					  << " lane_z: " << lane_id_z << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (cores_division_[1] == 1)
					if (!use_alt_blocked_)
						solve_slice_xyz_fused_transpose_blocked<index_t>(
							this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, az_, b1z_, a_scratchz_, c_scratchz_,
							dim_scratch_, get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.nx),
							get_diagonal_layout(this->problem_, this->problem_.ny), lane_scratchz_l,
							get_dim_scratch_layout(), s, s + s_step_length, block_z_begin, block_z_end, tid_z,
							cores_division_[2], epoch_z, countersz_[lane_id_z].value);
					else
						solve_slice_xyz_fused_transpose_blocked_alt<index_t>(
							this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, az_, b1z_, a_scratchz_, c_scratchz_,
							dim_scratch_, get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.nx),
							get_diagonal_layout(this->problem_, this->problem_.ny), lane_scratchz_l,
							get_dim_scratch_layout(), s, s + s_step_length, block_z_begin, block_z_end, tid_z,
							cores_division_[2], epoch_z, countersz_[lane_id_z].value);
				else
				{
					if (!use_alt_blocked_)
						solve_slice_xyz_fused_transpose_blocked<index_t>(
							this->substrates_, ax_, b1x_, cx_, ay_, b1y_, az_, b1z_, a_scratchy_, c_scratchy_,
							a_scratchz_, c_scratchz_, dim_scratch_, get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.nx), lane_scratchy_l, lane_scratchz_l,
							get_dim_scratch_layout(), s, s + s_step_length, block_y_begin, block_y_end, block_z_begin,
							block_z_end, tid_y, cores_division_[1], tid_z, cores_division_[2], epoch_y,
							countersy_[lane_id_y].value, epoch_z, countersz_[lane_id_z].value);
					else
						solve_slice_xyz_fused_transpose_blocked_alt<index_t>(
							this->substrates_, ax_, b1x_, cx_, ay_, b1y_, az_, b1z_, a_scratchy_, c_scratchy_,
							a_scratchz_, c_scratchz_, dim_scratch_, get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.nx), lane_scratchy_l, lane_scratchz_l,
							get_dim_scratch_layout(), s, s + s_step_length, block_y_begin, block_y_end, block_z_begin,
							block_z_end, tid_y, cores_division_[1], tid_z, cores_division_[2], epoch_y,
							countersy_[lane_id_y].value, epoch_z, countersz_[lane_id_z].value);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_f<real_t, aligned_x>::least_memory_thomas_solver_d_f(bool use_alt_blocked)
	: ax_(nullptr),
	  b1x_(nullptr),
	  cx_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  cy_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  cz_(nullptr),
	  a_scratchy_(nullptr),
	  c_scratchy_(nullptr),
	  a_scratchz_(nullptr),
	  c_scratchz_(nullptr),
	  dim_scratch_(nullptr),
	  use_alt_blocked_(use_alt_blocked)
{}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_f<real_t, aligned_x>::~least_memory_thomas_solver_d_f()
{
	if (cx_)
	{
		std::free(cx_);
		std::free(ax_);
		std::free(b1x_);
	}
	if (cy_)
	{
		std::free(cy_);
		std::free(ay_);
		std::free(b1y_);
	}
	if (cz_)
	{
		std::free(cz_);
		std::free(az_);
		std::free(b1z_);
	}

	if (a_scratchy_)
	{
		std::free(ay_);
		std::free(b1y_);
		std::free(a_scratchy_);
		std::free(c_scratchy_);
	}

	if (a_scratchz_)
	{
		std::free(az_);
		std::free(b1z_);
		std::free(a_scratchz_);
		std::free(c_scratchz_);
	}

	if (dim_scratch_)
	{
		std::free(dim_scratch_);
	}
}

template class least_memory_thomas_solver_d_f<float, true>;
template class least_memory_thomas_solver_d_f<double, true>;
