#include "co_thomas_solver.h"

#include <atomic>
#include <cstddef>
#include <iostream>

#include "omp_helper.h"
#include "vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, real_t*& c, index_t shape,
															index_t dims, index_t n)
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
void co_thomas_solver<real_t, aligned_x>::prepare(const max_problem_t& problem)
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

int t = 0;
int division_size = 32;

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	substrate_step_ =
		params.contains("substrate_step") ? (index_t)params["substrate_step"] : this->problem_.substrates_count;

	if (use_intrinsics_)
	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		x_tile_size_ = (x_tile_size_ + hn::Lanes(d) - 1) / hn::Lanes(d) * hn::Lanes(d);
		std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
		alignment_size_ = std::max(alignment_size_, vector_length * x_tile_size_ / hn::Lanes(d));
	}

	if (params.contains("t"))
		t = (int)params["t"];

	if (params.contains("div"))
		division_size = (int)params["div"];
}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, cx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, cy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, cz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);

	{
		const auto threads = get_max_threads();

		const index_t threads_per_s = threads / this->problem_.substrates_count;

		max_cores_groups_ = threads / threads_per_s;
	}

	auto scratch_layout = get_scratch_layout();

	a_scratch_ = (real_t*)std::malloc((scratch_layout | noarr::get_size()));
	c_scratch_ = (real_t*)std::malloc((scratch_layout | noarr::get_size()));

	countersz_count_ = max_cores_groups_;
	countersz_ = std::make_unique<aligned_atomic<long>[]>(countersz_count_);
}

template <typename real_t, bool aligned_x>
auto co_thomas_solver<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n)
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
	const real_t a_s, b1_s;
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
static void z_backward(const density_bag_t d, const diag_bag_t c, const index_t s)

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

template <typename index_t, typename real_t, typename scratch_bag_t>
struct inside_data_blocked
{
	const index_t begin;
	const real_t a_s, b1_s;
	scratch_bag_t c;
};

template <typename index_t, typename real_t, typename density_bag_t, typename scratch_bag_t>
static void z_forward_inside_y_blocked(const density_bag_t d, const index_t s, const index_t z, const index_t y,
									   const index_t x, const index_t z_begin, const real_t a_s, const real_t b1_s,
									   scratch_bag_t c, real_t data)

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

template <typename vec_t, typename tag, typename index_t, typename real_t, typename density_bag_t,
		  typename scratch_bag_t>
static void z_forward_inside_y_vectorized_blocked(const density_bag_t d, tag t, const index_t s, const index_t z,
												  const index_t y, const index_t z_begin, const real_t a_s,
												  const real_t b1_s, scratch_bag_t c, vec_t data)

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
static void z_forward_inside_y_blocked_next(const index_t s, const index_t z, const index_t z_len,
											const index_t z_begin, const real_t a_s, const real_t b1_s, scratch_bag_t a,
											scratch_bag_t c)

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
static void z_backward_blocked(const density_bag_t d, const real_t z_begin, const real_t z_end, const scratch_bag_t a,
							   const scratch_bag_t c, const index_t s)

{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();

	// Process the upper diagonal (backward)
	index_t i;
	for (i = z_end - 3; i >= z_begin + 1; i--)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		for (index_t y = 0; y < y_len; y++)
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', 'z'>(s, x, y, i) -=
					c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y, i + 1);


				// std::cout << "b0: " << i << " " << y << " " << x << " " << d.template at<'s', 'x', 'y', 'z'>(s, x, y,
				// i)
				// 		  << std::endl;
			}

		a[state] = a[state] - c[state] * a[next_state];
		c[state] = 0 - c[state] * c[next_state];
	}

	// Process the first row (backward)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto next_state = noarr::idx<'s', 'i'>(s, i + 1);

		const auto r = 1 / (1 - c[state] * a[next_state]);

		for (index_t y = 0; y < y_len; y++)
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', 'z'>(s, x, y, i) =
					r
					* (d.template at<'s', 'x', 'y', 'z'>(s, x, y, i)
					   - c[state] * d.template at<'s', 'x', 'y', 'z'>(s, x, y, i + 1));

				// std::cout << "b1: " << i << " " << y << " " << x << " " << d.template at<'s', 'x', 'y', 'z'>(s, x, y,
				// i)
				// 		  << std::endl;
			}

		a[state] = r * a[state];
		c[state] = r * (0 - c[state] * c[next_state]);
	}
}


template <typename index_t, typename density_bag_t, typename scratch_bag_t>
static void z_blocked_middle(density_bag_t d, scratch_bag_t a, scratch_bag_t c, const index_t block_size,
							 const index_t s)
{
	constexpr char dim = 'z';
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();
	const index_t n = d | noarr::get_length<dim>();

	const index_t blocks_count = (n + block_size - 1) / block_size;

	// const index_t block_tid = get_thread_num() % blocks_count;

	// const index_t y_block_size = (y_len + blocks_count - 1) / blocks_count;
	// const index_t y_begin = y_block_size * block_tid;
	// const index_t y_end = std::min(y_begin + y_block_size, y_len);


	const index_t y_begin = 0;
	const index_t y_end = y_len;


	// #pragma omp critical
	// std::cout << "Thread " << block_tid << " s_begin: " << s << " block_y_begin: " << y_begin
	// 		  << " block_y_end: " << y_end << std::endl;

	auto get_i = [block_size, n](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto i = block_idx * block_size + (equation_idx % 2) * (block_size - 1);
		return std::min(i, n - 1);
	};

	for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'s', 'i'>(s, i);
		const auto prev_state = noarr::idx<'s', 'i'>(s, prev_i);

		const auto r = 1 / (1 - a[state] * c[prev_state]);

		c[state] *= r;

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

	for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
					d.template at<'s', 'x', 'y', dim>(s, x, y, i)
					- c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, next_i);
			}
		}
	}
}

template <typename index_t, typename density_bag_t, typename scratch_bag_t>
static void z_blocked_end(density_bag_t d, scratch_bag_t a, scratch_bag_t c, const index_t z_begin, const index_t z_end,
						  const index_t s)
{
	constexpr char dim = 'z';
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();

	for (index_t i = z_begin + 1; i < z_end - 1; i++)
	{
		const auto state = noarr::idx<'s', 'i'>(s, i);

		for (index_t y = 0; y < y_len; y++)
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

template <bool update_c, typename index_t, typename real_t, typename density_bag_t>
static void y_forward_inside_x(const density_bag_t d, const index_t s, const index_t z, const index_t y,
							   const index_t x, const real_t a_s, const real_t b1_s, real_t& c_tmp, real_t data)

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

template <typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t>
static void y_backward(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z,
					   inside_data<real_t> z_data)

{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();

	for (index_t i = y_len - 2; i >= 0; i--)
	{
		const auto back_c = c.template at<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'z', 'y', 'x'>(s, z, i, x) -= back_c * d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x);

			z_forward_inside_y(d, s, z, i + 1, x, z_data.a_s, z_data.b1_s, z_data.c_tmp,
							   d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x));
		}
	}
}

template <typename index_t, typename density_bag_t, typename diag_bag_t, typename z_data_t>
static void y_backward_blocked(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z,
							   z_data_t z_data)

{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();

	for (index_t i = y_len - 2; i >= 0; i--)
	{
		const auto back_c = c.template at<'s', 'i'>(s, i);

		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'s', 'z', 'y', 'x'>(s, z, i, x) -= back_c * d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x);

			z_forward_inside_y_blocked(d, s, z, i + 1, x, z_data.begin, z_data.a_s, z_data.b1_s, z_data.c,
									   d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x));
		}
	}
}

template <typename index_t, typename real_t, typename density_bag_t>
static void x_forward(const density_bag_t d, const index_t s, const index_t z, const index_t y, const real_t a_s,
					  const real_t b1_s, real_t& a_tmp, real_t& b_tmp, real_t& c_tmp, real_t& prev)

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

template <typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t>
static void x_backward(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z, const index_t y,
					   real_t& prev, inside_data<real_t> y_data)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = x_len - 2; i >= 0; i--)
	{
		real_t curr = d.template at<'s', 'z', 'x', 'y'>(s, z, i, y);
		curr = curr - c.template at<'s', 'i'>(s, i) * prev;
		// d.template at<'s', 'z', 'x', 'y'>(s, z, i, y) = curr;

		y_forward_inside_x<false>(d, s, z, y, i + 1, y_data.a_s, y_data.b1_s, y_data.c_tmp, prev);

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

				x_backward(d, cx, s, z, y, prev, inside_data<real_t> { ay_s, b1y_s, c_tmp_y });

				y_forward_inside_x<true>(d, s, z, y, 0, ay_s, b1y_s, c_tmp_y, prev);
			}

			y_backward(d, cy, s, z, inside_data<real_t> { az_s, b1z_s, c_tmp_z });


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

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_xyz_fused_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ back_cy, const real_t* __restrict__ az, const real_t* __restrict__ b1z,
	real_t* __restrict__ a_data, real_t* __restrict__ c_data, const density_layout_t dens_l,
	const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l, const scratch_layout_t scratch_l, const index_t s,
	const index_t z_begin, const index_t z_end, const index_t z_block_size, index_t& epoch, std::atomic<long>& counter)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const auto d = noarr::make_bag(dens_l, densities);
	const auto cx = noarr::make_bag(diagx_l, back_cx);
	const auto cy = noarr::make_bag(diagy_l, back_cy);

	const auto a = noarr::make_bag(scratch_l, a_data);
	const auto c = noarr::make_bag(scratch_l, c_data);

	const real_t ax_s = ax[s];
	const real_t b1x_s = b1x[s];

	const real_t ay_s = ay[s];
	const real_t b1y_s = b1y[s];

	const real_t az_s = az[s];
	const real_t b1z_s = b1z[s];

	for (index_t z = z_begin; z < z_end; z++)
	{
		real_t c_tmp_y = ay_s;

		for (index_t y = 0; y < y_len; y++)
		{
			real_t a_tmp = 0;
			real_t b_tmp = b1x_s + ax_s;
			real_t c_tmp = ax_s;
			real_t prev = 0;

			x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

			x_backward(d, cx, s, z, y, prev, inside_data<real_t> { ay_s, b1y_s, c_tmp_y });

			y_forward_inside_x<true>(d, s, z, y, 0, ay_s, b1y_s, c_tmp_y, prev);
		}

		y_backward_blocked(d, cy, s, z, inside_data_blocked { z_begin, az_s, b1z_s, c });

		{
			for (index_t x = 0; x < x_len; x++)
			{
				z_forward_inside_y_blocked(d, s, z, 0, x, z_begin, az_s, b1z_s, c,
										   d.template at<'s', 'z', 'y', 'x'>(s, z, 0, x));
			}

			z_forward_inside_y_blocked_next(s, z, z_len, z_begin, az_s, b1z_s, a, c);
		}
	}

	z_backward_blocked(d, z_begin, z_end, a, c, s);

	index_t cooperating_threads = (z_len + z_block_size - 1) / z_block_size;

	{
		auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

		while (val < epoch * cooperating_threads)
		{
			val = counter.load(std::memory_order_acquire);
		}
	}

	// #pragma omp barrier

	if (z_begin == 0)
		z_blocked_middle(d, a, c, z_block_size, s);

	// #pragma omp barrier

	epoch++;

	{
		auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

		while (val < epoch * cooperating_threads)
		{
			val = counter.load(std::memory_order_acquire);
		}
	}

	z_blocked_end(d, a, c, z_begin, z_end, s);
}

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
constexpr static real_t z_forward_inside_y_vectorized(const density_layout_t d, simd_tag t, const index_t z,
													  const index_t y, const index_t z_len, const real_t az_s,
													  const real_t b1z_s, const real_t cz_tmp, vec_t data)
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
													const index_t y, const index_t x, const index_t s,
													const index_t length, const index_t y_len, const real_t ay_s,
													const real_t b1y_s, real_t ay_bl_tmp, real_t by_bl_tmp,
													real_t cy_bl_tmp)
{
	vec_t prev;

	if (y == 0)
		prev = hn::Zero(t);
	else
		prev = hn::Load(t, &(d.template at<'z', 'y', 'v', 'x', 's'>(z, y - 1, length - 1, x, s)));

	for (index_t v = 0; v < length; v++)
	{
		const real_t r = 1 / (by_bl_tmp - ay_bl_tmp * cy_bl_tmp);

		rows[v] = hn::Mul(hn::MulAdd(hn::Set(t, -ay_bl_tmp), prev, rows[v]), hn::Set(t, r));
		prev = rows[v];

		ay_bl_tmp = ay_s;
		by_bl_tmp = b1y_s + (y * length + v == y_len - 2 ? ay_s : 0);
		cy_bl_tmp = ay_s * r;
	}
}

template <typename vec_t, typename index_t, typename real_t, typename simd_tag, typename density_layout_t,
		  typename diag_bag_t>
constexpr static void y_backward_vectorized(const density_layout_t dens_l, real_t* __restrict__ densities,
											const diag_bag_t c, simd_tag t, const index_t z, const index_t s,
											const index_t simd_length, const index_t y_len, const index_t z_len,
											const real_t az_s, const real_t b1z_s, real_t& cz_tmp)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

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

			z_forward_inside_y_vectorized(d, t, z, i + 1, z_len, az_s, b1z_s, cz_tmp, prev);

			prev = curr;
		}

		r_z = z_forward_inside_y_vectorized(d, t, z, 0, z_len, az_s, b1z_s, cz_tmp, prev);
	}

	cz_tmp = az_s * r_z;
}

template <typename vec_t, typename index_t, typename real_t, typename simd_tag, typename density_layout_t,
		  typename diag_bag_t, typename scratch_bag_t>
constexpr static void y_backward_vectorized_blocked(const density_layout_t dens_l, real_t* __restrict__ densities,
													const diag_bag_t cy, simd_tag t, const index_t z, const index_t s,
													const index_t simd_length, const index_t y_len,
													const index_t z_begin, const real_t az_s, const real_t b1z_s,
													scratch_bag_t az_scratch, scratch_bag_t cz_scratch)
{
	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();
	const index_t z_len = blocked_dens_l | noarr::get_length<'z'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d = noarr::make_bag(
			blocked_dens_l ^ noarr::fix<'s', 'b', 'X', 'x'>(s, noarr::lit<0>, X, noarr::lit<0>), densities);

		vec_t prev = hn::Load(t, &d.template at<'z', 'y'>(z, y_len - 1));

		for (index_t i = y_len - 2; i >= 0; i--)
		{
			vec_t curr = hn::Load(t, &d.template at<'z', 'y'>(z, i));
			curr = hn::MulAdd(hn::Set(t, -cy.template at<'i'>(i)), prev, curr);
			// hn::Store(curr, t, &d.template at<'z', 'y'>(z, i));

			z_forward_inside_y_vectorized_blocked(d, t, s, z, i + 1, z_begin, az_s, b1z_s, cz_scratch, prev);

			prev = curr;
		}

		z_forward_inside_y_vectorized_blocked(d, t, s, z, 0, z_begin, az_s, b1z_s, cz_scratch, prev);
	}

	z_forward_inside_y_blocked_next(s, z, z_len, z_begin, az_s, b1z_s, az_scratch, cz_scratch);
}

template <typename simd_t, typename simd_tag, typename index_t, typename real_t, typename density_bag_t,
		  typename diag_bag_t>
static constexpr void xy_fused_transpose_part(const density_bag_t d, simd_tag t, const index_t simd_length,
											  const index_t y_len_full, const index_t z, const index_t s,
											  const real_t ax_s, const real_t b1x_s, const real_t ay_s,
											  const real_t b1y_s, const diag_bag_t cx, real_t& cy_tmp)
{
	real_t ay_tmp = 0;
	real_t by_tmp = b1y_s + ay_s;

	const index_t n = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	// vector registers that hold the to be transposed x*y plane
	simd_t* rows = new simd_t[simd_length + 1];

	for (index_t y = 0; y < y_len; y++)
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
				rows[v + 1] = hn::Load(t, &(d.template at<'z', 'y', 'v', 'x', 's'>(z, y, v, i, s)));

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
				hn::Store(rows[v + 1], t, &(d.template at<'z', 'y', 'v', 'x', 's'>(z, y, v, i, s)));
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			rows[0] = prev_x;

			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				rows[v + 1] = hn::Load(t, &(d.template at<'z', 'y', 'v', 'x', 's'>(z, y, v, full_n - simd_length, s)));

			// transposition to enable vectorization
			transpose(rows + 1);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			{
				if (remainder_work == 1)
					bx_tmp = b1x_s + ax_s;

				x_forward_vectorized<true>(rows + 1, t, remainder_work, ax_s, b1x_s, ax_tmp, bx_tmp, cx_tmp);
			}

			// the begin of backward part
			{
				x_backward_vectorized(cx, rows + 1, t, simd_length - 1, full_n - simd_length, s);

				prev_x = rows[1];
			}

			// transposition back to the original form
			transpose(rows + 1);

			y_forward_inside_x_vectorized(d, rows + 1, t, z, y, full_n - simd_length, s, simd_length, y_len_full, ay_s,
										  b1y_s, ay_tmp, by_tmp, cy_tmp);

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(rows[v + 1], t, &(d.template at<'z', 'y', 'v', 'x', 's'>(z, y, v, full_n - simd_length, s)));
		}

		// we continue with backwards substitution
		for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
		{
			rows[simd_length] = prev_x;

			// aligned loads
			for (index_t v = 0; v < simd_length; v++)
				rows[v] = hn::Load(t, &(d.template at<'z', 'y', 'v', 'x', 's'>(z, y, v, i, s)));

			// transposition to enable vectorization
			transpose(rows);

			// backward propagation
			{
				x_backward_vectorized(cx, rows, t, simd_length, i, s);

				prev_x = rows[0];
			}

			// transposition back to the original form
			transpose(rows);


			y_forward_inside_x_vectorized(d, rows, t, z, y, i, s, simd_length, y_len_full, ay_s, b1y_s, ay_tmp, by_tmp,
										  cy_tmp);

			// aligned stores
			for (index_t v = 0; v < simd_length; v++)
				hn::Store(rows[v], t, &(d.template at<'z', 'y', 'v', 'x', 's'>(z, y, v, i, s)));
		}

		for (index_t v = 0; v < simd_length; v++)
		{
			const real_t r = 1 / (by_tmp - ay_tmp * cy_tmp);

			ay_tmp = ay_s;
			by_tmp = b1y_s + (y * simd_length + v == y_len_full - 2 ? ay_s : 0);
			cy_tmp = ay_s * r;
		}
	}

	delete[] rows;
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_xyz_fused_transpose(real_t* __restrict__ densities, const real_t* __restrict__ ax,
											const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
											const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
											const real_t* __restrict__ back_cy, const real_t* __restrict__ az,
											const real_t* __restrict__ b1z, const real_t* __restrict__ back_cz,
											const density_layout_t dens_l, const diagonal_layout_t diagx_l,
											const diagonal_layout_t diagy_l, const diagonal_layout_t diagz_l,
											const index_t s_begin, const index_t s_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t y_len_full = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

#pragma omp for schedule(static) nowait
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
				const auto d = noarr::make_bag(body_dens_l, densities);

				xy_fused_transpose_part<simd_t>(d, t, simd_length, y_len_full, z, s, ax_s, b1x_s, ay_s, b1y_s, cx,
												cy_tmp);
			}

			// y remainder
			{
				const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z), densities);

				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t v_len = rem_dens_l | noarr::get_length<'v'>();

				for (index_t y = y_len_full - v_len; y < y_len_full; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1x_s + ax_s;
					real_t c_tmp = ax_s;
					real_t prev = 0;

					x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

					x_backward(d, cy, s, z, y, prev, inside_data<real_t> { ay_s, b1y_s, cy_tmp });

					y_forward_inside_x<true>(d, s, z, y, 0, ay_s, b1y_s, cy_tmp, prev);
				}
			}

			y_backward_vectorized<simd_t>(dens_l, densities, cy, t, z, s, simd_length, y_len_full, z_len, az_s, b1z_s,
										  cz_tmp);
		}

		z_backward<simd_t>(dens_l, densities, cz, t, s, simd_length, y_len_full, z_len);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_xyz_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
	const real_t* __restrict__ back_cx, const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
	const real_t* __restrict__ back_cy, const real_t* __restrict__ az, const real_t* __restrict__ b1z,
	real_t* __restrict__ a_data, real_t* __restrict__ c_data, const density_layout_t dens_l,
	const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l, const scratch_layout_t scratch_l, const index_t s,
	const index_t z_begin, const index_t z_end, const index_t z_block_size, index_t& epoch, std::atomic<long>& counter)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t y_len_full = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

	const real_t ax_s = ax[s];
	const real_t b1x_s = b1x[s];

	const real_t ay_s = ay[s];
	const real_t b1y_s = b1y[s];

	const real_t az_s = az[s];
	const real_t b1z_s = b1z[s];

	auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
	auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

	const auto a_scratch = noarr::make_bag(scratch_l, a_data);
	const auto c_scratch = noarr::make_bag(scratch_l, c_data);

	for (index_t z = z_begin; z < z_end; z++)
	{
		real_t cy_tmp = ay_s;

		{
			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const auto d = noarr::make_bag(body_dens_l, densities);

			xy_fused_transpose_part<simd_t>(d, t, simd_length, y_len_full, z, s, ax_s, b1x_s, ay_s, b1y_s, cx, cy_tmp);
		}

		// y remainder
		{
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z), densities);

			auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t v_len = rem_dens_l | noarr::get_length<'v'>();

			for (index_t y = y_len_full - v_len; y < y_len_full; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward(d, cy, s, z, y, prev, inside_data<real_t> { ay_s, b1y_s, cy_tmp });

				y_forward_inside_x<true>(d, s, z, y, 0, ay_s, b1y_s, cy_tmp, prev);
			}
		}

		y_backward_vectorized_blocked<simd_t>(dens_l, densities, cy, t, z, s, simd_length, y_len_full, z_begin, az_s,
											  b1z_s, a_scratch, c_scratch);
	}

	const auto d = noarr::make_bag(dens_l, densities);

	z_backward_blocked(d, z_begin, z_end, a_scratch, c_scratch, s);

	index_t cooperating_threads = (z_len + z_block_size - 1) / z_block_size;

	{
		auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

		while (val < epoch * cooperating_threads)
		{
			val = counter.load(std::memory_order_acquire);
		}
	}

	// #pragma omp barrier

	if (z_begin == 0)
		z_blocked_middle(d, a_scratch, c_scratch, z_block_size, s);

	// #pragma omp barrier

	epoch++;

	{
		auto val = counter.fetch_add(1, std::memory_order_acq_rel) + 1;

		while (val < epoch * cooperating_threads)
		{
			val = counter.load(std::memory_order_acquire);
		}
	}

	z_blocked_end(d, a_scratch, c_scratch, z_begin, z_end, s);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_2d_naive(real_t* __restrict__ densities, const real_t* __restrict__ ax,
						   const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
						   const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
						   const real_t* __restrict__ back_cy, const density_layout_t dens_l,
						   const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l)
{
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

#pragma omp parallel for
	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			const real_t a_s = ax[s];
			const real_t b1_s = b1x[s];
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y'>(s, y), densities);
			const auto c = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
			constexpr char dim = 'x';

			real_t a_tmp = 0;
			real_t b_tmp = b1_s + a_s;
			real_t c_tmp = a_s;
			real_t prev = 0;

			for (index_t i = 0; i < x_len; i++)
			{
				const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

				real_t curr = d.template at<dim>(i);
				curr = r * (curr - a_tmp * prev);
				d.template at<dim>(i) = curr;

				a_tmp = a_s;
				b_tmp = b1_s + (i == x_len - 2 ? a_s : 0);
				c_tmp = a_s * r;
				prev = curr;
			}

			for (index_t i = x_len - 2; i >= 0; i--)
			{
				real_t curr = d.template at<dim>(i);
				curr = curr - c.template at<'i'>(i) * prev;
				d.template at<dim>(i) = curr;

				prev = curr;
			}
		}

		{
			const real_t a_s = ay[s];
			const real_t b1_s = b1y[s];
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);
			const auto c = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);
			constexpr char dim = 'y';

			real_t c_tmp = a_s;

			{
				const real_t r = 1 / (b1_s + a_s);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(0, x) *= r;
				}

				c_tmp = a_s * r;
			}

			for (index_t i = 1; i < y_len; i++)
			{
				const real_t b_tmp = b1_s + (i == y_len - 1 ? a_s : 0);
				const real_t r = 1 / (b_tmp - a_s * c_tmp);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
				}

				c_tmp = a_s * r;
			}

			for (index_t i = y_len - 2; i >= 0; i--)
			{
				const auto back_c = c.template at<'i'>(i);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
			}
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_2d_tile(real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
						  const real_t* __restrict__ back_cx, const real_t* __restrict__ ay,
						  const real_t* __restrict__ b1y, const real_t* __restrict__ back_cy,
						  const density_layout_t dens_l, const diagonal_layout_t diagx_l,
						  const diagonal_layout_t diagy_l, const index_t x_tile_size)
{
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

#pragma omp parallel for
	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			const real_t a_s = ax[s];
			const real_t b1_s = b1x[s];
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y'>(s, y), densities);
			const auto c = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
			constexpr char dim = 'x';

			real_t a_tmp = 0;
			real_t b_tmp = b1_s + a_s;
			real_t c_tmp = a_s;
			real_t prev = 0;

			for (index_t i = 0; i < x_len; i++)
			{
				const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

				real_t curr = d.template at<dim>(i);
				curr = r * (curr - a_tmp * prev);
				d.template at<dim>(i) = curr;

				a_tmp = a_s;
				b_tmp = b1_s + (i == x_len - 2 ? a_s : 0);
				c_tmp = a_s * r;
				prev = curr;
			}

			for (index_t i = x_len - 2; i >= 0; i--)
			{
				real_t curr = d.template at<dim>(i);
				curr = curr - c.template at<'i'>(i) * prev;
				d.template at<dim>(i) = curr;

				prev = curr;
			}
		}

		auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
		const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

		for (index_t X = 0; X < X_len; X++)
		{
			const real_t a_s = ay[s];
			const real_t b1_s = b1y[s];
			const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'b', 'X'>(s, noarr::lit<0>, X), densities);
			const auto c = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);
			constexpr char dim = 'y';

			const index_t remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
			const index_t x_len_remainder = remainder == 0 ? x_tile_size : remainder;
			const index_t x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

			real_t c_tmp = a_s;

			{
				const real_t r = 1 / (b1_s + a_s);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(0, x) *= r;
				}

				c_tmp = a_s * r;
			}

			for (index_t i = 1; i < y_len; i++)
			{
				const real_t b_tmp = b1_s + (i == y_len - 1 ? a_s : 0);
				const real_t r = 1 / (b_tmp - a_s * c_tmp);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
				}

				c_tmp = a_s * r;
			}

			for (index_t i = y_len - 2; i >= 0; i--)
			{
				const auto back_c = c.template at<'i'>(i);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_2d_fused(real_t* __restrict__ densities, const real_t* __restrict__ ax,
						   const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
						   const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
						   const real_t* __restrict__ back_cy, const density_layout_t dens_l,
						   const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l)
{
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const auto d = noarr::make_bag(dens_l, densities);
	const auto cx = noarr::make_bag(diagx_l, back_cx);
	const auto cy = noarr::make_bag(diagy_l, back_cy);

#pragma omp parallel for
	for (index_t s = 0; s < s_len; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const index_t z = 0;

		{
			real_t c_tmp_y = ay_s;

			for (index_t y = 0; y < y_len; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward(d, cx, s, z, y, prev, inside_data<real_t> { ay_s, b1y_s, c_tmp_y });

				y_forward_inside_x<true>(d, s, z, y, 0, ay_s, b1y_s, c_tmp_y, prev);
			}

			for (index_t i = y_len - 2; i >= 0; i--)
			{
				const auto back_c = cy.template at<'s', 'i'>(s, i);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'z', 'y', 'x'>(s, z, i, x) -=
						back_c * d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x);
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void xf_part(density_layout_t d, const real_t a_s, const real_t b1_s, index_t& c_tmp, index_t x_begin,
					const index_t x_end, const index_t y)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	if (x_begin == 0)
	{
		const real_t r = 1 / (b1_s + a_s);

		d.template at<'y', 'x'>(y, 0) *= r;

		c_tmp = a_s * r;

		x_begin++;
	}

	for (index_t i = x_begin; i < x_end; i++)
	{
		const real_t b_tmp = b1_s + (i == x_len - 1 ? a_s : 0);
		const real_t r = 1 / (b_tmp - a_s * c_tmp);

		d.template at<'y', 'x'>(y, i) = r * (d.template at<'y', 'x'>(y, i) - a_s * d.template at<'y', 'x'>(y, i - 1));

		c_tmp = a_s * r;
	}
}

template <typename index_t, typename density_layout_t, typename diagonal_layout_t>
static void xb_part(density_layout_t d, diagonal_layout_t cx, index_t x_begin, const index_t x_end, const index_t y)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = std::min(x_end - 1, x_len - 2); i >= x_begin; i--)
	{
		const auto back_c = cx.template at<'i'>(i);

		d.template at<'y', 'x'>(y, i) -= back_c * d.template at<'y', 'x'>(y, i + 1);
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void yf_part(density_layout_t d, const real_t a_s, const real_t b1_s, index_t& c_tmp, index_t x_begin,
					const index_t x_end, index_t y_begin, const index_t y_end)
{
	const index_t y_len = d | noarr::get_length<'y'>();

	constexpr char dim = 'y';

	if (y_begin == 0)
	{
		const real_t r = 1 / (b1_s + a_s);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<dim, 'x'>(0, x) *= r;
		}

		c_tmp = a_s * r;

		y_begin++;
	}

	for (index_t i = y_begin; i < y_end; i++)
	{
		const real_t b_tmp = b1_s + (i == y_len - 1 ? a_s : 0);
		const real_t r = 1 / (b_tmp - a_s * c_tmp);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<dim, 'x'>(i, x) =
				r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
		}

		c_tmp = a_s * r;
	}
}

template <typename index_t, typename density_layout_t, typename diagonal_layout_t>
static void yb_part(density_layout_t d, diagonal_layout_t c, index_t x_begin, const index_t x_end, index_t y_begin,
					const index_t y_end)
{
	const index_t y_len = d | noarr::get_length<'y'>();

	constexpr char dim = 'y';

	for (index_t i = std::min(y_end - 1, y_len - 2); i >= y_begin; i--)
	{
		const auto back_c = c.template at<'i'>(i);

		for (index_t x = x_begin; x < x_end; x++)
			d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_xf_block(real_t* __restrict__ densities, const real_t a_s, const real_t b1_s,
						   const density_layout_t dens_l, const index_t s, index_t& c_tmp, index_t x_begin,
						   const index_t x_end, const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t c_tmp_copy = c_tmp;
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp_copy, x_begin, x_begin + x_size / 2, y_begin,
						   y_begin + y_size / 2);
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp_copy, x_begin + x_size / 2, x_end, y_begin,
						   y_begin + y_size / 2);

			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin, x_begin + x_size / 2, y_begin + y_size / 2,
						   y_end);
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin + x_size / 2, x_end, y_begin + y_size / 2,
						   y_end);
		}
		else if (x_size > division_size)
		{
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin, x_begin + x_size / 2, y_begin, y_end);
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin + x_size / 2, x_end, y_begin, y_end);
		}
		else
		{
			index_t c_tmp_copy = c_tmp;
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp_copy, x_begin, x_end, y_begin, y_begin + y_size / 2);
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin, x_end, y_begin + y_size / 2, y_end);
		}

		return;
	}

	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xf" << std::endl;


	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xf_part(d, a_s, b1_s, c_tmp, x_begin, x_end, y);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_xb_yf_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s, const real_t ay_s,
							  const real_t b1y_s, const density_layout_t dens_l, const index_t s, diagonal_layout_t cx,
							  index_t& cx_tmp, index_t& cy_tmp, index_t x_begin, const index_t x_end,
							  const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cx_tmp_copy = cx_tmp;
			index_t cy_tmp_copy = cy_tmp;
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp_copy,
							  x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
							  x_size / 2, y_begin, y_begin + y_size / 2);

			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp_copy,
							  x_begin + x_size / 2, x_end, y_begin + y_size / 2, y_end);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp, x_begin,
							  x_begin + x_size / 2, y_size / 2, y_end);
		}

		return;
	}

	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xb yf" << std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xb_part(d, cx, x_begin, x_end, y);
	}

	yf_part(d, ay_s, b1y_s, cy_tmp, x_begin, x_end, y_begin, y_end);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_xf_xb_yf_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s,
								 const real_t ay_s, const real_t b1y_s, const density_layout_t dens_l, const index_t s,
								 diagonal_layout_t cx, index_t& cx_tmp, index_t& cy_tmp, index_t x_begin,
								 const index_t x_end, const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cx_tmp_copy = cx_tmp;
			index_t cy_tmp_copy = cy_tmp;
			solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, x_begin, x_begin + x_size / 2, y_begin,
						   y_begin + y_size / 2);
			solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
							  x_begin + x_size / 2, y_begin, y_begin + y_size / 2);


			solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, x_begin, x_begin + x_size / 2,
						   y_begin + y_size / 2, y_end);
			solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin + y_size / 2, y_end);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
							  x_begin + x_size / 2, y_begin + y_size / 2, y_end);
		}
		// else if (x_size > division_size)
		// {
		// 	solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp, x_begin, x_begin + x_size / 2, y_begin, y_begin +
		// y_size / 2); 	solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp,
		// x_begin
		// + x_size / 2, x_end, 						 y_begin, y_begin + y_size / 2);
		// }
		// else
		// {
		// 	index_t cx_tmp_copy = cx_tmp;
		// 	solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
		// 						 x_end, y_begin, y_begin + y_size / 2);
		// 	solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp, x_begin, x_end,
		// 						 y_size / 2, y_end);
		// }

		return;
	}

	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xf xb yf" << std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xf_part(d, ax_s, b1x_s, cx_tmp, x_begin, x_end, y);

		xb_part(d, cx, x_begin, x_end, y);
	}

	yf_part(d, ay_s, b1y_s, cy_tmp, x_begin, x_end, y_begin, y_end);
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_xb_yf_yb_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s,
								 const real_t ay_s, const real_t b1y_s, const density_layout_t dens_l, const index_t s,
								 diagonal_layout_t cx, diagonal_layout_t cy, index_t& cx_tmp, index_t& cy_tmp,
								 index_t x_begin, const index_t x_end, const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cx_tmp_copy = cx_tmp;
			index_t cy_tmp_copy = cy_tmp;

			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin + y_size / 2, y_end);
			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp, x_begin,
								 x_begin + x_size / 2, y_begin + y_size / 2, y_end);

			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp, x_begin,
								 x_begin + x_size / 2, y_begin, y_begin + y_size / 2);
		}

		return;
	}


	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xb yf yb" << std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xb_part(d, cx, x_begin, x_end, y);
	}

	yf_part(d, ay_s, b1y_s, cy_tmp, x_begin, x_end, y_begin, y_end);

	yb_part(d, cy, x_begin, x_end, y_begin, y_end);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_yb_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s, const real_t ay_s,
						   const real_t b1y_s, const density_layout_t dens_l, const index_t s, diagonal_layout_t cx,
						   diagonal_layout_t cy, index_t& cx_tmp, index_t& cy_tmp, index_t x_begin, const index_t x_end,
						   const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cy_tmp_copy = cy_tmp;

			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp_copy, x_begin,
						   x_begin + x_size / 2, y_begin + y_size / 2, y_end);
			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp_copy, x_begin,
						   x_begin + x_size / 2, y_begin, y_begin + y_size / 2);

			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp, x_begin + x_size / 2,
						   x_end, y_size / 2, y_end);
			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp, x_begin + x_size / 2,
						   x_end, y_begin, y_begin + y_size / 2);
		}

		return;
	}


	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") yb" << std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	yb_part(d, cy, x_begin, x_end, y_begin, y_end);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_xf_xb_yf_yb_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s,
									const real_t ay_s, const real_t b1y_s, const density_layout_t dens_l,
									const index_t s, diagonal_layout_t cx, diagonal_layout_t cy, index_t& cx_tmp,
									index_t& cy_tmp, index_t x_begin, const index_t x_end, const index_t y_begin,
									const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cx_tmp_copy = cx_tmp;
			index_t cy_tmp_copy = cy_tmp;
			solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, x_begin, x_begin + x_size / 2, y_begin,
						   y_begin + y_size / 2);
			solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
							  x_begin + x_size / 2, y_begin, y_begin + y_size / 2);


			solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, x_begin, x_begin + x_size / 2,
						   y_begin + y_size / 2, y_end);
			solve_xf_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp_copy,
									x_begin + x_size / 2, x_end, y_begin + y_size / 2, y_end);
			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp, x_begin,
								 x_begin + x_size / 2, y_begin + y_size / 2, y_end);

			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp, x_begin,
						   x_begin + x_size / 2, y_begin, y_begin + y_size / 2);

			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp_copy,
						   x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
		}

		return;
	}


	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xf xb yf yb" <<
	// std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xf_part(d, ax_s, b1x_s, cx_tmp, x_begin, x_end, y);

		xb_part(d, cx, x_begin, x_end, y);
	}

	yf_part(d, ay_s, b1y_s, cy_tmp, x_begin, x_end, y_begin, y_end);

	yb_part(d, cy, x_begin, x_end, y_begin, y_end);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_2d_co(real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
						const real_t* __restrict__ back_cx, const real_t* __restrict__ ay,
						const real_t* __restrict__ b1y, const real_t* __restrict__ back_cy,
						const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l)
{
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

#pragma omp parallel for
	for (index_t s = 0; s < s_len; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		index_t cx_tmp = ax_s;
		index_t cy_tmp = ay_s;

		index_t cx_tmp_copy = ax_s;
		index_t cy_tmp_copy = ay_s;

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

		solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp, 0, x_len / 2, 0, y_len / 2);

		solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp, x_len / 2, x_len, 0,
							 y_len / 2);

		solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp_copy, 0, x_len / 2, 0,
						  y_len / 2);

		solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, 0, x_len / 2, y_len / 2, y_len);

		solve_xf_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp, x_len / 2,
								x_len, y_len / 2, y_len);

		solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp_copy, 0,
							 x_len / 2, y_len / 2, y_len);

		solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp_copy, 0, x_len / 2, 0,
					   y_len / 2);

		solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp, x_len / 2, x_len, 0,
					   y_len / 2);
	}
}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve_x()
{}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve_y()
{}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve_z()
{}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve()
{
	if (t == 0)
		solve_2d_naive<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<2>(),
								get_diagonal_layout(this->problem_, this->problem_.nx),
								get_diagonal_layout(this->problem_, this->problem_.ny));


	if (t == 1)
		solve_2d_tile<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<2>(),
							   get_diagonal_layout(this->problem_, this->problem_.nx),
							   get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);

	if (t == 2)
		solve_2d_fused<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<2>(),
								get_diagonal_layout(this->problem_, this->problem_.nx),
								get_diagonal_layout(this->problem_, this->problem_.ny));

	if (t == 3)
		solve_2d_co<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<2>(),
							 get_diagonal_layout(this->problem_, this->problem_.nx),
							 get_diagonal_layout(this->problem_, this->problem_.ny));
}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve_blocked()
{
	for (index_t i = 0; i < countersz_count_; i++)
	{
		countersz_[i].value = 0;
	}

#pragma omp parallel
	{
		const index_t s_len = this->problem_.substrates_count;
		const index_t z_len = this->problem_.nz;

		index_t epoch = 0;

		const auto threads = get_num_threads();

		const auto tid = get_thread_num();

		const index_t threads_per_s = threads / s_len;

		const index_t s = tid / threads_per_s;
		const index_t block_tid = tid % threads_per_s;

		const index_t block_size_z = (z_len + threads_per_s - 1) / threads_per_s;

		const auto block_z_begin = block_tid * block_size_z;
		const auto block_z_end = std::min(block_z_begin + block_size_z, z_len);



		// #pragma omp critical
		// std::cout << "Thread " << tid << " s_begin: " << s << " block_z_begin: " << block_z_begin
		// 		  << " block_z_end: " << block_z_end << std::endl;

		for (index_t i = 0; i < this->problem_.iterations; i++)
		{
			epoch++;

			if (use_intrinsics_)
			{
				solve_slice_xyz_fused_transpose_blocked<index_t>(
					this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, az_, b1z_, a_scratch_, c_scratch_,
					get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nx),
					get_diagonal_layout(this->problem_, this->problem_.ny), get_scratch_layout() ^ noarr::fix<'l'>(s),
					s, block_z_begin, block_z_end, block_size_z, epoch, countersz_[s].value);
			}
			else
			{
				solve_slice_xyz_fused_blocked<index_t>(
					this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, az_, b1z_, a_scratch_, c_scratch_,
					get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nx),
					get_diagonal_layout(this->problem_, this->problem_.ny), get_scratch_layout() ^ noarr::fix<'l'>(s),
					s, block_z_begin, block_z_end, block_size_z, epoch, countersz_[s].value);
			}
		}
	}
}

template <typename real_t, bool aligned_x>
co_thomas_solver<real_t, aligned_x>::co_thomas_solver(bool use_intrinsics, bool use_fused)
	: ax_(nullptr),
	  b1x_(nullptr),
	  cx_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  cy_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  cz_(nullptr),
	  a_scratch_(nullptr),
	  c_scratch_(nullptr),
	  use_intrinsics_(use_intrinsics),
	  use_blocked_(use_fused)
{}

template <typename real_t, bool aligned_x>
co_thomas_solver<real_t, aligned_x>::~co_thomas_solver()
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

	if (a_scratch_)
	{
		std::free(a_scratch_);
		std::free(c_scratch_);
	}
}

template class co_thomas_solver<float, false>;
template class co_thomas_solver<double, false>;

template class co_thomas_solver<float, true>;
template class co_thomas_solver<double, true>;
