#include "cubed_thomas_solver_t.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "noarr/structures/base/structs_common.hpp"
#include "noarr/structures/extra/shortcuts.hpp"
#include "omp_helper.h"

// a helper using for accessing static constexpr variables
using alg = cubed_thomas_solver_t<double, true>;

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, index_t shape, index_t dims,
																 index_t n, index_t& counters_count,
																 std::unique_ptr<aligned_atomic<long>[]>& counters,
																 index_t& max_threads)
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

	// index_t blocks_count = n / block_size_; // TODO: handle remainder
	// max_threads = std::max<index_t>(get_max_threads(), blocks_count);

	// counters_count = (max_threads + blocks_count - 1) / blocks_count;
	// counters = std::make_unique<aligned_atomic<long>[]>(counters_count);

	// for (index_t i = 0; i < counters_count; i++)
	// {
	// 	counters[i].value.store(0, std::memory_order_relaxed);
	// }
}

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void cubed_thomas_solver_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	block_size_ = params.contains("block_size") ? (std::size_t)params["block_size"] : 100;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
}

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, this->problem_.dx, this->problem_.dims, this->problem_.nx, countersx_count_,
						  countersx_, max_threadsx_);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, this->problem_.dy, this->problem_.dims, this->problem_.ny, countersy_count_,
						  countersy_, max_threadsy_);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, this->problem_.dz, this->problem_.dims, this->problem_.nz, countersz_count_,
						  countersz_, max_threadsz_);

	// cores_division_ = { 2, 4, 4 };
	cores_division_ = { 4, 4 };

	auto scratch_layout = get_scratch_layout();

	a_scratch_ = (real_t*)std::malloc((scratch_layout | noarr::get_size()));
	c_scratch_ = (real_t*)std::malloc((scratch_layout | noarr::get_size()));
}

template <typename real_t, bool aligned_x>
auto cubed_thomas_solver_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n)
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

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_x_start(real_t* __restrict__ densities, const real_t* __restrict__ ac,
								const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
								const density_layout_t dens_l, const scratch_layout_t scratch_l, const bool begin,
								const bool end)
{
	constexpr char dim = 'x';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	const auto tid = get_thread_num();

	const auto tid_x = tid % 4;
	const auto tid_y = tid / 4;

	const index_t block_size_x = 5;
	const index_t block_size_y = 5;

	const auto block_x_begin = tid_x * block_size_x;
	const auto block_x_end = std::min(block_x_begin + block_size_x, 20);
	const auto block_x_len = block_x_end - block_x_begin;

	const auto block_y_begin = tid_y * block_size_y;
	const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
	const auto block_y_len = block_y_end - block_y_begin;

	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			for (index_t y = 0; y < y_len; y++)
			{
				auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), a_data);
				auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), c_data);
				auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y', 'z'>(s, y, z), densities);

				// Normalize the first and the second equation
				index_t i;
				for (i = 0; i < 2; i++)
				{
					const auto state = noarr::idx<dim>(i);

					const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
					const auto b_tmp = b1[s] + ((begin && i == 0) || (end && i == n - 1) ? ac[s] : 0);
					const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

					a[state] = a_tmp / b_tmp;
					c[state] = c_tmp / b_tmp;

					d[state] /= b_tmp;

#pragma omp critical
					// if (y + block_y_begin == 0 && get_thread_num() < 4)
					{
						std::cout << "F i: " << i + block_x_begin << ", a_tmp: " << a_tmp << ", b_tmp: " << b_tmp
								  << ", c_tmp: " << c_tmp << ", a: " << a[state] << ", c: " << c[state]
								  << ", d: " << d[state] << std::endl;
					}
				}

				// Process the lower diagonal (forward)
				for (; i < n; i++)
				{
					const auto state = noarr::idx<dim>(i);
					const auto prev_state = noarr::idx<dim>(i - 1);

					const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
					const auto b_tmp = b1[s] + (end && i == n - 1 ? ac[s] : 0);
					const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

					const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

					a[state] = r * (0 - a_tmp * a[prev_state]);
					c[state] = r * c_tmp;

					d[state] = r * (d[state] - a_tmp * d[prev_state]);

#pragma omp critical
					// if (y + block_y_begin == 0 && get_thread_num() < 4)
					{
						std::cout << "F i: " << i + block_x_begin << ", a_tmp: " << a_tmp << ", b_tmp: " << b_tmp
								  << ", c_tmp: " << c_tmp << ", a: " << a[state] << ", c: " << c[state]
								  << ", d: " << d[state] << std::endl;
					}
				}

				// Process the upper diagonal (backward)
				for (i = n - 3; i >= 1; i--)
				{
					const auto state = noarr::idx<dim>(i);
					const auto next_state = noarr::idx<dim>(i + 1);

					d[state] -= c[state] * d[next_state];

					a[state] = a[state] - c[state] * a[next_state];
					c[state] = 0 - c[state] * c[next_state];

#pragma omp critical
					// if (y + block_y_begin == 0 && get_thread_num() < 4)
					{
						std::cout << "B i: " << i + block_x_begin << ", a: " << a[state] << ", c: " << c[state]
								  << ", d: " << d[state] << std::endl;
					}
				}

				// Process the first row (backward)
				{
					const auto state = noarr::idx<dim>(i);
					const auto next_state = noarr::idx<dim>(i + 1);

					const auto r = 1 / (1 - c[state] * a[next_state]);

					d[state] = r * (d[state] - c[state] * d[next_state]);

					a[state] = r * a[state];
					c[state] = r * (0 - c[state] * c[next_state]);

#pragma omp critical
					// if (y + block_y_begin == 0 && get_thread_num() < 4)
					{
						std::cout << "B i: " << i + block_x_begin << ", a: " << a[state] << ", c: " << c[state]
								  << ", d: " << d[state] << std::endl;
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_x_middle(real_t* __restrict__ densities, real_t* __restrict__ a_data,
								 real_t* __restrict__ c_data, const density_layout_t dens_l,
								 const scratch_layout_t scratch_l, const index_t block_size)
{
	constexpr char dim = 'x';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	const index_t blocks_count = (n + block_size - 1) / block_size;

	// const auto tid = get_thread_num();

	// const auto tid_x = tid % 4;
	// const auto tid_y = tid / 4;

	// const index_t block_size_x = 5;
	// const index_t block_size_y = 5;

	// const auto block_x_begin = tid_x * block_size_x;
	// const auto block_x_end = std::min(block_x_begin + block_size_x, 20);
	// const auto block_x_len = block_x_end - block_x_begin;

	// const auto block_y_begin = tid_y * block_size_y;
	// const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
	// const auto block_y_len = block_y_end - block_y_begin;

	// #pragma omp critical
	// 	std::cout << "tid: " << get_thread_num() << " blocks_count: " << blocks_count << std::endl;

	for (index_t s = 0; s < s_len; s++)
	{
		auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), a_data);
		auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), c_data);

		auto get_i = [block_size, n](index_t equation_idx) {
			const index_t block_idx = equation_idx / 2;
			const auto i = block_idx * block_size + (equation_idx % 2) * (block_size - 1);
			return std::min(i, n - 1);
		};

		for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
		{
			const auto state = noarr::idx<dim>(get_i(equation_idx));
			const auto prev_state = noarr::idx<dim>(get_i(equation_idx - 1));

			const auto r = 1 / (1 - a[state] * c[prev_state]);

			c[state] *= r;

			for (index_t z = 0; z < z_len; z++)
			{
				for (index_t y = 0; y < y_len; y++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y', 'z'>(s, y, z), densities);

					d[state] = r * (d[state] - a[state] * d[prev_state]);
				}
			}

			// #pragma omp critical
			// if (y + block_y_begin == 0 && get_thread_num() < 4)
			// {
			// 	std::cout << "TF i: " << get_i(equation_idx) << ", prev_i: " << get_i(equation_idx - 1)
			// 			  << " y: " << y + block_y_begin << ", c: " << c[state] << ", d: " << d[state] << std::endl;
			// }
		}

		for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto state = noarr::idx<dim>(get_i(equation_idx));
			const auto next_state = noarr::idx<dim>(get_i(equation_idx + 1));

			for (index_t z = 0; z < z_len; z++)
			{
				for (index_t y = 0; y < y_len; y++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y', 'z'>(s, y, z), densities);

					d[state] = d[state] - c[state] * d[next_state];
				}
			}

			// #pragma omp critical
			// if (y + block_y_begin == 0 && get_thread_num() < 4)
			// {
			// 	std::cout << "TB i: " << get_i(equation_idx) << ", next_i: " << get_i(equation_idx + 1)
			// 			  << " y: " << y + block_y_begin << ", c: " << c[state] << ", d: " << d[state] << std::endl;
			// }
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_x_end(real_t* __restrict__ densities, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
							  const density_layout_t dens_l, const scratch_layout_t scratch_l)
{
	constexpr char dim = 'x';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();


	const auto tid = get_thread_num();

	const auto tid_x = tid % 4;
	const auto tid_y = tid / 4;

	const index_t block_size_x = 5;
	const index_t block_size_y = 5;

	const auto block_x_begin = tid_x * block_size_x;
	const auto block_x_end = std::min(block_x_begin + block_size_x, 20);
	const auto block_x_len = block_x_end - block_x_begin;

	const auto block_y_begin = tid_y * block_size_y;
	const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
	const auto block_y_len = block_y_end - block_y_begin;

	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			for (index_t y = 0; y < y_len; y++)
			{
				auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), a_data);
				auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), c_data);
				auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y', 'z'>(s, y, z), densities);

				// Final part of modified thomas algorithm
				// Solve the rest of the unknowns
				{
					for (index_t i = 1; i < n - 1; i++)
					{
						const auto first_state = noarr::idx<dim>(0);
						const auto state = noarr::idx<dim>(i);
						const auto last_state = noarr::idx<dim>(n - 1);

						d[state] = d[state] - a[state] * d[first_state] - c[state] * d[last_state];

#pragma omp critical
						// if (y + block_y_begin == 0 && get_thread_num() < 4)
						{
							std::cout << "C i: " << i + block_x_begin << ", a: " << a[state] << ", c: " << c[state]
									  << ", d: " << d[state] << std::endl;
						}
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_y_start(real_t* __restrict__ densities, const real_t* __restrict__ ac,
								const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
								const density_layout_t dens_l, const scratch_layout_t scratch_l, const bool begin,
								const bool end)
{
	constexpr char dim = 'y';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), a_data);
			auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), c_data);

			// Normalize the first and the second equation
			index_t i;
			for (i = 0; i < 2; i++)
			{
				const auto state = noarr::idx<dim>(i);

				const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + ((begin && i == 0) || (end && i == n - 1) ? ac[s] : 0);
				const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'z'>(s, x, z), densities);

					d[state] /= b_tmp;
				}
			}

			// Process the lower diagonal (forward)
			for (; i < n; i++)
			{
				const auto state = noarr::idx<dim>(i);
				const auto prev_state = noarr::idx<dim>(i - 1);

				const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (end && i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'z'>(s, x, z), densities);

					d[state] = r * (d[state] - a_tmp * d[prev_state]);
				}
			}

			// Process the upper diagonal (backward)
			for (i = n - 3; i >= 1; i--)
			{
				const auto state = noarr::idx<dim>(i);
				const auto next_state = noarr::idx<dim>(i + 1);

				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'z'>(s, x, z), densities);

					d[state] -= c[state] * d[next_state];
				}

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<dim>(i);
				const auto next_state = noarr::idx<dim>(i + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'z'>(s, x, z), densities);

					d[state] = r * (d[state] - c[state] * d[next_state]);
				}

				a[state] = r * a[state];
				c[state] = r * (0 - c[state] * c[next_state]);
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_y_middle(real_t* __restrict__ densities, real_t* __restrict__ a_data,
								 real_t* __restrict__ c_data, const density_layout_t dens_l,
								 const scratch_layout_t scratch_l, const index_t block_size)
{
	constexpr char dim = 'y';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	const index_t blocks_count = (n + block_size - 1) / block_size;

	for (index_t s = 0; s < s_len; s++)
	{
		auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), a_data);
		auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), c_data);

		auto get_i = [block_size, n](index_t equation_idx) {
			const index_t block_idx = equation_idx / 2;
			const auto i = block_idx * block_size + (equation_idx % 2) * (block_size - 1);
			return std::min(i, n - 1);
		};

		for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
		{
			const auto state = noarr::idx<dim>(get_i(equation_idx));
			const auto prev_state = noarr::idx<dim>(get_i(equation_idx - 1));

			const auto r = 1 / (1 - a[state] * c[prev_state]);

			c[state] *= r;

			for (index_t z = 0; z < z_len; z++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'z'>(s, x, z), densities);

					d[state] = r * (d[state] - a[state] * d[prev_state]);
				}
			}
		}

		for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto state = noarr::idx<dim>(get_i(equation_idx));
			const auto next_state = noarr::idx<dim>(get_i(equation_idx + 1));

			for (index_t z = 0; z < z_len; z++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'z'>(s, x, z), densities);

					d[state] = d[state] - c[state] * d[next_state];
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_y_end(real_t* __restrict__ densities, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
							  const density_layout_t dens_l, const scratch_layout_t scratch_l)
{
	constexpr char dim = 'y';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), a_data);
			auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), c_data);

			// Final part of modified thomas algorithm
			// Solve the rest of the unknowns
			{
				for (index_t i = 1; i < n - 1; i++)
				{
					const auto first_state = noarr::idx<dim>(0);
					const auto state = noarr::idx<dim>(i);
					const auto last_state = noarr::idx<dim>(n - 1);

					for (index_t x = 0; x < x_len; x++)
					{
						auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'z'>(s, x, z), densities);

						d[state] = d[state] - a[state] * d[first_state] - c[state] * d[last_state];
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_z_start(real_t* __restrict__ densities, const real_t* __restrict__ ac,
								const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
								const density_layout_t dens_l, const scratch_layout_t scratch_l)
{
	constexpr char dim = 'z';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	for (index_t s = 0; s < s_len; s++)
	{
		auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), a_data);
		auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), c_data);

		// Normalize the first and the second equation
		index_t i;
		for (i = 0; i < 2; i++)
		{
			const auto state = noarr::idx<dim>(i);

			const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
			const auto b_tmp = b1[s] + (i == 0 || i == n - 1 ? ac[s] : 0);
			const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

			a[state] = a_tmp / b_tmp;
			c[state] = c_tmp / b_tmp;

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), densities);

					d[state] /= b_tmp;
				}
			}
		}

		// Process the lower diagonal (forward)
		for (; i < n; i++)
		{
			const auto state = noarr::idx<dim>(i);
			const auto prev_state = noarr::idx<dim>(i - 1);

			const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
			const auto b_tmp = b1[s] + (i == n - 1 ? ac[s] : 0);
			const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

			const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

			a[state] = r * (0 - a_tmp * a[prev_state]);
			c[state] = r * c_tmp;

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), densities);

					d[state] = r * (d[state] - a_tmp * d[prev_state]);
				}
			}
		}

		// Process the upper diagonal (backward)
		for (i = n - 3; i >= 1; i--)
		{
			const auto state = noarr::idx<dim>(i);
			const auto next_state = noarr::idx<dim>(i + 1);

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), densities);

					d[state] -= c[state] * d[next_state];
				}
			}

			a[state] = a[state] - c[state] * a[next_state];
			c[state] = 0 - c[state] * c[next_state];
		}

		// Process the first row (backward)
		{
			const auto state = noarr::idx<dim>(i);
			const auto next_state = noarr::idx<dim>(i + 1);

			const auto r = 1 / (1 - c[state] * a[next_state]);

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), densities);

					d[state] = r * (d[state] - c[state] * d[next_state]);
				}
			}

			a[state] = r * a[state];
			c[state] = r * (0 - c[state] * c[next_state]);
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_z_middle(real_t* __restrict__ densities, real_t* __restrict__ a_data,
								 real_t* __restrict__ c_data, const density_layout_t dens_l,
								 const scratch_layout_t scratch_l, const index_t block_size)
{
	constexpr char dim = 'z';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	const index_t blocks_count = (n + block_size - 1) / block_size;

	for (index_t s = 0; s < s_len; s++)
	{
		auto a = noarr::make_bag(scratch_l ^ noarr::fix<'s'>(s), a_data);
		auto c = noarr::make_bag(scratch_l ^ noarr::fix<'s'>(s), c_data);

		auto get_i = [block_size, n](index_t equation_idx) {
			const index_t block_idx = equation_idx / 2;
			const auto i = block_idx * block_size + (equation_idx % 2) * (block_size - 1);
			return std::min(i, n - 1);
		};

		for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
		{
			const auto state = noarr::idx<dim>(get_i(equation_idx));
			const auto prev_state = noarr::idx<dim>(get_i(equation_idx - 1));

			const auto r = 1 / (1 - a[state] * c[prev_state]);

			c[state] *= r;

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), densities);

					d[state] = r * (d[state] - a[state] * d[prev_state]);
				}
			}
		}

		for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto state = noarr::idx<dim>(get_i(equation_idx));
			const auto next_state = noarr::idx<dim>(get_i(equation_idx + 1));

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), densities);

					d[state] = d[state] - c[state] * d[next_state];
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_z_end(real_t* __restrict__ densities, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
							  const density_layout_t dens_l, const scratch_layout_t scratch_l)
{
	constexpr char dim = 'z';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	for (index_t s = 0; s < s_len; s++)
	{
		auto a = noarr::make_bag(scratch_l ^ noarr::fix<'s'>(s), a_data);
		auto c = noarr::make_bag(scratch_l ^ noarr::fix<'s'>(s), c_data);

		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		{
			for (index_t i = 1; i < n - 1; i++)
			{
				const auto first_state = noarr::idx<dim>(0);
				const auto state = noarr::idx<dim>(i);
				const auto last_state = noarr::idx<dim>(n - 1);

				for (index_t y = 0; y < y_len; y++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'x', 'y'>(s, x, y), densities);

						d[state] = d[state] - a[state] * d[first_state] - c[state] * d[last_state];
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_2d(real_t* __restrict__ densities, const density_layout_t dens_l, const real_t* __restrict__ acx,
					 const real_t* __restrict__ b1x, const real_t* __restrict__ acy, const real_t* __restrict__ b1y,
					 real_t* __restrict__ a_data, real_t* __restrict__ c_data, const scratch_layout_t scratch_l,
					 std::array<index_t, 3> cores_division)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t block_size_x = (x_len + cores_division[0] - 1) / cores_division[0];
	const index_t block_size_y = (y_len + cores_division[1] - 1) / cores_division[1];

#pragma omp parallel num_threads(cores_division[0] * cores_division[1])
	{
		const auto tid = get_thread_num();

		const auto tid_x = tid % cores_division[0];
		const auto tid_y = tid / cores_division[0];

		const auto block_x_begin = tid_x * block_size_x;
		const auto block_x_end = std::min(block_x_begin + block_size_x, x_len);
		const auto block_x_len = block_x_end - block_x_begin;

		const auto block_y_begin = tid_y * block_size_y;
		const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
		const auto block_y_len = block_y_end - block_y_begin;

		const auto thread_dens_l =
			dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len) ^ noarr::slice<'y'>(block_y_begin, block_y_len);

		// Do X
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_y);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_x_begin, block_x_len);

			const auto lane_dens_l = dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_len);

			solve_block_x_start<index_t>(densities, acx, b1x, a_data, c_data, thread_dens_l, thread_scratch_l,
										 tid_x == 0, tid_x == cores_division[0] - 1);

#pragma omp barrier

			if (tid_x == 0)
				solve_block_x_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_x);

#pragma omp barrier

			solve_block_x_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
		}

#pragma omp barrier

		// Do Y
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_x);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_y_begin, block_y_len);

			const auto lane_dens_l = dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len);

			solve_block_y_start<index_t>(densities, acy, b1y, a_data, c_data, thread_dens_l, thread_scratch_l,
										 tid_y == 0, tid_y == cores_division[1] - 1);

#pragma omp barrier

			if (tid_y == 0)
				solve_block_y_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_y);

#pragma omp barrier

			solve_block_y_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const density_layout_t dens_l,
									const real_t* __restrict__ ac, const real_t* __restrict__ b1,
									real_t* __restrict__ a_data, real_t* __restrict__ c_data,
									const scratch_layout_t scratch_l, std::array<index_t, 3> cores_division)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t block_size_x = (x_len + cores_division[0] - 1) / cores_division[0];
	const index_t block_size_y = (y_len + cores_division[1] - 1) / cores_division[1];

	// std::cout << "block_size_x: " << block_size_x << std::endl;
	// std::cout << "block_size_y: " << block_size_y << std::endl;

#pragma omp parallel num_threads(cores_division[0] * cores_division[1])
	{
		const auto tid = get_thread_num();

		const auto tid_x = tid % cores_division[0];
		const auto tid_y = tid / cores_division[0];

		const auto block_x_begin = tid_x * block_size_x;
		const auto block_x_end = std::min(block_x_begin + block_size_x, x_len);
		const auto block_x_len = block_x_end - block_x_begin;

		const auto block_y_begin = tid_y * block_size_y;
		const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
		const auto block_y_len = block_y_end - block_y_begin;

		// #pragma omp critical
		// 		std::cout << "tid: " << tid << " tid_x: " << tid_x << " block_x_begin: " << block_x_begin
		// 				  << " block_x_end: " << block_x_end << " block_x_len: " << block_x_len << " tid_y: " << tid_y
		// 				  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
		// 				  << " block_y_len: " << block_y_len << std::endl;

		const auto thread_dens_l =
			dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len) ^ noarr::slice<'y'>(block_y_begin, block_y_len);

		// Do X
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_y);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_x_begin, block_x_len);

			const auto lane_dens_l = dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_len);

			solve_block_x_start<index_t>(densities, ac, b1, a_data, c_data, thread_dens_l, thread_scratch_l, tid_x == 0,
										 tid_x == cores_division[0] - 1);

#pragma omp barrier

			if (tid_x == 0)
				solve_block_x_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_x);

#pragma omp barrier

			solve_block_x_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_y_2d_and_3d(real_t* __restrict__ densities, const density_layout_t dens_l,
									const real_t* __restrict__ ac, const real_t* __restrict__ b1,
									real_t* __restrict__ a_data, real_t* __restrict__ c_data,
									const scratch_layout_t scratch_l, std::array<index_t, 3> cores_division)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t block_size_x = (x_len + cores_division[0] - 1) / cores_division[0];
	const index_t block_size_y = (y_len + cores_division[1] - 1) / cores_division[1];

#pragma omp parallel num_threads(cores_division[0] * cores_division[1])
	{
		const auto tid = get_thread_num();

		const auto tid_x = tid % cores_division[0];
		const auto tid_y = tid / cores_division[0];

		const auto block_x_begin = tid_x * block_size_x;
		const auto block_x_end = std::min(block_x_begin + block_size_x, x_len);
		const auto block_x_len = block_x_end - block_x_begin;

		const auto block_y_begin = tid_y * block_size_y;
		const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
		const auto block_y_len = block_y_end - block_y_begin;

		// #pragma omp critical
		// 		std::cout << "tid: " << tid << " tid_x: " << tid_x << " tid_y: " << tid_y << " block_x_begin: " <<
		// block_x_begin
		// 				  << " block_x_end: " << block_x_end << " block_x_len: " << block_x_len
		// 				  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
		// 				  << " block_y_len: " << block_y_len << std::endl;

		const auto thread_dens_l =
			dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len) ^ noarr::slice<'y'>(block_y_begin, block_y_len);

		// Do Y
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_x);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_y_begin, block_y_len);

			const auto lane_dens_l = dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len);

			solve_block_y_start<index_t>(densities, ac, b1, a_data, c_data, thread_dens_l, thread_scratch_l, tid_y == 0,
										 tid_y == cores_division[1] - 1);

#pragma omp barrier

			if (tid_y == 0)
				solve_block_y_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_y);

#pragma omp barrier

			solve_block_y_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
		}
	}
}

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::solve_x()
{
	if (this->problem_.dims == 1)
	{
		return;
	}
	else
	{
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, get_substrates_layout<3>(), ax_, b1x_, a_scratch_,
										 c_scratch_, get_scratch_layout(), cores_division_);
	}
}

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::solve_y()
{
	solve_slice_y_2d_and_3d<index_t>(this->substrates_, get_substrates_layout<3>(), ay_, b1y_, a_scratch_, c_scratch_,
									 get_scratch_layout(), cores_division_);
}

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::solve_z()
{}

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
		return;
	}
	else if (this->problem_.dims == 2)
	{
		solve_2d(this->substrates_, get_substrates_layout<3>(), ax_, b1x_, ay_, b1y_, a_scratch_, c_scratch_,
				 get_scratch_layout(), cores_division_);
	}
	else {}
}

template <typename real_t, bool aligned_x>
cubed_thomas_solver_t<real_t, aligned_x>::cubed_thomas_solver_t()
	: ax_(nullptr),
	  b1x_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  a_scratch_(nullptr),
	  c_scratch_(nullptr)
{}

template <typename real_t, bool aligned_x>
cubed_thomas_solver_t<real_t, aligned_x>::~cubed_thomas_solver_t()
{
	if (b1x_)
	{
		std::free(ax_);
		std::free(b1x_);
	}
	if (b1y_)
	{
		std::free(ay_);
		std::free(b1y_);
	}
	if (b1z_)
	{
		std::free(az_);
		std::free(b1z_);
	}

	if (a_scratch_)
	{
		std::free(a_scratch_);
		std::free(c_scratch_);
	}
}

template class cubed_thomas_solver_t<float, false>;
template class cubed_thomas_solver_t<double, false>;

template class cubed_thomas_solver_t<float, true>;
template class cubed_thomas_solver_t<double, true>;
