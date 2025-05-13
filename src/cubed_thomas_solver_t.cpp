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
																 index_t, index_t&,
																 std::unique_ptr<aligned_atomic<long>[]>&, index_t&)
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

bool do_sync = true;

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	cores_division_ = params.contains("cores_division") ? (std::array<index_t, 3>)params["cores_division"]
														: std::array<index_t, 3> { 2, 2, 2 };


	if (params.contains("sync"))
		do_sync = (bool)params["sync"];
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

	if (this->problem_.dims == 2)
		cores_division_[2] = 1;

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

	auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), a_data);
	auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), c_data);
	auto d = noarr::make_bag(dens_l, densities);

	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			for (index_t y = 0; y < y_len; y++)
			{
				// Normalize the first and the second equation
				index_t i;
				for (i = 0; i < 2; i++)
				{
					const auto state = noarr::idx<'s', dim>(s, i);

					const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
					const auto b_tmp = b1[s] + ((begin && i == 0) || (end && i == n - 1) ? ac[s] : 0);
					const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

					a[state] = a_tmp / b_tmp;
					c[state] = c_tmp / b_tmp;

					d.template at<'s', 'y', 'z', dim>(s, y, z, i) /= b_tmp;
				}

				// Process the lower diagonal (forward)
				for (; i < n; i++)
				{
					const auto state = noarr::idx<'s', dim>(s, i);
					const auto prev_state = noarr::idx<'s', dim>(s, i - 1);

					const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
					const auto b_tmp = b1[s] + (end && i == n - 1 ? ac[s] : 0);
					const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

					const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

					a[state] = r * (0 - a_tmp * a[prev_state]);
					c[state] = r * c_tmp;

					d.template at<'s', 'y', 'z', dim>(s, y, z, i) =
						r
						* (d.template at<'s', 'y', 'z', dim>(s, y, z, i)
						   - a_tmp * d.template at<'s', 'y', 'z', dim>(s, y, z, i - 1));
				}

				// Process the upper diagonal (backward)
				for (i = n - 3; i >= 1; i--)
				{
					const auto state = noarr::idx<'s', dim>(s, i);
					const auto next_state = noarr::idx<'s', dim>(s, i + 1);

					d.template at<'s', 'y', 'z', dim>(s, y, z, i) -=
						c[state] * d.template at<'s', 'y', 'z', dim>(s, y, z, i + 1);

					a[state] = a[state] - c[state] * a[next_state];
					c[state] = 0 - c[state] * c[next_state];
				}

				// Process the first row (backward)
				{
					const auto state = noarr::idx<'s', dim>(s, i);
					const auto next_state = noarr::idx<'s', dim>(s, i + 1);

					const auto r = 1 / (1 - c[state] * a[next_state]);

					d.template at<'s', 'y', 'z', dim>(s, y, z, i) =
						r
						* (d.template at<'s', 'y', 'z', dim>(s, y, z, i)
						   - c[state] * d.template at<'s', 'y', 'z', dim>(s, y, z, i + 1));

					a[state] = r * a[state];
					c[state] = r * (0 - c[state] * c[next_state]);
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

	auto get_i = [block_size, n](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto i = block_idx * block_size + (equation_idx % 2) * (block_size - 1);
		return std::min(i, n - 1);
	};

	for (index_t s = 0; s < s_len; s++)
	{
		auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), a_data);
		auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>() ^ noarr::fix<'s'>(s), c_data);

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

	auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), a_data);
	auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), c_data);
	auto d = noarr::make_bag(dens_l, densities);

	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			for (index_t y = 0; y < y_len; y++)
			{
				// Final part of modified thomas algorithm
				// Solve the rest of the unknowns
				{
					for (index_t i = 1; i < n - 1; i++)
					{
						const auto state = noarr::idx<'s', dim>(s, i);

						d.template at<'s', 'y', 'z', dim>(s, y, z, i) =
							d.template at<'s', 'y', 'z', dim>(s, y, z, i)
							- a[state] * d.template at<'s', 'y', 'z', dim>(s, y, z, 0)
							- c[state] * d.template at<'s', 'y', 'z', dim>(s, y, z, n - 1);
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

	auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), a_data);
	auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), c_data);
	auto d = noarr::make_bag(dens_l, densities);

	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			// Normalize the first and the second equation
			index_t i;
			for (i = 0; i < 2; i++)
			{
				const auto state = noarr::idx<'s', dim>(s, i);

				const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + ((begin && i == 0) || (end && i == n - 1) ? ac[s] : 0);
				const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'x', 'z', dim>(s, x, z, i) /= b_tmp;
				}
			}

			// Process the lower diagonal (forward)
			for (; i < n; i++)
			{
				const auto state = noarr::idx<'s', dim>(s, i);
				const auto prev_state = noarr::idx<'s', dim>(s, i - 1);

				const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (end && i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'x', 'z', dim>(s, x, z, i) =
						r
						* (d.template at<'s', 'x', 'z', dim>(s, x, z, i)
						   - a_tmp * d.template at<'s', 'x', 'z', dim>(s, x, z, i - 1));
				}
			}

			// Process the upper diagonal (backward)
			for (i = n - 3; i >= 1; i--)
			{
				const auto state = noarr::idx<'s', dim>(s, i);
				const auto next_state = noarr::idx<'s', dim>(s, i + 1);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'x', 'z', dim>(s, x, z, i) -=
						c[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, i + 1);
				}

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<'s', dim>(s, i);
				const auto next_state = noarr::idx<'s', dim>(s, i + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'x', 'z', dim>(s, x, z, i) =
						r
						* (d.template at<'s', 'x', 'z', dim>(s, x, z, i)
						   - c[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, i + 1));
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

	auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), a_data);
	auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), c_data);
	auto d = noarr::make_bag(dens_l, densities);

	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			// Final part of modified thomas algorithm
			// Solve the rest of the unknowns
			{
				for (index_t i = 1; i < n - 1; i++)
				{
					const auto state = noarr::idx<'s', dim>(s, i);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<'s', 'x', 'z', dim>(s, x, z, i) =
							d.template at<'s', 'x', 'z', dim>(s, x, z, i)
							- a[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, 0)
							- c[state] * d.template at<'s', 'x', 'z', dim>(s, x, z, n - 1);
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_block_z_start(real_t* __restrict__ densities, const real_t* __restrict__ ac,
								const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
								const density_layout_t dens_l, const scratch_layout_t scratch_l, const bool begin,
								const bool end)
{
	constexpr char dim = 'z';
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), a_data);
	auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), c_data);
	auto d = noarr::make_bag(dens_l, densities);

	for (index_t s = 0; s < s_len; s++)
	{
		// Normalize the first and the second equation
		index_t i;
		for (i = 0; i < 2; i++)
		{
			const auto state = noarr::idx<'s', dim>(s, i);

			const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
			const auto b_tmp = b1[s] + ((begin && i == 0) || (end && i == n - 1) ? ac[s] : 0);
			const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

			a[state] = a_tmp / b_tmp;
			c[state] = c_tmp / b_tmp;

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'x', 'y', dim>(s, x, y, i) /= b_tmp;
				}
			}
		}

		// Process the lower diagonal (forward)
		for (; i < n; i++)
		{
			const auto state = noarr::idx<'s', dim>(s, i);
			const auto prev_state = noarr::idx<'s', dim>(s, i - 1);

			const auto a_tmp = ac[s] * (begin && i == 0 ? 0 : 1);
			const auto b_tmp = b1[s] + (end && i == n - 1 ? ac[s] : 0);
			const auto c_tmp = ac[s] * (end && i == n - 1 ? 0 : 1);

			const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

			a[state] = r * (0 - a_tmp * a[prev_state]);
			c[state] = r * c_tmp;

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
						r
						* (d.template at<'s', 'x', 'y', dim>(s, x, y, i)
						   - a_tmp * d.template at<'s', 'x', 'y', dim>(s, x, y, i - 1));
				}
			}
		}

		// Process the upper diagonal (backward)
		for (i = n - 3; i >= 1; i--)
		{
			const auto state = noarr::idx<'s', dim>(s, i);
			const auto next_state = noarr::idx<'s', dim>(s, i + 1);

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'x', 'y', dim>(s, x, y, i) -=
						c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, i + 1);
				}
			}

			a[state] = a[state] - c[state] * a[next_state];
			c[state] = 0 - c[state] * c[next_state];
		}

		// Process the first row (backward)
		{
			const auto state = noarr::idx<'s', dim>(s, i);
			const auto next_state = noarr::idx<'s', dim>(s, i + 1);

			const auto r = 1 / (1 - c[state] * a[next_state]);

			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
						r
						* (d.template at<'s', 'x', 'y', dim>(s, x, y, i)
						   - c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, i + 1));
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

	auto a = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), a_data);
	auto c = noarr::make_bag(scratch_l ^ noarr::rename<'i', dim>(), c_data);
	auto d = noarr::make_bag(dens_l, densities);

	for (index_t s = 0; s < s_len; s++)
	{
		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		{
			for (index_t i = 1; i < n - 1; i++)
			{
				const auto state = noarr::idx<'s', dim>(s, i);

				for (index_t y = 0; y < y_len; y++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<'s', 'x', 'y', dim>(s, x, y, i) =
							d.template at<'s', 'x', 'y', dim>(s, x, y, i)
							- a[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, 0)
							- c[state] * d.template at<'s', 'x', 'y', dim>(s, x, y, n - 1);
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_2d_and_3d(real_t* __restrict__ densities, const density_layout_t dens_l,
							const real_t* __restrict__ acx, const real_t* __restrict__ b1x,
							const real_t* __restrict__ acy, const real_t* __restrict__ b1y,
							const real_t* __restrict__ acz, const real_t* __restrict__ b1z, real_t* __restrict__ a_data,
							real_t* __restrict__ c_data, const scratch_layout_t scratch_l,
							std::array<index_t, 3> cores_division)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const index_t block_size_x = (x_len + cores_division[0] - 1) / cores_division[0];
	const index_t block_size_y = (y_len + cores_division[1] - 1) / cores_division[1];
	const index_t block_size_z = (z_len + cores_division[2] - 1) / cores_division[2];

#pragma omp parallel num_threads(cores_division[0] * cores_division[1] * cores_division[2])
	{
		const auto tid = get_thread_num();

		const auto tid_x = tid % cores_division[0];
		const auto tid_y = (tid / cores_division[0]) % cores_division[1];
		const auto tid_z = tid / (cores_division[0] * cores_division[1]);

		const auto block_x_begin = tid_x * block_size_x;
		const auto block_x_end = std::min(block_x_begin + block_size_x, x_len);
		const auto block_x_len = block_x_end - block_x_begin;

		const auto block_y_begin = tid_y * block_size_y;
		const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
		const auto block_y_len = block_y_end - block_y_begin;

		const auto block_z_begin = tid_z * block_size_z;
		const auto block_z_end = std::min(block_z_begin + block_size_z, z_len);
		const auto block_z_len = block_z_end - block_z_begin;

		const auto thread_dens_l = dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len)
								   ^ noarr::slice<'y'>(block_y_begin, block_y_len)
								   ^ noarr::slice<'z'>(block_z_begin, block_z_len);

		// do x
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_y * cores_division[2] + tid_z);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_x_begin, block_x_len);

			const auto lane_dens_l =
				dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_len) ^ noarr::slice<'z'>(block_z_begin, block_z_len);

			solve_block_x_start<index_t>(densities, acx, b1x, a_data, c_data, thread_dens_l, thread_scratch_l,
										 tid_x == 0, tid_x == cores_division[0] - 1);

			if (do_sync)
			{
#pragma omp barrier

				if (tid_x == 0)
					solve_block_x_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_x);

#pragma omp barrier
			}
			solve_block_x_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
		}

#pragma omp barrier

		// do Y
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_x * cores_division[2] + tid_z);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_y_begin, block_y_len);

			const auto lane_dens_l =
				dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len) ^ noarr::slice<'z'>(block_z_begin, block_z_len);

			solve_block_y_start<index_t>(densities, acy, b1y, a_data, c_data, thread_dens_l, thread_scratch_l,
										 tid_y == 0, tid_y == cores_division[1] - 1);

			if (do_sync)
			{
#pragma omp barrier

				if (tid_y == 0)
					solve_block_y_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_y);

#pragma omp barrier
			}
			solve_block_y_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
		}

		if (z_len > 1)
		{
#pragma omp barrier

			// do Z
			{
				const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_x * cores_division[1] + tid_y);
				const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_z_begin, block_z_len);

				const auto lane_dens_l = dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len)
										 ^ noarr::slice<'y'>(block_y_begin, block_y_len);

				solve_block_z_start<index_t>(densities, acz, b1z, a_data, c_data, thread_dens_l, thread_scratch_l,
											 tid_z == 0, tid_z == cores_division[2] - 1);

				if (do_sync)
				{
#pragma omp barrier
					if (tid_z == 0)
						solve_block_z_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_z);
#pragma omp barrier
				}
				solve_block_z_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
			}
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
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const index_t block_size_x = (x_len + cores_division[0] - 1) / cores_division[0];
	const index_t block_size_y = (y_len + cores_division[1] - 1) / cores_division[1];
	const index_t block_size_z = (z_len + cores_division[2] - 1) / cores_division[2];

#pragma omp parallel num_threads(cores_division[0] * cores_division[1] * cores_division[2])
	{
		const auto tid = get_thread_num();

		const auto tid_x = tid % cores_division[0];
		const auto tid_y = (tid / cores_division[0]) % cores_division[1];
		const auto tid_z = tid / (cores_division[0] * cores_division[1]);

		const auto block_x_begin = tid_x * block_size_x;
		const auto block_x_end = std::min(block_x_begin + block_size_x, x_len);
		const auto block_x_len = block_x_end - block_x_begin;

		const auto block_y_begin = tid_y * block_size_y;
		const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
		const auto block_y_len = block_y_end - block_y_begin;

		const auto block_z_begin = tid_z * block_size_z;
		const auto block_z_end = std::min(block_z_begin + block_size_z, z_len);
		const auto block_z_len = block_z_end - block_z_begin;

		const auto thread_dens_l = dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len)
								   ^ noarr::slice<'y'>(block_y_begin, block_y_len)
								   ^ noarr::slice<'z'>(block_z_begin, block_z_len);

		// do x
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_y * cores_division[2] + tid_z);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_x_begin, block_x_len);

			const auto lane_dens_l =
				dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_len) ^ noarr::slice<'z'>(block_z_begin, block_z_len);

			solve_block_x_start<index_t>(densities, ac, b1, a_data, c_data, thread_dens_l, thread_scratch_l, tid_x == 0,
										 tid_x == cores_division[0] - 1);

			if (do_sync)
			{
#pragma omp barrier

				if (tid_x == 0)
					solve_block_x_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_x);

#pragma omp barrier
			}

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
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const index_t block_size_x = (x_len + cores_division[0] - 1) / cores_division[0];
	const index_t block_size_y = (y_len + cores_division[1] - 1) / cores_division[1];
	const index_t block_size_z = (z_len + cores_division[2] - 1) / cores_division[2];

#pragma omp parallel num_threads(cores_division[0] * cores_division[1] * cores_division[2])
	{
		const auto tid = get_thread_num();

		const auto tid_x = tid % cores_division[0];
		const auto tid_y = (tid / cores_division[0]) % cores_division[1];
		const auto tid_z = tid / (cores_division[0] * cores_division[1]);

		const auto block_x_begin = tid_x * block_size_x;
		const auto block_x_end = std::min(block_x_begin + block_size_x, x_len);
		const auto block_x_len = block_x_end - block_x_begin;

		const auto block_y_begin = tid_y * block_size_y;
		const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
		const auto block_y_len = block_y_end - block_y_begin;

		const auto block_z_begin = tid_z * block_size_z;
		const auto block_z_end = std::min(block_z_begin + block_size_z, z_len);
		const auto block_z_len = block_z_end - block_z_begin;

		const auto thread_dens_l = dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len)
								   ^ noarr::slice<'y'>(block_y_begin, block_y_len)
								   ^ noarr::slice<'z'>(block_z_begin, block_z_len);

		// do Y
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_x * cores_division[2] + tid_z);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_y_begin, block_y_len);

			const auto lane_dens_l =
				dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len) ^ noarr::slice<'z'>(block_z_begin, block_z_len);

			solve_block_y_start<index_t>(densities, ac, b1, a_data, c_data, thread_dens_l, thread_scratch_l, tid_y == 0,
										 tid_y == cores_division[1] - 1);

			if (do_sync)
			{
#pragma omp barrier

				if (tid_y == 0)
					solve_block_y_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_y);

#pragma omp barrier
			}

			solve_block_y_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_z_2d_and_3d(real_t* __restrict__ densities, const density_layout_t dens_l,
									const real_t* __restrict__ ac, const real_t* __restrict__ b1,
									real_t* __restrict__ a_data, real_t* __restrict__ c_data,
									const scratch_layout_t scratch_l, std::array<index_t, 3> cores_division)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const index_t block_size_x = (x_len + cores_division[0] - 1) / cores_division[0];
	const index_t block_size_y = (y_len + cores_division[1] - 1) / cores_division[1];
	const index_t block_size_z = (z_len + cores_division[2] - 1) / cores_division[2];

#pragma omp parallel num_threads(cores_division[0] * cores_division[1] * cores_division[2])
	{
		const auto tid = get_thread_num();

		const auto tid_x = tid % cores_division[0];
		const auto tid_y = (tid / cores_division[0]) % cores_division[1];
		const auto tid_z = tid / (cores_division[0] * cores_division[1]);

		const auto block_x_begin = tid_x * block_size_x;
		const auto block_x_end = std::min(block_x_begin + block_size_x, x_len);
		const auto block_x_len = block_x_end - block_x_begin;

		const auto block_y_begin = tid_y * block_size_y;
		const auto block_y_end = std::min(block_y_begin + block_size_y, y_len);
		const auto block_y_len = block_y_end - block_y_begin;

		const auto block_z_begin = tid_z * block_size_z;
		const auto block_z_end = std::min(block_z_begin + block_size_z, z_len);
		const auto block_z_len = block_z_end - block_z_begin;

		const auto thread_dens_l = dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len)
								   ^ noarr::slice<'y'>(block_y_begin, block_y_len)
								   ^ noarr::slice<'z'>(block_z_begin, block_z_len);

		// do Z
		{
			const auto lane_scratch_l = scratch_l ^ noarr::fix<'l'>(tid_x * cores_division[1] + tid_y);
			const auto thread_scratch_l = lane_scratch_l ^ noarr::slice<'i'>(block_z_begin, block_z_len);

			const auto lane_dens_l =
				dens_l ^ noarr::slice<'x'>(block_x_begin, block_x_len) ^ noarr::slice<'y'>(block_y_begin, block_y_len);

			solve_block_z_start<index_t>(densities, ac, b1, a_data, c_data, thread_dens_l, thread_scratch_l, tid_z == 0,
										 tid_z == cores_division[2] - 1);
			if (do_sync)
			{
#pragma omp barrier
				if (tid_z == 0)
					solve_block_z_middle(densities, a_data, c_data, lane_dens_l, lane_scratch_l, block_size_z);
#pragma omp barrier
			}

			solve_block_z_end<index_t>(densities, a_data, c_data, thread_dens_l, thread_scratch_l);
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
{
	solve_slice_z_2d_and_3d<index_t>(this->substrates_, get_substrates_layout<3>(), az_, b1z_, a_scratch_, c_scratch_,
									 get_scratch_layout(), cores_division_);
}

template <typename real_t, bool aligned_x>
void cubed_thomas_solver_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
		return;
	}
	else
	{
		solve_2d_and_3d(this->substrates_, get_substrates_layout<3>(), ax_, b1x_, ay_, b1y_, az_, b1z_, a_scratch_,
						c_scratch_, get_scratch_layout(), cores_division_);
	}
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
