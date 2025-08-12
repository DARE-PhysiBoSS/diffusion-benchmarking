#include "reference_thomas_solver.h"

template <typename real_t>
void sdd_reference_thomas_solver<real_t>::precompute_values(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b,
															std::unique_ptr<real_t[]>& c, index_t shape, index_t dims)
{
	auto substrates_layout = get_substrates_layout();

	a = std::make_unique<real_t[]>((substrates_layout | noarr::get_size()) / sizeof(real_t));
	b = std::make_unique<real_t[]>((substrates_layout | noarr::get_size()) / sizeof(real_t));
	c = std::make_unique<real_t[]>((substrates_layout | noarr::get_size()) / sizeof(real_t));

	auto a_bag = noarr::make_bag(substrates_layout, a.get());
	auto b_bag = noarr::make_bag(substrates_layout, b.get());
	auto c_bag = noarr::make_bag(substrates_layout, c.get());

	auto get_diffusion_coefficients = [&](index_t x, index_t y, index_t z, index_t s) {
		return this->problem_.diffusion_coefficients[s];
	};

	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		for (index_t x = 0; x < this->problem_.nx; x++)
			for (index_t y = 0; y < this->problem_.ny; y++)
				for (index_t z = 0; z < this->problem_.nz; z++)
				{
					auto idx = noarr::idx<'x', 'y', 'z', 's'>(x, y, z, s);

					a_bag[idx] = -this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
					b_bag[idx] = 1 + this->problem_.dt * this->problem_.decay_rates[s] / dims
								 + 2 * this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
					c_bag[idx] = -this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
				}
}

template <typename real_t>
void sdd_reference_thomas_solver<real_t>::initialize()
{
	precompute_values(a_, b_, c_, this->problem_.dz, this->problem_.dims);
	b_scratch_ = std::make_unique<real_t[]>(std::max({ this->problem_.nx, this->problem_.ny, this->problem_.nz })
											* this->problem_.substrates_count);
}

template <typename real_t>
void sdd_reference_thomas_solver<real_t>::solve_x()
{
	auto dens_l = get_substrates_layout();
	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::vectors<'s', 'x'>(this->problem_.substrates_count, this->problem_.nx);

	auto a_bag = noarr::make_bag(dens_l, a_.get());
	auto b_bag = noarr::make_bag(dens_l, b_.get());
	auto c_bag = noarr::make_bag(dens_l, c_.get());

	auto d = noarr::make_bag(dens_l, this->substrates_);

	auto scratch = noarr::make_bag(diag_l, b_scratch_.get());

	for (index_t z = 0; z < this->problem_.nz; z++)
	{
		for (index_t y = 0; y < this->problem_.ny; y++)
		{
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, 0, y, z);
				scratch[idx] = b_bag[idx];
			}

			for (index_t x = 1; x < this->problem_.nx; x++)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, z);
					auto prev_idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x - 1, y, z);

					auto r = a_bag[prev_idx] / scratch[prev_idx];

					scratch[idx] = b_bag[idx] - c_bag[idx] * r;

					d[idx] -= r * d[prev_idx];
				}
			}

			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, this->problem_.nx - 1, y, z);
				d[idx] /= scratch[idx];
			}

			for (index_t x = this->problem_.nx - 2; x >= 0; x--)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, z);
					auto next_idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x + 1, y, z);

					d[idx] = (d[idx] - c_bag[next_idx] * d[next_idx]) / scratch[idx];
				}
			}
		}
	}
}

template <typename real_t>
void sdd_reference_thomas_solver<real_t>::solve_y()
{
	auto dens_l = get_substrates_layout();
	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::vectors<'s', 'x'>(this->problem_.substrates_count, this->problem_.nx);

	auto a_bag = noarr::make_bag(dens_l, a_.get());
	auto b_bag = noarr::make_bag(dens_l, b_.get());
	auto c_bag = noarr::make_bag(dens_l, c_.get());

	auto d = noarr::make_bag(dens_l, this->substrates_);

	auto scratch = noarr::make_bag(diag_l, b_scratch_.get());

	for (index_t z = 0; z < this->problem_.nz; z++)
	{
		for (index_t x = 0; x < this->problem_.nx; x++)
		{
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, 0, z);
				scratch[idx] = b_bag[idx];
			}

			for (index_t y = 1; y < this->problem_.ny; y++)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, z);
					auto prev_idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y - 1, z);

					auto r = a_bag[prev_idx] / scratch[prev_idx];

					scratch[idx] = b_bag[idx] - c_bag[idx] * r;

					d[idx] -= r * d[prev_idx];
				}
			}

			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, this->problem_.ny - 1, z);
				d[idx] /= scratch[idx];
			}

			for (index_t y = this->problem_.ny - 2; y >= 0; y--)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, z);
					auto next_idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y + 1, z);

					d[idx] = (d[idx] - c_bag[next_idx] * d[next_idx]) / scratch[idx];
				}
			}
		}
	}
}

template <typename real_t>
void sdd_reference_thomas_solver<real_t>::solve_z()
{
	auto dens_l = get_substrates_layout();
	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::vectors<'s', 'x'>(this->problem_.substrates_count, this->problem_.nx);

	auto a_bag = noarr::make_bag(dens_l, a_.get());
	auto b_bag = noarr::make_bag(dens_l, b_.get());
	auto c_bag = noarr::make_bag(dens_l, c_.get());

	auto d = noarr::make_bag(dens_l, this->substrates_);

	auto scratch = noarr::make_bag(diag_l, b_scratch_.get());

	for (index_t y = 0; y < this->problem_.ny; y++)
	{
		for (index_t x = 0; x < this->problem_.nx; x++)
		{
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, 0);
				scratch[idx] = b_bag[idx];
			}

			for (index_t z = 1; z < this->problem_.nz; z++)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, z);
					auto prev_idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, z - 1);

					auto r = a_bag[prev_idx] / scratch[prev_idx];

					scratch[idx] = b_bag[idx] - c_bag[idx] * r;

					d[idx] -= r * d[prev_idx];
				}
			}

			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, this->problem_.nz - 1);
				d[idx] /= scratch[idx];
			}

			for (index_t z = this->problem_.nz - 2; z >= 0; z--)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					auto idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, z);
					auto next_idx = noarr::idx<'s', 'x', 'y', 'z'>(s, x, y, z + 1);
					d[idx] = (d[idx] - c_bag[next_idx] * d[next_idx]) / scratch[idx];
				}
			}
		}
	}
}

template <typename real_t>
void sdd_reference_thomas_solver<real_t>::solve()
{
	for (index_t i = 0; i < this->problem_.iterations; i++)
	{
		if (this->problem_.dims == 1)
		{
			solve_x();
		}
		else if (this->problem_.dims == 2)
		{
			solve_x();
			solve_y();
		}
		else if (this->problem_.dims == 3)
		{
			solve_x();
			solve_y();
			solve_z();
		}
	}
}

template class sdd_reference_thomas_solver<float>;
template class sdd_reference_thomas_solver<double>;
