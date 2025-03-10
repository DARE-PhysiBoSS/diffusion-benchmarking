#include "lapack_thomas_solver.h"

#include <fstream>
#include <iostream>

#include "solver_utils.h"

extern "C"
{
	extern void spttrf_(const int* n, float* d, float* e, int* info);
	extern void spttrs_(const int* n, const int* nrhs, const float* d, const float* e, float* b, const int* ldb,
						int* info);

	extern void dpttrf_(const int* n, double* d, double* e, int* info);
	extern void dpttrs_(const int* n, const int* nrhs, const double* d, const double* e, double* b, const int* ldb,
						int* info);

	extern void sptsv_(const int* n, const int* nrhs, float* d, float* e, float* b, const int* ldb, int* info);
	extern void dptsv_(const int* n, const int* nrhs, double* d, double* e, double* b, const int* ldb, int* info);
}

template <>
void lapack_thomas_solver<float>::pttrf(const int* n, float* d, float* e, int* info)
{
	spttrf_(n, d, e, info);
}

template <>
void lapack_thomas_solver<double>::pttrf(const int* n, double* d, double* e, int* info)
{
	dpttrf_(n, d, e, info);
}

template <>
void lapack_thomas_solver<float>::pttrs(const int* n, const int* nrhs, const float* d, const float* e, float* b,
										const int* ldb, int* info)
{
	spttrs_(n, nrhs, d, e, b, ldb, info);
}

template <>
void lapack_thomas_solver<double>::pttrs(const int* n, const int* nrhs, const double* d, const double* e, double* b,
										 const int* ldb, int* info)
{
	dpttrs_(n, nrhs, d, e, b, ldb, info);
}

template <>
void lapack_thomas_solver<float>::ptsv(const int* n, const int* nrhs, float* d, float* e, float* b, const int* ldb,
									   int* info)
{
	sptsv_(n, nrhs, d, e, b, ldb, info);
}

template <>
void lapack_thomas_solver<double>::ptsv(const int* n, const int* nrhs, double* d, double* e, double* b, const int* ldb,
										int* info)
{
	dptsv_(n, nrhs, d, e, b, ldb, info);
}

template <typename real_t>
void lapack_thomas_solver<real_t>::precompute_values(std::vector<std::unique_ptr<real_t[]>>& a,
													 std::vector<std::unique_ptr<real_t[]>>& b, index_t shape,
													 index_t dims, index_t n)
{
	for (index_t s_idx = 0; s_idx < problem_.substrates_count; s_idx++)
	{
		auto single_substr_a = std::make_unique<real_t[]>(n - 1);
		auto single_substr_b = std::make_unique<real_t[]>(n);
		for (index_t i = 0; i < n; i++)
		{
			if (i != n - 1)
				single_substr_a[i] = -problem_.dt * problem_.diffusion_coefficients[s_idx] / (shape * shape);

			single_substr_b[i] = 1 + problem_.dt * problem_.decay_rates[s_idx] / dims
								 + 2 * problem_.dt * problem_.diffusion_coefficients[s_idx] / (shape * shape);

			if (i == 0 || i == n - 1)
				single_substr_b[i] -= problem_.dt * problem_.diffusion_coefficients[s_idx] / (shape * shape);
		}

		int info;
		pttrf(&n, single_substr_b.get(), single_substr_a.get(), &info);

		if (info != 0)
			throw std::runtime_error("LAPACK pttrf failed with error code " + std::to_string(info));

		a.emplace_back(std::move(single_substr_a));
		b.emplace_back(std::move(single_substr_b));
	}
}

template <typename real_t>
auto lapack_thomas_solver<real_t>::get_substrates_layout(const problem_t<index_t, real_t>& problem)
{
	return noarr::scalar<real_t>()
		   ^ noarr::vectors<'x', 'y', 'z', 's'>(problem.nx, problem.ny, problem.nz, problem.substrates_count);
}

template <typename real_t>
void lapack_thomas_solver<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout(problem_);

	solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_);
}

template <typename real_t>
void lapack_thomas_solver<real_t>::initialize()
{
	if (problem_.dims >= 1)
		precompute_values(ax_, bx_, problem_.dx, problem_.dims, problem_.nx);
	if (problem_.dims >= 2)
		precompute_values(ay_, by_, problem_.dy, problem_.dims, problem_.ny);
	if (problem_.dims >= 3)
		precompute_values(az_, bz_, problem_.dz, problem_.dims, problem_.nz);
}

template <typename real_t>
void lapack_thomas_solver<real_t>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"] : 1;
}

template <typename real_t>
void lapack_thomas_solver<real_t>::solve_x()
{
	auto dens_l = get_substrates_layout(problem_) ^ noarr::merge_blocks<'y', 'z', 'm'>();

	for (index_t s = 0; s < problem_.substrates_count; s++)
	{
#pragma omp for schedule(static, 1) nowait
		for (index_t yz = 0; yz < problem_.ny * problem_.nz; yz += work_items_)
		{
			const index_t begin_offset = (dens_l | noarr::offset<'x', 'm', 's'>(0, yz, s)) / sizeof(real_t);

			int info;
			int rhs = std::min((int)work_items_, problem_.ny * problem_.nz - yz);
			pttrs(&problem_.nx, &rhs, bx_[s].get(), ax_[s].get(), substrates_.get() + begin_offset, &problem_.nx,
				  &info);

			if (info != 0)
				throw std::runtime_error("LAPACK pttrs failed with error code " + std::to_string(info));
		}
	}
}

template <typename real_t>
void lapack_thomas_solver<real_t>::solve_y()
{
	throw std::runtime_error("Not implemented");
}

template <typename real_t>
void lapack_thomas_solver<real_t>::solve_z()
{
	throw std::runtime_error("Not implemented");
}

template <typename real_t>
void lapack_thomas_solver<real_t>::solve()
{
	if (problem_.dims == 1)
	{
		solve_x();
	}
	else if (problem_.dims == 2)
	{
		solve_x();
		solve_y();
	}
	else if (problem_.dims == 3)
	{
		solve_x();
		solve_y();
		solve_z();
	}
}

template <typename real_t>
void lapack_thomas_solver<real_t>::save(const std::string& file) const
{
	auto dens_l = get_substrates_layout(problem_);

	std::ofstream out(file);

	for (index_t z = 0; z < problem_.nz; z++)
		for (index_t y = 0; y < problem_.ny; y++)
			for (index_t x = 0; x < problem_.nx; x++)
			{
				for (index_t s = 0; s < problem_.substrates_count; s++)
					out << (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) << " ";
				out << std::endl;
			}

	out.close();
}

template <typename real_t>
double lapack_thomas_solver<real_t>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
	auto dens_l = get_substrates_layout(problem_);

	return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
}

template class lapack_thomas_solver<float>;
template class lapack_thomas_solver<double>;
