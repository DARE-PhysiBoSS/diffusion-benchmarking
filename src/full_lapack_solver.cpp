#include "full_lapack_solver.h"

#include <fstream>
#include <iostream>

#include "solver_utils.h"

extern "C"
{
	extern void spbtrf_(const char* uplo, const int* n, const int* kd, float* ab, const int* ldab, int* info);
	extern void spbtrs_(const char* uplo, const int* n, const int* kd, const int* nrhs, const float* ab,
						const int* ldab, float* b, const int* ldb, int* info);

	extern void dpbtrf_(const char* uplo, const int* n, const int* kd, double* ab, const int* ldab, int* info);
	extern void dpbtrs_(const char* uplo, const int* n, const int* kd, const int* nrhs, const double* ab,
						const int* ldab, double* b, const int* ldb, int* info);
}

template <>
void full_lapack_solver<float>::pbtrf(const char* uplo, const int* n, const int* kd, float* ab, const int* ldab,
									  int* info)
{
	spbtrf_(uplo, n, kd, ab, ldab, info);
}

template <>
void full_lapack_solver<double>::pbtrf(const char* uplo, const int* n, const int* kd, double* ab, const int* ldab,
									   int* info)
{
	dpbtrf_(uplo, n, kd, ab, ldab, info);
}

template <>
void full_lapack_solver<float>::pbtrs(const char* uplo, const int* n, const int* kd, const int* nrhs, const float* ab,
									  const int* ldab, float* b, const int* ldb, int* info)
{
	spbtrs_(uplo, n, kd, nrhs, ab, ldab, b, ldb, info);
}

template <>
void full_lapack_solver<double>::pbtrs(const char* uplo, const int* n, const int* kd, const int* nrhs, const double* ab,
									   const int* ldab, double* b, const int* ldb, int* info)
{
	dpbtrs_(uplo, n, kd, nrhs, ab, ldab, b, ldb, info);
}

template <typename real_t>
void full_lapack_solver<real_t>::precompute_values()
{
	auto substrates_layout = get_substrates_layout(problem_);

	int kd = 1 * (problem_.dims >= 2 ? problem_.nx : 1) * (problem_.dims >= 3 ? problem_.ny : 1);

	auto ab_layout =
		noarr::scalar<real_t>() ^ noarr::vectors<'i', 'j'>(kd + 1, problem_.nx * problem_.ny * problem_.nz);

	for (index_t s_idx = 0; s_idx < problem_.substrates_count; s_idx++)
	{
		auto single_substr_ab = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * (kd + 1));

		std::fill(single_substr_ab.get(), single_substr_ab.get() + problem_.nx * problem_.ny * problem_.nz * (kd + 1),
				  0);

		auto r_x = -problem_.dt * problem_.diffusion_coefficients[s_idx] / (problem_.dx * problem_.dx);
		auto r_y = -problem_.dt * problem_.diffusion_coefficients[s_idx] / (problem_.dy * problem_.dy);
		auto r_z = -problem_.dt * problem_.diffusion_coefficients[s_idx] / (problem_.dz * problem_.dz);

		for (index_t z = 0; z < problem_.nz; z++)
			for (index_t y = 0; y < problem_.ny; y++)
				for (index_t x = 0; x < problem_.nx; x++)
				{
					auto i = (substrates_layout | noarr::offset<'x', 'y', 'z', 's'>(x, y, z, 0)) / sizeof(real_t);

					real_t x_neighbors = 0;
					real_t y_neighbors = 0;
					real_t z_neighbors = 0;

					if (x > 0)
					{
						auto j =
							(substrates_layout | noarr::offset<'x', 'y', 'z', 's'>(x - 1, y, z, 0)) / sizeof(real_t);
						(ab_layout | noarr::get_at<'i', 'j'>(single_substr_ab.get(), i - j, j)) = r_x;
						x_neighbors++;
					}
					if (x < problem_.nx - 1)
					{
						x_neighbors++;
					}

					if (y > 0)
					{
						auto j =
							(substrates_layout | noarr::offset<'x', 'y', 'z', 's'>(x, y - 1, z, 0)) / sizeof(real_t);
						(ab_layout | noarr::get_at<'i', 'j'>(single_substr_ab.get(), i - j, j)) = r_y;
						y_neighbors++;
					}
					if (y < problem_.ny - 1)
					{
						y_neighbors++;
					}

					if (z > 0)
					{
						auto j =
							(substrates_layout | noarr::offset<'x', 'y', 'z', 's'>(x, y, z - 1, 0)) / sizeof(real_t);
						(ab_layout | noarr::get_at<'i', 'j'>(single_substr_ab.get(), i - j, j)) = r_z;
						z_neighbors++;
					}
					if (z < problem_.nz - 1)
					{
						z_neighbors++;
					}

					(ab_layout | noarr::get_at<'i', 'j'>(single_substr_ab.get(), i - i, i)) =
						1 + problem_.dt * problem_.decay_rates[s_idx]
						+ problem_.dt * problem_.diffusion_coefficients[s_idx]
							  * (x_neighbors / (problem_.dx * problem_.dx) + y_neighbors / (problem_.dy * problem_.dy)
								 + z_neighbors / (problem_.dz * problem_.dz));
				}

		int info;
		int n = problem_.nx * problem_.ny * problem_.nz;
		int ldab = kd + 1;
		pbtrf("L", &n, &kd, single_substr_ab.get(), &ldab, &info);

		if (info != 0)
			throw std::runtime_error("LAPACK pbtrf failed with error code " + std::to_string(info));

		ab_.emplace_back(std::move(single_substr_ab));
	}
}

template <typename real_t>
auto full_lapack_solver<real_t>::get_substrates_layout(const problem_t<index_t, real_t>& problem)
{
	return noarr::scalar<real_t>()
		   ^ noarr::vectors<'x', 'y', 'z', 's'>(problem.nx, problem.ny, problem.nz, problem.substrates_count);
}

template <typename real_t>
void full_lapack_solver<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout(problem_);

	solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_);
}

template <typename real_t>
void full_lapack_solver<real_t>::initialize()
{
	precompute_values();
}

template <typename real_t>
void full_lapack_solver<real_t>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"] : 1;
}

template <typename real_t>
void full_lapack_solver<real_t>::solve()
{
	auto dens_l = get_substrates_layout(problem_);

#pragma omp for schedule(static, 1) nowait
	for (index_t s = 0; s < problem_.substrates_count; s++)
	{
		const index_t begin_offset = (dens_l | noarr::offset<'x', 'y', 'z', 's'>(0, 0, 0, s)) / sizeof(real_t);

		int info;
		int n = problem_.nx * problem_.ny * problem_.nz;
		int kd = 1 * (problem_.dims >= 2 ? problem_.nx : 1) * (problem_.dims >= 3 ? problem_.ny : 1);
		int rhs = 1;
		int ldab = kd + 1;
		pbtrs("L", &n, &kd, &rhs, ab_[s].get(), &ldab, substrates_.get() + begin_offset, &n, &info);

		if (info != 0)
			throw std::runtime_error("LAPACK pbtrs failed with error code " + std::to_string(info));
	}
}

template <typename real_t>
void full_lapack_solver<real_t>::save(const std::string& file) const
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
double full_lapack_solver<real_t>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
	auto dens_l = get_substrates_layout(problem_);

	return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
}

template class full_lapack_solver<float>;
template class full_lapack_solver<double>;
