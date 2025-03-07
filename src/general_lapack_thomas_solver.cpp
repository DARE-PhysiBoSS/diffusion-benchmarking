#include "general_lapack_thomas_solver.h"

#include <fstream>
#include <iostream>

#include "solver_utils.h"

extern "C"
{
	extern void sgttrf_(const int* n, float* dl, float* d, float* du, float* du2, int* ipiv, int* info);
	extern void sgttrs_(const char* trans, const int* n, const int* nrhs, const float* dl, const float* d,
						const float* du, const float* du2, const int* ipiv, float* b, const int* ldb, int* info);

	extern void dgttrf_(const int* n, double* dl, double* d, double* du, double* du2, int* ipiv, int* info);
	extern void dgttrs_(const char* trans, const int* n, const int* nrhs, const double* dl, const double* d,
						const double* du, const double* du2, const int* ipiv, double* b, const int* ldb, int* info);
}

template <>
void general_lapack_thomas_solver<float>::gttrf(const int* n, float* dl, float* d, float* du, float* du2, int* ipiv,
												int* info)
{
	sgttrf_(n, dl, d, du, du2, ipiv, info);
}

template <>
void general_lapack_thomas_solver<double>::gttrf(const int* n, double* dl, double* d, double* du, double* du2,
												 int* ipiv, int* info)
{
	dgttrf_(n, dl, d, du, du2, ipiv, info);
}

template <>
void general_lapack_thomas_solver<float>::gttrs(const char* trans, const int* n, const int* nrhs, const float* dl,
												const float* d, const float* du, const float* du2, const int* ipiv,
												float* b, const int* ldb, int* info)
{
	sgttrs_(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info);
};

template <>
void general_lapack_thomas_solver<double>::gttrs(const char* trans, const int* n, const int* nrhs, const double* dl,
												 const double* d, const double* du, const double* du2, const int* ipiv,
												 double* b, const int* ldb, int* info)
{
	dgttrs_(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info);
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::precompute_values(std::vector<std::unique_ptr<real_t[]>>& dls,
															 std::vector<std::unique_ptr<real_t[]>>& ds,
															 std::vector<std::unique_ptr<real_t[]>>& dus,
															 std::vector<std::unique_ptr<real_t[]>>& du2s,
															 std::vector<std::unique_ptr<int[]>>& ipivs, index_t shape,
															 index_t dims, index_t n)
{
	for (index_t s_idx = 0; s_idx < problem_.substrates_count; s_idx++)
	{
		auto dl = std::make_unique<real_t[]>(n - 1);
		auto d = std::make_unique<real_t[]>(n);
		auto du = std::make_unique<real_t[]>(n - 1);
		auto du2 = std::make_unique<real_t[]>(n - 2);
		auto ipiv = std::make_unique<int[]>(n);
		for (index_t i = 0; i < n; i++)
		{
			if (i != n - 1)
			{
				dl[i] = -problem_.dt * problem_.diffusion_coefficients[s_idx] / (shape * shape);
				du[i] = -problem_.dt * problem_.diffusion_coefficients[s_idx] / (shape * shape);
			}

			d[i] = 1 + problem_.dt * problem_.decay_rates[s_idx] / dims
				   + 2 * problem_.dt * problem_.diffusion_coefficients[s_idx] / (shape * shape);

			if (i == 0 || i == n - 1)
				d[i] -= problem_.dt * problem_.diffusion_coefficients[s_idx] / (shape * shape);
		}

		int info;
		gttrf(&n, dl.get(), d.get(), du.get(), du2.get(), ipiv.get(), &info);

		if (info != 0)
			throw std::runtime_error("LAPACK spttrf failed with error code " + std::to_string(info));

		dls.emplace_back(std::move(dl));
		ds.emplace_back(std::move(d));
		dus.emplace_back(std::move(du));
		du2s.emplace_back(std::move(du2));
		ipivs.emplace_back(std::move(ipiv));
	}
}

template <typename real_t>
auto general_lapack_thomas_solver<real_t>::get_substrates_layout(const problem_t<index_t, real_t>& problem)
{
	return noarr::scalar<real_t>()
		   ^ noarr::vectors<'x', 'y', 'z', 's'>(problem.nx, problem.ny, problem.nz, problem.substrates_count);
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout(problem_);

	solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_);
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::initialize()
{
	if (problem_.dims >= 1)
		precompute_values(dlx_, dx_, dux_, du2x_, ipivx_, problem_.dx, problem_.dims, problem_.nx);
	if (problem_.dims >= 2)
		precompute_values(dly_, dy_, duy_, du2y_, ipivy_, problem_.dy, problem_.dims, problem_.ny);
	if (problem_.dims >= 3)
		precompute_values(dlz_, dz_, duz_, du2z_, ipivz_, problem_.dz, problem_.dims, problem_.nz);
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"] : 1;
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::solve_x()
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

			char c = 'N';
			gttrs(&c, &problem_.nx, &rhs, dlx_[s].get(), dx_[s].get(), dux_[s].get(), du2x_[s].get(), ipivx_[s].get(),
				  substrates_.get() + begin_offset, &problem_.nx, &info);

			if (info != 0)
				throw std::runtime_error("LAPACK spttrs failed with error code " + std::to_string(info));
		}
	}
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::solve_y()
{}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::solve_z()
{}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::save(const std::string& file) const
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
double general_lapack_thomas_solver<real_t>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
	auto dens_l = get_substrates_layout(problem_);

	return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
}

template class general_lapack_thomas_solver<float>;
template class general_lapack_thomas_solver<double>;
