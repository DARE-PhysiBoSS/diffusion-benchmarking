#pragma once

#include <memory>

#include <noarr/structures/interop/bag.hpp>

#include "problem.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <hwy/aligned_allocator.h>
#pragma GCC diagnostic pop

template <typename real_t>
struct least_memory_thomas_solver_data
{
	real_t* a = nullptr;
	real_t* b1 = nullptr;
	real_t* b = nullptr;

	template <typename index_t, typename diagonal_layout_t>
	void initialize(index_t shape, index_t dims, index_t n, problem_t<index_t, real_t> problem,
					diagonal_layout_t layout, bool aligned = false)
	{
		if (aligned)
		{
			a_storage_aligned_ = hwy::MakeUniqueAlignedArray<real_t>(problem.substrates_count);
			b1_storage_aligned_ = hwy::MakeUniqueAlignedArray<real_t>(problem.substrates_count);
			b_storage_aligned_ = hwy::MakeUniqueAlignedArray<real_t>(n * problem.substrates_count);

			a = a_storage_aligned_.get();
			b1 = b1_storage_aligned_.get();
			b = b_storage_aligned_.get();
		}
		else
		{
			a_storage_ = std::make_unique<real_t[]>(problem.substrates_count);
			b1_storage_ = std::make_unique<real_t[]>(problem.substrates_count);
			b_storage_ = std::make_unique<real_t[]>(n * problem.substrates_count);

			a = a_storage_.get();
			b1 = b1_storage_.get();
			b = b_storage_.get();
		}

		auto b_diag = noarr::make_bag(layout, b);

		// compute a
		for (index_t s = 0; s < problem.substrates_count; s++)
			a[s] = -problem.dt * problem.diffusion_coefficients[s] / (shape * shape);

		// compute b1
		for (index_t s = 0; s < problem.substrates_count; s++)
			b1[s] = 1 + problem.decay_rates[s] * problem.dt / dims
					+ 2 * problem.dt * problem.diffusion_coefficients[s] / (shape * shape);

		// compute b_i
		{
			std::array<index_t, 2> indices = { 0, n - 1 };

			for (index_t i : indices)
				for (index_t s = 0; s < problem.substrates_count; s++)
					b_diag.template at<'i', 's'>(i, s) =
						1 + problem.decay_rates[s] * problem.dt / dims
						+ problem.dt * problem.diffusion_coefficients[s] / (shape * shape);

			for (index_t i = 1; i < n - 1; i++)
				for (index_t s = 0; s < problem.substrates_count; s++)
					b_diag.template at<'i', 's'>(i, s) =
						1 + problem.decay_rates[s] * problem.dt / dims
						+ 2 * problem.dt * problem.diffusion_coefficients[s] / (shape * shape);
		}

		// compute b_i'
		{
			for (index_t s = 0; s < problem.substrates_count; s++)
				b_diag.template at<'i', 's'>(0, s) = 1 / b_diag.template at<'i', 's'>(0, s);

			for (index_t i = 1; i < n; i++)
				for (index_t s = 0; s < problem.substrates_count; s++)
				{
					b_diag.template at<'i', 's'>(i, s) =
						1 / (b_diag.template at<'i', 's'>(i, s) - a[s] * a[s] * b_diag.template at<'i', 's'>(i - 1, s));
				}
		}
	}

private:
	std::unique_ptr<real_t[]> a_storage_, b1_storage_, b_storage_;
	hwy::AlignedUniquePtr<real_t[]> a_storage_aligned_, b1_storage_aligned_, b_storage_aligned_;
};
