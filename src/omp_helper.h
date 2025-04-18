#pragma once

#ifdef _OPENMP
	#include <omp.h>
#endif

template <typename T, typename F>
inline void omp_trav_for_each(const T& trav, const F& f)
{
#pragma omp parallel for
	for (auto trav_inner : trav)
		trav_inner.for_each(f);
}

inline auto get_thread_num()
{
#ifdef _OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}

inline auto get_num_threads()
{
#ifdef _OPENMP
	return omp_get_num_threads();
#else
	return 1;
#endif
}

inline auto get_max_threads()
{
#ifdef _OPENMP
	return omp_get_max_threads();
#else
	return 1;
#endif
}