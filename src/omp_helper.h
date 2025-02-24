#pragma once

template <typename T, typename F>
inline void omp_trav_for_each(const T& trav, const F& f)
{
#pragma omp parallel for
	for (auto trav_inner : trav)
		trav_inner.for_each(f);
}
