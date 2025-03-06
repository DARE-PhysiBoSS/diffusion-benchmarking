#pragma once

#include <noarr/traversers.hpp>

#include "noarr/structures/extra/funcs.hpp"
#include "omp_helper.h"

class solver_utils
{
public:
	template <typename real_t>
	static void initialize_substrate_with_initial_conditions(auto substrates_layout, real_t* substrates,
															 const real_t* initial_conditions)
	{
		omp_trav_for_each(noarr::traverser(substrates_layout), [&](auto state) {
			auto s_idx = noarr::get_index<'s'>(state);

			(substrates_layout | noarr::get_at(substrates, state)) = initial_conditions[s_idx];
		});
	}

	template <typename real_t>
	static void initialize_substrate_linear(auto substrates_layout, real_t* substrates)
	{
		omp_trav_for_each(noarr::traverser(substrates_layout), [&](auto state) {
			auto offset = (substrates_layout | noarr::offset(state)) / sizeof(real_t);

			(substrates_layout | noarr::get_at(substrates, state)) = offset;
		});
	}
};
