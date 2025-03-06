#include "tridiagonal_solver.h"

void tridiagonal_solver::solve()
{
	solve_x();
	solve_y();
	solve_z();
}
