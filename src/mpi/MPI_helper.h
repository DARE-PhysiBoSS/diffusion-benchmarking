#pragma once

#include <math.h>
#include <iostream>
#include <fstream>
#include <noarr/traversers.hpp>
#include <mpi.h>

#include "noarr/structures/extra/funcs.hpp"
#include "../omp_helper.h"
#include "../problem.h"

bool MPI_writter(){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}



