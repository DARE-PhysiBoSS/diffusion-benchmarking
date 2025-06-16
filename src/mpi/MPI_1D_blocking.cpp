#include "MPI_1D_blocking.h"
#include <immintrin.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <type_traits>

template <typename T>
MPI_Datatype get_mpi_type()
{
    if constexpr (std::is_same_v<T, float>)
        return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, double>)
        return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, long double>)
        return MPI_LONG_DOUBLE;
    else
        static_assert(sizeof(T) == 0, "Unsupported type for MPI communication");
}

template <typename real_t>
void MPI_1D_blocking<real_t>::precompute_values()
{

    thomas_i_jump = problem_.substrates_count;
    thomas_j_jump = thomas_i_jump * problem_.nx;
    thomas_k_jump = thomas_j_jump * problem_.ny;


    std::vector<real_t> zero(problem_.substrates_count, 0.0);
    std::vector<real_t> one(problem_.substrates_count, 1.0);
    real_t dt = problem_.dt;

    //Thomas initialization
    bx_.resize(local_x_nodes, zero); // sizeof(x_coordinates) = local_x_nodes, denomx is the main diagonal elements
    cx_.resize(local_x_nodes, zero);     // Both b and c of tridiagonal matrix are equal, hence just one array needed

    by_.resize(local_y_nodes, zero);
    cy_.resize(local_y_nodes, zero);

    bz_.resize(local_z_nodes, zero);
    cz_.resize(local_z_nodes, zero);

    constant1 = problem_.diffusion_coefficients; // dt*D/dx^2
    std::vector<real_t> constant1a = zero;                  // -dt*D/dx^2;
    std::vector<real_t> constant2 = problem_.decay_rates;            // (1/3)* dt*lambda
    std::vector<real_t> constant3 = one;                    // 1 + 2*constant1 + constant2;
    std::vector<real_t> constant3a = one;                   // 1 + constant1 + constant2;

    for (index_t d = 0; d < problem_.substrates_count; d++)
    {
        constant1[d] *= dt;
        constant1[d] /= problem_.dx; //dx
        constant1[d] /= problem_.dx; //dx

        constant1a[d] = constant1[d];
        constant1a[d] *= -1.0;

        constant2[d] *= dt;
        constant2[d] /= 3.0; // for the LOD splitting of the source, division by 3 is for 3-D

        constant3[d] += constant1[d];
        constant3[d] += constant1[d];
        constant3[d] += constant2[d];

        constant3a[d] += constant1[d];
        constant3a[d] += constant2[d];
    }


    // Thomas solver coefficients

    cx_.assign(local_x_nodes, constant1a); // Fill b and c elements with -D * dt/dx^2
    bx_.assign(local_x_nodes, constant3);  // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dx^2)
    bx_[0] = constant3a; 
    bx_[ local_x_nodes-1 ] = constant3a; 
    if( local_x_nodes == 1 )
    { 
        bx_[0] = one; 
        for (index_t d = 0; d < problem_.substrates_count; d++)
            bx_[0][d] += constant2[d]; 
    } 

    for(index_t d = 0; d < problem_.substrates_count; d++)
        cx_[0][d] /= bx_[0][d]; 
    for( index_t i=1 ; i <= local_x_nodes-1 ; i++ )
    {
        for (index_t d = 0; d < problem_.substrates_count; d++)
        {
            bx_[i][d] += constant1[d] * cx_[i-1][d]; 
            cx_[i][d] /= bx_[i][d]; // the value at  size-1 is not actually used
        }   
    }

    cy_.assign(local_y_nodes, constant1a); // Fill b and c elements with -D * dt/dy^2
    by_.assign(local_y_nodes, constant3);  // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dy^2)
    by_[0] = constant3a; 
    by_[ local_y_nodes-1 ] = constant3a; 
    if( local_y_nodes == 1 )
    { 
        by_[0] = one;
        for(index_t d = 0; d < problem_.substrates_count; d++) 
            by_[0][d] += constant2[d]; 
    } 

    for(index_t d = 0; d < problem_.substrates_count; d++)
        cy_[0][d] /= by_[0][d]; 
    for( index_t i=1 ; i <= local_y_nodes-1 ; i++ )
    {
        for (index_t d = 0; d < problem_.substrates_count; d++)
        {
            by_[i][d] += constant1[d] * cy_[i-1][d]; 
            cy_[i][d] /= by_[i][d]; // the value at  size-1 is not actually used
        }   
    }
    //Distributed dimension
    index_t step_size = (local_x_nodes * local_y_nodes) / mpi_blocks;
    snd_data_size = step_size * problem_.substrates_count; // Number of data elements to be sent
    rcv_data_size = step_size * problem_.substrates_count; // All p_density_vectors elements have same size, use anyone

    snd_data_size_last = ((local_x_nodes * local_y_nodes) % mpi_blocks) * problem_.substrates_count; // Number of data elements to be sent
    rcv_data_size_last = ((local_x_nodes * local_y_nodes) % mpi_blocks) * problem_.substrates_count;
    cz_.assign(local_z_nodes, constant1a); // Fill b and c elements with -D * dt/dz^2
    bz_.assign(local_z_nodes, constant3);  // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dz^2)
    if (mpi_rank == 0) bz_[0] = constant3a;

    if (mpi_rank == (mpi_size -1 )) bz_[ local_z_nodes-1 ] = constant3a; 
    
    if (mpi_rank == 0 )
    {
        if( local_z_nodes == 1 )
        { 
            bz_[0] = one; 
            for(index_t d = 0; d < problem_.substrates_count; d++)
                bz_[0][d] += constant2[d]; 
        }
    } 

    if (mpi_rank == 0) {
        for(index_t d = 0; d < problem_.substrates_count; d++)
            cz_[0][d] /= bz_[0][d];
    }
    for (index_t ser_ctr = 0; ser_ctr <= mpi_size - 1; ser_ctr++)
    {
        if (mpi_rank == ser_ctr)
        {
            if (mpi_rank == 0 && mpi_rank <= mpi_size - 1) // If size=1, then this process does not send data
            {

                for (index_t i = 1; i <= local_z_nodes - 1; i++)
                {
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        bz_[i][d] += constant1[d] * cz_[i-1][d]; 
                        cz_[i][d] /= bz_[i][d]; // the value at  size-1 is not actually used
                    }   
                }
            }
            else
            {
                for (index_t i = 1; i <= local_z_nodes - 1; i++)
                {
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        bz_[i][d] += constant1[d] * cz_[i-1][d]; 
                        cz_[i][d] /= bz_[i][d]; // the value at  size-1 is not actually used
                    }   
                }
            }

            if (mpi_rank < (mpi_size - 1))
            {
                MPI_Request send_req;
                MPI_Isend(&(cz_[local_z_nodes - 1][0]), cz_[local_z_nodes - 1].size(), mpi_type, ser_ctr + 1, 1111, mpi_comm, &send_req);
            }
        }

        if (mpi_rank == (ser_ctr + 1) && (ser_ctr + 1) <= (mpi_size - 1))
        {

            std::vector<double> temp_cz(cz_[0].size());
            MPI_Request recv_req;
            MPI_Irecv(&temp_cz[0], temp_cz.size(), mpi_type, ser_ctr, 1111, mpi_comm, &recv_req);
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

            for (index_t d = 0; d < problem_.substrates_count; d++)
            {
                bz_[0][d] += constant1[d] * temp_cz[d]; 
                cz_[0][d] /= bz_[0][d]; // the value at  size-1 is not actually used
            }   
        }
        MPI_Barrier(mpi_comm);
    } 
}

/*
std::int32_t gcd(std::int32_t a, std::int32_t b) {
    while (b != 0) {
        std::int32_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Function to compute the least common multiple (LCM)
std::int32_t lcm(std::int32_t a, std::int32_t b) {
    return (a * b) / gcd(a, b);
}

template <typename real_t>
void MPI_1D_blocking<real_t>::precompute_values_vec( std::int32_t vl)
{

    //Vectorization initialization
    gvec_size = lcm(problem_.substrates_count, vl);

    vconstant1.resize(gvec_size, 0.0);
    auto dest_iter =  vconstant1.begin();
    for (index_t j = 0; j < gvec_size; j+=problem_.substrates_count){
        copy(constant1.begin(), constant1.end(), dest_iter);
        dest_iter+=problem_.substrates_count;
    }

    //Y-diffusion

    vby_.resize(problem_.ny);
    vcy_.resize(problem_.ny);
    for (index_t j = 0; j < problem_.ny; ++j){
        vby_[j].resize(gvec_size, 0.0);
        vcy_[j].resize(gvec_size, 0.0);
        auto dest_denomy = vby_[j].begin();
        auto dest_cy = vcy_[j].begin();
        for (index_t d = 0; d < gvec_size; d+=problem_.substrates_count){
            copy(by_[j].begin(), by_[j].end(), dest_denomy);
            copy(cy_[j].begin(), cy_[j].end(), dest_cy);
            dest_denomy+=problem_.substrates_count;
            dest_cy+=problem_.substrates_count;
        }
    }
    //Z - diffusion

    vbz_.resize(problem_.nz);
    vcz_.resize(problem_.nz);
    for (index_t j = 0; j < problem_.nz; ++j){
        vbz_[j].resize(gvec_size, 0.0);
        vcz_[j].resize(gvec_size, 0.0);
        auto dest_denomz = vbz_[j].begin();
        auto dest_cz = vcz_[j].begin();
        for (index_t d = 0; d < gvec_size; d+=problem_.substrates_count){
            copy(bz_[j].begin(), bz_[j].end(), dest_denomz);
            copy(cz_[j].begin(), cz_[j].end(), dest_cz);
            dest_denomz+=problem_.substrates_count;
            dest_cz+=problem_.substrates_count;
        }
    }
}
*/
template <typename real_t>
void MPI_1D_blocking<real_t>::initialize()
{
	precompute_values();
}
template <typename real_t>
auto MPI_1D_blocking<real_t>::get_substrates_layout(const problem_t<index_t, real_t>& problem, const index_t x_nodes, const index_t y_nodes, const index_t z_nodes)
{
    
	return noarr::scalar<real_t>()
		   ^ noarr::vectors<'s', 'x', 'y', 'z'>(problem.substrates_count, x_nodes, y_nodes,z_nodes);
}


template <typename real_t>
void MPI_1D_blocking<real_t>::prepare(const max_problem_t& problem)
{
    mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    local_x_nodes = problem.nx;
    local_y_nodes = problem.ny;
    local_z_nodes = problem.nz/mpi_size;
    init_z_node = local_z_nodes * mpi_rank + std::min(static_cast<index_t>(mpi_rank), static_cast<index_t>(problem.nz % mpi_size)); //calculate the initial z node for each rank
    if (mpi_rank < (problem.nz % mpi_size)) ++local_z_nodes;
    mpi_type = get_mpi_type<real_t>();

	problem_ = problems::cast<std::int32_t, real_t>(problem);
    long long int size = local_z_nodes * problem_.ny;
    size *= problem_.nx;
    size *= problem_.substrates_count;
	substrates_ = std::make_unique<real_t[]>(size);
    //set the 3D subdomain
    z_min = init_z_node;
    z_max = init_z_node + local_z_nodes;
    y_min = 0;
    y_max = problem.ny;
    x_min = 0;
    x_max = problem.nx;
    precompute_values();

    std::vector<index_t> subdomain = {x_min, x_max, y_min, y_max, z_min, z_max};

	// Initialize substrates

	auto substrates_layout = get_substrates_layout(problem_, local_x_nodes, local_y_nodes, local_z_nodes); 

	//solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_, init_z_node);
    for (int i = 0; i < mpi_size; i++)
    {
        if (i == mpi_rank)
        {
            solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_, subdomain);
        
        }
        MPI_Barrier(mpi_comm);
    }
}
    

template <typename real_t>
void MPI_1D_blocking<real_t>::solve()
{
	solve_x();
	solve_y();
	solve_z();
}

template <typename real_t>
void MPI_1D_blocking<real_t>::backward( index_t start, index_t end, index_t step, index_t d, const std::vector<std::vector<real_t>>& c){
    index_t prev = start;
    index_t i = c.size() - 1; // Start from the last index
    for (index_t curr = start + step; curr >= end; curr += step){
        substrates_[curr] -= c[i][d] * substrates_[prev];
        prev = curr;
        --i; 
    }
}

template <typename real_t>
void MPI_1D_blocking<real_t>::forward( index_t start, index_t end, index_t step, 
            const std::vector<real_t>& cons, index_t d, 
            const std::vector<std::vector<real_t>>& b){
            
    substrates_[start] /= b[0][d];
    index_t prev = start;
    index_t i = 1;
    for (index_t curr = start + step; curr <= end; curr += step){
        substrates_[curr] += cons[d] * substrates_[prev];
        substrates_[curr] /= b[i][d];
        prev = curr;
        ++i; 
    }
}

template <typename real_t>
void MPI_1D_blocking<real_t>::solve_x()
{
	#pragma omp parallel for collapse(3)
    for (index_t k = 0; k < local_z_nodes; k++)
        {
        for (index_t j = 0; j < local_y_nodes; j++)
        {
            for (index_t d = 0; d < problem_.substrates_count; d++)
            {
                index_t start = k * thomas_k_jump + j * thomas_j_jump + d;
                index_t end = start + (local_x_nodes - 1) * thomas_i_jump;
                forward(start, end, thomas_i_jump, constant1, d, bx_);
                backward(end, start, -thomas_i_jump, d, cx_);
            }
            
        }
    }
}



/*
template <typename real_t>
void MPI_1D_blocking<real_t>::solve_y()
{
	#pragma omp parallel for collapse(2)
	for (index_t k = 0; k < local_z_nodes; k++)
	{
		for (index_t i = 0; i < local_x_nodes; i++)
		{
			index_t index = k * thomas_k_jump + i * thomas_i_jump;
			for (index_t d = 0; d < problem_.substrates_count; d++)
			{
				substrates_[index + d] /= by_[0][d];
			}

			for (index_t j = 1; j < problem_.ny; j++)
			{
				index_t index_inc = index + thomas_j_jump;
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_inc + d] += constant1[d] * substrates_[index + d];
				}
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_inc + d] /= by_[j][d];
				}

				index = index_inc;
			}

			index = k * thomas_k_jump + i * thomas_i_jump + (thomas_j_jump * (local_y_nodes - 1));
			for (index_t j = problem_.ny - 2; j >= 0; j--)
			{
				index_t index_dec = index - thomas_j_jump;
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_dec + d] -= cy_[j][d] * substrates_[index + d];
				}
				index = index_dec;
			}
		}
	}
}*/

template <typename real_t>
void MPI_1D_blocking<real_t>::solve_y()
{
	#pragma omp parallel for collapse(3)
	for (index_t k = 0; k < local_z_nodes; k++)
	{
		for (index_t i = 0; i < local_x_nodes; i++)
		{
            for (index_t d = 0; d < problem_.substrates_count; d++) {
                index_t start = k * thomas_k_jump + i * thomas_i_jump + d;
                index_t end = start + (local_y_nodes - 1) * thomas_j_jump;
                forward(start, end, thomas_j_jump, constant1, d, by_);
                backward(end, start, -thomas_j_jump, d, cy_);
            }
        }
    }		
}




template <typename real_t>
void MPI_1D_blocking<real_t>::solve_z()
{
    std::vector<real_t> block3d(local_x_nodes*local_y_nodes*problem_.substrates_count);
    std::vector<MPI_Request> send_req(mpi_blocks + 1, MPI_REQUEST_NULL);  // Initialize to MPI_REQUEST_NULL
    std::vector<MPI_Request> recv_req(mpi_blocks + 1, MPI_REQUEST_NULL);  // Initialize to MPI_REQUEST_NULL
    
    if (mpi_rank == 0)
    {
        for (index_t step = 0; step < mpi_blocks; ++step)
        {
            index_t initial_index = step * snd_data_size;
            #pragma omp parallel for
            for (index_t index = initial_index; index < initial_index + snd_data_size; index += problem_.substrates_count)
            {
                index_t index_dec = index; 
                for (index_t d = 0; d < problem_.substrates_count; d++)
                {
                    substrates_[index + d] /= bz_[0][d];
                }

                for (index_t i = 1; i < local_z_nodes; i++)
                {
                    
                    index_t index_inc = index_dec + thomas_k_jump;
                    // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - thomas_k_jump]);
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index_inc + d] += constant1[d] * substrates_[index_dec + d];
                    }

                    //(*M.p_density_vectors)[n] /= bz_[i];
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index_inc + d] /= bz_[i][d];
                    }
                    index_dec = index_inc;
                }
            }

            if (mpi_size > 1) {
                index_t z_end = local_z_nodes - 1;
                index_t offset = step * snd_data_size;
                MPI_Isend(&(substrates_[z_end * thomas_k_jump + offset]), snd_data_size, mpi_type, mpi_rank + 1, step, mpi_comm, &send_req[step]);
            }
        }
    
        //Last iteration
        if (rcv_data_size_last > 0) {
            index_t initial_index = mpi_blocks * snd_data_size;
            #pragma omp parallel for
            for (index_t index = initial_index; index < initial_index + snd_data_size_last; index += problem_.substrates_count)
            {
                index_t index_dec = index; 
                for (index_t d = 0; d < problem_.substrates_count; d++)
                {
                    substrates_[index + d] /= bz_[0][d];
                }

                for (index_t i = 1; i < local_z_nodes; i++)
                {
                    index_t index_inc = index_dec + thomas_k_jump;
                    // axpy(&(*(*M.p_density_vectors))[n], M.thomas_constant1, (*(*M.p_density_vectors))[n - thomas_k_jump]);
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index_inc + d] += constant1[d] * substrates_[index_dec + d];
                    }
                    //(*substrates_)[n] /= bz_[i];
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index_inc + d] /= bz_[i][d];
                    }
                    index_dec = index_inc;
                }
            }

            if (mpi_size > 1) {
                index_t z_end = local_z_nodes - 1;
                index_t offset = mpi_blocks * snd_data_size;
                MPI_Isend(&(substrates_[z_end * thomas_k_jump + offset]), snd_data_size_last, mpi_type, mpi_rank + 1, mpi_blocks, mpi_comm, &send_req[mpi_blocks]);
                
            }
        }
    }
    else
    {
        if (mpi_rank >= 1 && mpi_rank <= (mpi_size - 1))
        {
            for (index_t step = 0; step < mpi_blocks; ++step)
            {
                index_t initial_index = step * snd_data_size;
                MPI_Irecv(&(block3d[initial_index]), rcv_data_size, mpi_type, mpi_rank-1, step, mpi_comm, &(recv_req[step]));
            }
            if (rcv_data_size_last > 0)
                MPI_Irecv(&(block3d[mpi_blocks*snd_data_size]), rcv_data_size_last, mpi_type, mpi_rank-1, mpi_blocks, mpi_comm, &(recv_req[mpi_blocks]));
            for (index_t step = 0; step < mpi_blocks; ++step)
            {
                index_t initial_index = step * snd_data_size;
                MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (index_t index = initial_index; index < initial_index + snd_data_size; index += problem_.substrates_count)
                {
                    // axpy(&(*substrates_)[n], M.thomas_constant1, block3d[k][j]);
                    index_t index_dec = index;
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index + d] += constant1[d] * block3d[index + d];
                    }
                    //(*substrates_)[n] /= bz_[0];
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index + d] /= bz_[0][d];
                    }
                    for (index_t i = 1; i < local_z_nodes; i++)
                    {
                        index_t index_inc = index_dec + thomas_k_jump;
                        // axpy(&(*substrates_)[n], M.thomas_constant1, (*substrates_)[n - thomas_k_jump]);
                        for (index_t d = 0; d < problem_.substrates_count; d++)
                        {
                            substrates_[index_inc + d] += constant1[d] * substrates_[index_dec + d];
                        }
                        //(*substrates_)[n] /= bz_[i];
                        for (index_t d = 0; d < problem_.substrates_count; d++)
                        {
                            substrates_[index_inc + d] /= bz_[i][d];
                        }
                        index_dec = index_inc;
                    }
                }
                if (mpi_rank < (mpi_size - 1))
                {
                    index_t z_end = local_z_nodes - 1;
                    MPI_Isend(&(substrates_[(z_end * thomas_k_jump) + initial_index]), snd_data_size, mpi_type, mpi_rank + 1, step, mpi_comm, &send_req[step]);
                }
            }
            if (snd_data_size_last > 0)
            {
                index_t initial_index = mpi_blocks * snd_data_size;
                MPI_Wait(&recv_req[mpi_blocks], MPI_STATUS_IGNORE); 
                #pragma omp parallel for
                for (index_t index = initial_index; index < initial_index + snd_data_size_last; index += problem_.substrates_count)
                {
                    // axpy(&(*substrates_)[n], M.thomas_constant1, block3d[k][j]);
                    index_t index_dec = index;
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index + d] += constant1[d] * block3d[index + d];
                    }
                    //(*substrates_)[n] /= bz_[0];
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index + d] /= bz_[0][d];
                    }

                    for (index_t i = 1; i < local_z_nodes; i++)
                    {
                        index_t index_inc = index_dec + thomas_k_jump;
                        // axpy(&(*substrates_)[n], M.thomas_constant1, (*substrates_)[n - thomas_k_jump]);
                        for (index_t d = 0; d < problem_.substrates_count; d++)
                        {
                            substrates_[index_inc + d] += constant1[d] * substrates_[index_dec + d];
                        }
                        //(*substrates_)[n] /= bz_[i];
                        for (index_t d = 0; d < problem_.substrates_count; d++)
                        {
                            substrates_[index_inc + d] /= bz_[i][d];
                        }

                        index_dec = index_inc;
                    }
                    
                }
                // End of computation region
                if (mpi_rank < (mpi_size - 1))
                {
                    index_t z_end = local_z_nodes - 1;
                    MPI_Isend(&(substrates_[z_end * thomas_k_jump + initial_index]), snd_data_size_last, mpi_type, mpi_rank + 1, mpi_blocks, mpi_comm, &send_req[mpi_blocks]);
                    
                }
            }
        }
    } 
        /*-----------------------------------------------------------------------------------*/
        /*                         CODE FOR BACK SUBSITUTION                                 */
        /*-----------------------------------------------------------------------------------*/
        
        if (mpi_rank == (mpi_size - 1))
        {
            for (index_t step = 0; step < mpi_blocks; ++step)
            {
                index_t initial_index = ((local_z_nodes - 1)*thomas_k_jump) + (step * snd_data_size);
                #pragma omp parallel for
                for (index_t index = initial_index; index < initial_index + snd_data_size; index += problem_.substrates_count)
                {
                    index_t index_aux = index;
                    for (index_t i = local_z_nodes - 2; i >= 0; i--)
                    {

                        index_t index_dec = index_aux - thomas_k_jump;
                        // naxpy(&(*substrates_)[n], cz_[i], (*substrates_)[n + thomas_k_jump]);
                        for (index_t d = 0; d < problem_.substrates_count; d++)
                        {
                            substrates_[index_dec + d] -= cz_[i][d] * substrates_[index_aux + d];
                        }
                        index_aux = index_dec;
                    }
                }
                if (mpi_size > 1) {
                    MPI_Isend(&(substrates_[step * snd_data_size]), snd_data_size, mpi_type, mpi_rank - 1, step, mpi_comm, &send_req[step]);
                }
            }

            //Last iteration
            if (snd_data_size_last > 0) {
                index_t initial_index = ((local_z_nodes - 1)*thomas_k_jump) + (mpi_blocks * snd_data_size);
                #pragma omp parallel for
                for (index_t index = initial_index; index < initial_index + snd_data_size_last; index += problem_.substrates_count)
                {
                    index_t index_aux = index;
                    for (index_t i = local_z_nodes - 2; i >= 0; i--)
                    {

                        index_t index_dec = index_aux - thomas_k_jump;
                        // naxpy(&(*substrates_)[n], cz_[i], (*substrates_)[n + thomas_k_jump]);
                        for (index_t d = 0; d < problem_.substrates_count; d++)
                        {
                            substrates_[index_dec + d] -= cz_[i][d] * substrates_[index_aux + d];
                        }
                        index_aux = index_dec;
                    }
                }
                if (mpi_size > 1) {
                    MPI_Isend(&(substrates_[mpi_blocks * snd_data_size]), snd_data_size_last, mpi_type, mpi_rank - 1, mpi_blocks, mpi_comm, &send_req[mpi_blocks]);
                }
            
            }
        }
        else
        {
            for (index_t step = 0; step < mpi_blocks; ++step) {
                MPI_Irecv(&(block3d[step*snd_data_size]), rcv_data_size, mpi_type, mpi_rank+1, step, mpi_comm, &recv_req[step]);}
            if (rcv_data_size_last > 0)
                MPI_Irecv(&(block3d[mpi_blocks*snd_data_size]), rcv_data_size_last, mpi_type, mpi_rank+1, mpi_blocks, mpi_comm, &recv_req[mpi_blocks]);
            
            for (index_t step = 0; step < mpi_blocks; ++step)
            {
                index_t initial_index = ((local_z_nodes - 1)*thomas_k_jump) + (step * snd_data_size);
                index_t index_3d_initial = (step * snd_data_size);
                MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (index_t offset = 0; offset < snd_data_size; offset += problem_.substrates_count)
                {
                    index_t index_aux = initial_index + offset;
                    index_t index_3d = index_3d_initial + offset;
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index_aux + d] -= cz_[local_z_nodes - 1][d] * block3d[index_3d + d];
                    }

                    for (index_t i = local_z_nodes - 2; i >= 0; i--)
                    {

                        index_t index_dec = index_aux - thomas_k_jump;
                        // naxpy(&(*substrates_)[n], cz_[i], (*substrates_)[n + thomas_k_jump]);
                        for (index_t d = 0; d < problem_.substrates_count; d++)
                        {
                            substrates_[index_dec + d] -= cz_[i][d] * substrates_[index_aux + d];
                        }
                        index_aux = index_dec;
                        
                    }
                }
                if (mpi_rank > 0)
                {
                    MPI_Isend(&(substrates_[step * snd_data_size]), snd_data_size, mpi_type, mpi_rank - 1, step, mpi_comm, &send_req[step]);
                }
            }
            if (rcv_data_size_last > 0)
            {
                index_t initial_index = ((local_z_nodes - 1)*thomas_k_jump) + (mpi_blocks * snd_data_size);
                index_t index_3d_initial = (mpi_blocks * snd_data_size);
                MPI_Wait(&recv_req[mpi_blocks], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (index_t offset = 0; offset < snd_data_size_last; offset += problem_.substrates_count)
                {
                    index_t index_aux = initial_index + offset;
                    index_t index_3d = index_3d_initial + offset;
                    //naxpy(&(*substrates_)[n], cz_[M.mesh.x_coordinates.size() - 1], block3d[k][j]);
                    for (index_t d = 0; d < problem_.substrates_count; d++)
                    {
                        substrates_[index_aux + d] -= cz_[local_z_nodes - 1][d] * block3d[index_3d + d];
                    }
                    for (index_t i = local_z_nodes - 2; i >= 0; i--)
                    {
                        index_t index_dec = index_aux - thomas_k_jump;
                        // naxpy(&(*substrates_)[n], cz_[i], (*substrates_)[n + thomas_k_jump]);
                        for (index_t d = 0; d < problem_.substrates_count; d++)
                        {
                            substrates_[index_dec + d] -= cz_[i][d] * substrates_[index_aux + d];
                        }
                        index_aux = index_dec;
                    }
                }
                if (mpi_rank > 0)
                {
                    MPI_Isend(&(substrates_[mpi_blocks * snd_data_size]), snd_data_size_last, mpi_type, mpi_rank - 1, mpi_blocks, mpi_comm, &send_req[mpi_blocks]);
                }
            }
        }
        MPI_Barrier(mpi_comm);
}
template <typename real_t>
void MPI_1D_blocking<real_t>::save(std::ostream& out) const
{
	auto dens_l = get_substrates_layout(problem_, local_x_nodes, local_y_nodes, local_z_nodes);

    for (index_t ser_ctr = 0; ser_ctr < mpi_size; ++ser_ctr) {
        if (ser_ctr == mpi_rank) {
            for (index_t z = 0; z < local_z_nodes; z++)
                for (index_t y = 0; y < local_y_nodes; y++)
                    for (index_t x = 0; x < local_x_nodes; x++)
                    {
                        out << "(" << x << ", " << y << ", " << init_z_node + z << ")";
                        for (index_t s = 0; s < problem_.substrates_count; s++)
                            out << (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) << " ";
                        out << std::endl;
                    }
        }
        
        MPI_Barrier(mpi_comm);
    }
}

template <typename real_t>
double MPI_1D_blocking<real_t>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
    //std::cout << "###############################################" <<  std::endl;
	auto dens_l = get_substrates_layout(problem_, local_x_nodes, local_y_nodes, local_z_nodes);

    auto value = (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
	return value;
}


template <typename real_t>
void MPI_1D_blocking<real_t>::tune(const nlohmann::json& params)
{
    if (params.contains("blocks"))
        mpi_blocks = params["blocks"];
    else{
        std::cout << "Warning! No blocks parameter found, using default value of 1" << std::endl;
        mpi_blocks = 1;
    }
}
template class MPI_1D_blocking<float>;
template class MPI_1D_blocking<double>;
