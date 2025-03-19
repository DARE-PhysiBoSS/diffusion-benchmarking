#include "simd.h"
#include <immintrin.h>
#include <fstream>
#include <iostream>
#include <vector>

template <typename real_t>
void simd<real_t>::precompute_values()
{

    thomas_i_jump = problem_.substrates_count;
    thomas_j_jump = thomas_i_jump * problem_.nx;
    thomas_k_jump = thomas_j_jump * problem_.ny;


    std::vector<real_t> zero(problem_.substrates_count, 0.0);
    std::vector<real_t> one(problem_.substrates_count, 1.0);
    real_t dt = problem_.dt;

    //Thomas initialization
    bx_.resize(problem_.nx, zero); // sizeof(x_coordinates) = local_x_nodes, denomx is the main diagonal elements
    cx_.resize(problem_.nx, zero);     // Both b and c of tridiagonal matrix are equal, hence just one array needed

    by_.resize(problem_.ny, zero);
    cy_.resize(problem_.ny, zero);

    bz_.resize(problem_.nz, zero);
    cz_.resize(problem_.nz, zero);

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

    cx_.assign(problem_.nx, constant1a); // Fill b and c elements with -D * dt/dx^2
    bx_.assign(problem_.nx, constant3);  // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dx^2)
    bx_[0] = constant3a; 
    bx_[ problem_.nx-1 ] = constant3a; 
    if( problem_.nx == 1 )
    { 
        bx_[0] = one; 
        for (index_t d = 0; d < problem_.substrates_count; d++)
            bx_[0][d] += constant2[d]; 
    } 

    for(index_t d = 0; d < problem_.substrates_count; d++)
        cx_[0][d] /= bx_[0][d]; 
    for( index_t i=1 ; i <= problem_.nx-1 ; i++ )
    {
        for (index_t d = 0; d < problem_.substrates_count; d++)
        {
            bx_[i][d] += constant1[d] * cx_[i-1][d]; 
            cx_[i][d] /= bx_[i][d]; // the value at  size-1 is not actually used
        }   
    }

    cy_.assign(problem_.ny, constant1a); // Fill b and c elements with -D * dt/dy^2
    by_.assign(problem_.ny, constant3);  // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dy^2)
    by_[0] = constant3a; 
    by_[ problem_.ny-1 ] = constant3a; 
    if( problem_.ny == 1 )
    { 
        by_[0] = one;
        for(int d = 0; d < problem_.substrates_count; d++) 
            by_[0][d] += constant2[d]; 
    } 

    for(int d = 0; d < problem_.substrates_count; d++)
        cy_[0][d] /= by_[0][d]; 
    for( index_t i=1 ; i <= problem_.ny-1 ; i++ )
    {
        for (index_t d = 0; d < problem_.substrates_count; d++)
        {
            by_[i][d] += constant1[d] * cy_[i-1][d]; 
            cy_[i][d] /= by_[i][d]; // the value at  size-1 is not actually used
        }   
    }

    cz_.assign(problem_.nz, constant1a); // Fill b and c elements with -D * dt/dz^2
    bz_.assign(problem_.nz, constant3);  // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dz^2)
    bz_[0] = constant3a; 
    bz_[ problem_.nz-1 ] = constant3a; 
    if( problem_.nz == 1 )
    { 
        bz_[0] = one; 
        for(index_t d = 0; d < problem_.substrates_count; d++)
            bz_[0][d] += constant2[d]; 
    } 

    for(index_t d = 0; d < problem_.substrates_count; d++)
        cz_[0][d] /= bz_[0][d]; 
    for( index_t i=1 ; i <= problem_.nz-1 ; i++ )
    {
        for (index_t d = 0; d < problem_.substrates_count; d++)
        {
            bz_[i][d] += constant1[d] * cz_[i-1][d]; 
            cz_[i][d] /= bz_[i][d]; // the value at  size-1 is not actually used
        }   
    }
}

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
void simd<real_t>::precompute_values_vec( std::int32_t vl)
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

template <typename real_t>
void simd<real_t>::initialize()
{
	precompute_values();
    vl_ = 256 / (8 * sizeof(real_t)); // AVX-256 / sizeof(template)
    precompute_values_vec(vl_);
}
template <typename real_t>
auto simd<real_t>::get_substrates_layout(const problem_t<index_t, real_t>& problem)
{
	return noarr::scalar<real_t>()
		   ^ noarr::vectors<'s', 'x', 'y', 'z'>(problem.substrates_count, problem.nx, problem.ny, problem.nz);
}


template <typename real_t>
void simd<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);
	substrates_ = std::make_unique<real_t[]>(problem_.nx * problem_.ny * problem_.nz * problem_.substrates_count);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout(problem_); 

	solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_);
}

template <typename real_t>
void simd<real_t>::solve()
{
	solve_x();
	solve_y();
	solve_z();
}

template <typename real_t>
void simd<real_t>::solve_x()
{
	#pragma omp parallel for collapse(2)
    for (index_t k = 0; k < problem_.nz; k++)
        {
        for (index_t j = 0; j < problem_.ny; j++)
        {
            index_t index = k * thomas_k_jump + j * thomas_j_jump;
            //(*(*M.p_density_vectors))[n] /= M.thomas_denomz[0];
            for (index_t d = 0; d < problem_.substrates_count; d++)
            {
                substrates_[index + d] /= bx_[0][d];
            }

            // should be an empty loop if mesh.z_coordinates.size() < 2
            for (index_t i = 1; i < problem_.nx; i++)
            {

                index_t index_inc = index + thomas_i_jump;
                // axpy(&(*(*M.p_density_vectors))[n], M.thomas_constant1, (*(*M.p_density_vectors))[n - M.thomas_k_jump]);
                for (index_t d = 0; d < problem_.substrates_count; d++)
                {
                    substrates_[index_inc + d] += constant1[d] * substrates_[index + d]; 
                }
                //(*(*M.p_density_vectors))[n] /= M.thomas_denomz[k];
                for (index_t d = 0; d < problem_.substrates_count; d++)
                {
                    substrates_[index_inc + d] /= bx_[i][d];
                }

                index = index_inc;
            }

            index = k * thomas_k_jump + j * thomas_j_jump + (thomas_i_jump * (problem_.nx - 1));
            for (index_t i = problem_.nx - 2; i >= 0; i--)
            {
                index_t index_dec = index - thomas_i_jump;
                // naxpy(&(*(*M.p_density_vectors))[n], M.thomas_cz[k], (*(*M.p_density_vectors))[n + M.thomas_k_jump]);
                for (index_t d = 0; d < problem_.substrates_count; d++)
                {
                    substrates_[index_dec + d] -= cx_[i][d] * substrates_[index + d];
                }
                index = index_dec;
            }
        }
    }
}

template <typename real_t>
void simd<real_t>::solve_y()
{
	#pragma omp parallel for
    for (index_t k = 0; k < problem_.nz; k++)
    {
        //Forward Elimination
        //J = 0
        index_t gd = 0; //Density pointer
        index_t index = k * thomas_k_jump;
        index_t zd;
        for (zd = 0; zd + vl_ < thomas_j_jump; zd+=vl_)
        {
            __m256d denomy1 = _mm256_loadu_pd(&vby_[0][gd]);
            __m256d density1 = _mm256_loadu_pd(&substrates_[index + zd]);
            gd+=vl_;
            if (gd == gvec_size) gd = 0;
            __m256d aux1 = _mm256_div_pd(density1, denomy1);
            _mm256_storeu_pd(&substrates_[index + zd], aux1);
        }
        //Epilogo
        index_t ep = thomas_j_jump - zd;
        index_t x_ini = zd / problem_.substrates_count;
        index_t d_ini = zd % problem_.substrates_count;
        index = index + (x_ini * thomas_i_jump);
        for (index_t i = x_ini; i < problem_.nx; ++i)
        {
            for (index_t d = d_ini; d < problem_.substrates_count; ++d){
                d_ini = 0;
                substrates_[index + d] /= by_[0][d];
            }
            index+=thomas_i_jump;
        }
        //J = 1..(y_size-1)
        for (index_t j = 1; j < problem_.ny; j++)
        {
            index_t index_base = k * thomas_k_jump +  (j-1)*thomas_j_jump;
            index_t index_inc =  index_base + thomas_j_jump;
            index_t zd;
            gd = 0;
            for (zd = 0; zd + vl_ < thomas_j_jump; zd+=vl_)
            {
                __m256d constant1 = _mm256_loadu_pd(&vconstant1[gd]);
                __m256d density_curr1 = _mm256_loadu_pd(&substrates_[index_base + zd]);
                __m256d density_inc1 = _mm256_loadu_pd(&substrates_[index_inc + zd]);
                __m256d denomy1 = _mm256_loadu_pd(&vby_[j][gd]);
                gd+=vl_;
                if (gd == gvec_size) gd = 0;
                density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
                density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                _mm256_storeu_pd(&substrates_[index_inc + zd], density_curr1);
            }
            //Epilogo
            ep = thomas_j_jump - zd;
            x_ini = zd / problem_.substrates_count;
            d_ini = zd % problem_.substrates_count;
            index_base = index_base + (x_ini * thomas_i_jump);
            index_inc = index_inc + (x_ini * thomas_i_jump);
            for (index_t i = x_ini; i < problem_.nx; ++i)
            {
                for (index_t d = d_ini; d < problem_.substrates_count; ++d){
                    d_ini = 0;
                    substrates_[index_inc + d] += constant1[d] * substrates_[index_base + d];
                    substrates_[index_inc + d] /= by_[j][d];
                }
                index_base+=thomas_i_jump;
                index_inc+=thomas_i_jump;
            }
        }
        // Back substitution
        for (index_t j = problem_.ny - 2; j >= 0; j--)
        {
            index_t index_base = k * thomas_k_jump + (j+1) * thomas_j_jump;
            index_t index_dec = index_base - thomas_j_jump;
            index_t zd;
            gd = 0;
            for ( zd = 0; zd + vl_ < thomas_j_jump; zd+=vl_)
            {
                __m256d cy1 = _mm256_loadu_pd(&vcy_[j][gd]);
                __m256d density_curr1 = _mm256_loadu_pd(&substrates_[index_base + zd]);
                __m256d density_dec1 = _mm256_loadu_pd(&substrates_[index_dec+ zd]);
                gd+=vl_;
                if (gd == gvec_size) gd = 0;

                density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                _mm256_storeu_pd(&substrates_[index_dec + zd], density_curr1);
                
            }

            //Epilogo
            index_t ep = thomas_j_jump - zd;
            index_t x_ini = zd / problem_.substrates_count;
            index_t d_ini = zd % problem_.substrates_count;
            index_base = index_base + x_ini * thomas_i_jump;
            index_dec = index_dec + x_ini * thomas_i_jump;
            for (index_t i = x_ini; i < problem_.nx; ++i)
            {
                for (index_t d = d_ini; d < problem_.substrates_count; ++d){
                    d_ini = 0;
                    substrates_[index_dec + d] -= cy_[j][d] * substrates_[index_base + d];
                }
                index_base+=thomas_i_jump;
                index_dec+=thomas_i_jump;
            }

        }
    }
}

template <typename real_t>
void simd<real_t>::solve_z()
{
    
    //Forward Elimination
    index_t initial_index = 0;
    index_t limit = initial_index + thomas_k_jump;
    index_t limit_vec = limit -(thomas_k_jump%vl_);
    #pragma omp parallel for
    for (index_t index = initial_index; index < limit_vec; index += vl_)
    {
        index_t index_dec = index;
        index_t gd = index%gvec_size;

        __m256d denomx1 = _mm256_loadu_pd(&vbz_[0][gd]);
        __m256d density1 = _mm256_loadu_pd(&substrates_[index]);
        __m256d aux1 = _mm256_div_pd(density1, denomx1);

        _mm256_storeu_pd(&substrates_[index], aux1);

        for (index_t k = 1; k < problem_.nz; k++)
        {
            index_t index_inc = index_dec + thomas_k_jump;
            __m256d constant1 = _mm256_loadu_pd(&vconstant1[gd]);
            __m256d density_curr1 = _mm256_loadu_pd(&substrates_[index_dec]);
            __m256d density_inc1 = _mm256_loadu_pd(&substrates_[index_inc]);
            __m256d denomy1 = _mm256_loadu_pd(&vbz_[k][gd]);
        
            density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
    
            density_curr1 = _mm256_div_pd(density_curr1, denomy1);
            _mm256_storeu_pd(&substrates_[index_inc], density_curr1);
            
            index_dec = index_inc;
        }
    }

    //Epilogo vectorization
    for (index_t index = limit_vec; index < limit; ++index)
    {
        index_t index_dec = index;
        index_t d = index % problem_.substrates_count; 

        substrates_[index] /= bz_[0][d];

        for (index_t k = 1; k < problem_.nz; k++)
        {
            index_t index_inc = index_dec + thomas_k_jump;
            // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
            substrates_[index_inc] += constant1[d] * substrates_[index_dec];
            
            //(*M.microenvironment)[n] /= M.thomas_denomx[i];
            
            substrates_[index_inc] /= bz_[k][d];
            
            index_dec = index_inc;
        }
    }

    // Back substitution

    index_t last_zplane = ((problem_.nz - 1)*thomas_k_jump);
    initial_index = last_zplane;
    limit = initial_index + thomas_k_jump;
    limit_vec = limit - (thomas_k_jump%vl_);
    #pragma omp parallel for 
    for (index_t index = initial_index; index < limit_vec; index += vl_)
    {
        index_t index_aux = index;
        index_t gd = (index - last_zplane)%gvec_size;
        for (index_t k = problem_.nz - 2; k >= 0; k--)
        {
            index_t index_dec = index_aux - thomas_k_jump;
            __m256d cy1 = _mm256_loadu_pd(&vcz_[k][gd]);
            __m256d density_curr1 = _mm256_loadu_pd(&substrates_[index_aux]);
            __m256d density_dec1 = _mm256_loadu_pd(&substrates_[index_dec]);

            density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

            _mm256_storeu_pd(&substrates_[index_dec], density_curr1);
            index_aux = index_dec;
        }
    }

    //Epilogo Vectorizacion Back Last rank
    
    for (index_t index = limit_vec; index < limit; ++index){
        index_t index_aux = index;
        index_t d = (index - last_zplane) % problem_.substrates_count;
        for (index_t k = problem_.nz - 2; k >= 0; k--)
        {
            index_t index_dec = index_aux - thomas_k_jump;
            // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
            substrates_[index_dec] -= cz_[k][d] * substrates_[index_aux];
            
            index_aux = index_dec;
        }
    }
}

template <typename real_t>
void simd<real_t>::save(std::ostream& out) const
{
	auto dens_l = get_substrates_layout(problem_);

	for (index_t z = 0; z < problem_.nz; z++)
		for (index_t y = 0; y < problem_.ny; y++)
			for (index_t x = 0; x < problem_.nx; x++)
			{
                out << "(" << x << ", " << y << ", " << z << ")";
				for (index_t s = 0; s < problem_.substrates_count; s++)
					out << (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) << " ";
				out << std::endl;
			}
}

template <typename real_t>
double simd<real_t>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
    
	auto dens_l = get_substrates_layout(problem_);

	return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
    //return substrates_[z * thomas_k_jump + y * thomas_j_jump + x * thomas_i_jump + s];
}

//template class simd<float>;
template class simd<double>;
