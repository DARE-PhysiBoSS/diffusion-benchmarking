#include <fstream>
#include <iostream>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

#include "../algorithms.h"


#include <mpi.h>


int main(int argc, char** argv)
{
	// #define CSR_FLUSH_TO_ZERO (1 << 15)
	// 	unsigned csr = __builtin_ia32_stmxcsr();
	// 	csr |= CSR_FLUSH_TO_ZERO;
	// 	__builtin_ia32_ldmxcsr(csr);

	
	int rank, size;
	try
	{
		int provided_thread_level;
		MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
		if (provided_thread_level != MPI_THREAD_FUNNELED)
		{
			std::cerr << "ERROR! The MPI implementation does not provide the required thread level" << std::endl;
			return 1;
		}

		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
	

	argparse::ArgumentParser program("diffuse_mpi");

	std::string alg;
	program.add_argument("--alg").help("Algorithm to use").required().store_into(alg);

	std::string params_file;
	program.add_argument("--params").help("A file with the algorithm specific parameters").store_into(params_file);

	std::string problem_file;
	program.add_argument("--problem")
		.help("A file describing the problem instance")
		.required()
		.store_into(problem_file);

	bool double_precision;
	program.add_argument("--double").help("Use double precision").flag().store_into(double_precision);

	bool verbose;
	program.add_argument("-v").help("Add verbosity").flag().store_into(verbose);

	auto& group = program.add_mutually_exclusive_group();

	bool validate;
	group.add_argument("--validate")
		.help("Check the validity of the algorithm by running single iteration per each dimension and comparing to the "
			  "reference implementation")
		.flag()
		.store_into(validate);

	std::string output_file;
	group.add_argument("--run_and_save")
		.help("If present, the provided problem will be solved by the algorithm and results will be stored in the "
			  "output file")
		.store_into(output_file);

	bool benchmark;
	group.add_argument("--benchmark")
		.help("The run of the algorithm will be benchmarked and outputed to standard output")
		.flag()
		.store_into(benchmark);

	try
	{
		// program.parse_args({ "./diffuse", "--alg", "full_lapack", "--problem", "../example-problems/toy.json",
		// "--validate"});
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err)
	{

		std::cerr << "Error in Rank" << rank << ": " <<  err.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
		
		return 1;
	}

	algorithms algs(double_precision, verbose);

	max_problem_t problem;

	try
	{
		problem = problems::read_problem(problem_file);
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
		
		return 1;
	}

	nlohmann::json params;

	if (!params_file.empty())
	{
		std::ifstream ifs(params_file);
		if (!ifs)
		{
			
			if (rank == 0) std::cerr << "Cannot open file " << params_file << std::endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
			
			return 1;
		}

		ifs >> params;
	}

	try
	{
		if (validate)
		{
			algs.validate(alg, problem, params);
		}
		else if (!output_file.empty())
		{
			algs.run(alg, problem, params, output_file);
		}
		else if (benchmark)
		{
			algs.benchmark(alg, problem, params);
		}
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl;
		return 1;
	}

	
	MPI_Finalize();
	

	return 0;
}
