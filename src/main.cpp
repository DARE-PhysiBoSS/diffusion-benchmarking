#include <fstream>
#include <iostream>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

#include "algorithms.h"

int main(int argc, char** argv)
{
	argparse::ArgumentParser program("diffuse");

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

	auto& group = program.add_mutually_exclusive_group();

	bool validate;
	group.add_argument("--validate")
		.help("Check the validity of the algorithm by running single iteration per each dimension and comparing to the "
			  "reference implementation")
		.flag()
		.store_into(validate);

	std::string output_file;
	group.add_argument("--run_and_save")
		.help("If present, the provided problem will be solved  by the algorithm and results will be stored in the "
			  "output file")
		.store_into(output_file);

	bool benchmark;
	group.add_argument("--benchmark")
		.help("The run of the algorithm will be benchmarked and outputed to standard output")
		.flag()
		.store_into(benchmark);

	try
	{
		// program.parse_args({ "./diffuse", "--alg", "omp", "--problem", "../example-problems/toy.json", "--validate" });
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	algorithms algs(double_precision);

	auto problem = problems::read_problem(problem_file);

	nlohmann::json params;

	if (!params_file.empty())
	{
		std::ifstream ifs(params_file);
		if (!ifs)
		{
			std::cerr << "Cannot open file " << params_file << std::endl;
			return 1;
		}

		ifs >> params;
	}

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
		algs.measure(alg, problem, params);
	}

	return 0;
}
