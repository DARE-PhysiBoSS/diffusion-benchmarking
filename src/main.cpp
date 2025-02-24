#include <iostream>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

int main(int argc, char** argv)
{
	argparse::ArgumentParser program("diffuse");

	bool validate;
	program.add_argument("--validate").help("Validate the results").flag().store_into(validate);

	std::string alg;
	program.add_argument("--alg").help("Algorithm to use").required().store_into(alg);

	std::string problem_file;
	program.add_argument("--problem").help("A file describing the problem size").required().store_into(problem_file);

	std::string output_file;
	program.add_argument("--output")
		.help("If present, the diffusion results will be stored in the output file")
		.store_into(output_file);

	std::string params_file;
	program.add_argument("--params").help("A file with the algorithm specific parameters").store_into(params_file);

	bool double_precision;
	program.add_argument("--double").help("Use double precision").flag().store_into(double_precision);

	try
	{
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	return 0;
}
