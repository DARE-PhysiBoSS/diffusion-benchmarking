"""
Benchmarking Data Generator and Runner

This script automates the generation of benchmarking data and the execution of diffusion benchmarks for various algorithms and parameter combinations.

Features:
- Parses a benchmark JSON file describing groups, problem sizes, algorithms, and parameter sweeps.
- Generates input problem files and parameter files for each group and algorithm.
- Iterates over all combinations of algorithm parameters using Cartesian product.
- Runs the specified executable for each problem/parameter combination and collects results.
- Supports running benchmarks for all groups or a selected group.

Usage:
    python benchmark.py <benchmark_json_path> <executable_path> [-g GROUP_NAME]

Arguments:
    benchmark_json_path: Path to the benchmark JSON file.
    executable_path: Path to the diffusion benchmark executable.
    -g, --group: (Optional) Name of the group to benchmark. If omitted, all groups are processed.

JSON Schema:
    {
        "name": <benchmark_name>,
        "groups": {
            <group_name>: {
                "sizes": ["<x_dim>x<y_dim>x<z_dim>x<substrates_count>x<iterations>", ...],
                "data_type": <"float" or "double">,
                "default_params": {<param_name>: <value>, ...},
                "runs": {
                    <run_name>: {
                        "alg": <algorithm_name>,
                        "params": {<param_name>: [<values>], ...}
                    },
                    ...
                }
            },
            ...
        }
    }

Outputs:
- Problem and parameter files are generated in a directory structure under <benchmark_name>/<group_name>/.
- Results are saved as CSV files for each run/parameter/problem combination.
"""

import sys
import json
import itertools
import os
from typing import TypedDict, List, Dict, Any
import argparse
import subprocess


def create_problems_for_group(group_directory: str, sizes: list[str]):
    """
    Generate problem JSON files for each specified size in a group.

    Args:
        group_directory (str): Directory to store problem files.
        sizes (list[str]): List of problem size strings (e.g., "32x32x32x1").
    """
    for problem_size in sizes:
        parts = [int(x) for x in problem_size.split('x')]
        dims = len(parts) - 2
        if (dims < 2):
            print(
                f"Error: Incorrect problem size: {problem_size}", file=sys.stderr)
            sys.exit(1)
        iterations: int = int(parts[-1])
        substrate_count: int = int(parts[-2])
        nx = int(parts[0])
        ny = int(parts[1])
        nz = int(parts[2]) if dims == 3 else 1

        template: dict[str, object] = {
            "dims": dims,
            "dx": 20,
            "dy": 20,
            "dz": 20,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "substrates_count": substrate_count,
            "iterations": iterations,
            "dt": 0.01,
            "diffusion_coefficients": 10000,
            "decay_rates": 0.01,
            "initial_conditions": 1000,
            "gaussian_pulse": False
        }

        input_dir = f"{group_directory}/data"
        os.makedirs(input_dir, exist_ok=True)

        output_path = f"{input_dir}/{problem_size}.json"
        with open(output_path, "w") as outfile:
            json.dump(template, outfile, indent=2)


def create_param_file_for_alg(algorithm_directory: str, default_params: dict[str, object],  params: dict[str, list[int]]):
    """
    Generate parameter JSON files for each combination of algorithm parameters.

    Args:
        algorithm_directory (str): Directory to store parameter files.
        default_params (dict): Default parameters to include in each file.
        params (dict): Dictionary of parameter lists to sweep over.
    """
    keys = list(params.keys())
    values = [params[k] for k in keys]

    for combination in itertools.product(*values):
        combo_dict = dict(zip(keys, combination))
        combo_dict: dict[str, object] = {**default_params, **combo_dict}

        args = "_".join(str(x) for x in combination)

        param_dir = f"{algorithm_directory}/{args}"
        os.makedirs(param_dir, exist_ok=True)

        output_path = f"{algorithm_directory}/{args}/params.json"
        with open(output_path, "w") as outfile:
            json.dump(combo_dict, outfile, indent=2)


class RunDict(TypedDict):
    """
    TypedDict for a single run configuration.
    Fields:
        alg (str): Algorithm name.
        params (Dict[str, List[int]]): Parameter sweep dictionary.
    """
    alg: str
    params: Dict[str, List[int]]


class GroupDict(TypedDict):
    """
    TypedDict for a benchmark group configuration.
    Fields:
        sizes (List[str]): List of problem size strings.
        default_params (Dict[str, Any]): Default parameters for the group.
        data_type (str): Data type ("float" or "double").
        runs (Dict[str, RunDict]): Dictionary of run configurations.
    """
    sizes: List[str]
    default_params: Dict[str, Any]
    data_type: str
    runs: Dict[str, RunDict]


def generate_data(bench_name: str, group_name: str, group: GroupDict):
    """
    Generate all problem and parameter files for a benchmark group.

    Args:
        bench_name (str): Benchmark name.
        group_name (str): Group name.
        group (GroupDict): Group configuration dictionary.
    """
    group_dir = f"{bench_name}/{group_name}"
    os.makedirs(group_dir, exist_ok=True)

    create_problems_for_group(group_dir, group["sizes"])

    for run_name, run in group["runs"].items():
        if run_name == "data":
            print("Run name can not be named 'data'.",  file=sys.stderr)
            sys.exit(1)
        alg_dir = f'{group_dir}/{run_name}'
        os.makedirs(alg_dir, exist_ok=True)

        create_param_file_for_alg(
            alg_dir, group["default_params"], run["params"])


def diffuse(executable_path: str, problem_path: str, params_path: str, alg: str, double: bool):
    """
    Run the benchmark executable for a given problem and parameter set.

    Args:
        executable_path (str): Path to the benchmark executable.
        problem_path (str): Path to the problem JSON file.
        params_path (str): Path to the parameter JSON file.
        alg (str): Algorithm name.
        double (bool): Whether to use double precision.
    """
    problems_descriptor = os.path.basename(problem_path)
    params_descriptor = os.path.basename(os.path.dirname(params_path))
    out_csv_path = os.path.join(os.path.dirname(
        params_path), f"{os.path.splitext(problems_descriptor)[0]}.csv")

    cmd = [
        executable_path,
        "--alg", alg,
        "--problem", problem_path,
        "--params", params_path,
        "--benchmark"
    ]

    if double:
        cmd.append("--double")

    if os.path.exists(out_csv_path):
        print(f"{alg} {problems_descriptor} {params_descriptor} skipped.")
        return

    try:
        with open(out_csv_path, "w") as outfile:
            subprocess.run(cmd, stdout=outfile,
                           stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        os.remove(out_csv_path)
        print(f"{alg} {problems_descriptor} {params_descriptor} Failed.")


def run_single_group(executable: str, bench_name: str, group_name: str, group: GroupDict):
    """
    Run all benchmarks for a single group, iterating over problems and parameter sets.

    Args:
        executable (str): Path to the benchmark executable.
        bench_name (str): Benchmark name.
        group_name (str): Group name.
        group (GroupDict): Group configuration dictionary.
    """
    group_dir = f"{bench_name}/{group_name}"
    double: bool = group["data_type"] == "double"

    for problem_file in os.listdir(os.path.join(group_dir, "data")):
        problem_path = os.path.join(group_dir, "data", problem_file)

        for run_name in os.listdir(group_dir):
            if run_name == "data":
                continue

            alg_dir = os.path.join(group_dir, run_name)
            alg = group["runs"][run_name]["alg"]

            for param_name in os.listdir(alg_dir):
                params_path = os.path.join(
                    group_dir, run_name, param_name, "params.json")

                diffuse(executable, problem_path, params_path, alg, double)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking data generator")
    parser.add_argument("benchmark_path", help="Path to JSON file")
    parser.add_argument("executable_path", help="Path to executable")
    parser.add_argument(
        "-g", "--group", required=False, action='store', help="Group name to benchmark")

    args = parser.parse_args()
    json_path = args.benchmark_path
    exe_path = args.executable_path

    selected_group_name = args.group

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error parsing JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    name = data["name"]
    groups = data["groups"]

    print("Generating data...")

    if selected_group_name is None:
        for group_name, group in groups.items():
            generate_data(name, group_name, group)
    else:
        if selected_group_name in groups.keys():
            print(
                f"Error: Group '{selected_group_name}' not found.", file=sys.stderr)
            sys.exit(1)
        generate_data(name, selected_group_name, groups[selected_group_name])

    print("Running...")

    if selected_group_name is None:
        for group_name, group in groups.items():
            run_single_group(exe_path, name, group_name, group)
    else:
        run_single_group(exe_path, name, selected_group_name,
                         groups[selected_group_name])
