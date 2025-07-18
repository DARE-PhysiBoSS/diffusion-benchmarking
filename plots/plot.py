import re
import os
import json

import matplotlib.pyplot as plt
from collections import defaultdict
import csv
from pprint import pprint


def get_sum(path, counter):
    if not os.path.isfile(path):
        return None

    with open(path, 'r') as f:
        data = json.load(f)

    threads = data.get('threads', {})
    total = 0
    for thread in threads.values():
        regions = thread.get('regions', {})
        for region in regions.values():
            total += int(region.get(counter, 0))

    return total

    # print(f"Total {counter} across all threads: {total}")


def rename_files_in_dir(directory):
    for d in os.listdir(directory):
        if d.endswith(']'):
            pattern = r"profile_([^_]+?)_([^_]+?)_([^_]+?)x([^_]+?)_(.+?)_(\[.+?\])$"
            match = re.match(pattern, d)
            algorithm, precision, side_length, substrate_count, counters, params = match.groups()
            new_dirname = f"profile_{algorithm}_{precision}_{side_length}x{substrate_count}_{params}_{counters}"
        else:
            pattern = r"profile_([^_]+?)_([^_]+?)_([^_]+?)x([^_]+?)_([^[]+)$"
            match = re.match(pattern, d)
            if match:
                algorithm, precision, side_length, substrate_count, counters = match.groups()
                new_dirname = f"profile_{algorithm}_{precision}_{side_length}x{substrate_count}_[]_{counters}"

        if match:
            print(f"Renaming {d} to {new_dirname}")
            new_path = os.path.join(directory, new_dirname)
            old_path = os.path.join(directory, d)
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)


def extract_info_from_dir(dirname):
    # Example: profile_biofvm_s_100x8_params_PAPI_BR_MSP
    pattern = r"profile_(.+?)_(.+?)_(.+?)x(.+?)_\[(.*)\]_(.+?)$"
    match = re.match(pattern, dirname)
    if match:
        algorithm, precision, side_length, substrate_count, params, counters = match.groups()
        return algorithm, precision, side_length, substrate_count, params, counters
    return None, None, None, None, None, None


def find_unique_prefixes_without_counter(base_dir):
    prefixes = set()
    pattern = r"^(profile_.+?_.+?_.+?x.+?_\[.*\]_)"
    for d in os.listdir(base_dir):
        if d.startswith('profile'):
            match = re.match(pattern, d)
            if match:
                prefixes.add(match.group(1))
    return list(prefixes)


def get_benchmark_time(bench_file):
    bench_time = None
    if os.path.isfile(bench_file):
        min_std = None
        min_time = None
        with open(bench_file, 'r') as bf:
            
            # Replace all occurrences of '[' and ']' by '"'
            content = bf.read().replace('[', '"').replace(']', '"')
            bf.seek(0)
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                std_dev = float(row.get('std_dev', 'inf'))
                time_val = float(row.get('time', 'nan'))
                init_val = float(row.get('init_time', 'nan'))
                if min_std is None or std_dev < min_std:
                    min_std = std_dev
                    min_time = time_val + init_val
            bench_time = min_time

    if bench_time is None:
        # print(f"None at {bench_file}")
        bench_time = 10**12

    return bench_time


def build_table(counters_dir):
    dir_prefixes = find_unique_prefixes_without_counter(counters_dir)
    table = []
    for prefix in dir_prefixes:
        br_msp, fp_ins, fp_ops, l3_tcm, l2_tcm, l1_tcm, l2_dcm, l1_dcm, vec_ins, tot_ins = \
            None, None, None, None, None, None, None, None, None, None
        algorithm = precision = side_length = substrate_count = d = None
        for d in [d for d in os.listdir(counters_dir) if d.startswith(prefix)]:
            algorithm, precision, side_length, substrate_count, params, counters = extract_info_from_dir(
                d)
            json_file_path = os.path.join(
                counters_dir, d, 'papi_hl_output', 'rank_000000.json')
            if counters == 'PAPI_BR_MSP':
                br_msp = get_sum(json_file_path, 'PAPI_BR_MSP')
            if counters == 'PAPI_FP_OPS,PAPI_FP_INS':
                fp_ins = get_sum(json_file_path, 'PAPI_FP_INS')
                fp_ops = get_sum(json_file_path, 'PAPI_FP_OPS')
            if counters == 'PAPI_L3_TCM,PAPI_L2_TCM,PAPI_L1_TCM':
                l3_tcm = get_sum(json_file_path, 'PAPI_L3_TCM')
                l2_tcm = get_sum(json_file_path, 'PAPI_L2_TCM')
                l1_tcm = get_sum(json_file_path, 'PAPI_L1_TCM')
            if counters == 'PAPI_L2_DCM,PAPI_L1_DCM':
                l2_dcm = get_sum(json_file_path, 'PAPI_L2_DCM')
                l1_dcm = get_sum(json_file_path, 'PAPI_L1_DCM')
            if counters == 'PAPI_VEC_INS,PAPI_TOT_INS':
                vec_ins = get_sum(json_file_path, 'PAPI_VEC_INS')
                tot_ins = get_sum(json_file_path, 'PAPI_TOT_INS')

        if (br_msp is None or fp_ins is None or fp_ops is None or
                l3_tcm is None or l2_tcm is None or l1_tcm is None or
                l2_dcm is None or l1_dcm is None or vec_ins is None or
                tot_ins is None):
            # print(f"Missing data in {d}, skipping")
            continue

        # Read benchmark CSV and select time with smallest std_dev
        params_part = '' if params == '' else f'_[{params}]'
        bench_file = os.path.join(
            counters_dir.replace('counters', 'bench'),
            f'benchmark_{algorithm}_{precision}_{side_length}x{substrate_count}{params_part}.out'
        )
        bench_time = get_benchmark_time(bench_file)

        domain = 0
        if params != '':
            cx, cy, cz = params.split(',')
            l = int(side_length)
            domain = l/int(cx) * (l/int(cy)) * (l/int(cz))
            domain = domain * (4 if precision == 's' else 8)

            if l/int(cx) < 4 or l/int(cy) < 4 or l/int(cz) < 4:
                continue

        row = {
            'dir': d,
            'algorithm': algorithm,
            'precision': precision,
            'side_length': side_length,
            'substrate_count': substrate_count,
            'params': params,
            'domain': domain,
            'br_msp': br_msp,
            'fp_ins': fp_ins,
            'fp_ops': fp_ops,
            'l3_tcm': l3_tcm,
            'l2_tcm': l2_tcm,
            'l1_tcm': l1_tcm,
            'l2_dcm': l2_dcm,
            'l1_dcm': l1_dcm,
            'vec_ins': vec_ins,
            'tot_ins': tot_ins,
            'time': bench_time
        }
        table.append(row)

    # Print the table in a readable format
    # pprint(table)
    return table


iterations = 50


def get_problem_size(precision, side_length, substrates):
    data_type_size = 4 if precision == 's' else 8
    return iterations * data_type_size * substrates * side_length ** 3


def get_problem_theoretical_ops(side_length, substrates):
    dims = 3
    ops_per_element = 3
    return dims * ops_per_element * substrates * iterations * side_length ** dims


def plot_parametrized(rows):
    data_by_substrate = defaultdict(list)
    for row in table:
        if row['br_msp'] is not None and int(row['side_length']) <= 250 and row['algorithm'] is not None and row['substrate_count'] is not None and row['params'] != '':
            data_by_substrate[row['substrate_count']].append(row)

    # Collect all params across all data for consistent mapping
    all_params = set()
    for rows in data_by_substrate.values():
        for row in rows:
            if row['params']:
                all_params.add(row['params'])

    marker_styles = ['o', 's', '^', 'D', 'v', 'P',
                     '*', 'X', 'h', '1', '2', '3', '4', '8']
    color_styles = plt.cm.get_cmap('tab10', len(all_params))
    alg_list = sorted(all_params)
    alg_marker_map = {param: marker_styles[i % len(
        marker_styles)] for i, param in enumerate(alg_list)}
    alg_color_map = {alg: color_styles(i) for i, alg in enumerate(alg_list)}

    for substrate_count, rows in data_by_substrate.items():
        # Split rows by precision
        single_rows = [row for row in rows if row['precision'] == 's']
        double_rows = [row for row in rows if row['precision'] == 'd']

        # Group by param for each precision
        single_alg_data = defaultdict(list)
        for row in single_rows:
            single_alg_data[row['params']].append(row)
        double_alg_data = defaultdict(list)
        for row in double_rows:
            double_alg_data[row['params']].append(row)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['br_msp']/r['tot_ins'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('BR_MSP / TOT_INS')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['br_msp']/r['tot_ins'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_relative_br_msp_{substrate_count}.png')
        plt.close(fig)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['vec_ins']/r['fp_ins'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('VEC_INS / FP_INS')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['vec_ins']/r['fp_ins'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_vec_{substrate_count}.png')
        plt.close(fig)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l1_dcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 4) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('L1 Misses / data size in B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l1_dcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 8) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_l1_{substrate_count}.png')
        plt.close(fig)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l2_dcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 4) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('L2 Misses / data size in B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l2_dcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 8) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_l2_{substrate_count}.png')
        plt.close(fig)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l3_tcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 4) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('L3 Misses / data size in B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l3_tcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 8) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_l3_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

            # Left subplot: single precision
            ax = axes[0]
            for alg, alg_rows in single_alg_data.items():
                alg_rows_sorted = sorted(
                    alg_rows, key=lambda r: int(r['side_length']))
                x = [int(r['side_length']) for r in alg_rows_sorted]
                y = [r['fp_ops']/(r['time'] * 10**3) for r in alg_rows_sorted]
                ax.plot(x, y, marker=alg_marker_map[alg],
                        color=alg_color_map[alg], label=alg)
            ax.set_xlabel('Side Length')
            ax.set_ylabel('GFLOPS')
            ax.set_title('Single Precision')
            ax.legend()

            # Right subplot: double precision
            ax = axes[1]
            for alg, alg_rows in double_alg_data.items():
                alg_rows_sorted = sorted(
                    alg_rows, key=lambda r: int(r['side_length']))
                x = [int(r['side_length']) for r in alg_rows_sorted]
                y = [r['fp_ops']/(r['time'] * 10**3) for r in alg_rows_sorted]
                ax.plot(x, y, marker=alg_marker_map[alg],
                        color=alg_color_map[alg], label=alg)
            ax.set_xlabel('Side Length')
            ax.set_title('Double Precision')
            ax.legend()

            fig.suptitle(f'Substrate Count: {substrate_count}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'params_flops_{substrate_count}.png')
            plt.close(fig)
        except:
            pass

        # One Plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

            # Left subplot: single precision
            ax = axes[0]
            for alg, alg_rows in single_alg_data.items():
                alg_rows_sorted = sorted(
                    alg_rows, key=lambda r: int(r['side_length']))
                x = [int(r['side_length']) for r in alg_rows_sorted]
                y = [r['time'] for r in alg_rows_sorted]
                ax.plot(x, y, marker=alg_marker_map[alg],
                        color=alg_color_map[alg], label=alg)
            ax.set_xlabel('Side Length')
            ax.set_ylabel('time (us)')
            ax.set_yscale('log')
            ax.set_title('Single Precision')
            ax.legend()

            # Right subplot: double precision
            ax = axes[1]
            for alg, alg_rows in double_alg_data.items():
                alg_rows_sorted = sorted(
                    alg_rows, key=lambda r: int(r['side_length']))
                x = [int(r['side_length']) for r in alg_rows_sorted]
                y = [r['time'] for r in alg_rows_sorted]
                ax.plot(x, y, marker=alg_marker_map[alg],
                        color=alg_color_map[alg], label=alg)
            ax.set_xlabel('Side Length')
            ax.set_title('Double Precision')
            ax.legend()

            fig.suptitle(f'Substrate Count: {substrate_count}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'params_time_{substrate_count}.png')
            plt.close(fig)
        except:
            pass

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['time'] / get_problem_theoretical_ops(
                int(r['side_length']), int(substrate_count)) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('time per op (us)')
        ax.set_yscale('log')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['time'] / get_problem_theoretical_ops(
                int(r['side_length']), int(substrate_count)) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_relative_time_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / get_problem_size('s', int(r['side_length']), int(
                substrate_count)) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('arith. int.')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / get_problem_size('d', int(r['side_length']), int(
                substrate_count)) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_arithmetic_intensity_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l1_dcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('FP_OPS / L1 missed B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l1_dcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_arithmetic_intensity_l1_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l2_dcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('FP_OPS / L2 missed B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l2_dcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_arithmetic_intensity_l2_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l3_tcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('FP_OPS / L3 missed B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l3_tcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'params_arithmetic_intensity_l3_{substrate_count}.png')
        plt.close(fig)


def plot_parametrized_x(rows, size):
    data_by_substrate = defaultdict(list)
    for row in table:
        if row['br_msp'] is not None and int(row['side_length']) == size and row['algorithm'] is not None and row['substrate_count'] is not None and row['params'] != '':
            data_by_substrate[row['substrate_count']].append(row)

    # Collect all params across all data for consistent mapping
    all_params = set()
    for rows in data_by_substrate.values():
        for row in rows:
            if row['params']:
                all_params.add(row['params'])

    marker_styles = ['o', 's', '^', 'D', 'v', 'P',
                     '*', 'X', 'h', '1', '2', '3', '4', '8']
    color_styles = plt.cm.get_cmap('tab10', len(all_params))
    param_list = sorted(all_params)
    param_marker_map = {param: marker_styles[i % len(
        marker_styles)] for i, param in enumerate(param_list)}
    param_color_map = {alg: color_styles(i)
                       for i, alg in enumerate(param_list)}

    for substrate_count, rows in data_by_substrate.items():
        # Split rows by precision
        single_rows = [row for row in rows if row['precision'] == 's']
        double_rows = [row for row in rows if row['precision'] == 'd']

        # Group by param for each precision
        single_alg_data = defaultdict(list)
        for row in single_rows:
            single_alg_data[row['params']].append(row)
        double_alg_data = defaultdict(list)
        for row in double_rows:
            double_alg_data[row['params']].append(row)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))

            x = [int(r['domain']) for r in alg_rows_sorted]
            y = [r['time'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=param_marker_map[alg],
                    color=param_color_map[alg], label=alg)
        ax.set_xlabel('Domain size')
        ax.set_xscale('log')
        ax.set_ylabel('time')
        ax.set_yscale('log')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['domain']) for r in alg_rows_sorted]
            y = [r['time'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=param_marker_map[alg],
                    color=param_color_map[alg], label=alg)
        ax.set_xlabel('Domain size')
        ax.set_xscale('log')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'domain_{size}_{substrate_count}.png')
        plt.close(fig)


if __name__ == "__main__":
    counters_dir = '../data/counters'
    rename_files_in_dir(counters_dir)
    table = build_table(counters_dir)

    plot_parametrized(table)
    plot_parametrized_x(table, 100)
    plot_parametrized_x(table, 150)
    plot_parametrized_x(table, 200)
    plot_parametrized_x(table, 250)


    # Filter: For algorithm 'lstmfpabni', for each (side_length, precision, substrate_count), keep only the row with the minimum time
    filtered_table = []
    # Group lstmfpabni rows by (side_length, precision, substrate_count)
    from collections import defaultdict
    lstm_groups = defaultdict(list)
    for row in table:
        if row['algorithm'] == 'lstmfpabni':
            key = (row['side_length'], row['precision'], row['substrate_count'])
            lstm_groups[key].append(row)
        else:
            filtered_table.append(row)
    # For each group, keep only the row with the minimum time
    for group_rows in lstm_groups.values():
        min_row = min(group_rows, key=lambda r: r['time'])
        filtered_table.append(min_row)

    table = filtered_table


    # Organize data by substrate_count
    data_by_substrate = defaultdict(list)
    for row in table:
        if row['br_msp'] is not None and int(row['side_length']) > 25 and row['algorithm'] in ["biofvm", "lstcta", "lstmfpabni"] and row['substrate_count'] is not None:
            data_by_substrate[row['substrate_count']].append(row)

    # Rename 'biofvm' to 'reference' in data_by_substrate
    for rows in data_by_substrate.values():
        for row in rows:
            if row['algorithm'] == 'biofvm':
                row['algorithm'] = 'Reference'
            if row['algorithm'] == 'lstcta':
                row['algorithm'] = 'Temporal'
            if row['algorithm'] == 'lstmfpabni':
                row['algorithm'] = 'Cache-aware'

    # Collect all algorithms across all data for consistent mapping
    all_algorithms = set()
    for rows in data_by_substrate.values():
        for row in rows:
            if row['algorithm']:
                all_algorithms.add(row['algorithm'])

    marker_styles = ['o', 's', '^', 'D', 'v', 'P',
                     '*', 'X', 'h', '1', '2', '3', '4', '8']
    color_styles = plt.cm.get_cmap('tab10', len(all_algorithms))
    alg_list = sorted(all_algorithms)
    alg_marker_map = {alg: marker_styles[i % len(
        marker_styles)] for i, alg in enumerate(alg_list)}
    alg_color_map = {alg: color_styles(i) for i, alg in enumerate(alg_list)}

    for substrate_count, rows in data_by_substrate.items():
        # Split rows by precision
        single_rows = [row for row in rows if row['precision'] == 's']
        double_rows = [row for row in rows if row['precision'] == 'd']

        # Group by algorithm for each precision
        single_alg_data = defaultdict(list)
        for row in single_rows:
            single_alg_data[row['algorithm']].append(row)
        double_alg_data = defaultdict(list)
        for row in double_rows:
            double_alg_data[row['algorithm']].append(row)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['br_msp']/r['tot_ins'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('BR_MSP / TOT_INS')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['br_msp']/r['tot_ins'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'relative_br_msp_{substrate_count}.png')
        plt.close(fig)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['vec_ins']/r['fp_ins'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('VEC_INS / FP_INS')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['vec_ins']/r['fp_ins'] for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'vec_{substrate_count}.png')
        plt.close(fig)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l1_dcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 4) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('L1 Misses / data size in B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l1_dcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 8) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'l1_{substrate_count}.png')
        plt.close(fig)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l2_dcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 4) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('L2 Misses / data size in B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l2_dcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 8) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'l2_{substrate_count}.png')
        plt.close(fig)

        # One Plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l3_tcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 4) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('L3 Misses / data size in B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [int(r['l3_tcm']) / ((int(r['side_length']) ** 3)
                                     * int(substrate_count) * 50 * 8) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'l3_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

            # Left subplot: single precision
            ax = axes[0]
            for alg, alg_rows in single_alg_data.items():
                alg_rows_sorted = sorted(
                    alg_rows, key=lambda r: int(r['side_length']))
                x = [int(r['side_length']) for r in alg_rows_sorted]
                y = [r['fp_ops']/(r['time'] * 10**3) for r in alg_rows_sorted]
                ax.plot(x, y, marker=alg_marker_map[alg],
                        color=alg_color_map[alg], label=alg)
            ax.set_xlabel('Side Length')
            ax.set_ylabel('GFLOPS')
            ax.set_title('Single Precision')
            ax.legend()

            # Right subplot: double precision
            ax = axes[1]
            for alg, alg_rows in double_alg_data.items():
                alg_rows_sorted = sorted(
                    alg_rows, key=lambda r: int(r['side_length']))
                x = [int(r['side_length']) for r in alg_rows_sorted]
                y = [r['fp_ops']/(r['time'] * 10**3) for r in alg_rows_sorted]
                ax.plot(x, y, marker=alg_marker_map[alg],
                        color=alg_color_map[alg], label=alg)
            ax.set_xlabel('Side Length')
            ax.set_title('Double Precision')
            ax.legend()

            fig.suptitle(f'Substrate Count: {substrate_count}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'flops_{substrate_count}.png')
            plt.close(fig)
        except:
            pass

        # One Plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

            # Left subplot: single precision
            ax = axes[0]
            for alg, alg_rows in single_alg_data.items():
                alg_rows_sorted = sorted(
                    alg_rows, key=lambda r: int(r['side_length']))
                x = [int(r['side_length']) for r in alg_rows_sorted]
                y = [r['time'] for r in alg_rows_sorted]
                ax.plot(x, y, marker=alg_marker_map[alg],
                        color=alg_color_map[alg], label=alg)
            ax.set_xlabel('Side Length')
            ax.set_ylabel('time (us)')
            ax.set_yscale('log')
            ax.set_title('Single Precision')
            ax.legend()

            # Right subplot: double precision
            ax = axes[1]
            for alg, alg_rows in double_alg_data.items():
                alg_rows_sorted = sorted(
                    alg_rows, key=lambda r: int(r['side_length']))
                x = [int(r['side_length']) for r in alg_rows_sorted]
                y = [r['time'] for r in alg_rows_sorted]
                ax.plot(x, y, marker=alg_marker_map[alg],
                        color=alg_color_map[alg], label=alg)
            ax.set_xlabel('Side Length')
            ax.set_title('Double Precision')
            ax.legend()

            fig.suptitle(f'Substrate Count: {substrate_count}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'time_{substrate_count}.png')
            plt.close(fig)
        except:
            pass

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [(10 ** 6) * r['time'] / get_problem_theoretical_ops(
                int(r['side_length']), int(substrate_count)) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('time per op (ps)')
        ax.set_yscale('log')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [(10 ** 6) * r['time'] / get_problem_theoretical_ops(
                int(r['side_length']), int(substrate_count)) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'relative_time_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / get_problem_size('s', int(r['side_length']), int(
                substrate_count)) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('arith. int.')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / get_problem_size('d', int(r['side_length']), int(
                substrate_count)) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'arithmetic_intensity_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l1_dcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('FP_OPS / L1 missed B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l1_dcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'arithmetic_intensity_l1_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l2_dcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('FP_OPS / L2 missed B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l2_dcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'arithmetic_intensity_l2_{substrate_count}.png')
        plt.close(fig)

        # One Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Left subplot: single precision
        ax = axes[0]
        for alg, alg_rows in single_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l3_tcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_ylabel('FP_OPS / L3 missed B')
        ax.set_title('Single Precision')
        ax.legend()

        # Right subplot: double precision
        ax = axes[1]
        for alg, alg_rows in double_alg_data.items():
            alg_rows_sorted = sorted(
                alg_rows, key=lambda r: int(r['side_length']))
            x = [int(r['side_length']) for r in alg_rows_sorted]
            y = [r['fp_ops'] / (r['l3_tcm'] * 64) for r in alg_rows_sorted]
            ax.plot(x, y, marker=alg_marker_map[alg],
                    color=alg_color_map[alg], label=alg)
        ax.set_xlabel('Side Length')
        ax.set_title('Double Precision')
        ax.legend()

        fig.suptitle(f'Substrate Count: {substrate_count}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'arithmetic_intensity_l3_{substrate_count}.png')
        plt.close(fig)
