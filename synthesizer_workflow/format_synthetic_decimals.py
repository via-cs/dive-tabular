"""Match synthetic CSV decimal precision to a reference training split.

The maximum observed number of decimal places is inferred independently for
each numeric reference column. Synthetic values are rounded and written in
fixed-point notation with exactly that many decimal places.

Input synthetic files are overwritten unless --output-dir is supplied.
"""

import argparse
from pathlib import Path

import pandas as pd

from .decimal_formatting import format_synthetic_file, infer_decimal_places


def collect_csv_paths(inputs):
    """Expand file and directory inputs into a de-duplicated CSV path list."""
    paths = []
    seen = set()
    for input_path in map(Path, inputs):
        if input_path.is_dir():
            candidates = sorted(input_path.glob('*.csv'))
        elif input_path.is_file():
            candidates = [input_path]
        else:
            raise FileNotFoundError(f'Synthetic input not found: {input_path}')
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                paths.append(path)
                seen.add(resolved)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--reference',
        type=Path,
        required=True,
        help='Training CSV used to infer maximum precision per column.',
    )
    parser.add_argument(
        'synthetic_inputs',
        nargs='+',
        type=Path,
        help='Synthetic CSV files or directories containing synthetic CSVs.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Write files here instead of overwriting their inputs.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reference = pd.read_csv(args.reference)
    decimal_places = infer_decimal_places(reference)
    synthetic_paths = collect_csv_paths(args.synthetic_inputs)
    if not synthetic_paths:
        raise FileNotFoundError('No CSV files found in the synthetic inputs.')
    for synthetic_path in synthetic_paths:
        output_path = (
            None
            if args.output_dir is None
            else args.output_dir / synthetic_path.name
        )
        print(
            format_synthetic_file(
                reference,
                synthetic_path,
                output_path,
                decimal_places=decimal_places,
            )
        )


if __name__ == '__main__':
    main()
