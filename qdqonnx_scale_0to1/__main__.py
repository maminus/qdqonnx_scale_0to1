import sys
import logging
import argparse
from pathlib import Path
from .qdq_scale_converter import convert_onnx_file


def _get_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument('-i', '--in-file', type=Path, help='input onnx file path')
    parser.add_argument('-o', '--out-file', type=Path, help='output onnx file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='output messages verbosely')
    args = parser.parse_args(argv[1:])
    return args


def _get_logger(is_verbose):
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if is_verbose else logging.INFO
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(asctime)s, %(levelname)s, %(filename)s, %(lineno)d, %(message)s'))
    logger.addHandler(handler)
    return logger


def main():
    args = _get_args(sys.argv)

    input_filepath = args.in_file
    output_filepath = args.out_file
    logger = _get_logger(args.verbose)

    convert_onnx_file(input_filepath, output_filepath, logger=logger)


if __name__ == '__main__':
    main()
