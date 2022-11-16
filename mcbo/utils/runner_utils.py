r"""
A set of helper functions for the runner.py script.
"""
import argparse


def check_nonnegative(value):
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0:
        raise argparse.ArgumentTypeError("%s is an invalid nonnegative float value" % f)
    return f
