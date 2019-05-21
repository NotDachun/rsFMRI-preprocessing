from rsfmri_pp_afni import PreprocessingPipeline
from multiprocessing import cpu_count
import argparse
import os
import numpy as np
import sys

def parse_args(args):
    """
    Creates a parser and parses arguments.

    View help for valid flags and inputs.

    Make sure that subjects are aligned on the same line for all text files. For instance, if a subject's
    path to epi data is found on line 10 of the file containing paths to epi data, the subject's anatomical,
    output directory, and, optionally, subject id information should also be found on line 10 of the inputted
    text files.

    Parameters:
            args (List): A list of arguments appropriate to the parser, often just sys.argv[-1:]
    Returns:
            A Namespace object that maps flags to their value according to the inputted args
    """

    parser = argparse.ArgumentParser(
        description="""Generates and runs an afni_proc.py script to preprocess resting state fMRI data""",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional Flags
    parser.add_argument("-t", "--trs_remove", action="store", default=5, type=int, metavar='TRs',
                        help="""number of trs to remove at the beginning of the epi data
                                (default = 5 trs)""")
    parser.add_argument("-d", "--dim_voxel", action="store", default=2.0, type=float, metavar='MM',
                        help="voxel dimensions in mm that processed epi will be resampled to (default = 2.0 mm)")
    parser.add_argument("-b", "--bandpass", action="store", default=[0.01, 0.25], nargs=2, type=float, metavar="F",
                        help="bandpass frequencies lower and upper limits (default = 0.01 0.25)")
    parser.add_argument("-v", "--volumes", action="store", default=0, type=int, metavar="V",
                        help="""truncate the epi data to the inputted number of volumes, useful if subjects have data 
                        with different numbers of volumes (default = no truncation)""")
    parser.add_argument("-f", "--fwhm", action="store", default=5.0, type=float, metavar="MM",
                        help="the full width half maximum that is used when blurring (default = 5.0 mm)")
    parser.add_argument("-c", "--cores", action="store", default=cpu_count(), type=int, metavar="C",
                        help="number of cores supplied to 3dDeconvolve (default = all cores)")
    parser.add_argument("-s", "--subj_id", action="store", default="sub", metavar="SUB",
                        help="text file of subject ids (default = sub)")
    parser.add_argument("-T", "--time_step", action="store", default=0, type=float, metavar="TS",
                        help="set the time step for bandpassing (default = ts in header info")

    parser.add_argument("-g", "--global_signal_regression", action="store_false", default=True,
                        help="do not perform global signal regression (default = perform gsr)")

    parser.add_argument("-r", "--rerun", action="store_true", default=False,
                        help="""rerun preprocessing, override and delete previous results in 
                        'Processed' folder (default = don't override)""")
    parser.add_argument("-m", "--motion_param", action="store_true", default=False,
                        help="use 12 motion parameters for regression (default = 6 motion parameters)")
    parser.add_argument("-G", "--gm_blur", action="store_true", default=False,
                        help="blur only in grey matter mask (default = blur in whole brain)")
    parser.add_argument("-n", "--nl_reg", action="store_true", default=False,
                        help="use non-linear warp between anatomical and MNI template (default = linear warp)")

    # Required Inputs
    required = parser.add_argument_group("required arguments")
    required.add_argument("-e", "--epi", action="store", required=True,
                          help="text file of paths to raw epi data")
    required.add_argument("-a", "--anat", action="store", required=True,
                          help="text file of paths to raw anatomical data")
    required.add_argument("-o", "--out_dir", action="store", required=True, metavar="OUT",
                          help="text file of paths to output directory")
    result = parser.parse_args(args)

    # Make sure inputted parameters are legal
    assert (os.path.isfile(result.epi)), "{} does not exist or is not a file".format(result.epi)
    assert (os.path.isfile(result.anat)), "{} does not exist or is not a file".format(result.ant)
    assert (result.trs_remove >= 0), "Cannot remove negative trs"
    assert (result.dim_voxel >= 0), "Cannot have a negative voxel dimension"
    assert (np.all(np.array(result.bandpass) > 0)), "Cannot have a negative frequency limit for bandpassing"
    assert (result.volumes > -1), "Number of volumes must be greater than 0"
    assert (result.cores > 0), "Number of cores used must be greater than 0"
    assert (result.time_step > -1), "Time step must be greater than 0"

    return result

def run_multi_subject(pipeline, args, **kwargs):
    pipeline.new_outfile()
    if kwargs is not None:
        temp = vars(args)
        for key, value in kwargs:
            temp[key] = value.strip()
            pipeline.run(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    pipeline = PreprocessingPipeline("preprocessing_output.txt")

    with open(args.epi, "r") as epi, open(args.anat, "r") as anat, open(args.out_dir, "r") as out_dir:
        if not args.subj_id == "sub":
            with open(args.subj_id, "r") as sub:
                for e, a, o, s in zip(epi, anat, out_dir, sub):
                    run_multi_subject(pipeline, args, epi=e, anat=a, out_dir=o, subj_id=s)
        else:
            for e, a, o in zip(epi, anat, out_dir):
                # args.epi = e.strip()
                # args.anat = a.strip()
                # args.out_dir = o.strip()
                # pipeline.run(args)
                run_multi_subject(pipeline, args, epi=e, anat=a, out_dir=o)
