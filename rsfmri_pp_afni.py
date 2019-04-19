#! /usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:22:36 2019

@author: dzhu99

Resting state fMRI preprocessing script rewritten in Python

Rationale:
    -More tool support
    -Easier to maintain/change in the future
    -Better for larger scale programs
    -Ability to do unit testing

ToDo:
	-Test parse_args
	-Test override ability
	-Move files to correct folder
	-Test EPI on MNI snapshot
	-Create dynamic DMN QC
"""

import argparse
import os
import subprocess
import sys
import shutil
import numpy as np
from multiprocessing import cpu_count

# All output is written to rsfmri_pp_afni_output.txt for verification
# in case errors occur
name = os.path.basename(__file__)
output_file = open(name[:name.find(".py")] + "_output.txt", "w")

def record(text):
	"""
	Prints to both console and the specified output file 

	Parameters:
		text (String): the text to be written to the console and output file
	"""

	print(text)
	if output_file:
		output_file.write(str(text) + "\n")

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
	"""
	Checks if two floats are equal

	Parameters:
		a (float/double): float to be checked for equivalence
		b (float/double): float be to checked for equivalence
		rel_tol (float/double): relative tolerance 
		abs_tol (float/double): absolute tolerance
	Returns:
		True if floats are equal, otherwise false
	"""
	return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def parse_args(args):
	"""
	Creates a parser and parses arguments.

	View help for valid flags and inputs.

	Parameters:
		args (List): A list of arguments appropiate to the parser
	Returns:
		A Namespace object that maps flags to their value according to the inputted args
	"""
	
	parser = argparse.ArgumentParser(
	    description="""Generates and runs an afni_proc.py script to preprocess 
	                resting state fMRI data""",
	    formatter_class=argparse.RawDescriptionHelpFormatter)

	# Optional Flags
	parser.add_argument("-t", "--trs_remove", action="store", default=5, type=int, metavar='TRs',
	                    help="""number of seconds of trs to remove at the beginning of the epi data
	                            (default = 5 trs)""")
	parser.add_argument("-d", "--dim_voxel", action="store", default=2.0, type=float, metavar='MM',
	                    help="voxel dimensions in mm (default = 2.0 mm)")
	parser.add_argument("-b", "--bandpass", action="store", default=[0.01, 0.25], nargs=2, type=float, metavar="F",
	                    help="bandpass frequencies limits (default = 0.01 0.25)")
	parser.add_argument("-v", "--volumes", action="store", default=0, type=int, metavar="V",
						help="""truncate the epi data to the inputted number of volumes, useful if subjects have data 
						with different numbers of volumes (default = no truncation)""")
	parser.add_argument("-f", "--fwhm", action="store", default=5.0, type=float, metavar="MM",
						help="the full width half maximum that is used when blurring (default = 5.0 mm)")
	parser.add_argument("-c", "--cores", action="store", default=cpu_count(), type=int, metavar="C",
	                    help="number of cores supplied to 3dDeconvolve (default = all cores)")
	parser.add_argument("-s", "--subj_id", action="store", default="sub", metavar="SUB",
	                    help="subject id (default = sub")
	parser.add_argument("-B", "--bandpass_ts", action="store", default=0, type=float, metavar="TS",
						help="set the time step for bandpassing (default = ts in header info")


	parser.add_argument("-r", "--rerun", action="store_true", default=False,
	                    help="rerun preprocessing")
	parser.add_argument("-m", "--motion_param", action="store_true", default=False,
	                    help="use 12 motion parameters for regression (default = 6 motion parameters)")
	parser.add_argument("-g", "--global_signal_regression", action="store_false", default=True,
						help="perform global signal regression (default = perform gsr)")
	# parser.add_argument("-g", "--gm_blur", action="store_true", default=False,
	#                     help="blur only in grey matter mask (default = blur in whole brain)")
	parser.add_argument("-n", "--nl_reg", action="store_true", default=False,
	                    help="use non-linear warp between anatomical and MNI template (default = linear warp)")

	# Required Inputs
	required = parser.add_argument_group("required arguments")
	required.add_argument("-e", "--epi", action="store", required=True,
	                      help="path to raw epi data")
	required.add_argument("-a", "--anat", action="store", required=True,
	                      help="path to raw anatomical data")
	required.add_argument("-o", "--out_dir", action="store", required=True, metavar="OUT",
	                      help="path to output directory")
	result = parser.parse_args(args)

	# Make sure inputted parameters are legal
	assert (os.path.isfile(result.epi)), "{} does not exist or is not a file".format(result.epi)
	assert (os.path.isfile(result.anat)), "{} does not exist or is not a file".format(result.ant)
	assert (result.trs_remove >= 0), "Cannot remove negative trs"
	assert (result.dim_voxel >= 0), "Cannot have a negative voxel dimension"
	assert (np.all(np.array(result.bandpass) > 0)), "Cannot have a negative frequency limit for bandpassing"
	assert (result.volumes > -1), "Number of volumes must be greater than 0"
	assert (result.cores > 0), "Number of cores used must be greater than 0"
	assert (result.bandpass_ts > -1), "Time step must be greater than 0"

	return result

def create_outdir(out_dir):
	"""
	Creates the output directory if it does not already exist. Intermediate
	directories will be created as well.

	Parameters: 
		out_dir (String): Path to location of where output directory is created
	Returns:
		Path to the created output directory. Should be of form "out_dir/Processed"
	"""
	output_dir = out_dir + ("/" if out_dir[-1] != "/" else "") + "Processed/"
	if not os.path.exists(output_dir):
	    record("Creating output directory: " + output_dir)
	    os.makedirs(output_dir)
	else:
		record("Output directory already exists")
	return output_dir

def run_SSwarper(out_dir, subj_id, anat, template):
	"""
	Runs AFNI's @SSwarper with the inputted parameters if it has not already been performed

	Parameters:
		out_dir (String): Path to the output directory
		subj_id (String): Subject id
		anat (String): Path to the anatomical data
		mni (String): Path to the template to be used (Look at @SSWarper help page for more 
		information on templates)
	"""

	assert (os.path.isdir(out_dir)), "Output directory does not exist"
	assert (os.path.isfile(anat)), "Path to anatomical data does not exist"
	assert (os.path.isfile(template)), "Path to template does not exist"

	sswarper_outdir = out_dir + ("/" if out_dir[-1] != "/" else "") + "SSwarper_Output/"

    # Check if desired output exists already
	if (os.path.isdir(sswarper_outdir) and
        os.path.isfile(sswarper_outdir + "anatQQ.{}.nii".format(subj_id)) and
        os.path.isfile(sswarper_outdir + "anatQQ.{}.aff12.1D".format(subj_id)) and
        os.path.isfile(sswarper_outdir + "anatQQ.{}_WARP.nii".format(subj_id))):
		record("@SSwarper has already been performed")

	else:
		record("Performing @SSwarper")
		record("Creating " + sswarper_outdir)

		if os.path.exists(sswarper_outdir):
			shutil.rmtree(sswarper_outdir)
		os.makedirs(sswarper_outdir)

		record(subprocess.check_output([
		    "@SSwarper",
		    "-input", anat,
		    "-base", template,
		    "-subid", subj_id,
		    "-odir", sswarper_outdir], stderr=subprocess.STDOUT))

def truncate_epi(epi, out_dir, volumes):
	"""
	Truncates a copy of EPI data to the inputted number of volumes. Places this
	copy in the passed output directory

	Parameters:
		epi (String): Path to EPI data
		out_dir (String): Path to output directory where truncated EPI is placed
		volumes: The number of volumes that the EPI data is truncated to
	Returns:
		Path to the truncated EPI data
	"""

	assert (os.path.isfile(epi)), "Path to EPI data does not exist"
	assert (os.path.exists(out_dir)), "Output directory does not exist"
	assert (volumes > 0), "Desired volumes must be greater than 0"

	# Use afni's 3dinfo to check number of volumes in epi
	# Equivalent to shell command: 3dinfo -nv $epi
	subj_vol = int(subprocess.check_output([
		"3dinfo", "-nv", epi]))
	record("The current number of volumes is {}".format(subj_vol))

	if (subj_vol > volumes):
		record("Truncating to first {} volumes".format(volumes))
		epi_toVolumes = out_dir + "to{}".format(volumes)

		# Use afni's 3dTcat to truncate epi to inputted number of volumes
		# Equivalent to shell command: 3dTcat -prefix "${outdir}/to990" $epi'[0..989]
		record(subprocess.check_output([
			"3dTcat -prefix {} {}'[0..{}]'".format(epi_toVolumes, epi, volumes - 1)], 
			shell=True, stderr=subprocess.STDOUT))
		epi_toVolumes += "+orig."
		epi = epi_toVolumes

	check_vol = int(subprocess.check_output([
		"3dinfo", "-nv", epi]))
	assert (check_vol == volumes), "TCat EPI does not have the correct number of volumes"

	return epi

def generate_afni_proc(args, template):
	"""
	Generates the afni_proc.py script according to the inputted parameters

	Parameters:
		args (Namespace): Container that holds the desired parameters for preprocessing that are mapped to flags
		template: The template that will be the base for alignment
	Returns:
		afni_proc.py script in list format separated by spaces
	"""

	sswarper_outdir = args.out_dir + ("/" if args.out_dir[-1] != "/" else "") + "SSwarper_Output/"

	# Create afni_proc.py shell command
	record("Generating afni_proc.py script")
	afni_proc = [
	"afni_proc.py", 
	"-subj_id", args.subj_id,
	"-script", "RSproc.{}".format(args.subj_id), "-scr_overwrite",
	"-blocks", "despike", "align", "tlrc", "volreg", "mask", "blur", "scale", "regress",
	"-copy_anat", sswarper_outdir + "anatSS.{}.nii".format(args.subj_id),
	"-anat_has_skull", "no",
	"-dsets", args.epi,
	"-tcat_remove_first_trs", str(args.trs_remove),
	"-align_opts_aea", "-giant_move", "-cost", "lpc+zz",
	"-volreg_align_to", "MIN_OUTLIER",
	"-volreg_align_e2a",
	"-volreg_tlrc_warp",
	"-tlrc_base", template,
	"-volreg_warp_dxyz", str(args.dim_voxel)]

	# Pass SSwarper output to afni_proc.py
	if args.nl_reg:
		afni_proc.extend([
			"-tlrc_NL_warp",
			"-tlrc_NL_warped_dsets",
			sswarper_outdir + "anatQQ.{}.nii".format(args.subj_id),
			sswarper_outdir + "anatQQ.{}.aff12.1D".format(args.subj_id),
			sswarper_outdir + "anatQQ.{}_WARP.nii".format(args.subj_id)])

	afni_proc.extend([
		"-blur_size", str(args.fwhm),
		"-mask_segment_anat", "yes",
		"-mask_segment_erode", "yes",
		"-regress_motion_per_run",
		"-regress_bandpass", str(args.bandpass[0]), str(args.bandpass[1]),
		"-regress_apply_mot_types", "demean"])

	# Regress 12 motion parameters
	if args.motion_param:
		afni_proc.append("deriv")

	# Global Signal Regression
	if args.global_signal_regression:
		afni_proc.extend(["-regress_ROI", "brain"])


	afni_proc.extend([
		"-regress_opts_3dD",
		"-jobs", str(args.cores),
		"-regress_run_clustersim", "no"])

	assert (all(isinstance(param, str) for param in afni_proc)), "Not all parameters are strings"
	return afni_proc

def set_epi_tr(epi, tr):
	"""
	Modifies the tr header information of the EPI data

	Parameters:
		epi (String): Path to the EPI dat
		tr (float): The time step
	"""

	record("Ensuring TR in EPI header info is correct.")
	# Equivalent to shell command: 3dinfo -tr $epi
	curr_tr = float(subprocess.check_output(["3dinfo", "-tr", epi]))
	record("The current TR is {}".format(curr_tr))

	if (not isclose(curr_tr, tr)):
		record("Modifying TR from {} to be {}".format(curr_tr, tr))
		# Equivalent to shell command: 3drefit -TR $epi
		record(subprocess.check_output(["3drefit", "-TR", str(tr), epi]))

	check_tr = float(subprocess.check_output(["3dinfo", "-tr", epi]))

	assert (isclose(check_tr, tr)), "TR was not modified correctly"

def create_EM_snapshot(final_epi, subj_id, template):
	"""
	Creates a snapshot to check for alignment between the processed epi and
	an inputted template

	Parameters:
		final_epi (String): Path to the processed epi
		subj_id (String): Subject id
		template (String): Path to the template that was used for registration
	"""
	# Equivalent to shell command: @snapshot_volreg $template $subj.results/pb03.$subj.r01.volreg+tlrc. EM_$subj.jpg
	# Names snapshot EM_$subj.jpg
	record(subprocess.check_output([
		"@snapshot_volreg", template, final_epi, "EM_{}.jpg"format(subj_id)]))

def clean(subj_id):
	"""
	Removes previous {sub}.results directories from pwd

	Parameter:
		subj_id (String): Subject Id
	"""
	old_dir = "{}.results".format(subj_id)

	if (os.path.isdir(old_dir)):
		record("Clean - Removing {}".format(old_dir))
		shutil.rmtree(old_dir)

def run():
	"""
	Runs this script

	"""
	mni = "MNI152_T1_2mm_SSW.nii.gz"

	args = parse_args(sys.argv[1:])

	clean(args.subj_id)

	os.environ['OMP_NUM_THREADS'] = args.cores
	final_result = args.out_dir + create_outdir(args.out_dir)

	result = "{}.results/errts.sub.tproject+tlrc".format(args.subj_id)
	if (os.path.isfile(final_result + ".HEAD") and os.path.isfile(final_result + ".BRIK")
		and not args.rerun):
		record("Data has already been preprocessed.")
		record("Results of preprocessing are the following files: ")
		record(final_result + ".HEAD")
		record(final_result + ".BRIK")
		record("Use flag -r/--rerun to override results and rerun the preprocessing")
		return
	
	else:
		shutil.rmtree(args.out_dir)
		os.mkdir(args.out_dir)

		if args.nl_reg:
			run_SSwarper(args.out_dir, args.subj_id, args.anat, mni)
		if args.volumes:
			args.epi = truncate_epi()
		if args.bandpass_ts:
			set_epi_tr(args.epi, args.bandpass_ts)

		afni_proc = generate_afni_proc(args, mni)
		# Execute afni_proc.py, creates RSproc.$subj script
		record(subprocess.check_output(afni_proc, stderr=subprocess.STDOUT))

		# Execute RSproc.$subj
		record(subprocess.check_output(["tcsh", "-xef", "RSproc.{}".format(args.subj_id)]))

if __name__ == "__main__":
	run()
	