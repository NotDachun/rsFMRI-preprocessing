#! /usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:22:36 2019

@author: dzhu99

Resting state fMRI preprocessing script rewritten in Python
Call this script using: python rsfmri_pp_afni.py [options]

Rationale:
    -More tool support
    -Easier to maintain/change in the future
    -Better for larger scale programs
    -Ability to do unit testing

ToDo:
    -Test override ability

    -Run preprocessing on multiple subjects
"""

import argparse
import os
import subprocess
import sys
import shutil
import numpy as np
import dynamic_connectivity as dc
from multiprocessing import cpu_count

"""
PreprocessingPipeline is meant to preprocess rsFMRI data through the AFNI
software package. 
"""
class PreprocessingPipeline(object):

    mni = "MNI152_T1_2mm_SSW.nii"

    def __init__(self, output_file="rsfmri_pp_afni_output.txt"):
        self.output_file = open(output_file, "w")

    def record(self, text):
        """
        Prints to both console and the specified output file 

        Parameters:
                text (String): the text to be written to the console and output file
        """

        print(text)
        self.output_file.write(str(text) + "\n")

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=0.0):
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
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def cpe_output(self, exception, text):
        """
        Error message to be printed when a CalledProcessError is raised

        Parameters:
                exception (CallProcessError): caught exception
                text (String): error message to be printed
        """
        self.record("Error raised on command: ")
        self.record(exception.cmd)
        self.record(exception.output)
        self.record(text)

    def create_outdir(self, out_dir):
        """
        Creates the output directory if it does not already exist. Intermediate
        directories will be created as well.

        Parameters: 
                out_dir (String): Path to location of where output directory is created
        Returns:
                Path to the created output directory. Should be of form "out_dir/Processed"
        """

        output_dir = os.path.join(out_dir, "Processed/")
        if not os.path.exists(output_dir):
            self.record("Creating output directory: " + output_dir)
            os.makedirs(output_dir)
        else:
            self.record("Output directory already exists")

        assert os.path.isdir(output_dir), "Output directory was not created"
        return output_dir

    def run_SSwarper(self, out_dir, subj_id, anat, template):
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

        sswarper_outdir = os.path.join(out_dir, "SSwarper_Output/")

    # Check if desired output exists already
        if (os.path.isdir(sswarper_outdir) and
            os.path.isfile(os.path.join(sswarper_outdir, "anatQQ.{}.nii".format(subj_id))) and
            os.path.isfile(os.path.join(sswarper_outdir, "anatQQ.{}.aff12.1D".format(subj_id))) and
                os.path.isfile(os.path.join(sswarper_outdir, "anatQQ.{}_WARP.nii".format(subj_id)))):
            self.record("@SSwarper has already been performed")

        else:
            self.record("Performing @SSwarper")
            self.record("Creating " + sswarper_outdir)

            if os.path.exists(sswarper_outdir):
                shutil.rmtree(sswarper_outdir)
            os.makedirs(sswarper_outdir)

            try:
                self.record(subprocess.check_output([
                    "@SSwarper",
                    "-input", anat,
                    "-base", template,
                    "-subid", subj_id,
                    "-odir", sswarper_outdir], stderr=subprocess.STDOUT))

            except subprocess.CalledProcessError as e:
                self.cpe_output(e, "@SSWarper failed.")

        assert os.path.isdir(sswarper_outdir)
        assert os.path.isfile(os.path.join(sswarper_outdir, "anatQQ.{}.nii".format(subj_id)))
        assert os.path.isfile(os.path.join(sswarper_outdir, "anatQQ.{}.aff12.1D".format(subj_id)))
        assert os.path.isfile(os.path.join(sswarper_outdir, "anatQQ.{}_WARP.nii".format(subj_id)))

    def truncate_epi(self, epi, out_dir, volumes):
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
        subj_vol = int(subprocess.check_output(["3dinfo", "-nv", epi]))
        self.record("The current number of volumes is {}".format(subj_vol))

        if (subj_vol > volumes):
            self.record("Truncating to first {} volumes".format(volumes))
            epi_toVolumes = out_dir + "to{}".format(volumes)

            # Use afni's 3dTcat to truncate epi to inputted number of volumes
            # Equivalent to shell command: 3dTcat -prefix "${outdir}/to990" $epi'[0..989]
            try:
                self.record(subprocess.check_output([
                    "3dTcat -prefix {} {}'[0..{}]'".format(epi_toVolumes, epi, volumes - 1)],
                    shell=True, stderr=subprocess.STDOUT))
                epi_toVolumes += "+orig."
                epi = epi_toVolumes
            except subprocess.CalledProcessError as e:
                self.cpe_output(e, "3dTcat failed.")

        try:
            check_vol = int(subprocess.check_output([
                "3dinfo", "-nv", epi]))
        except subprocess.CalledProcessError as e:
            self.cpe_output(e, "EPI does not exist")

        assert (check_vol == volumes), "TCat EPI does not have the correct number of volumes"
        return epi

    def set_gm_mask(self, rsproc):
        """
        Modifies a RSproc file by supplying the grey matter mask as an input to the blur_in_mask
        function

        Parameters:
                rsproc (String): Path to the RSproc
        """

        assert os.path.isfile(rsproc), "File does not exist"

        with open(rsproc, "r") as file:
            filedata = file.read()

        filedata = filedata.replace("-Mmask full_mask.$subj+tlrc", "-mask mask_GM_resam+tlrc")

        with open(rsproc, "w") as file:
            file.write(filedata)

    def generate_afni_proc(self, args, template):
        """
        Generates the afni_proc.py script according to the inputted parameters

        Parameters:
                args (Namespace): Container that holds the desired parameters for preprocessing that are mapped to flags
                template: The template that will be the base for alignment
        Returns:
                afni_proc.py script in list format separated by spaces.
                A name that corresponds to the preprocessing steps, used by to rename the output later
        """

        assert os.path.isfile(template), "Template does not exist"

        name = "rsfmri_despike_toMNI_"
        sswarper_outdir = os.path.join(args.out_dir, "SSwarper_Output/")

        # Create afni_proc.py shell command
        self.record("Generating afni_proc.py script")
        afni_proc = [
            "afni_proc.py",
            "-subj_id", args.subj_id,
            "-script", "RSproc.{}".format(args.subj_id), "-scr_overwrite",
            "-blocks", "despike", "align", "tlrc", "volreg", "mask", "blur", "scale", "regress",
            "-copy_anat"]

        if args.nl_reg:
            assert (os.path.isfile(os.path.join(sswarper_outdir, "anatSS.{}.nii".format(args.subj_id))))

            afni_proc.append(os.path.join(sswarper_outdir, "anatSS.{}.nii".format(args.subj_id)))
        else:
            assert (os.path.isfile(args.anat))

            afni_proc.append(args.anat)

        afni_proc.extend(["-anat_has_skull", "no",
                          "-dsets", args.epi,
                          "-tcat_remove_first_trs", str(args.trs_remove),
                          "-align_opts_aea", "-giant_move", "-cost", "lpc+zz",
                          "-volreg_align_to", "MIN_OUTLIER",
                          "-volreg_align_e2a",
                          "-volreg_tlrc_warp",
                          "-tlrc_base", template,
                          "-volreg_warp_dxyz", str(args.dim_voxel)])

        # Pass SSwarper output to afni_proc.py
        if args.nl_reg:
            afni_proc.extend([
                "-tlrc_NL_warp",
                "-tlrc_NL_warped_dsets",
                os.path.join(sswarper_outdir, "anatQQ.{}.nii".format(args.subj_id)),
                os.path.join(sswarper_outdir, "anatQQ.{}.aff12.1D".format(args.subj_id)),
                os.path.join(sswarper_outdir, "anatQQ.{}_WARP.nii".format(args.subj_id))])

        # If blurring only in grey matter, supply the blur in mask option.
        # But this does not supply the grey matter mask, must call self.set_gm_mask
        # afterwards to correctly modify the RSproc script to use the AFNI
        # generated grey matter mask.
        afni_proc.extend(["-blur_size", str(args.fwhm)])
        name += str(args.fwhm) + "mmSmooth_"

        if args.gm_blur:
            afni_proc.extend(["-blur_in_mask", "yes"])

        afni_proc.extend([
            "-mask_segment_anat", "yes",
            "-mask_segment_erode", "yes",
            "-regress_motion_per_run",
            "-regress_bandpass", str(args.bandpass[0]), str(args.bandpass[1]),
            "-regress_apply_mot_types", "demean"])
        name += "bandpass_"

        # Regress 12 motion parameters
        if args.motion_param:
            afni_proc.append("deriv")

        # Global Signal Regression
        if args.global_signal_regression:
            afni_proc.extend(["-regress_ROI", "brain"])
            name += "gsregress_"

        afni_proc.extend([
            "-regress_opts_3dD",
            "-jobs", str(args.cores),
            "-regress_run_clustsim", "no"])

        assert (all(isinstance(param, str) for param in afni_proc)), "Not all parameters are strings"

        if name[-1] == "_":
            name = name[:-1]

        return afni_proc, name

    def set_epi_tr(self, epi, tr):
        """
        Modifies the tr header information of the EPI data

        Parameters:
                epi (String): Path to the EPI dat
                tr (float): The time step
        """

        assert (os.path.isfile(epi))
        assert (tr > 0)

        self.record("Ensuring TR in EPI header info is correct.")
        # Equivalent to shell command: 3dinfo -tr $epi
        curr_tr = float(subprocess.check_output(["3dinfo", "-tr", epi]))
        self.record("The current TR is {}".format(curr_tr))

        if (not self.isclose(curr_tr, tr)):
            self.record("Modifying TR from {} to be {}".format(curr_tr, tr))
            # Equivalent to shell command: 3drefit -TR $epi
            self.record(subprocess.check_output(["3drefit", "-TR", str(tr), epi]))

        check_tr = float(subprocess.check_output(["3dinfo", "-tr", epi]))

        assert (self.isclose(check_tr, tr)), "TR was not modified correctly"

    def create_EM_snapshot(self, volreg_epi, subj_id, out_dir, template):
        """
        Creates a snapshot to check for alignment between the processed epi and
        an inputted template

        Parameters:
                volreg_epi (String): Path to the processed epi
                subj_id (String): Subject id
                out_dir (String): Path to output directory
                template (String): Path to the template that was used for registration
        """

        assert (os.path.isfile(volreg_epi + "HEAD"))
        assert (os.path.isfile(volreg_epi + "BRIK"))
        assert (os.path.isdir(out_dir))
        assert (os.path.isfile(template))

        try:
            # Equivalent to shell command: @snapshot_volreg $template $subj.results/pb03.$subj.r01.volreg+tlrc. $outdir/EM_$subj.jpg
            # Names snapshot EM_$subj.jpg
            self.record(subprocess.check_output([
                "@snapshot_volreg", template, volreg_epi, os.path.join(out_dir, "EM_{}.jpg".format(subj_id))], stderr=subprocess.STDOUT))
        except subprocess.CalledProcessError as e:
            self.cpe_output(e, "Error creating the EPI on MNI registration snapshot")

    def create_seed_based_network(self, epi, out_dir, seed):
        """
        Seed based analysis - Creates a correlation map based on the time course 
        of the seed region. Will create intermediate directories when saving to 
        output directory.

        Parameters:
                epi (String): Path to the processed EPI data
                out_dir (String): Path to the output directory
                seed (String): Path to a seed ROI in NIFTI format
        """
        assert (os.path.isfile(epi))
        assert (os.path.isfile(seed))

        self.record("Creating seed based correlation map")
        data, seed, header = dc.read_data(epi, seed)
        seed_tc = dc.extract_seed_tc(seed, data)
        correlation = dc.calculate_static_corr(seed_tc, data)
        output_name = os.path.basename(epi)
        output_name = output_name[:output_name.find(".nii")]
        dc.save_output(correlation, header, output_name + "_DMNStaticCorr", out_dir)

    def afni_to_nifti(self, afni_file, name=""):
        """
        Converts an AFNI file format (.HEAD/.BRIK) to NIFTI file format (.nii).
        Names the outputed NIFTI file if passed a name parameter

        Parameter:
                afni_file (String): Path to the file in AFNI format
                name (String): Name of converted file
        """

        assert (os.path.isfile(afni_file + "HEAD"))
        assert (os.path.isfile(afni_file + "BRIK"))

        self.record("Converting from AFNI to NIFTI")
        try:
            # Equivalent to shell command:
            # 3dAFNItoNIFTI -prefix $name $afni_file
            command = ["3dAFNItoNIFTI"]
            if name:
                command.extend(["-prefix", name])
            command.append(afni_file)
            self.record(subprocess.check_output(command, stderr=subprocess.STDOUT))

        except subprocess.CalledProcessError as e:
            self.cpe_output(e, "Error converting file to NIFTI")

    def move_to_outdir(self, out_dir, *args):
        """
        Moves any number of files into out_dir

        Parameters:
                *args (String): Path to files
                out_dir (String): Path to the output directory
        """

        assert (os.path.isdir(out_dir))

        for file in args:
            shutil.move(file, out_dir)

    def clean(self, subj_id):
        """
        Removes previous {sub}.results directories from pwd

        Parameter:
                subj_id (String): Subject Id
        """
        old_dir = "{}.results".format(subj_id)

        if (os.path.isdir(old_dir)):
            self.record("Clean - Removing {}".format(old_dir))
            shutil.rmtree(old_dir)

    def run(self, args):
        """
        Runs this preprocessing pipeline

        """
        self.clean(args.subj_id)

        os.environ['OMP_NUM_THREADS'] = args.cores

        # outdir = outdir/Processed/
        args.out_dir = self.create_outdir(args.out_dir)
        results_dir = "{}.results".format(args.subj_id)
        afni_final = os.path.join(results_dir, "errts.{}.tproject+tlrc".format(subj_id))
        result = os.path.join(args.out_dir, afni_final)

        if (os.path.isfile(result + ".HEAD") and os.path.isfile(result + ".BRIK")
                and not args.rerun):
            self.record("Data has already been preprocessed.")
            self.record("Results of preprocessing are the following files: ")
            self.record(result + ".HEAD")
            self.record(result + ".BRIK")
            self.record("Use flag -r/--rerun to override results and rerun the preprocessing")
            return

        else:
            if os.path.isdir(args.out_dir):
                self.record("Overriding and deleting previous 'Processed' directory.")
                shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)

            if args.nl_reg:
                self.run_SSwarper(args.out_dir, args.subj_id, args.anat, mni)
            if args.volumes:
                args.epi = self.truncate_epi()
            if args.time_step:
                self.set_epi_tr(args.epi, args.time_step)

            afni_proc, name = self.generate_afni_proc(args, mni)
            # Execute afni_proc.py, creates RSproc.$subj script
            try:
                self.record(subprocess.check_output(afni_proc, stderr=subprocess.STDOUT))
            except subprocess.CalledProcessError as e:
                self.cpe_output(e, "Failed to generate RSproc.")

            rsproc = "RSproc.{}".format(args.subj_id)
            if args.gm_blur:
                self.set_gm_mask(rsproc)

            # Execute RSproc.$subj
            try:
                self.record(subprocess.check_output(["tcsh", "-xef", rsproc]), stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                self.cpe_output(e, "Preprocessing failed in RSproc")

            # Rename and convert errts.tproject and move $sub.results
            self.afni_to_nifti(afni_final, name)
            shutil.move(name + ".nii", results_dir)

            self.record("Finished preprocessing, moving results to output directory")
            self.move_to_outdir(args.out_dir, rsproc, file, results_dir)
