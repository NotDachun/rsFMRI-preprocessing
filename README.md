# rsFMRI-preprocessing

### Preprocessing relies on AFNI. Install it at:
https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/index.html

### rsfmri_preprocessing.py 
Creates a pipeline based on the supplied parameters and runs it on all subjects.

<pre>
usage: rsfmri_preprocessing.py [-h] [-t TRs] [-d MM] [-b F F] [-v V] [-f MM]
                               [-c C] [-s SUB] [-T TS] [-g] [-r] [-m] [-G]
                               [-n] -e EPI -a ANAT -o OUT

Generates and runs an afni_proc.py script to preprocess resting state fMRI data

optional arguments:
  -h, --help            show this help message and exit
  -t TRs, --trs_remove TRs
                        number of trs to remove at the beginning of the epi
                        data (default = 5 trs)
  -d MM, --dim_voxel MM
                        voxel dimensions in mm that processed epi will be
                        resampled to (default = 2.0 mm)
  -b F F, --bandpass F F
                        bandpass frequencies lower and upper limits (default =
                        0.01 0.25)
  -v V, --volumes V     truncate the epi data to the inputted number of
                        volumes, useful if subjects have data with different
                        numbers of volumes (default = no truncation)
  -f MM, --fwhm MM      the full width half maximum that is used when blurring
                        (default = 5.0 mm)
  -c C, --cores C       number of cores supplied to 3dDeconvolve (default =
                        all cores)
  -s SUB, --subj_id SUB
                        text file of paths (default = sub)
  -T TS, --time_step TS
                        set the time step for bandpassing (default = ts in
                        header info
  -g, --global_signal_regression
                        do not perform global signal regression (default =
                        perform gsr)
  -r, --rerun           rerun preprocessing, override and delete previous
                        results in 'Processed' folder (default = don't
                        override)
  -m, --motion_param    use 12 motion parameters for regression (default = 6
                        motion parameters)
  -G, --gm_blur         blur only in grey matter mask (default = blur in whole
                        brain)
  -n, --nl_reg          use non-linear warp between anatomical and MNI
                        template (default = linear warp)

required arguments:
  -e EPI, --epi EPI     text file of paths to raw epi data
  -a ANAT, --anat ANAT  text file of paths to raw anatomical data
  -o OUT, --out_dir OUT
                        text file of paths to output directory
</pre>

## Output
* Processed Data:
  * EPI: rsfri_despike_..._MNI.nii (Named accordingly to preprocessing steps)
  * ANAT: anat_final
* Quality Control
  * EM_sub.jpg: snapshot of alignment of EPI on MNI
  * DMNStatic(Corr/Z): static DMN map
  * preprocessing_output.txt: console output of program
  * RSproc: afni_proc.py script that was run
  * WIP!
