import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# /mnt/tpp/hesam/SU_ADRC/Hannah/preprocessing/processed/15-09-29.1_3T3_h/Processed/sub.results/rsfmri_despike_toMNI_5.0mmSmooth_bandpass_gsregress.nii

s = nib.load("/mnt/tpp/hesam/SU_ADRC/Hannah/preprocessing/processed/15-09-29.1_3T3_h/Processed/sub.results/rsfmri_despike_toMNI_5.0mmSmooth_bandpass_gsregress.nii", mmap=False)
data = s.get_data()
means = np.mean(data, axis=(0,1,2))
p = plt.plot(np.arange(0, len(means)), means)
plt.show()
