Node: correct_b1_with_b0 (correct_b1_map (utility)
==================================================


 Hierarchy : prepare_data.correct_b1_with_b0.correct_b1_map
 Exec ID : correct_b1_map.a0


Original Inputs
---------------


* b0_map_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/fa260552b6d4853a3fe8ca53582b8892c6678bf3/reslice/b0map_flirt.nii.gz
* b1_map_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/flirt_apply/image_smoothed_flirt.nii.gz
* fa_b1_in_degrees : 60
* fa_nominal_in_degrees : 20
* function_str : def correct_b1_map(b1_map_file, b0_map_file, fa_b1_in_degrees,
                   fa_nominal_in_degrees, pulse_duration_in_seconds):
    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()
    b1_map_file_nib = nib.load(b1_map_file)
    b0_map_file_nib = nib.load(b0_map_file)

    b1_map_image = b1_map_file_nib.get_fdata()
    b0_map_image = b0_map_file_nib.get_fdata()

    def cos_omega_eff(omega_eff, delta_omega, tau=2.445e-3):
        return np.cos(omega_eff * tau) + ((delta_omega / omega_eff) ** 2) * (
                    1 - np.cos(omega_eff * tau))

    def gamma_b1(alpha, tau=2.445e-3):
        return alpha / tau

    tau = pulse_duration_in_seconds
    fa_nominal = np.deg2rad(fa_nominal_in_degrees)
    fa_actual = fa_nominal * b1_map_image / (fa_b1_in_degrees * 10.0)
    delta_omega = b0_map_image * 2.0 * np.pi

    omega_eff = np.sqrt(
        (delta_omega) ** 2 + (gamma_b1(fa_actual, tau=tau)) ** 2)
    cosine = cos_omega_eff(omega_eff=omega_eff, delta_omega=delta_omega,
                           tau=tau)
    fa_actual_with_offresonance = np.arccos(cosine)
    b1_map = 100 * fa_actual_with_offresonance / fa_nominal

    b1_output_filename = os.path.join(base_dir, 'b1_map_corr.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_map, b0_map_file_nib.affine,
                                   b0_map_file_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename

* pulse_duration_in_seconds : 0.00246

