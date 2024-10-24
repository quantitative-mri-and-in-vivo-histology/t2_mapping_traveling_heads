import math


def compute_t2_t1_amplitude_maps(magnitude_file,
                                 phase_file,
                                 b1_map_file,
                                 repetition_time,
                                 flip_angle,
                                 rf_phase_increments,
                                 mask_file=None
                                 ):
    from T2T1AM import cal_T2T1AM
    import os
    import nibabel as nib
    import numpy as np

    base_dir = os.getcwd()
    output_dir = base_dir

    if mask_file is None:
        # Create a mask of ones with the same shape as the input image
        img = nib.load(magnitude_file)
        mask_data = np.ones(img.shape[0:3], dtype=np.uint8)
        mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
        mask_file = os.path.join(base_dir, "mask.nii.gz")
        nib.save(mask_img, mask_file)

    # compute repetition time in milliseconds for cal_T2T1AM
    repetition_time_ms = 1000 * repetition_time

    # fit T1, T2, AM
    cal_T2T1AM(magnitude_file, phase_file, mask_file, b1_map_file,
               repetition_time_ms, flip_angle, rf_phase_increments,
               outputdir=output_dir)

    # define fixed output paths (used by cal_T2T1AM)
    t2_map_file = os.path.join(base_dir, "T2_.nii.gz")
    t1_map_file = os.path.join(base_dir, "T1_.nii.gz")
    am_map_file = os.path.join(base_dir, "Am_.nii.gz")

    # compute T1 in seconds and save
    t1_map_msec_nib = nib.load(t1_map_file)
    t1_map_msec = t1_map_msec_nib.get_fdata()
    t1_map_sec = t1_map_msec / 1000.0
    t1_map_sec_nib = nib.Nifti1Image(t1_map_sec, t1_map_msec_nib.affine,
                                     t1_map_msec_nib.header)
    nib.save(t1_map_sec_nib, t1_map_file)

    # compute T2 in seconds and save
    t2_map_msec_nib = nib.load(t2_map_file)
    t2_map_msec = t2_map_msec_nib.get_fdata()
    t2_map_sec = t2_map_msec / 1000.0
    t2_map_sec_nib = nib.Nifti1Image(t2_map_sec, t2_map_msec_nib.affine,
                                     t2_map_msec_nib.header)
    nib.save(t2_map_sec_nib, t2_map_file)

    return t2_map_file, t1_map_file, am_map_file


def subtract_background_phase(magnitude_file, phase_file):
    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()
    mag_nib = nib.load(magnitude_file)
    phase_nib = nib.load(phase_file)
    mag = mag_nib.get_fdata()
    phase = phase_nib.get_fdata()

    com = mag * np.exp(1.0j * phase)
    hip = com[..., 1::2] * np.conj(com[..., 0::2])

    phase_bg_sub = np.angle(hip) / 2.0
    mag_bg_sub = np.sqrt(np.abs(hip))

    phase_bg_sub_nii = nib.Nifti1Image(phase_bg_sub, phase_nib.affine,
                                       phase_nib.header)
    mag_bg_sub_nii = nib.Nifti1Image(mag_bg_sub, mag_nib.affine,
                                     mag_nib.header)

    magnitude_out_file = os.path.join(base_dir,
                                      "{}{}".format("magnitude", ".nii.gz"))
    phase_out_file = os.path.join(base_dir,
                                  "{}{}".format("phase", ".nii.gz"))

    nib.save(mag_bg_sub_nii, magnitude_out_file)
    nib.save(phase_bg_sub_nii, phase_out_file)

    return magnitude_out_file, phase_out_file


def correct_b1_map(b1_map_file, b0_map_file,
                   fa_nominal_in_degrees, pulse_duration_in_seconds):
    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()
    b1_map_file_nib = nib.load(b1_map_file)
    b0_map_file_nib = nib.load(b0_map_file)

    b1_map_image = b1_map_file_nib.get_fdata()
    b0_map_image = b0_map_file_nib.get_fdata()

    def cos_omega_eff(omega_eff, delta_omega, tau):
        return np.cos(omega_eff * tau) + ((delta_omega / omega_eff) ** 2) * (
                1 - np.cos(omega_eff * tau))

    def gamma_b1(alpha, tau):
        return alpha / tau

    tau = pulse_duration_in_seconds
    fa_nominal = np.deg2rad(fa_nominal_in_degrees)
    fa_actual = fa_nominal * b1_map_image
    delta_omega = b0_map_image * 2.0 * np.pi

    omega_eff = np.sqrt(
        (delta_omega) ** 2 + (gamma_b1(fa_actual, tau=tau)) ** 2)
    cosine = cos_omega_eff(omega_eff=omega_eff, delta_omega=delta_omega,
                           tau=tau)
    fa_actual_with_offresonance = np.arccos(cosine)
    b1_map = fa_actual_with_offresonance / fa_nominal

    b1_output_filename = os.path.join(base_dir, 'b1_map_corr.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_map, b0_map_file_nib.affine,
                                   b0_map_file_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename


def create_qi_jsr_config(t1w_metadata_dicts, t2w_metadata_dicts):

    def assert_all_similar(values, tolerance=1e-9):
        # Ensure all values in the list are similar within the given tolerance
        first_value = values[0]
        assert all(
            math.isclose(value, first_value, rel_tol=tolerance,
                         abs_tol=tolerance)
            for value in values), \
            "Not all values are similar"

    qi_jsr_config_dict = dict()

    # collect T1w metadata
    t1w_flip_angles = []
    t1w_repetition_times = []
    t1w_echo_times = []
    for t1w_metadata_dict in t1w_metadata_dicts:
        t1w_flip_angles.append(t1w_metadata_dict['FlipAngle'])
        t1w_repetition_times.append(
            t1w_metadata_dict['RepetitionTimeExcitation'])
        t1w_echo_times.append(
            t1w_metadata_dict['EchoTime'])

    # pulse duration and repetition time need to be consistent across T2w images
    assert_all_similar(t1w_repetition_times)
    assert_all_similar(t1w_echo_times)

    # create SPGR config entry
    qi_jsr_config_dict["SPGR"] = dict(
        TR=t1w_repetition_times[0],
        TE=t1w_echo_times[0],
        FA=t1w_flip_angles)

    # collect T2w metadata
    t2w_rf_phase_increments = []
    t2w_flip_angles = []
    t2w_rf_pulse_durations = []
    t2w_repetition_times = []
    for t2w_metadata_dict in t2w_metadata_dicts:
        t2w_rf_phase_increments.append(t2w_metadata_dict['RfPhaseIncrement'])
        t2w_flip_angles.append(t2w_metadata_dict['FlipAngle'])
        t2w_rf_pulse_durations.append(t2w_metadata_dict['RfPulseDuration'])
        t2w_repetition_times.append(
            t2w_metadata_dict['RepetitionTimeExcitation'])

    # pulse duration and repetition time need to be consistent across T2w images
    assert_all_similar(t2w_rf_pulse_durations)
    assert_all_similar(t2w_repetition_times)

    # create SSFP config entry
    qi_jsr_config_dict["SSFP"] = dict(
        TR=t2w_repetition_times[0],
        Trf=t2w_rf_pulse_durations[0],
        FA=t2w_flip_angles,
        PhaseInc=t2w_rf_phase_increments)

    return qi_jsr_config_dict
