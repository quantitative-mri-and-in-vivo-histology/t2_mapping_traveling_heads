import os
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    TraitedSpec, File, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix


def compute_t2_t1_amplitude_maps(magnitude_file,
                                  phase_file,
                                  mask_file,
                                  b1_map_file,
                                  repetition_time,
                                  flip_angle,
                                  rf_phase_increments
                                  ):
    from T2T1AM import cal_T2T1AM
    import os

    base_dir = os.getcwd()
    output_dir = base_dir

    cal_T2T1AM(magnitude_file, phase_file, mask_file, b1_map_file,
               repetition_time, flip_angle, rf_phase_increments, outputdir=output_dir)

    t2_map_file = os.path.join(base_dir, "T2_.nii.gz")
    t1_map_file = os.path.join(base_dir, "T1_.nii.gz")
    am_map_file = os.path.join(base_dir, "Am_.nii.gz")

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


def correct_b1_map(b1_map_file, b0_map_file, fa_b1_in_degrees,
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


# Define the input specification for QiTgv
class QiTgvInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, desc='Input file', mandatory=True, position=0,
                   argstr='%s')
    out_file = File(desc='Output file', position=1,
                    argstr='--out=%s')  # Optional
    alpha = traits.Float(desc='Alpha parameter', position=2,
                         argstr='--alpha=%f', usedefault=False)  # Optional


# Define the output specification for QiTgv
class QiTgvOutputSpec(TraitedSpec):
    out_file = File(desc='Output file', exists=True)


# Define the custom command-line wrapper for QiTgv
class QiTgv(CommandLine):
    _cmd = 'qi tgv'  # The command should map to "qi tgv"
    input_spec = QiTgvInputSpec
    output_spec = QiTgvOutputSpec

    # Override _list_outputs to auto-generate the out_file in the current working directory
    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_file'] = os.path.abspath(
                self.inputs.out_file)  # Ensure full path
        else:
            # Use fname_presuffix to add '_tgv' to the basename, ensure it's in the current working directory
            outputs['out_file'] = fname_presuffix(self.inputs.in_file,
                                                  suffix='_tgv',
                                                  newpath=os.getcwd())
        return outputs


# Define the input specification for QiJsr
class QiJsrInputSpec(CommandLineInputSpec):
    spgr_file = File(exists=True, desc='SPGR input file', mandatory=True,
                     position=0, argstr='%s')
    ssfp_file = File(exists=True, desc='SSFP input file', mandatory=True,
                     position=1, argstr='%s')
    b1_file = File(exists=True, desc='B1 map file', mandatory=True, position=2,
                   argstr='--B1=%s')
    mask_file = File(exists=True, desc='Mask file', position=3,
                     argstr='--mask=%s', mandatory=False)  # Optional
    npsi = traits.Int(6, desc='Number of PSI components', usedefault=True,
                      position=4, argstr='--npsi=%d')
    json_file = File(exists=True, desc='Input JSON file', position=5,
                     argstr='--json=%s')


# Define the output specification for QiJsr
class QiJsrOutputSpec(TraitedSpec):
    t2_map_file = File(desc='Path to the generated T2 map file', exists=True)
    t1_map_file = File(desc='Path to the generated T1 map file', exists=True)


# Define the custom command-line wrapper for QiJsr
class QiJsr(CommandLine):
    _cmd = 'qi jsr'  # The command should map to "qi jsr"
    input_spec = QiJsrInputSpec
    output_spec = QiJsrOutputSpec

    # Override _list_outputs to specify the expected output files.
    def _list_outputs(self):
        outputs = self.output_spec().get()

        # Use the working directory where the node is executed.
        output_dir = os.path.abspath(self.inputs.cwd) \
            if hasattr(self, 'inputs') and hasattr(
            self.inputs, 'cwd') else os.getcwd()

        # Define the paths to the output files based on the output directory.
        outputs['t2_map_file'] = os.path.join(output_dir, 'JSR_T2.nii.gz')
        outputs['t1_map_file'] = os.path.join(output_dir, 'JSR_T1.nii.gz')

        return outputs
