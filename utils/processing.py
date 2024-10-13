import os
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    TraitedSpec, File, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix


def unwrap_phase_b0_siemens(b0_phase_diff_file, b0_te_delta):
    import nibabel as nib
    import os

    base_dir = os.getcwd()

    # read b0 image
    b0_phase_diff_file_nib = nib.load(b0_phase_diff_file)
    b0_phase_diff_image = b0_phase_diff_file_nib.get_fdata()

    # Compute the B0 map
    phase_unwrap_factor = 1.0 / (4096 * b0_te_delta * 2)
    b0_map = phase_unwrap_factor * b0_phase_diff_image

    # write b1 map
    b0_output_filename = os.path.join(base_dir, 'b0map.nii.gz')
    b0_image_nib = nib.Nifti1Image(b0_map, b0_phase_diff_file_nib.affine,
                                   b0_phase_diff_file_nib.header)
    nib.save(b0_image_nib, b0_output_filename)

    return b0_output_filename


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


def cut_and_merge_image(in_file, n_voxels, axis):
    """
    Shift a specified number of voxels along a given dimension and merge them to the other side.

    Parameters:
    in_file (str): Path to the input NIfTI file.
    n_voxels (int): Number of voxels to shift.
    axis (int): Dimension along which to shift (0 for x, 1 for y, 2 for z).
    out_file (str): Path for the output NIfTI file.

    Returns:
    out_file (str): Path to the modified NIfTI file.
    """
    import os
    import nibabel as nib
    import numpy as np

    base_dir = os.getcwd()

    # Load the NIfTI file
    image_nib = nib.load(in_file)
    image = image_nib.get_fdata()
    affine = image_nib.affine

    # Create the mask to identify the touched region
    untouched_mask = np.zeros(image.shape, dtype=bool)
    slicer = [slice(None)] * len(image.shape)
    slicer[axis] = slice(-(n_voxels - 1), None)
    untouched_mask[tuple(slicer)] = True

    voxel_shift = np.zeros(4)
    voxel_shift[axis] = n_voxels * affine[axis, axis]
    adjusted_affine = affine.copy()
    adjusted_affine[:3, 3] -= voxel_shift[:3]
    shift_amount = image.shape[axis] - n_voxels
    voxel_size = affine[
        axis, axis]  # Get the voxel size along the specified axis (diagonal element)
    shift_in_mm = shift_amount * voxel_size
    adjusted_affine = affine.copy()
    adjusted_affine[
        axis, 3] -= shift_in_mm  # Adjust the translation component of the affine matrix

    # Perform the shift
    image_shifted = np.roll(image, shift=-n_voxels, axis=axis)

    # save image
    image_shifted_nib = nib.Nifti1Image(image_shifted, adjusted_affine,
                                        image_nib.header)
    out_file = os.path.join(base_dir, 'image_shifted.nii.gz')
    nib.save(image_shifted_nib, out_file)

    # save untouched mask
    untouched_mask_nib = nib.Nifti1Image(untouched_mask, adjusted_affine,
                                         image_nib.header)
    untouched_mask_file = os.path.join(base_dir, 'image_untouched_mask.nii.gz')
    nib.save(untouched_mask_nib, untouched_mask_file)

    return out_file, untouched_mask_file


def create_brain_mask_from_anatomical_b1(in_file, threshold=200, fwhm=8):
    """
    Shift a specified number of voxels along a given dimension and merge them to the other side.

    Parameters:
    in_file (str): Path to the input NIfTI file.
    n_voxels (int): Number of voxels to shift.
    axis (int): Dimension along which to shift (0 for x, 1 for y, 2 for z).
    out_file (str): Path for the output NIfTI file.

    Returns:
    out_file (str): Path to the modified NIfTI file.
    """
    import os
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter

    base_dir = os.getcwd()

    # Load the NIfTI file
    image_nib = nib.load(in_file)
    image = image_nib.get_fdata()

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    smoothed_data = gaussian_filter(image, sigma=sigma)
    brain_mask = smoothed_data > threshold

    brain_mask_nib = nib.Nifti1Image(brain_mask, image_nib.affine,
                                     image_nib.header)
    out_file = os.path.join(base_dir, 'brain_mask.nii.gz')
    nib.save(brain_mask_nib, out_file)

    return out_file


def inpaint(in_file, brain_mask_file, tissue_threshold=0, fwhm=2,
            iterations=10):
    """
    Shift a specified number of voxels along a given dimension and merge them to the other side.

    Parameters:
    in_file (str): Path to the input NIfTI file.
    n_voxels (int): Number of voxels to shift.
    axis (int): Dimension along which to shift (0 for x, 1 for y, 2 for z).
    out_file (str): Path for the output NIfTI file.

    Returns:
    out_file (str): Path to the modified NIfTI file.
    """
    import os
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter

    base_dir = os.getcwd()

    # Load the NIfTI file
    image_nib = nib.load(in_file)
    image = image_nib.get_fdata()

    brain_mask_nib = nib.load(brain_mask_file)
    brain_mask = brain_mask_nib.get_fdata().astype(np.bool)

    value_mask = image > tissue_threshold
    smoothing_mask = np.logical_and(brain_mask, value_mask)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Create a copy of the data with zeros where inpainting is needed
    inpainted_data = np.copy(image)
    inpainted_data[~smoothing_mask] = 0
    # Number of iterations for diffusion-based inpainting

    # Iteratively smooth and update the masked regions
    for _ in range(iterations):
        # Smooth the entire data
        smoothed_data = gaussian_filter(inpainted_data, sigma=sigma)

        # Update only the masked regions with the smoothed values
        inpainted_data[~smoothing_mask] = smoothed_data[~smoothing_mask]
    #
    #
    # smoothing_area = np.logical_and(brain_mask, ~smoothing_mask)
    final_image = np.copy(inpainted_data)
    final_image[~smoothing_mask] = inpainted_data[~smoothing_mask]
    final_image[~brain_mask] = image[~brain_mask]
    # Save the modified data to a new NIfTI image

    image_smoothed_nib = nib.Nifti1Image(final_image, image_nib.affine,
                                         image_nib.header)
    out_file = os.path.join(base_dir, 'image_smoothed.nii.gz')
    nib.save(image_smoothed_nib, out_file)

    return out_file


def compute_b1_magnitude_image_from_ste_lte(b1_ste_file, b1_fid_file):
    import nibabel as nib
    import os

    base_dir = os.getcwd()

    # read ste and fid b1 magnitude images
    b1_ste_file_nib = nib.load(b1_ste_file)
    b1_fid_image_nib = nib.load(b1_fid_file)
    b1_ste_image = b1_ste_file_nib.get_fdata()
    b1_fid_image = b1_fid_image_nib.get_fdata()

    # Compute the B1 ref
    b1_ref = (2 * b1_ste_image + b1_fid_image)

    # write b1 map
    b1_output_filename = os.path.join(base_dir, 'b1ref.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_ref, b1_ste_file_nib.affine,
                                   b1_ste_file_nib.header)
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
