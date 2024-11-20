import argparse
import math
import multiprocessing
import os

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from nipype import Node
from nipype import Workflow
from nipype.interfaces.utility import IdentityInterface, Function

from nodes.io import BidsOutputWriter
from utils.bids_config import (DEFAULT_NIFTI_READ_EXT_ENTITY,
                               STANDARDIZED_ENTITY_OVERRIDES_T1W,
                               STANDARDIZED_ENTITY_OVERRIDES_B0_PHASEDIFF_MAP,
                               STANDARDIZED_ENTITY_OVERRIDES_B0_ANAT_REF,
                               STANDARDIZED_ENTITY_OVERRIDES_B1_MAP,
                               STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF,
                               STANDARDIZED_ENTITY_OVERRIDES_T2W_MAG,
                               STANDARDIZED_ENTITY_OVERRIDES_T2W_PHASE)
from utils.io import write_minimal_bids_dataset_description, find_image_and_json
from workflows.processing import correct_b1_with_b0


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
    brain_mask = brain_mask_nib.get_fdata().astype(bool)

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


def correct_phase_wrap_around_workflow(base_dir=os.getcwd(),
                                       name="correct_phase_wrap_around"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = Node(
        IdentityInterface(
            fields=['b1_anat_ref_file',
                    'b1_map_file',
                    'axis',
                    'n_voxels']),
        name='input_node'
    )
    output_node = pe.Node(pe.utils.IdentityInterface(
        fields=['b1_anat_ref_file',
                'b1_map_file',
                'brain_mask_file',
                'untouched_mask_file']),
        name='output_node')

    cut_and_merge_b1_map = Node(Function(
        input_names=['in_file', 'n_voxels', 'axis'],
        output_names=['out_file', 'untouched_mask_file'],
        function=cut_and_merge_image),
        name='cut_and_merge_b1_map')
    wf.connect(input_node, "b1_map_file",
               cut_and_merge_b1_map, "in_file")
    wf.connect(input_node, "axis",
               cut_and_merge_b1_map, "axis")
    wf.connect(input_node, "n_voxels",
               cut_and_merge_b1_map, "n_voxels")

    cut_and_merge_b1_anat_ref = Node(Function(
        input_names=['in_file', 'n_voxels', 'axis'],
        output_names=['out_file', 'untouched_mask_file'],
        function=cut_and_merge_image),
        name='cut_and_merge_b1_anat_ref')
    wf.connect(input_node, "b1_anat_ref_file",
               cut_and_merge_b1_anat_ref, "in_file")
    wf.connect(input_node, "axis",
               cut_and_merge_b1_anat_ref, "axis")
    wf.connect(input_node, "n_voxels",
               cut_and_merge_b1_anat_ref, "n_voxels")

    create_brain_mask_node = Node(Function(
        input_names=['in_file', 'fwhm'],
        output_names=['out_file'],
        function=create_brain_mask_from_anatomical_b1),
        name='create_brain_mask')
    create_brain_mask_node.inputs.fwhm = 8
    wf.connect(cut_and_merge_b1_anat_ref, "out_file",
               create_brain_mask_node, "in_file")

    inpaint_b1_map = Node(Function(
        input_names=['in_file', 'brain_mask_file', 'fwhm', 'iterations'],
        output_names=['out_file'],
        function=inpaint),
        name='inpaint_b1_map')
    inpaint_b1_map.inputs.fwhm = 2
    inpaint_b1_map.inputs.iterations = 15
    wf.connect(cut_and_merge_b1_map, "out_file",
               inpaint_b1_map, "in_file")
    wf.connect(create_brain_mask_node, "out_file",
               inpaint_b1_map, "brain_mask_file")

    inpaint_b1_anat_ref = Node(Function(
        input_names=['in_file', 'brain_mask_file', 'fwhm', 'iterations'],
        output_names=['out_file'],
        function=inpaint),
        name='inpaint_b1_anat_ref')
    inpaint_b1_anat_ref.inputs.fwhm = 2
    inpaint_b1_anat_ref.inputs.iterations = 10
    wf.connect(cut_and_merge_b1_anat_ref, "out_file",
               inpaint_b1_anat_ref, "in_file")
    wf.connect(create_brain_mask_node, "out_file",
               inpaint_b1_anat_ref, "brain_mask_file")

    # set outputs
    wf.connect(cut_and_merge_b1_map, "untouched_mask_file",
               output_node, "untouched_mask_file")
    wf.connect(create_brain_mask_node, "out_file",
               output_node, "brain_mask_file")
    wf.connect(inpaint_b1_map, "out_file",
               output_node, "b1_map_file")
    wf.connect(inpaint_b1_anat_ref, "out_file",
               output_node, "b1_anat_ref_file")

    return wf


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SSFP-BEAT dataset from UKE, Hamburg.")
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Path to the output derivatives folder.')
    parser.add_argument('-t', '--temp_dir', default=os.getcwd(),
                        help='Directory for intermediate outputs (default: current working directory).')
    parser.add_argument('--n_procs', type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of processors to use (default: all available cores).')
    parser.add_argument('--subject', help='Process a specific subject.')
    parser.add_argument('--session', help='Process a specific session.')
    parser.add_argument('--run', help='Process a specific run.')
    args = parser.parse_args()

    # write minimal dataset description for output derivatives
    os.makedirs(args.output_dir, exist_ok=True)
    write_minimal_bids_dataset_description(
        dataset_root=args.output_dir,
        dataset_name=os.path.dirname(args.output_dir)
    )

    # Define the reusable run settings in a dictionary
    run_settings = dict(plugin='MultiProc',
                        plugin_args={'n_procs': args.n_procs})

    # collect inputs
    layout = BIDSLayout(args.input_dir,
                        derivatives=True,
                        validate=False)

    # collect data for each independent subject-session-run combination
    inputs = []
    subjects = [args.subject] if args.subject else layout.get_subjects()
    for subject in subjects:
        sessions = [args.session] if args.session else layout.get_sessions(
            subject=subject)
        if sessions:
            for session in sessions:

                test_image_entities = dict(subject=subject,
                                           session=session,
                                           suffix="T2w",
                                           part="mag",
                                           **DEFAULT_NIFTI_READ_EXT_ENTITY)

                test_images = layout.get(**test_image_entities)
                if len(test_images) == 0:
                    continue

                valid_runs = layout.get(return_type='id',
                                        target='run',
                                        **test_image_entities)

                runs = [args.run] if args.run else valid_runs

                if len(runs) == 0:
                    runs = [None]

                for run in runs:
                    input_dict = dict()
                    input_dict["subject"] = subject
                    input_dict["session"] = session
                    input_dict["run"] = run

                    (input_dict["t1w_file"],
                     input_dict["t1w_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="dznebnt1wmprage1isoComb",
                        suffix="T1w",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["t2w_mag_file"],
                     input_dict["t2w_mag_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        suffix="T2w",
                        part="mag",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["t2w_phase_file"],
                     input_dict["t2w_phase_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        suffix="T2w",
                        part="phase",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b1_map_file"],
                     input_dict["b1_map_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        suffix="TB1map",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b1_ste_file"],
                     input_dict["b1_ste_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="dznebnB1",
                        suffix="magnitude1",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b1_fid_file"],
                     input_dict["b1_fid_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="dznebnB1",
                        suffix="magnitude2",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    b0_run_id = 1 if run is None else (run - 1) * 2 + 1
                    (input_dict["b0_mag1_file"],
                     input_dict["b0_mag1_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=b0_run_id,
                        acquisition="dznebnB0",
                        suffix="magnitude1",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["b0_mag1_entity_overrides"] = dict(
                        run=run, **STANDARDIZED_ENTITY_OVERRIDES_B0_ANAT_REF)

                    (input_dict["b0_phasediff_file"],
                     input_dict[
                         "b0_phasediff_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=b0_run_id,
                        acquisition="dznebnB0",
                        suffix="phase2",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["b0_phasediff_entity_overrides"] = dict(
                        run=run,
                        **STANDARDIZED_ENTITY_OVERRIDES_B0_PHASEDIFF_MAP)

                    # add B1 map normalization factor (to 1)
                    fa_b1_in_degrees = input_dict["b1_map_json_dict"][
                        "FlipAngle"]
                    input_dict["b1_normalization_factor"] = 1.0 / (
                            fa_b1_in_degrees * 10)

                    # add phase unwrap factor
                    b0_te_delta = input_dict["b0_phasediff_json_dict"][
                                      "EchoTime2"] - \
                                  input_dict["b0_phasediff_json_dict"][
                                      "EchoTime1"]
                    input_dict["b0_phase_unwrap_factor"] = 1.0 / (
                            4096 * b0_te_delta * 2)

                    # add wrap-around correction parameters
                    input_dict["axis_wrap_around"] = 1
                    input_dict["n_voxels_wrap_around"] = 47

                    # add flip angles
                    input_dict["fa_b1_in_degrees"] = \
                        fa_b1_in_degrees
                    input_dict["fa_nominal_in_degrees"] = input_dict[
                        "t2w_mag_json_dict"]["FlipAngle"]

                    # add RF pulse duration and phase increments to T2w metadata
                    input_dict["rf_pulse_duration"] = 2.46e-3
                    input_dict[
                        "t2w_mag_json_dict"]["RfPulseDuration"] = input_dict[
                        "rf_pulse_duration"]
                    input_dict[
                        "t2w_phase_json_dict"]["RfPulseDuration"] = input_dict[
                        "rf_pulse_duration"]
                    rf_phase_increments = [1, 2, 4, 1.5, 3, 5]
                    input_dict[
                        "t2w_mag_json_dict"][
                        "RfPhaseIncrement"] = rf_phase_increments
                    input_dict[
                        "t2w_phase_json_dict"]["RfPhaseIncrement"] = \
                        rf_phase_increments

                    inputs.append(input_dict)

    for input_index, input_dict in enumerate(inputs):

        # create new workflow for
        # subject-run-session combination
        wf_id = "{}_{}_{}".format(input_dict["subject"], input_dict["session"],
                                  input_dict["run"])
        wf = Workflow(name="prepare_3depi_dzne_dataset_{}".format(wf_id))
        wf.base_dir = args.temp_dir

        # create input node using entries in input_dict for
        # subject-run-session combination
        input_node = Node(
            IdentityInterface(fields=list(input_dict.keys())),
            name='input_node')
        for key, value in input_dict.items():
            setattr(input_node.inputs, key, value)

        # scale phase to radian
        scaling_factor = math.pi / 4096.0
        scale_phase_from_siemens_to_radian = pe.Node(
            fsl.ImageMaths(
                op_string='-mul {}'.format(scaling_factor)),
            name="scale_phase_from_siemens_to_rad")
        wf.connect(input_node, "t2w_phase_file",
                   scale_phase_from_siemens_to_radian, "in_file")

        # convert B0 map to radian
        unwrap_phase_b0 = pe.Node(fsl.BinaryMaths(operation="mul"),
                                  name="unwrap_phase_b0")
        wf.connect(input_node, "b0_phasediff_file",
                   unwrap_phase_b0, "in_file")
        wf.connect(input_node, "b0_phase_unwrap_factor",
                   unwrap_phase_b0, "operand_value")

        # normalize B1 map (to 1)
        normalize_b1 = pe.Node(fsl.BinaryMaths(operation="mul"),
                               name="normalize_b1")
        wf.connect(input_node, "b1_map_file",
                   normalize_b1, "in_file")
        wf.connect(input_node, "b1_normalization_factor",
                   normalize_b1, "operand_value")

        # Compute B1 anatomical reference image as 2 * ste + fid
        multiply_ste_by_two = pe.Node(
            fsl.BinaryMaths(operation="mul",
                            operand_value=2),
            name="multiply_ste")
        wf.connect(input_node, "b1_ste_file",
                   multiply_ste_by_two, "in_file")
        add_fid = pe.Node(fsl.BinaryMaths(operation="add"),
                          name="add_fid")
        wf.connect(multiply_ste_by_two, "out_file", add_fid,
                   "in_file")
        wf.connect(input_node, "b1_fid_file", add_fid,
                   "operand_file")

        # create dummy node that
        # - holds phase-wrap corrected B1 map (for subjects 002-004)
        # - passes through B1 map from previous otherwise
        phase_wrap_b1_node = pe.Node(IdentityInterface(fields=[
            "b1_map_file", "b1_anat_ref_file"
        ]), name="phase_wrap_b1_node")

        # only do phase wrap-around correction for phy002 to phy004
        if input_dict["subject"] in ["phy002", "phy003", "phy004"]:
            correct_phase_wrap_around_wf = correct_phase_wrap_around_workflow()
            wf.connect(add_fid, "out_file",
                       correct_phase_wrap_around_wf,
                       "input_node.b1_anat_ref_file")
            wf.connect(normalize_b1, "out_file",
                       correct_phase_wrap_around_wf, "input_node.b1_map_file")
            wf.connect(input_node, "axis_wrap_around",
                       correct_phase_wrap_around_wf, "input_node.axis")
            wf.connect(input_node, "n_voxels_wrap_around",
                       correct_phase_wrap_around_wf, "input_node.n_voxels")

            # use phase-wrap corrected B1 map
            wf.connect(correct_phase_wrap_around_wf, "output_node.b1_map_file",
                       phase_wrap_b1_node, "b1_map_file")
            wf.connect(correct_phase_wrap_around_wf,
                       "output_node.b1_anat_ref_file",
                       phase_wrap_b1_node, "b1_anat_ref_file")
        else:
            # use B1 map from previous step
            wf.connect(normalize_b1, "out_file",
                       phase_wrap_b1_node, "b1_map_file")
            wf.connect(add_fid, "out_file",
                       phase_wrap_b1_node, "b1_anat_ref_file")

        # B1 correction due for T2w images due to large difference in pulse duration
        correct_b1_with_b0_wf = correct_b1_with_b0()
        wf.connect(unwrap_phase_b0, "out_file",
                   correct_b1_with_b0_wf, "input_node.b0_map_file")
        wf.connect(phase_wrap_b1_node,
                   "b1_map_file",
                   correct_b1_with_b0_wf, "input_node.b1_map_file")
        wf.connect(input_node, "b0_mag1_file",
                   correct_b1_with_b0_wf,
                   "input_node.b0_anat_ref_file")
        wf.connect(phase_wrap_b1_node,
                   "b1_anat_ref_file",
                   correct_b1_with_b0_wf,
                   "input_node.b1_anat_ref_file")
        wf.connect(input_node, "fa_nominal_in_degrees",
                   correct_b1_with_b0_wf,
                   "input_node.fa_nominal_in_degrees")
        wf.connect(input_node, "rf_pulse_duration",
                   correct_b1_with_b0_wf,
                   "input_node.pulse_duration_in_seconds")

        # write B0 map
        b0_map_file_writer = pe.Node(BidsOutputWriter(),
                                     name="b0_map_file_writer")
        b0_map_file_writer.inputs.output_dir = args.output_dir
        wf.connect(unwrap_phase_b0, "out_file",
                   b0_map_file_writer, "in_file")
        wf.connect(input_node, "b0_phasediff_json_dict",
                   b0_map_file_writer, "json_dict")
        wf.connect(input_node, "b0_phasediff_file",
                   b0_map_file_writer, "template_file")
        wf.connect(input_node, "b0_phasediff_entity_overrides",
                   b0_map_file_writer, "entity_overrides")

        # write B0 anatomical reference magnitude image
        b0_anat_ref_file_writer = pe.Node(BidsOutputWriter(),
                                          name="b0_anat_ref_file_writer")
        b0_anat_ref_file_writer.inputs.output_dir = args.output_dir
        wf.connect(input_node, "b0_mag1_file",
                   b0_anat_ref_file_writer, "in_file")
        wf.connect(input_node, "b0_mag1_json_dict",
                   b0_anat_ref_file_writer, "json_dict")
        wf.connect(input_node, "b0_mag1_file",
                   b0_anat_ref_file_writer, "template_file")
        wf.connect(input_node, "b0_mag1_entity_overrides",
                   b0_anat_ref_file_writer, "entity_overrides")

        # write B1 map
        b1_map_file_writer = pe.Node(BidsOutputWriter(),
                                     name="b1_map_file_writer")
        b1_map_file_writer.inputs.output_dir = args.output_dir
        b1_map_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_B1_MAP
        wf.connect(correct_b1_with_b0_wf, "output_node.out_file",
                   b1_map_file_writer, "in_file")
        wf.connect(input_node, "b1_map_json_dict",
                   b1_map_file_writer, "json_dict")
        wf.connect(input_node, "b1_map_file",
                   b1_map_file_writer, "template_file")

        # write B1 anatomical reference magnitude image
        b1_anat_ref_file_writer = pe.Node(BidsOutputWriter(),
                                          name="b1_anat_ref_file_writer")
        b1_anat_ref_file_writer.inputs.output_dir = args.output_dir
        b1_anat_ref_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF
        wf.connect(phase_wrap_b1_node, "b1_anat_ref_file",
                   b1_anat_ref_file_writer, "in_file")
        wf.connect(input_node, "b1_ste_json_dict",
                   b1_anat_ref_file_writer, "json_dict")
        wf.connect(input_node, "b1_ste_file",
                   b1_anat_ref_file_writer, "template_file")

        # write T2w magngitude image
        t2w_mag_file_writer = pe.Node(BidsOutputWriter(),
                                      name="t2w_mag_file_writer")
        t2w_mag_file_writer.inputs.output_dir = args.output_dir
        t2w_mag_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_T2W_MAG
        wf.connect(input_node, "t2w_mag_file",
                   t2w_mag_file_writer, "in_file")
        wf.connect(input_node, "t2w_mag_file",
                   t2w_mag_file_writer, "template_file")
        wf.connect(input_node, "t2w_mag_json_dict",
                   t2w_mag_file_writer, "json_dict")

        # write T2w phase image
        t2w_phase_file_writer = pe.Node(BidsOutputWriter(),
                                        name="t2w_phase_file_writer")
        t2w_phase_file_writer.inputs.output_dir = args.output_dir
        t2w_phase_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_T2W_PHASE
        wf.connect(scale_phase_from_siemens_to_radian, "out_file",
                   t2w_phase_file_writer, "in_file")
        wf.connect(input_node, "t2w_phase_file",
                   t2w_phase_file_writer, "template_file")
        wf.connect(input_node, "t2w_phase_json_dict",
                   t2w_phase_file_writer, "json_dict")

        # write T1w image
        t1w_file_writer = pe.Node(BidsOutputWriter(),
                                  name="t1w_file_writer")
        t1w_file_writer.inputs.output_dir = args.output_dir
        t1w_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_T1W
        wf.connect(input_node, "t1w_file",
                   t1w_file_writer, "in_file")
        wf.connect(input_node, "t1w_file",
                   t1w_file_writer, "template_file")
        wf.connect(input_node, "t1w_json_dict",
                   t1w_file_writer, "json_dict")

        wf.run(**run_settings)


if __name__ == "__main__":
    main()
