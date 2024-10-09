import os
import sys
from nipype import Node, Function, Workflow
import nibabel as nib
import numpy as np
import re
import os
import argparse
import multiprocessing
from nipype.pipeline import Workflow
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from nipype import Node, Function
from pathlib import Path
from bids.layout import BIDSLayout
from nipype_utils import BidsRename, BidsOutputFormatter, create_output_folder

num_cores = multiprocessing.cpu_count()


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
    output_node = pe.Node(util.IdentityInterface(
        fields=['b1_anat_ref_file',
                'b1_map_file'
                'brain_mask_file',
                'uncorrected_data_mask_file']),
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

    impaint_b1_map = Node(Function(
        input_names=['in_file', 'brain_mask_file', 'fwhm', 'iterations'],
        output_names=['out_file'],
        function=inpaint),
        name='impaint_b1_map')
    impaint_b1_map.inputs.fwhm = 2
    impaint_b1_map.inputs.iterations = 15
    wf.connect(cut_and_merge_b1_map, "out_file",
               impaint_b1_map, "in_file")
    wf.connect(create_brain_mask_node, "out_file",
               impaint_b1_map, "brain_mask_file")

    impaint_b1_anat_ref = Node(Function(
        input_names=['in_file', 'brain_mask_file', 'fwhm', 'iterations'],
        output_names=['out_file'],
        function=inpaint),
        name='impaint_b1_anat_ref')
    impaint_b1_anat_ref.inputs.fwhm = 2
    impaint_b1_anat_ref.inputs.iterations = 10
    wf.connect(cut_and_merge_b1_anat_ref, "out_file",
               impaint_b1_anat_ref, "in_file")
    wf.connect(create_brain_mask_node, "out_file",
               impaint_b1_anat_ref, "brain_mask_file")

    # set outputs
    wf.connect(cut_and_merge_b1_map, "uncorrected_data_mask_file",
               output_node, "uncorrected_data_mask_file")
    wf.connect(create_brain_mask_node, "out_file",
               output_node, "brain_mask_file")
    wf.connect(impaint_b1_map, "out_file",
               output_node, "b1_map_file")
    wf.connect(impaint_b1_anat_ref, "out_file",
               output_node, "b1_anat_ref_file")

    return wf


def collect_bids_bonn_inputs(input_directory, input_derivatives, subject_id=None,
                             session_id=None, run_id=None):
    layout = BIDSLayout(input_directory, derivatives=input_derivatives, validate=False)

    subjects = [subject_id] if subject_id else layout.get_subjects()
    inputs = []

    for subject in subjects:
        sessions = [session_id] if session_id else layout.get_sessions(subject=subject)
        if sessions:
            for session in sessions:
                valid_runs = layout.get(return_type='id', subject=subject,
                                        session=session, target='run',
                                        suffix='T2w',
                                        part="phase", extension="nii.gz")
                runs = [run_id] if isinstance(run_id,
                                              str) else valid_runs

                if len(runs) == 0:
                    runs = [None]

                for run in runs:
                    b1_map_files = layout.get(subject=subject,
                                                     session=session,
                                                     suffix="TB1map",
                                                     extension="nii.gz",
                                                     run=run)

                    b1_anat_ref_files = layout.get(subject=subject, session=session,
                                              suffix="magnitude",
                                              extension="nii.gz",
                                              acquisition="b1anat",
                                              run=run)

                    b1_anat_ref_files = [
                        f for f in b1_anat_ref_files if
                        not re.search(r'_desc-[^_]+', f.path)
                    ]

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       b1_map_file=b1_map_files[0],
                                       b1_anat_ref_file=b1_anat_ref_files[0]))

    return inputs


def main_bids(args):
    # Define a parser specifically for BIDS mode
    parser = argparse.ArgumentParser(
        description='BIDS-based B1 map correction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_directory", '-i', required=True, type=str,
                        help='BIDS input dataset root.')
    parser.add_argument("--input_derivatives", '-d', type=str, nargs='*',
                        help='BIDS derivative dataset roots (optional).')
    parser.add_argument("--subject_id", '-s', type=str,
                        help='Subject to process.')
    parser.add_argument("--session_id", '-t', type=str,
                        help='Session to process.')
    parser.add_argument("--run_id", '-r', type=str,
                        help='Run ID to process.')
    parser.add_argument("--output_directory", '-o', required=True, type=str,
                        help='Output directory for results.')
    parser.add_argument('--n_procs', '-n', type=int, default=num_cores,
                        help='Number of cores for parallel processing.')

    args = parser.parse_args(args)

    # Use BIDS input handling and create the workflow

    inputs = collect_bids_bonn_inputs(args.input_directory, args.input_derivatives,
                                      subject_id=args.subject_id,
                                      session_id=args.session_id,
                                      run_id=args.run_id)

    for i in range(0, len(inputs)):
        inputs[i]["axis"] = 1
        inputs[i]["n_voxels"] = 47

    # generate input node from collected inputs
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='input_node')
    keys = inputs[0].keys()
    input_node.iterables = [(key, [input_dict[key] for input_dict in inputs])
                            for key in keys]
    input_node.synchronize = True

    # Initialize workflow
    wf = correct_phase_wrap_around_workflow(base_dir=args.output_directory)
    wf.config['execution'] = {
        'remove_unnecessary_outputs': False,
        'sequential': True  # This is key to process one subject at a time
    }

    # set up bids input node
    bids_input_node = Node(IdentityInterface(fields=list(inputs[0].keys())), name='bids_input_node')
    keys = inputs[0].keys()
    bids_input_node.iterables = [(key, [input_dict[key] for input_dict in inputs]) for key in keys]
    bids_input_node.synchronize = True

    # connect bids inputs to workflow inputs
    wf_input_node = wf.get_node('input_node')
    input_keys = [key for key in keys if key not in ["subject", "session", "run"]]
    for key in input_keys:
        wf.connect(bids_input_node, key, wf_input_node, key)

    # set up output folder
    Path(args.output_directory).mkdir(exist_ok=True, parents=True)
    data_sink = pe.Node(nio.DataSink(), name='data_sink')
    data_sink.inputs.base_directory = args.output_directory
    output_folder_node = Node(
        Function(input_names=['subject', 'session', 'datatype'],
                 output_names=['output_folder'],
                 function=create_output_folder),
        name='output_folder_node')
    output_folder_node.inputs.datatype = "fmap"
    wf.connect(bids_input_node, 'subject', output_folder_node, 'subject')
    wf.connect(bids_input_node, 'session', output_folder_node, 'session')
    wf.connect(output_folder_node, 'output_folder', data_sink, 'container')

    # write outputs in bids format
    wf_output_node = wf.get_node('output_node')

    b1_map_pattern = "sub-{subject}_ses-{session}_run-{run}_desc-phaseWrapCorrected_B1map.nii.gz"
    rename_bids_b1_map_file = pe.Node(BidsOutputFormatter(), name="rename_bids_b1_map")
    rename_bids_b1_map_file.inputs.pattern = b1_map_pattern
    wf.connect(wf_output_node, 'b1_map_file',
               rename_bids_b1_map_file, 'in_file')
    wf.connect(rename_bids_b1_map_file, 'out_file',
               data_sink, '@b1_map_file')

    b1_anat_ref_file_pattern = "sub-{subject}_ses-{session}_acq-b1anat_run-{run}_desc-phaseWrapCorrected_magnitude.nii.gz"
    rename_bids_b1_anat_ref_file = pe.Node(BidsOutputFormatter(), name="rename_bids_b1_anat_ref")
    rename_bids_b1_anat_ref_file.inputs.pattern = b1_anat_ref_file_pattern
    wf.connect(wf_output_node, 'b1_anat_ref_file',
               rename_bids_b1_anat_ref_file, 'in_file')
    wf.connect(rename_bids_b1_anat_ref_file, 'out_file',
               data_sink, '@b1_anat_ref_file')

    bids_formatter_nodes = [rename_bids_b1_map_file, rename_bids_b1_anat_ref_file]
    for bids_formatter_node in bids_formatter_nodes:
        wf.connect(bids_input_node, 'subject', bids_formatter_node, 'subject')
        wf.connect(bids_input_node, 'session', bids_formatter_node, 'session')
        wf.connect(bids_input_node, 'run', bids_formatter_node, 'run')

    # run workflow
    # wf.write_graph(graph2use='hierarchical', format='svg', simple_form=False, dotfilename="correct_phase_wrap_around")
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


def main_non_bids(args):
    # Define a parser specifically for non-BIDS mode
    parser = argparse.ArgumentParser(
        description='Non-BIDS-based B1 map correction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--b1_anat_ref_file", type=str, required=True,
                        help='Path to the phase difference file used for generating the B0 map. '
                             'This file typically represents the phase differences between two echoes.')
    parser.add_argument("--b1_map_file", type=str, required=True,
                        help='Path to the phase difference file used for generating the B0 map. '
                             'This file typically represents the phase differences between two echoes.')
    parser.add_argument("--output_directory", '-o', required=True, type=str,
                        help='Path to the directory where the corrected B1 map and other output files will be saved.')
    parser.add_argument('--n_procs', '-n', type=int, default=num_cores,
                        help='Number of parallel processes to use for computation. '
                             'Defaults to the total number of available CPU cores.')
    args = parser.parse_args(args)

    # Initialize the workflow with inputs
    wf = correct_phase_wrap_around_workflow(base_dir=args.output_directory)
    wf.config['execution'] = {
        'remove_unnecessary_outputs': False,
        'sequential': True  # This is key to process one subject at a time
    }

    wf.inputs.input_node.b1_anat_ref_file = args.b1_anat_ref_file
    wf.inputs.input_node.b1_map_file = args.b1_map_file
    wf.inputs.input_node.axis = 1
    wf.inputs.input_node.n_voxels = 47

    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform B1 map correction with BIDS support or explicit file paths',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bids", action="store_true",
                        help='Use BIDS dataset layout for input files.')
    initial_args, remaining_args = parser.parse_known_args()

    # Depending on the presence of --bids, delegate to the appropriate main function
    if initial_args.bids:
        main_bids(remaining_args)
    else:
        main_non_bids(remaining_args)
