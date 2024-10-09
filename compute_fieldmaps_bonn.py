import os
import sys
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


# Core function to compute the B1 map
def unwrap_phase_b0(b0_phase_diff_file, b0_te_delta):
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
    b0_image_nib = nib.Nifti1Image(b0_map, b0_phase_diff_file_nib.affine, b0_phase_diff_file_nib.header)
    nib.save(b0_image_nib, b0_output_filename)

    return b0_output_filename


# Core function to compute the B1 map
def compute_b1_ref(b1_ste_file, b1_fid_file):
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
    b1_image_nib = nib.Nifti1Image(b1_ref, b1_ste_file_nib.affine, b1_ste_file_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename


def bonn_fieldmaps_workflow(base_dir=os.getcwd(), name="bonn_fieldmaps"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(util.IdentityInterface(
        fields=['b0_phase_diff_file',
                'b0_te_delta',
                'b1_ste_file',
                'b1_fid_file']),
        name='input_node')
    output_node = pe.Node(util.IdentityInterface(fields=['b0_map_file',
                                                         'b1_anat_ref_file']),
                          name='output_node')

    # unwrap b0 map
    unwrap_phase_b0_node = pe.Node(interface=util.Function(
        input_names=['b0_phase_diff_file', 'b0_te_delta'],
        output_names=['out_file'],
        function=unwrap_phase_b0),
        name='unwrap_phase_b0')
    wf.connect(input_node, 'b0_phase_diff_file', unwrap_phase_b0_node, 'b0_phase_diff_file')
    wf.connect(input_node, 'b0_te_delta', unwrap_phase_b0_node, 'b0_te_delta')

    # unwrap b0 map
    compute_b1_ref_node = pe.Node(interface=util.Function(
        input_names=['b1_ste_file', 'b1_fid_file'],
        output_names=['out_file'],
        function=compute_b1_ref),
        name='compute_b1_ref')
    wf.connect(input_node, 'b1_ste_file', compute_b1_ref_node, 'b1_ste_file')
    wf.connect(input_node, 'b1_fid_file', compute_b1_ref_node, 'b1_fid_file')

    wf.connect(unwrap_phase_b0_node, "out_file", output_node, "b0_map_file")
    wf.connect(compute_b1_ref_node, "out_file", output_node, "b1_anat_ref_file")

    return wf


def collect_bids_inputs(input_directory, input_derivatives, subject_id=None,
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

                    # to be fixed: use IntendedFor metadata field instead
                    b_maps_run_id = 1
                    if run is not None:
                        b_maps_run_id = (run - 1) * 2 + 1  # run = 1 -> 1; run 2 -> 3; ...

                    b0_phase_diff_files = layout.get(subject=subject,
                                                     session=session,
                                                     suffix="phase2",
                                                     extension="nii.gz",
                                                     acquisition="dznebnB0",
                                                     run=b_maps_run_id)

                    b0_te_delta = b0_phase_diff_files[0].entities["EchoTime2"] - b0_phase_diff_files[0].entities[
                        "EchoTime1"]

                    b1_ste_files = layout.get(subject=subject, session=session,
                                              suffix="magnitude1",
                                              extension="nii.gz",
                                              acquisition="dznebnB1",
                                              run=run)

                    b1_fid_files = layout.get(subject=subject, session=session,
                                              suffix="magnitude2",
                                              extension="nii.gz",
                                              acquisition="dznebnB1",
                                              run=run)

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       b0_phase_diff_file=b0_phase_diff_files[0],
                                       b0_te_delta=b0_te_delta,
                                       b1_ste_file=b1_ste_files[0],
                                       b1_fid_file=b1_fid_files[0]))
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
    inputs = collect_bids_inputs(args.input_directory, args.input_derivatives,
                                      subject_id=args.subject_id,
                                      session_id=args.session_id,
                                      run_id=args.run_id)

    # generate input node from collected inputs
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='input_node')
    keys = inputs[0].keys()
    input_node.iterables = [(key, [input_dict[key] for input_dict in inputs])
                            for key in keys]
    input_node.synchronize = True

    # Initialize workflow
    wf = bonn_fieldmaps_workflow(base_dir=args.output_directory)
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
    output_folder_node = Node(Function(input_names=['subject', 'session'],
                                       output_names=['output_folder'],
                                       function=create_output_folder),
                              name='output_folder_node')
    wf.connect(bids_input_node, 'subject', output_folder_node, 'subject')
    wf.connect(bids_input_node, 'session', output_folder_node, 'session')
    wf.connect(output_folder_node, 'output_folder', data_sink, 'container')

    # write outputs in bids format
    wf_output_node = wf.get_node('output_node')

    b0_map_pattern = "sub-{subject}_ses-{session}_run-{run}_B0map.nii.gz"
    rename_bids_b0_map_file = pe.Node(BidsOutputFormatter(), name="rename_bids_b0_map")
    rename_bids_b0_map_file.inputs.pattern = b0_map_pattern
    wf.connect(wf_output_node, 'b0_map_file',
               rename_bids_b0_map_file, 'in_file')
    wf.connect(rename_bids_b0_map_file, 'out_file',
               data_sink, '@b0_map_file')

    b1_anat_ref_file_pattern = "sub-{subject}_ses-{session}_acq-anat_run-{run}_magnitude.nii.gz"
    rename_bids_b1_anat_ref_file = pe.Node(BidsOutputFormatter(), name="rename_bids_b1_anat_ref")
    rename_bids_b1_anat_ref_file.inputs.pattern = b1_anat_ref_file_pattern
    wf.connect(wf_output_node, 'b1_anat_ref_file',
               rename_bids_b1_anat_ref_file, 'in_file')
    wf.connect(rename_bids_b1_anat_ref_file, 'out_file',
               data_sink, '@b1_anat_ref_file')

    bids_formatter_nodes = [rename_bids_b0_map_file, rename_bids_b1_anat_ref_file]
    for bids_formatter_node in bids_formatter_nodes:
        wf.connect(bids_input_node, 'subject', bids_formatter_node, 'subject')
        wf.connect(bids_input_node, 'session', bids_formatter_node, 'session')
        wf.connect(bids_input_node, 'run', bids_formatter_node, 'run')

    # run workflow
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


def main_non_bids(args):
    # Define a parser specifically for non-BIDS mode
    parser = argparse.ArgumentParser(
        description='Non-BIDS-based B1 map correction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--b0_phase_diff_file", type=str, required=True,
                        help='Path to the phase difference file used for generating the B0 map. '
                             'This file typically represents the phase differences between two echoes.')
    parser.add_argument("--b0_te_delta", type=str, required=True,
                        help='Difference in echo times (TE) between the two acquisitions used for '
                             'calculating the B0 map, provided in seconds (e.g., "0.00246" for 2.46 ms).')
    parser.add_argument("--b1_ste_file", type=str, required=True,
                        help='Path to the STE (Stimulated Echo) image file for B1 mapping. '
                             'This file is used as part of the B1 mapping process.')
    parser.add_argument("--b1_fid_file", type=str, required=True,
                        help='Path to the FID (Free Induction Decay) image file for B1 mapping. '
                             'This file complements the STE image and is used to compute the B1 map.')
    parser.add_argument("--output_directory", '-o', required=True, type=str,
                        help='Path to the directory where the corrected B1 map and other output files will be saved.')
    parser.add_argument('--n_procs', '-n', type=int, default=num_cores,
                        help='Number of parallel processes to use for computation. '
                             'Defaults to the total number of available CPU cores.')
    args = parser.parse_args(args)

    # Initialize the workflow with inputs
    wf = bonn_fieldmaps_workflow(base_dir=args.output_directory)
    wf.config['execution'] = {
        'remove_unnecessary_outputs': False,
        'sequential': True  # This is key to process one subject at a time
    }
    wf.inputs.input_node.b0_phase_diff_file = args.b0_phase_diff_file
    wf.inputs.input_node.b0_te_delta = args.b0_te_delta
    wf.inputs.input_node.b1_ste_file = args.b1_ste_file
    wf.inputs.input_node.b1_fid_file = args.b1_fid_file

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
