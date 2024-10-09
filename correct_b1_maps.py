import os
import argparse
import sys
import nipype.interfaces.io as nio
import multiprocessing
from pathlib import Path
from bids.layout import BIDSLayout
from nipype import Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype_utils import BidsRename, BidsOutputFormatter, create_output_folder

num_cores = multiprocessing.cpu_count()


def register_image(base_dir=os.getcwd(), name="register_image"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir

    input_node = pe.Node(util.IdentityInterface(
        fields=['moving_file', 'reference_file', 'target_file']),
        name='input_node')

    output_node = pe.Node(util.IdentityInterface(fields=['out_file']),
                          name='output_node')

    flirt_estimate = pe.Node(fsl.FLIRT(uses_qform=True, dof=6),
                             "flirt_estimate")
    flirt_apply = pe.Node(fsl.FLIRT(apply_xfm=True, uses_qform=True, dof=6),
                          "flirt_apply")

    first_volume_extractor = Node(fsl.ExtractROI(),
                                  name="first_volume_extractor")
    first_volume_extractor.inputs.t_min = 0
    first_volume_extractor.inputs.t_size = 1

    bet_target = Node(fsl.BET(), name="bet_target")
    bet_target.inputs.robust = True

    bet_reference = Node(fsl.BET(), name="bet_reference")
    bet_reference.inputs.robust = True

    workflow.connect(input_node, "reference_file", bet_reference, "in_file")
    workflow.connect(input_node, "target_file", bet_target, "in_file")

    workflow.connect(bet_target, "out_file", first_volume_extractor, "in_file")
    workflow.connect(bet_reference, "out_file", flirt_estimate, "in_file")
    workflow.connect(first_volume_extractor, "roi_file", flirt_estimate, "reference")

    workflow.connect(input_node, "moving_file", flirt_apply, "in_file")
    workflow.connect(first_volume_extractor, "roi_file", flirt_apply, "reference")
    workflow.connect(flirt_estimate, "out_matrix_file", flirt_apply, "in_matrix_file")

    workflow.connect(flirt_apply, "out_file", output_node, "out_file")


    # bet_target = Node(fsl.BET(), name="bet_target")
    # bet_target.inputs.robust = True
    #
    # bet_reference = Node(fsl.BET(), name="bet_reference")
    # bet_reference.inputs.robust = True
    #
    # workflow.connect(input_node, "reference_file", bet_reference, "in_file")
    # workflow.connect(input_node, "target_file", bet_target, "in_file")
    #
    # workflow.connect(input_node, "target_file", first_volume_extractor, "in_file")
    # workflow.connect(input_node, "reference_file", flirt_estimate, "in_file")
    # workflow.connect(first_volume_extractor, "roi_file", flirt_estimate, "reference")
    #
    # workflow.connect(input_node, "moving_file", flirt_apply, "in_file")
    # workflow.connect(first_volume_extractor, "roi_file", flirt_apply, "reference")
    # workflow.connect(flirt_estimate, "out_matrix_file", flirt_apply, "in_matrix_file")
    #
    # workflow.connect(flirt_apply, "out_file", output_node, "out_file")

    return workflow


def correct_b1_map(b1_map_file, b0_map_file, fa_b1_in_degrees, fa_nominal_in_degrees, pulse_duration_in_seconds):
    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()
    b1_map_file_nib = nib.load(b1_map_file)
    b0_map_file_nib = nib.load(b0_map_file)

    b1_map_image = b1_map_file_nib.get_fdata()
    b0_map_image = b0_map_file_nib.get_fdata()

    def cos_omega_eff(omega_eff, delta_omega, tau=2.445e-3):
        return np.cos(omega_eff * tau) + ((delta_omega / omega_eff) ** 2) * (1 - np.cos(omega_eff * tau))

    def gamma_b1(alpha, tau=2.445e-3):
        return alpha / tau

    tau = pulse_duration_in_seconds
    fa_nominal = np.deg2rad(fa_nominal_in_degrees)
    fa_actual = fa_nominal * b1_map_image / (fa_b1_in_degrees * 10.0)
    delta_omega = b0_map_image * 2.0 * np.pi

    omega_eff = np.sqrt((delta_omega) ** 2 + (gamma_b1(fa_actual, tau=tau)) ** 2)
    cosine = cos_omega_eff(omega_eff=omega_eff, delta_omega=delta_omega, tau=tau)
    fa_actual_with_offresonance = np.arccos(cosine)
    b1_map = 100 * fa_actual_with_offresonance / fa_nominal

    b1_output_filename = os.path.join(base_dir, 'b1_map_corr.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_map, b0_map_file_nib.affine, b0_map_file_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename


def b1_correction_workflow(base_dir=os.getcwd(), name="b1_correction"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(util.IdentityInterface(
        fields=['b0_map_file',
                'b1_map_file',
                'b0_anat_ref_file',
                'b1_anat_ref_file',
                'fa_b1_in_degrees',
                'fa_nominal_in_degrees',
                'pulse_duration_in_seconds']),
        name='input_node')
    output_node = pe.Node(util.IdentityInterface(fields=['out_file']),
                          name='output_node')

    # Register B1 map to B0 map
    register_b1_map_to_b0_map = register_image(name="register_b1_map_to_b0_map")
    wf.connect(input_node, "b1_map_file", register_b1_map_to_b0_map, "input_node.moving_file")
    wf.connect(input_node, "b1_anat_ref_file", register_b1_map_to_b0_map, "input_node.reference_file")
    wf.connect(input_node, "b0_anat_ref_file", register_b1_map_to_b0_map, "input_node.target_file")

    # Correct B1 map
    correct_b1_map_node = Node(Function(
        input_names=['b1_map_file', 'b0_map_file', 'fa_b1_in_degrees',
                     'fa_nominal_in_degrees', 'pulse_duration_in_seconds'],
        output_names=['b1_output_file'],
        function=correct_b1_map),
        name='correct_b1_map')

    wf.connect(register_b1_map_to_b0_map, "output_node.out_file", correct_b1_map_node, "b1_map_file")
    wf.connect(input_node, "b0_map_file", correct_b1_map_node, "b0_map_file")
    wf.connect(input_node, "fa_b1_in_degrees", correct_b1_map_node, "fa_b1_in_degrees")
    wf.connect(input_node, "fa_nominal_in_degrees", correct_b1_map_node, "fa_nominal_in_degrees")
    wf.connect(input_node, "pulse_duration_in_seconds", correct_b1_map_node, "pulse_duration_in_seconds")
    wf.connect(correct_b1_map_node, "b1_output_file", output_node, "out_file")

    return wf


def collect_bids_hamburg_inputs(input_directory, input_derivatives, subject_id=None,
                        session_id=None, run_id=None):
    raise NotImplementedError()


def collect_bids_bonn_inputs(input_directory, input_derivatives, subject_id=None,
                        session_id=None, run_id=None):
    layout = BIDSLayout(input_directory, derivatives=input_derivatives, validate=False)

    # dataset constants
    fa_b1_in_degrees = 60
    pulse_duration_in_seconds = 2.46e-3

    subjects = [subject_id] if subject_id else layout.get_subjects()
    inputs = []

    for subject in subjects:
        sessions = [session_id] if session_id else layout.get_sessions(subject=subject)
        if sessions:
            for session in sessions:
                valid_runs = layout.get(
                    return_type='id',
                    subject=subject,
                    session=session,
                    target='run',
                    suffix='T2w',
                    part="phase",
                    extension="nii.gz"
                )
                runs = [run_id] if isinstance(run_id,
                                              str) else valid_runs

                if len(runs) == 0:
                    runs = [None]

                for run in runs:
                    b0_map_files = layout.get(
                        subject=subject,
                        session=session,
                        acquisition="b0",
                        suffix="phasediff",
                        extension="nii.gz",
                        run=run
                    )
                    b_maps_run_id = 1 if run is None else (run - 1) * 2 + 1 # dirty hack; todo: fix with IntendedFor field

                    b0_anat_ref_files = layout.get(
                        subject=subject,
                        session=session,
                        acquisition="dznebnB0",
                        suffix="magnitude1",
                        extension="nii.gz",
                        run=b_maps_run_id
                    )

                    b1_map_files = layout.get(
                        subject=subject,
                        session=session,
                        suffix="B1map",
                        extension="nii.gz",
                        run=run
                    )

                    b1_map_files = [f for f in b1_map_files
                                         if f.entities.get(
                            "desc") == "phaseWrapCorrected"]

                    b1_anat_ref_files = layout.get(
                        subject=subject,
                        session=session,
                        acquisition="b1anat",
                        suffix="magnitude",
                        extension="nii.gz",
                        run=run
                    )

                    b1_anat_ref_files = [f for f in b1_anat_ref_files
                                         if f.entities.get("desc") == "phaseWrapCorrected"]

                    inputs.append({
                        "subject": subject,
                        "session": session,
                        "run": run,
                        "b0_map_file": b0_map_files[0],
                        "b0_anat_ref_file": b0_anat_ref_files[0],
                        "b1_map_file": b1_map_files[0],
                        "b1_anat_ref_file": b1_anat_ref_files[0],
                        "fa_b1_in_degrees": fa_b1_in_degrees,
                        "pulse_duration_in_seconds": pulse_duration_in_seconds
                    })
    return inputs


def main_bids(args):
    # Define a parser specifically for BIDS mode
    parser = argparse.ArgumentParser(
        description='BIDS-based B1 map correction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_name", '-x', required=True, type=str,
                        help='BIDS input dataset name ("bonn" or "hamburg"')
    parser.add_argument("--input_directory", '-i', required=True, type=str,
                        help='BIDS input dataset root.')
    parser.add_argument("--input_derivatives", '-d', type=str, nargs='*',
                        help='BIDS derivative dataset roots (optional).')
    parser.add_argument("--fa_nominal_in_degrees", type=float, required=True,
                        help='Nominal flip angle in degrees.')
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
    if args.dataset_name == "bonn":
        inputs = collect_bids_bonn_inputs(args.input_directory, args.input_derivatives,
                                 subject_id=args.subject_id,
                                 session_id=args.session_id,
                                 run_id=args.run_id)
    elif args.dataset_name == "hamburg":
        inputs = collect_bids_hamburg_inputs(args.input_directory, args.input_derivatives,
                                     subject_id=args.subject_id,
                                     session_id=args.session_id,
                                     run_id=args.run_id)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset_name))

    # add nominal flip angle to each input set
    for i in range(0, len(inputs)):
        inputs[i]["fa_nominal_in_degrees"] = args.fa_nominal_in_degrees

    # Initialize workflow
    wf = b1_correction_workflow(base_dir=args.output_directory)

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
    b1_map_pattern = "sub-{subject}_ses-{session}_run-{run}_desc-b0Corrected_B1map.nii.gz"
    rename_bids_b1_map = pe.Node(BidsOutputFormatter(), name="rename_bids_b1_map")
    rename_bids_b1_map.inputs.pattern = b1_map_pattern
    wf.connect(bids_input_node, 'subject',
               rename_bids_b1_map, 'subject')
    wf.connect(bids_input_node, 'session',
               rename_bids_b1_map, 'session')
    wf.connect(bids_input_node, 'run',
               rename_bids_b1_map, 'run')
    wf.connect(wf_output_node, 'out_file',
               rename_bids_b1_map, 'in_file')
    wf.connect(rename_bids_b1_map, 'out_file',
               data_sink, '@b1_map_file')

    # run workflow
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


def main_non_bids(args):
    # Define a parser specifically for non-BIDS mode
    parser = argparse.ArgumentParser(
        description='Non-BIDS-based B1 map correction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--b1_map_file", type=str, required=True,
                        help='Path to the B1 map file.')
    parser.add_argument("--b0_map_file", type=str, required=True,
                        help='Path to the B0 map file.')
    parser.add_argument("--b0_anat_ref_file", type=str, required=True,
                        help='Path to the anatomical reference file for the B0 map.')
    parser.add_argument("--b1_anat_ref_file", type=str, required=True,
                        help='Path to the anatomical reference file for the B1 map.')
    parser.add_argument("--fa_b1_in_degrees", type=float, required=True,
                        help='Flip angle for the B1 map in degrees.')
    parser.add_argument("--fa_nominal_in_degrees", type=float, required=True,
                        help='Nominal flip angle in degrees.')
    parser.add_argument("--pulse_duration_in_seconds", type=float, required=True,
                        help='Duration of the pulse in seconds.')
    parser.add_argument("--output_directory", '-o', required=True, type=str,
                        help='Output directory for results.')
    parser.add_argument('--n_procs', '-n', type=int, default=num_cores,
                        help='Number of cores for parallel processing.')

    args = parser.parse_args(args)

    # Initialize the workflow with inputs
    wf = b1_correction_workflow(base_dir=args.output_directory)
    wf.inputs.input_node.b0_map_file = args.b0_map_file
    wf.inputs.input_node.b1_map_file = args.b1_map_file
    wf.inputs.input_node.b0_anat_ref_file = args.b0_anat_ref_file
    wf.inputs.input_node.b1_anat_ref_file = args.b1_anat_ref_file
    wf.inputs.input_node.fa_b1_in_degrees = args.fa_b1_in_degrees
    wf.inputs.input_node.fa_nominal_in_degrees = args.fa_nominal_in_degrees
    wf.inputs.input_node.pulse_duration_in_seconds = args.pulse_duration_in_seconds

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
