import argparse
import os
import multiprocessing
from nipype import Workflow
import shutil

import json
import os
import math

from PyQt6.uic.pyuic import generate
from nipype.interfaces.utility import IdentityInterface
import nipype.pipeline.engine as pe
from nipype import Node, Function
from abc import abstractmethod, ABC
from bids.layout import BIDSLayout
from datasets.bids_dataset import BidsDataset
from workflows.preprocessing_workflows import \
    denoise_mag_and_phase_in_complex_domain_workflow, \
    motion_correction_mag_and_phase_workflow, create_brain_mask_workflow, \
    register_image_workflow
from utils.io import ExplicitPathDataSink
import nipype.interfaces.fsl as fsl
from nipype_utils import BidsRename, BidsOutputFormatter, BidsOutputWriter
from workflows.preprocessing_workflows import preprocess_ssfp, \
    preprocess_ssfp_multi_file
from workflows.parameter_estimation_workflows import \
    estimate_relaxation_ssfp_multi_file
from utils.io import write_minimal_bids_dataset_description, find_image_and_json


def assert_all_similar(values, tolerance=1e-9):
    # Ensure all values in the list are similar within the given tolerance
    first_value = values[0]
    assert all(
        math.isclose(value, first_value, rel_tol=tolerance, abs_tol=tolerance)
        for value in values), \
        "Not all values are similar"


def main():
    parser = argparse.ArgumentParser(
        description="Process a dataset with optional steps.")
    parser.add_argument('-i', '--bids_root', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-d', '--derivatives', nargs='*', required=False,
                        help='One or more derivatives directories to use.')
    parser.add_argument('-o', '--output_derivative_dir', required=True,
                        help='Path to the output derivatives folder.')
    parser.add_argument('--base_dir', default=os.getcwd(),
                        help='Base directory for processing (default: current working directory).')
    parser.add_argument(
        '--preprocess_only', action='store_true', default=False,
        help="Preprocess the data only, without parameter estimation"
    )
    parser.add_argument('--subject', default=None,
                        help='Specify a subject to process (e.g., sub-01). If not provided, all subjects are processed.')
    parser.add_argument('--session', default=None,
                        help='Specify a session to process (e.g., ses-01). If not provided, all sessions are processed.')
    parser.add_argument('--run', default=None,
                        help='Specify a run to process (e.g., run-01). If not provided, all runs are processed.')
    parser.add_argument('--n_procs', type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of processors to use (default: all available cores).')
    args = parser.parse_args()

    # write minimal dataset description for output derivatives
    os.makedirs(args.output_derivative_dir, exist_ok=True)
    write_minimal_bids_dataset_description(
        dataset_root=args.output_derivative_dir,
        dataset_name=os.path.dirname(args.output_derivative_dir)
    )

    # Define the reusable run settings in a dictionary
    run_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {'n_procs': args.n_procs}
    }

    # collect inputs
    layout = BIDSLayout(args.bids_root,
                        derivatives=args.derivatives,
                        validate=False)
    inputs = []
    subjects = layout.get_subjects()

    for subject in subjects:
        sessions = layout.get_sessions(subject=subject)
        if sessions:  # Only add subjects with existing sessions
            for session in sessions:
                runs = layout.get_runs(subject=subject, session=session)

                if len(runs) == 0:
                    runs = [None]

                for run in runs:
                    t1w_files = layout.get(subject=subject,
                                           session=session,
                                           suffix="T1w",
                                           extension="nii.gz",
                                           run=run)

                    t1w_flip_angles = [t1w_file.entities["FlipAngle"] for
                                       t1w_file in t1w_files]
                    t1w_echo_times = [t1w_file.entities["EchoTime"] for
                                      t1w_file in t1w_files]
                    assert_all_similar(t1w_echo_times)
                    t1w_repetition_times = [
                        t1w_file.entities["RepetitionTimeExcitation"] for
                        t1w_file in t1w_files]
                    assert_all_similar(t1w_repetition_times)

                    max_flip_angle_index = t1w_flip_angles.index(
                        max(t1w_flip_angles))
                    t1w_reg_target_file = t1w_files[max_flip_angle_index]

                    t2w_files = layout.get(subject=subject,
                                           session=session,
                                           suffix="T2w",
                                           extension="nii.gz",
                                           run=run)

                    t2w_rf_phase_increments = [
                        t2w_file.entities["RfPhaseIncrement"] for
                        t2w_file in t2w_files]
                    t2w_flip_angles = [t2w_file.entities["FlipAngle"] for
                                       t2w_file in t2w_files]
                    t2w_rf_pulse_durations = [
                        t2w_file.entities["RfPulseDuration"] for
                        t2w_file in t2w_files]
                    assert_all_similar(t2w_rf_pulse_durations)
                    t2w_repetition_times = [
                        t2w_file.entities["RepetitionTimeExcitation"] for
                        t2w_file in t2w_files]
                    assert_all_similar(t2w_repetition_times)

                    spgr_dict = dict(TR=t1w_repetition_times[0],
                                     TE=t1w_echo_times[0],
                                     FA=t1w_flip_angles)
                    ssfp_dict = dict(TR=t2w_repetition_times[0],
                                     Trf=t2w_rf_pulse_durations[0],
                                     FA=t2w_flip_angles,
                                     PhaseInc=t2w_rf_phase_increments)
                    qi_jsr_config_dict = dict(SPGR=spgr_dict, SSFP=ssfp_dict)

                    (b1_map_file,
                     b1_map_json_dict) = find_image_and_json(
                        layout, subject=subject,
                        session=session,
                        suffix="B1Map",
                        acquisition="B1",
                        extension="nii.gz",
                        run=run)

                    (b1_anat_ref_file,
                     b1_anat_json_dict) = find_image_and_json(
                        layout, subject=subject,
                        session=session,
                        suffix="magnitude",
                        extension="nii.gz",
                        acquisition="B1Ref",
                        run=run)

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       t1w_files=t1w_files,
                                       t1w_reg_target_file=t1w_reg_target_file,
                                       t2w_files=t2w_files,
                                       b1_map_file=b1_map_file,
                                       b1_anat_ref_file=b1_anat_ref_file,
                                       qi_jsr_config_dict=qi_jsr_config_dict))

    wf = pe.Workflow(name="process_ssfp")
    wf.base_dir = args.base_dir

    # set up bids input node
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    preprocess_ssfp_wf = preprocess_ssfp_multi_file()

    wf.connect([(input_node, preprocess_ssfp_wf, [
        ('b1_map_file', 'input_node.b1_map_file'),
        ('b1_anat_ref_file', 'input_node.b1_anat_ref_file'),
        ('t1w_files', 'input_node.t1w_files'),
        ('t2w_files', 'input_node.t2w_files'),
        ('t1w_reg_target_file', 'input_node.reg_target_file')
    ])])

    out_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                  'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                  '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'

    b1_map_file_writer = pe.Node(BidsOutputWriter(),
                                 name="b1_map_file_writer")
    b1_map_file_writer.inputs.output_dir = args.output_derivative_dir
    b1_map_file_writer.inputs.pattern = out_pattern
    b1_map_file_writer.inputs.entity_overrides = dict(acquisition="B1",
                                                      suffix="B1Map",
                                                      desc="registered")
    wf.connect(preprocess_ssfp_wf, "output_node.b1_map_file",
               b1_map_file_writer, "in_file")
    wf.connect(input_node, "b1_map_file",
               b1_map_file_writer, "template_file")

    b1_anat_ref_file_writer = pe.Node(BidsOutputWriter(),
                                 name="b1_anat_ref_file_writer")
    b1_anat_ref_file_writer.inputs.output_dir = args.output_derivative_dir
    b1_anat_ref_file_writer.inputs.pattern = out_pattern
    b1_anat_ref_file_writer.inputs.entity_overrides = dict(acquisition="B1ref",
                                                      suffix="magnitude",
                                                      desc="registered")
    wf.connect(preprocess_ssfp_wf, "output_node.b1_anat_ref_file",
               b1_anat_ref_file_writer, "in_file")
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "template_file")

    t1w_file_writer = pe.MapNode(BidsOutputWriter(),
                                 iterfield=['in_file', 'template_file'],
                                 name="t1w_file_writer")
    t1w_file_writer.inputs.output_dir = args.output_derivative_dir
    t1w_file_writer.inputs.pattern = out_pattern
    t1w_file_writer.inputs.entity_overrides = dict(part=None, desc="preproc")
    wf.connect(preprocess_ssfp_wf, "output_node.t1w_files",
               t1w_file_writer, "in_file")
    wf.connect(input_node, "t1w_files",
               t1w_file_writer, "template_file")

    t2w_file_writer = pe.MapNode(BidsOutputWriter(),
                                 iterfield=['in_file', 'template_file'],
                                 name="t2w_file_writer")
    t2w_file_writer.inputs.output_dir = args.output_derivative_dir
    t2w_file_writer.inputs.pattern = out_pattern
    t2w_file_writer.inputs.entity_overrides = dict(part=None, desc="preproc")
    wf.connect(preprocess_ssfp_wf, "output_node.t2w_files",
               t2w_file_writer, "in_file")
    wf.connect(input_node, "t2w_files",
               t2w_file_writer, "template_file")

    t1w_reg_target_writer = pe.Node(BidsOutputWriter(),
                                     name="t1w_reg_target_writer")
    t1w_reg_target_writer.inputs.output_dir = args.output_derivative_dir
    t1w_reg_target_writer.inputs.pattern = out_pattern
    t1w_reg_target_writer.inputs.entity_overrides = dict(part=None, desc="preproc", acquisition="T1wRef")
    wf.connect(preprocess_ssfp_wf, "output_node.reg_target_file",
               t1w_reg_target_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               t1w_reg_target_writer, "template_file")

    brain_mask_file_writer = pe.Node(BidsOutputWriter(),
                                     name="brain_mask_file_writer")
    brain_mask_file_writer.inputs.output_dir = args.output_derivative_dir
    brain_mask_file_writer.inputs.pattern = out_pattern
    brain_mask_file_writer.inputs.entity_overrides = dict(part=None,
                                                          desc="brain",
                                                          suffix="mask",
                                                          acquisition=None)
    wf.connect(preprocess_ssfp_wf, "output_node.brain_mask_file",
               brain_mask_file_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               brain_mask_file_writer, "template_file")

    if not args.preprocess_only:
        estimate_relaxation_ssfp_wf = estimate_relaxation_ssfp_multi_file()

        wf.connect([(preprocess_ssfp_wf, estimate_relaxation_ssfp_wf, [
            ('output_node.b1_map_file', 'input_node.b1_map_file'),
            ('output_node.t1w_files', 'input_node.t1w_files'),
            ('output_node.t2w_files', 'input_node.t2w_files'),
            ('output_node.brain_mask_file', 'input_node.brain_mask_file')
        ])])
        wf.connect(input_node, 'qi_jsr_config_dict',
                   estimate_relaxation_ssfp_wf, 'input_node.qi_jsr_config_dict')

        # write output files
        out_maps = dict(
            R1map="output_node.r1_map_file",
            R2map="output_node.r2_map_file",
            T1map="output_node.t1_map_file",
            T2map="output_node.t2_map_file"
        )
        for out_map_suffix, out_map_name in out_maps.items():
            file_writer = pe.Node(BidsOutputWriter(),
                                  name="file_writer_{}".format(out_map_suffix))
            file_writer.inputs.output_dir = args.output_derivative_dir
            file_writer.inputs.pattern = out_pattern
            file_writer.inputs.entity_overrides = dict(part=None,
                                                       suffix=out_map_suffix,
                                                       acquisition=None)
            wf.connect(estimate_relaxation_ssfp_wf, out_map_name,
                       file_writer, "in_file")
            wf.connect(input_node, "t1w_reg_target_file",
                       file_writer, "template_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
