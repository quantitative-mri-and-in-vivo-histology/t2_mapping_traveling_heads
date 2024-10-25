import argparse
import math
import os
import multiprocessing

import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from nipype import Node
from nipype.interfaces.utility import IdentityInterface
from nodes.io import BidsOutputWriter

from utils.io import write_minimal_bids_dataset_description, find_image_and_json
from workflows.parameter_estimation import \
    estimate_relaxation_ssfp
from workflows.processing import preprocess_ssfp_spgr, create_brain_mask
from utils.bids_config import (DEFAULT_NIFTI_READ_EXT_ENTITY,
                               STANDARDIZED_ENTITY_OVERRIDES_T1W,
                               STANDARDIZED_ENTITY_OVERRIDES_T2W,
                               STANDARDIZED_ENTITY_OVERRIDES_B1_MAP,
                               STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF,
                               PROCESSED_ENTITY_OVERRIDES_B1_MAP,
                               PROCESSED_ENTITY_OVERRIDES_B1_ANAT_REF,
                               PROCESSED_ENTITY_OVERRIDES_T1W,
                               PROCESSED_ENTITY_OVERRIDES_T2W,
                               PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE,
                               PROCESSED_ENTITY_OVERRIDES_R1_MAP,
                               PROCESSED_ENTITY_OVERRIDES_R2_MAP,
                               PROCESSED_ENTITY_OVERRIDES_T1_MAP,
                               PROCESSED_ENTITY_OVERRIDES_T2_MAP,
                               PROCESSED_ENTITY_OVERRIDES_AM_MAP,
                               PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK)
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
        description="Process SSFP dataset.")
    parser.add_argument('-i', '--bids_root', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Path to the output derivatives folder.')
    parser.add_argument('-t', '--temp_dir', default=os.getcwd(),
                        help='Directory for intermediate outputs (default: current working directory).')
    parser.add_argument(
        '--preprocess_only', action='store_true', default=False,
        help="Preprocess the data only, without parameter estimation"
    )
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
    layout = BIDSLayout(args.bids_root,
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
                runs = [args.run] if args.run else layout.get_runs(
                    subject=subject, session=session)

                if len(runs) == 0:
                    runs = [None]

                for run in runs:
                    t1w_files = layout.get(subject=subject,
                                           session=session,
                                           run=run,
                                           extension="nii.gz",
                                           **STANDARDIZED_ENTITY_OVERRIDES_T1W)

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
                                           run=run,
                                           extension="nii.gz",
                                           **STANDARDIZED_ENTITY_OVERRIDES_T2W
                                           )

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
                        run=run,
                        extension="nii.gz",
                        **STANDARDIZED_ENTITY_OVERRIDES_B1_MAP)

                    (b1_anat_ref_file,
                     b1_anat_json_dict) = find_image_and_json(
                        layout, subject=subject,
                        session=session,
                        run=run,
                        extension="nii.gz",
                        **STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF)

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       t1w_files=t1w_files,
                                       t1w_reg_target_file=t1w_reg_target_file,
                                       t2w_files=t2w_files,
                                       b1_map_file=b1_map_file,
                                       b1_anat_ref_file=b1_anat_ref_file,
                                       qi_jsr_config_dict=qi_jsr_config_dict))

    # set up workflow
    wf = pe.Workflow(name="process_ssfp_dataset")
    wf.base_dir = args.temp_dir

    # create input node using entries in input_dict and
    # use independent subject-session-run combinations as iterables
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    preprocess_ssfp_wf = preprocess_ssfp_spgr()

    wf.connect([(input_node, preprocess_ssfp_wf, [
        ('b1_map_file', 'input_node.b1_map_file'),
        ('b1_anat_ref_file', 'input_node.b1_anat_ref_file'),
        ('t1w_files', 'input_node.t1w_files'),
        ('t2w_files', 'input_node.t2w_files'),
        ('t1w_reg_target_file', 'input_node.reg_target_file')
    ])])

    create_brain_mask_wf = create_brain_mask()
    wf.connect(preprocess_ssfp_wf, "output_node.reg_target_file",
               create_brain_mask_wf, "input_node.in_file")

    b1_map_file_writer = pe.Node(BidsOutputWriter(),
                                 name="b1_map_file_writer")
    b1_map_file_writer.inputs.output_dir = args.output_dir
    b1_map_file_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_B1_MAP
    wf.connect(preprocess_ssfp_wf, "output_node.b1_map_file",
               b1_map_file_writer, "in_file")
    wf.connect(input_node, "b1_map_file",
               b1_map_file_writer, "template_file")

    b1_anat_ref_file_writer = pe.Node(BidsOutputWriter(),
                                      name="b1_anat_ref_file_writer")
    b1_anat_ref_file_writer.inputs.output_dir = args.output_dir
    b1_anat_ref_file_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_B1_ANAT_REF
    wf.connect(preprocess_ssfp_wf, "output_node.b1_anat_ref_file",
               b1_anat_ref_file_writer, "in_file")
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "template_file")

    t1w_file_writer = pe.MapNode(BidsOutputWriter(),
                                 iterfield=['in_file', 'template_file'],
                                 name="t1w_file_writer")
    t1w_file_writer.inputs.output_dir = args.output_dir
    t1w_file_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_T1W
    wf.connect(preprocess_ssfp_wf, "output_node.t1w_files",
               t1w_file_writer, "in_file")
    wf.connect(input_node, "t1w_files",
               t1w_file_writer, "template_file")

    t2w_file_writer = pe.MapNode(BidsOutputWriter(),
                                 iterfield=['in_file', 'template_file'],
                                 name="t2w_file_writer")
    t2w_file_writer.inputs.output_dir = args.output_dir
    t2w_file_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_T2W
    wf.connect(preprocess_ssfp_wf, "output_node.t2w_files",
               t2w_file_writer, "in_file")
    wf.connect(input_node, "t2w_files",
               t2w_file_writer, "template_file")

    t1w_reg_target_writer = pe.Node(BidsOutputWriter(),
                                    name="t1w_reg_target_writer")
    t1w_reg_target_writer.inputs.output_dir = args.output_dir
    t1w_reg_target_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE
    wf.connect(preprocess_ssfp_wf, "output_node.reg_target_file",
               t1w_reg_target_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               t1w_reg_target_writer, "template_file")

    brain_mask_file_writer = pe.Node(BidsOutputWriter(),
                                     name="brain_mask_file_writer")
    brain_mask_file_writer.inputs.output_dir = args.output_dir
    brain_mask_file_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK
    wf.connect(create_brain_mask_wf, "output_node.out_file",
               brain_mask_file_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               brain_mask_file_writer, "template_file")

    if not args.preprocess_only:
        estimate_relaxation_ssfp_wf = estimate_relaxation_ssfp()

        wf.connect([(preprocess_ssfp_wf, estimate_relaxation_ssfp_wf, [
            ('output_node.b1_map_file', 'input_node.b1_map_file'),
            ('output_node.t1w_files', 'input_node.t1w_files'),
            ('output_node.t2w_files', 'input_node.t2w_files'),
        ])])
        wf.connect(input_node, 'qi_jsr_config_dict',
                   estimate_relaxation_ssfp_wf, 'input_node.qi_jsr_config_dict')
        wf.connect(create_brain_mask_wf, 'output_node.out_file',
                   estimate_relaxation_ssfp_wf, 'input_node.brain_mask_file')

        r1_map_writer = pe.Node(BidsOutputWriter(),
                                name="r1_map_writer")
        r1_map_writer.inputs.output_dir = args.output_dir
        r1_map_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_R1_MAP
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.r1_map_file",
                   r1_map_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   r1_map_writer, "template_file")

        r2_map_writer = pe.Node(BidsOutputWriter(),
                                name="r2_map_writer")
        r2_map_writer.inputs.output_dir = args.output_dir
        r2_map_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_R2_MAP
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.r2_map_file",
                   r2_map_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   r2_map_writer, "template_file")

        t1_map_writer = pe.Node(BidsOutputWriter(),
                                name="t1_map_writer")
        t1_map_writer.inputs.output_dir = args.output_dir
        t1_map_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_T1_MAP
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.t1_map_file",
                   t1_map_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   t1_map_writer, "template_file")

        t2_map_writer = pe.Node(BidsOutputWriter(),
                                name="t2_map_writer")
        t2_map_writer.inputs.output_dir = args.output_dir
        t2_map_writer.inputs.entity_overrides = PROCESSED_ENTITY_OVERRIDES_T2_MAP
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.t2_map_file",
                   t2_map_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   t2_map_writer, "template_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
