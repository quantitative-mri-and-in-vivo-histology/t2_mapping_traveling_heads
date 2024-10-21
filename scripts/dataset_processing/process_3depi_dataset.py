import argparse
import math
import multiprocessing
import os

import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from nipype import Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces import fsl

from nodes.io import BidsOutputWriter
from utils.bids_config import DEFAULT_NIFTI_READ_EXT_ENTITY, \
    STANDARDIZED_ENTITY_OVERRIDES_T1W, \
    STANDARDIZED_ENTITY_OVERRIDES_T2W_MAG, \
    STANDARDIZED_ENTITY_OVERRIDES_T2W_PHASE, \
    STANDARDIZED_ENTITY_OVERRIDES_B1_MAP, STANDARDIZED_ENTITY_OVERRIDES_B1_REF
from utils.io import write_minimal_bids_dataset_description, find_image_and_json
from workflows.parameter_estimation import \
    estimate_relaxation_3d_epi
from workflows.processing import preprocess_3depi_workflow, create_brain_mask
from nodes.registration import create_default_ants_rigid_registration_node


def assert_all_similar(values, tolerance=1e-9):
    # Ensure all values in the list are similar within the given tolerance
    first_value = values[0]
    assert all(
        math.isclose(value, first_value, rel_tol=tolerance, abs_tol=tolerance)
        for value in values), \
        "Not all values are similar"


def main():
    parser = argparse.ArgumentParser(
        description="Process 3D-EPI dataset.")
    parser.add_argument('-i', '--bids_root', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-o', '--output_derivative_dir', required=True,
                        help='Path to the output derivatives folder.')
    parser.add_argument('--base_dir', default=os.getcwd(),
                        help='Base directory for processing (default: current working directory).')
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
    os.makedirs(args.output_derivative_dir, exist_ok=True)
    write_minimal_bids_dataset_description(
        dataset_root=args.output_derivative_dir,
        dataset_name=os.path.dirname(args.output_derivative_dir)
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
                    input_dict = dict()

                    (input_dict["t1w_file"],
                     input_dict["t1w_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        **STANDARDIZED_ENTITY_OVERRIDES_T1W,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["t2w_mag_file"],
                     input_dict["t2w_mag_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        **STANDARDIZED_ENTITY_OVERRIDES_T2W_MAG,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["t2w_phase_file"],
                     input_dict["t2w_phase_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        **STANDARDIZED_ENTITY_OVERRIDES_T2W_PHASE,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b1_map_file"],
                     input_dict["b1_map_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        **STANDARDIZED_ENTITY_OVERRIDES_B1_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b1_anat_ref_file"],
                     input_dict["b1_anat_ref_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        **STANDARDIZED_ENTITY_OVERRIDES_B1_REF,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["echo_time"] = input_dict[
                        "t2w_mag_json_dict"]["EchoTime"]
                    input_dict["repetition_time"] = input_dict[
                        "t2w_mag_json_dict"]["RepetitionTimeExcitation"]
                    input_dict["flip_angle"] = input_dict[
                        "t2w_mag_json_dict"]["FlipAngle"]
                    input_dict["rf_phase_increments"] = input_dict[
                        "t2w_mag_json_dict"]["RfPhaseIncrement"]

                    inputs.append(input_dict)

    # set up workflow
    wf = pe.Workflow(name="process_3depi_dataset")
    wf.base_dir = args.base_dir

    # create input node using entries in input_dict and
    # use independent subject-session-run combinations as iterables
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    preprocess_3depi_wf = preprocess_3depi_workflow()

    wf.connect([(input_node, preprocess_3depi_wf, [
        ('b1_map_file', 'input_node.b1_map_file'),
        ('b1_anat_ref_file', 'input_node.b1_anat_ref_file'),
        ('t2w_mag_file', 'input_node.magnitude_file'),
        ('t2w_phase_file', 'input_node.phase_file'),
        # ('t1w_file', 'input_node.t1w_file'),
    ])])

    mag_first_volume_extractor = Node(fsl.ExtractROI(),
                                      name="mag_first_volume_extractor")
    mag_first_volume_extractor.inputs.t_min = 0
    mag_first_volume_extractor.inputs.t_size = 1
    wf.connect(preprocess_3depi_wf, "output_node.magnitude_file",
               mag_first_volume_extractor, "in_file")

    register_t1w_to_t2w = pe.Node(create_default_ants_rigid_registration_node(),
                                          name="register_t1w_to_t2w")
    wf.connect(mag_first_volume_extractor, "roi_file",
               register_t1w_to_t2w, "fixed_image")
    wf.connect(input_node, "t1w_file",
               register_t1w_to_t2w, "moving_image")

    create_brain_mask_wf = create_brain_mask()
    wf.connect(register_t1w_to_t2w, "warped_image",
               create_brain_mask_wf, "input_node.in_file")


    b1_map_file_writer = pe.Node(BidsOutputWriter(),
                                 name="b1_map_file_writer")
    b1_map_file_writer.inputs.output_dir = args.output_derivative_dir
    b1_map_file_writer.inputs.entity_overrides = dict(acquisition="B1",
                                                      suffix="B1map",
                                                      desc="registered")
    wf.connect(preprocess_3depi_wf, "output_node.b1_map_file",
               b1_map_file_writer, "in_file")
    wf.connect(input_node, "b1_map_file",
               b1_map_file_writer, "template_file")

    b1_anat_ref_file_writer = pe.Node(BidsOutputWriter(),
                                      name="b1_anat_ref_file_writer")
    b1_anat_ref_file_writer.inputs.output_dir = args.output_derivative_dir
    b1_anat_ref_file_writer.inputs.entity_overrides = dict(acquisition="B1ref",
                                                           suffix="magnitude",
                                                           desc="registered")
    wf.connect(preprocess_3depi_wf, "output_node.b1_anat_ref_file",
               b1_anat_ref_file_writer, "in_file")
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "template_file")

    t2w_mag_file_writer = pe.MapNode(BidsOutputWriter(),
                                     iterfield=['in_file', 'template_file'],
                                     name="t2w_mag_file_writer")
    t2w_mag_file_writer.inputs.output_dir = args.output_derivative_dir
    t2w_mag_file_writer.inputs.entity_overrides = dict(desc="preproc")
    wf.connect(preprocess_3depi_wf, "output_node.magnitude_file",
               t2w_mag_file_writer, "in_file")
    wf.connect(input_node, "t2w_mag_file",
               t2w_mag_file_writer, "template_file")

    t2w_phase_file_writer = pe.MapNode(BidsOutputWriter(),
                                       iterfield=['in_file', 'template_file'],
                                       name="t2w_phase_file_writer")
    t2w_phase_file_writer.inputs.output_dir = args.output_derivative_dir
    t2w_phase_file_writer.inputs.entity_overrides = dict(desc="preproc")
    wf.connect(preprocess_3depi_wf, "output_node.phase_file",
               t2w_phase_file_writer, "in_file")
    wf.connect(input_node, "t2w_phase_file",
               t2w_phase_file_writer, "template_file")

    t1w_reg_target_writer = pe.Node(BidsOutputWriter(),
                                    name="t1w_reg_target_writer")
    t1w_reg_target_writer.inputs.output_dir = args.output_derivative_dir
    t1w_reg_target_writer.inputs.entity_overrides = dict(part=None,
                                                         desc="preproc",
                                                         acquisition="T1wRef")
    wf.connect(register_t1w_to_t2w, "warped_image",
               t1w_reg_target_writer, "in_file")
    wf.connect(input_node, "t1w_file",
               t1w_reg_target_writer, "template_file")

    brain_mask_file_writer = pe.Node(BidsOutputWriter(),
                                     name="brain_mask_file_writer")
    brain_mask_file_writer.inputs.output_dir = args.output_derivative_dir
    brain_mask_file_writer.inputs.entity_overrides = dict(part=None,
                                                          desc="brain",
                                                          suffix="mask",
                                                          acquisition=None)
    wf.connect(create_brain_mask_wf, "output_node.out_file",
               brain_mask_file_writer, "in_file")
    wf.connect(input_node, "t1w_file",
               brain_mask_file_writer, "template_file")

    if not args.preprocess_only:

        # estimate relaxation parameter maps
        estimate_relaxation_3d_epi_wf = estimate_relaxation_3d_epi()
        wf.connect([(preprocess_3depi_wf, estimate_relaxation_3d_epi_wf, [
            ('output_node.b1_map_file', 'input_node.b1_map_file'),
            ('output_node.magnitude_file', 'input_node.t2w_magnitude_file'),
            ('output_node.phase_file', 'input_node.t2w_phase_file'),
            ('output_node.brain_mask_file', 'input_node.brain_mask_file')
        ])])
        wf.connect([(input_node, estimate_relaxation_3d_epi_wf, [
            ('rf_phase_increments', 'input_node.rf_phase_increments'),
            ('repetition_time', 'input_node.repetition_time'),
            ('flip_angle', 'input_node.flip_angle')
        ])])

        # write relaxation parameter map files
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
            file_writer.inputs.entity_overrides = dict(part=None,
                                                       suffix=out_map_suffix,
                                                       acquisition=None)
            wf.connect(estimate_relaxation_3d_epi_wf, out_map_name,
                       file_writer, "in_file")
            wf.connect(input_node, "t2w_mag_file",
                       file_writer, "template_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
