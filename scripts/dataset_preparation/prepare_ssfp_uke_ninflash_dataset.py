import argparse
import multiprocessing
import os

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from nipype import Node
from nipype import Workflow
from nipype.interfaces.utility import IdentityInterface

from nodes.io import BidsOutputWriter
from utils.bids_config import (DEFAULT_NIFTI_READ_EXT_ENTITY,
                               STANDARDIZED_ENTITY_OVERRIDES_T1W, \
                               STANDARDIZED_ENTITY_OVERRIDES_T2W, \
                               STANDARDIZED_ENTITY_OVERRIDES_B1_MAP,
                               STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF)
from utils.io import write_minimal_bids_dataset_description, find_image_and_json


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SSFP-NINFLASH dataset from UKE, Hamburg.")
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
                                           acquisition="t2Ssfp3A12RF180",
                                           part="mag",
                                           suffix='T2w',
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

                    input_dict = dict(
                        subject=subject,
                        session=session,
                        run=run
                    )

                    try:
                        (t1w_a2_file, t1w_a2_json_dict) = find_image_and_json(
                            layout,
                            subject=subject,
                            session=session,
                            acquisition="t2Ssfp3A2",
                            run=run,
                            part="mag",
                            suffix="T1w",
                            **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        t1w_a13_files = layout.get(subject=subject,
                                                   session=session,
                                                   acquisition="t2Ssfp3A13",
                                                   run=run,
                                                   part="mag",
                                                   suffix="T1w",
                                                   **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        t1w_a12_files = layout.get(subject=subject,
                                                   session=session,
                                                   acquisition="t2Ssfp3A12",
                                                   run=run,
                                                   part="mag",
                                                   suffix="T1w",
                                                   **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        if len(t1w_a13_files) == 0 and len(t1w_a12_files) == 1:
                            (t1w_a13_file,
                             t1w_a13_json_dict) = find_image_and_json(
                                layout,
                                subject=subject,
                                session=session,
                                acquisition="t2Ssfp3A12",
                                run=run,
                                part="mag",
                                suffix="T1w",
                                **DEFAULT_NIFTI_READ_EXT_ENTITY)
                        else:
                            (t1w_a13_file,
                             t1w_a13_json_dict) = find_image_and_json(
                                layout,
                                subject=subject,
                                session=session,
                                acquisition="t2Ssfp3A13",
                                run=run,
                                part="mag",
                                suffix="T1w",
                                **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        (t2w_a12rf180_file,
                         t2w_a12rf180_json_dict) = find_image_and_json(
                            layout,
                            subject=subject,
                            session=session,
                            acquisition="t2Ssfp3A12RF180",
                            run=run,
                            part="mag",
                            suffix="T2w",
                            **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        (t2w_a49rf0_file,
                         t2w_a49rf0_json_dict) = find_image_and_json(
                            layout,
                            subject=subject,
                            session=session,
                            acquisition="t2Ssfp3A49RF0",
                            run=run,
                            part="mag",
                            suffix="T2w",
                            **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        (t2w_a49rf180_file,
                         t2w_a49rf180_json_dict) = find_image_and_json(
                            layout,
                            subject=subject,
                            session=session,
                            acquisition="t2Ssfp3A49RF180",
                            run=run,
                            suffix="T2w",
                            part="mag",
                            **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        (input_dict["b1_map_file"],
                         input_dict["b1_map_json_dict"]) = find_image_and_json(
                            layout,
                            subject=subject,
                            session=session,
                            acquisition="B1",
                            run=run,
                            suffix="B1Map",
                            **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        (input_dict["b1_anat_ref_file"],
                         input_dict[
                             "b1_anat_ref_json_dict"]) = find_image_and_json(
                            layout,
                            subject=subject,
                            session=session,
                            acquisition="B1Ref",
                            run=run,
                            suffix="magnitude",
                            **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    except ValueError as e:
                        # Print the error message
                        print(f"A ValueError occurred: {e}")
                        continue

                    # collect T1w files and json metadata
                    input_dict["t1w_files"] = [t1w_a2_file, t1w_a13_file]
                    input_dict["t1w_json_dicts"] = [t1w_a2_json_dict,
                                                    t1w_a13_json_dict]

                    # store repetition time in t1w metadata
                    for t1w_json_dict in input_dict["t1w_json_dicts"]:
                        t1w_json_dict["RepetitionTimeExcitation"] = 0.0062

                    # collect T2w files
                    input_dict["t2w_files"] = [t2w_a12rf180_file,
                                               t2w_a49rf0_file,
                                               t2w_a49rf180_file]

                    # add missing entries in T2w metadata (file-specific)
                    t2w_a12rf180_json_dict["RfPhaseIncrement"] = 180
                    t2w_a49rf0_json_dict["RfPhaseIncrement"] = 0
                    t2w_a49rf180_json_dict["RfPhaseIncrement"] = 180

                    # collect T2w json metadata
                    input_dict["t2w_json_dicts"] = [t2w_a12rf180_json_dict,
                                                    t2w_a49rf0_json_dict,
                                                    t2w_a49rf180_json_dict]

                    # add missing entries in T2w metadata (common entries)
                    for t2w_json_dict in input_dict["t2w_json_dicts"]:
                        t2w_json_dict["RepetitionTimeExcitation"] = 0.006
                        t2w_json_dict["RfPulseDuration"] = 0.0003

                    # compute B1 map normalization factor (norm to. 1)
                    input_dict["b1_normalization_factor"] = 1.0 / 100

                    # add to inputs
                    inputs.append(input_dict)

    # set up workflow
    wf = Workflow(name="prepare_ssfp_uke_ninflash_dataset")
    wf.base_dir = args.temp_dir

    # create input node using entries in input_dict and
    # use independent subject-session-run combinations as iterables
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    # normalize B1 map (to 1)
    normalize_b1 = pe.Node(
        fsl.BinaryMaths(operation='mul'),
        name="normalize_b1")
    wf.connect(input_node, "b1_map_file",
               normalize_b1, "in_file")
    wf.connect(input_node, "b1_normalization_factor",
               normalize_b1, "operand_value")

    # write B1 map
    b1_map_file_writer = pe.Node(BidsOutputWriter(),
                                 name="b1_map_file_writer")
    b1_map_file_writer.inputs.output_dir = args.output_dir
    b1_map_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_B1_MAP
    wf.connect(normalize_b1, "out_file",
               b1_map_file_writer, "in_file")
    wf.connect(input_node, "b1_map_json_dict",
               b1_map_file_writer, "json_dict")
    wf.connect(input_node, "b1_map_file",
               b1_map_file_writer, "template_file")

    # write B1 anatomical reference magnitude image
    b1_anat_ref_file_writer = pe.Node(BidsOutputWriter(),
                                      name="b1_anat_ref_file_writer")
    b1_anat_ref_file_writer.inputs.output_dir = args.output_dir
    b1_anat_ref_file_writer.inputs.entity_overrides = \
        STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "in_file")
    wf.connect(input_node, "b1_anat_ref_json_dict",
               b1_anat_ref_file_writer, "json_dict")
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "template_file")

    # write T1w images
    t1w_file_writer = pe.MapNode(BidsOutputWriter(),
                                 iterfield=['in_file', 'template_file',
                                            'json_dict'],
                                 name="t1w_file_writer")
    t1w_file_writer.inputs.output_dir = args.output_dir
    t1w_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_T1W
    wf.connect(input_node, "t1w_files",
               t1w_file_writer, "in_file")
    wf.connect(input_node, "t1w_files",
               t1w_file_writer, "template_file")
    wf.connect(input_node, "t1w_json_dicts",
               t1w_file_writer, "json_dict")

    # write T2w images
    t2w_file_writer = pe.MapNode(BidsOutputWriter(),
                                 iterfield=['in_file', 'template_file',
                                            'json_dict'],
                                 name="t2w_file_writer")
    t2w_file_writer.inputs.output_dir = args.output_dir
    t2w_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_T2W
    wf.connect(input_node, "t2w_files",
               t2w_file_writer, "in_file")
    wf.connect(input_node, "t2w_files",
               t2w_file_writer, "template_file")
    wf.connect(input_node, "t2w_json_dicts",
               t2w_file_writer, "json_dict")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
