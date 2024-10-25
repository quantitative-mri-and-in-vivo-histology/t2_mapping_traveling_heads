import argparse
import math
import os
import multiprocessing
from nipype import Workflow
from nipype.interfaces.utility import IdentityInterface
import nipype.pipeline.engine as pe
from nipype import Node
from bids.layout import BIDSLayout
import nipype.interfaces.fsl as fsl
from workflows.processing import correct_b1_with_b0
from nodes.io import BidsOutputWriter
from utils.io import write_minimal_bids_dataset_description, find_image_and_json
from utils.bids_config import (DEFAULT_NIFTI_READ_EXT_ENTITY,
                               STANDARDIZED_ENTITY_OVERRIDES_T1W,
                               STANDARDIZED_ENTITY_OVERRIDES_B0_PHASEDIFF_MAP,
                               STANDARDIZED_ENTITY_OVERRIDES_B0_ANAT_REF,
                               STANDARDIZED_ENTITY_OVERRIDES_B1_MAP,
                               STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF,
                               STANDARDIZED_ENTITY_OVERRIDES_T2W_MAG,
                               STANDARDIZED_ENTITY_OVERRIDES_T2W_PHASE)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SSFP dataset from King's College Hospital, London.")
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

    inputs = []
    subjects = layout.get_subjects()

    for subject in subjects:
        sessions = layout.get_sessions(subject=subject)
        sessions = [ses for ses in sessions if not ses.startswith("failed")]
        if sessions:  # Only add subjects with existing sessions
            for session in sessions:

                test_image_entities = dict(subject=subject,
                                           session=session,
                                           suffix="T2w",
                                           part="mag",
                                           extension="nii.gz")

                test_images = layout.get(**test_image_entities)
                if len(test_images) == 0:
                    continue

                valid_runs = layout.get(return_type='id',
                                        target='run',
                                        **test_image_entities)
                runs = valid_runs

                if len(runs) == 0:
                    runs = [None]

                for run in runs:
                    input_dict = dict(
                        subject=subject,
                        session=session,
                        run=run
                    )

                    (input_dict["t1w_file"],
                     input_dict["t1w_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="dzneep3df0a107fT1w",
                        mt="off",
                        echo=1,
                        part="mag",
                        suffix="MPM",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["t2w_mag_file"],
                     input_dict["t2w_mag_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="dzneep3df0a107f",
                        suffix="T2w",
                        part="mag",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["t2w_phase_file"],
                     input_dict["t2w_phase_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="dzneep3df0a107f",
                        suffix="T2w",
                        part="phase",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b1_map_file"],
                     input_dict["b1_map_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="B1",
                        suffix="B1Map",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b1_anat_ref_file"],
                     input_dict["b1_anat_ref_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="B1Ref",
                        suffix="magnitude",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b0_mag1_file"],
                     input_dict["b0_mag1_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="grefieldmap1acqrlshortAntoine",
                        suffix="magnitude1",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    (input_dict["b0_phasediff_file"],
                     input_dict[
                         "b0_phasediff_json_dict"]) = find_image_and_json(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        acquisition="grefieldmap1acqrlshortAntoine",
                        suffix="phasediff",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["b0_mag1_entity_overrides"] = dict(
                        run=run,
                        **STANDARDIZED_ENTITY_OVERRIDES_B0_ANAT_REF)

                    input_dict["b0_phasediff_entity_overrides"] = dict(
                        run=run,
                        **STANDARDIZED_ENTITY_OVERRIDES_B0_PHASEDIFF_MAP)

                    b0_te_delta = input_dict["b0_phasediff_json_dict"][
                                      "EchoTime2"] - \
                                  input_dict["b0_phasediff_json_dict"][
                                      "EchoTime1"]
                    input_dict["b0_phase_unwrap_factor"] = 1.0 / (
                            4096 * b0_te_delta * 2)

                    input_dict["fa_nominal_in_degrees"] = input_dict[
                        "t2w_mag_json_dict"]["FlipAngle"]

                    input_dict["b1_normalization_factor"] = 1.0 / 100
                    input_dict["rf_pulse_duration"] = 2.46e-3

                    # add metadata
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

    # set up bids input node
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    wf = Workflow(name="prepare_3depi_uke_dataset")
    wf.base_dir = args.temp_dir

    # normalize B1 map
    normalize_b1 = pe.Node(
        fsl.BinaryMaths(operation='mul'),
        name="normalize_b1")
    wf.connect(input_node, "b1_map_file",
               normalize_b1, "in_file")
    wf.connect(input_node, "b1_normalization_factor",
               normalize_b1, "operand_value")

    # convert B0 map to radian
    unwrap_phase_b0 = pe.Node(fsl.BinaryMaths(operation="mul"),
                              name="unwrap_phase_b0")
    wf.connect(input_node, "b0_phasediff_file",
               unwrap_phase_b0, "in_file")
    wf.connect(input_node, "b0_phase_unwrap_factor",
               unwrap_phase_b0, "operand_value")

    # B1 adjustment for T2w images
    correct_b1_with_b0_wf = correct_b1_with_b0()
    wf.connect(unwrap_phase_b0, "out_file",
               correct_b1_with_b0_wf, "input_node.b0_map_file")
    wf.connect(normalize_b1,
               "out_file",
               correct_b1_with_b0_wf, "input_node.b1_map_file")
    wf.connect(input_node, "b0_mag1_file",
               correct_b1_with_b0_wf,
               "input_node.b0_anat_ref_file")
    wf.connect(input_node,
               "b1_anat_ref_file",
               correct_b1_with_b0_wf,
               "input_node.b1_anat_ref_file")
    wf.connect(input_node, "fa_nominal_in_degrees",
               correct_b1_with_b0_wf,
               "input_node.fa_nominal_in_degrees")
    wf.connect(input_node, "rf_pulse_duration",
               correct_b1_with_b0_wf,
               "input_node.pulse_duration_in_seconds")

    # scale phase to radian
    scaling_factor = math.pi / 4096.0
    scale_phase_from_siemens_to_radian = pe.Node(
        fsl.ImageMaths(
            op_string='-mul {}'.format(scaling_factor)),
        name="scale_phase_from_siemens_to_rad")
    wf.connect(input_node, "t2w_phase_file",
               scale_phase_from_siemens_to_radian, "in_file")

    # extract first T1w volume
    t1w_first_volume_extractor = Node(fsl.ExtractROI(),
                                      name="mag_first_volume_extractor")
    t1w_first_volume_extractor.inputs.t_min = 0
    t1w_first_volume_extractor.inputs.t_size = 1
    wf.connect(input_node, "t1w_file",
               t1w_first_volume_extractor, "in_file")

    # write B1 map
    b1_map_file_writer = pe.Node(BidsOutputWriter(),
                                 name="b1_map_file_formatter")
    b1_map_file_writer.inputs.output_dir = args.output_dir
    b1_map_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_B1_MAP
    wf.connect(correct_b1_with_b0_wf, "output_node.out_file",
               b1_map_file_writer, "in_file")
    wf.connect(input_node, "b1_map_json_dict",
               b1_map_file_writer, "json_dict")
    wf.connect(input_node, "b1_map_file",
               b1_map_file_writer, "template_file")

    # write B1 anatomical reference image
    b1_anat_ref_file_writer = pe.Node(BidsOutputWriter(),
                                      name="b1_anat_ref_file_writer")
    b1_anat_ref_file_writer.inputs.output_dir = args.output_dir
    b1_anat_ref_file_writer.inputs.entity_overrides = STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "in_file")
    wf.connect(input_node, "b1_anat_ref_json_dict",
               b1_anat_ref_file_writer, "json_dict")
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "template_file")

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

    # write B0 anatomical reference image
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

    # write T2w magnitude images
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

    # write T2w phase images
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
    wf.connect(t1w_first_volume_extractor, "roi_file",
               t1w_file_writer, "in_file")
    wf.connect(input_node, "t1w_file",
               t1w_file_writer, "template_file")
    wf.connect(input_node, "t1w_json_dict",
               t1w_file_writer, "json_dict")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
