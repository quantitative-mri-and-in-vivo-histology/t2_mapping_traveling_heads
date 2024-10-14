import argparse
import json
import os
import multiprocessing
from nipype import Workflow
from nipype.interfaces.utility import IdentityInterface
import nipype.pipeline.engine as pe
from nipype import Node
from bids.layout import BIDSLayout
import nipype.interfaces.fsl as fsl
from nipype_utils import BidsOutputWriter
from utils.io import write_minimal_bids_dataset_description, find_image_and_json


def main():
    parser = argparse.ArgumentParser(
        description="Process a dataset with optional steps.")
    parser.add_argument('-i', '--bids_root', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-d', '--derivatives', nargs='+', required=True,
                        help='One or more derivatives directories to use.')
    parser.add_argument('-o', '--output_derivative_dir', required=True,
                        help='Path to the output derivatives folder.')
    parser.add_argument('--base_dir', default=os.getcwd(),
                        help='Base directory for processing (default: current working directory).')
    parser.add_argument('--n_procs', type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of processors to use (default: all available cores).')
    args = parser.parse_args()

    # Ensure `derivatives` is a list with one or more entries
    if not args.derivatives or len(args.derivatives) == 0:
        raise ValueError(
            "At least one derivatives directory must be specified with the -d option.")

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
        sessions = [ses for ses in sessions if not ses.startswith("failed")]
        if sessions:  # Only add subjects with existing sessions
            for session in sessions:

                test_image_entities = dict(subject=subject,
                                           session=session,
                                           acquisition="t2Ssfp3A12RF180",
                                           part="mag",
                                           suffix='T2w',
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

                    try:
                        (t1w_a2_file, t1w_a2_json_dict) = find_image_and_json(
                            layout, subject=subject,
                            session=session,
                            suffix="T1w",
                            acquisition="t2Ssfp3A2",
                            part="mag",
                            extension="nii.gz",
                            run=run)

                        t1w_a13_files = layout.get(subject=subject,
                                                   session=session,
                                                   suffix="T1w",
                                                   acquisition="t2Ssfp3A13",
                                                   part="mag",
                                                   extension="nii.gz",
                                                   run=run)

                        t1w_a12_files = layout.get(subject=subject,
                                                   session=session,
                                                   suffix="T1w",
                                                   acquisition="t2Ssfp3A12",
                                                   part="mag",
                                                   extension="nii.gz",
                                                   run=run)

                        if len(t1w_a13_files) == 0 and len(t1w_a12_files) == 1:
                            (t1w_a13_file,
                             t1w_a13_json_dict) = find_image_and_json(
                                layout, subject=subject,
                                session=session,
                                suffix="T1w",
                                acquisition="t2Ssfp3A12",
                                part="mag",
                                extension="nii.gz",
                                run=run)
                        else:
                            (t1w_a13_file,
                             t1w_a13_json_dict) = find_image_and_json(
                                layout, subject=subject,
                                session=session,
                                suffix="T1w",
                                acquisition="t2Ssfp3A13",
                                part="mag",
                                extension="nii.gz",
                                run=run)


                        (t2w_a12rf180_file,
                         t2w_a12rf180_json_dict) = find_image_and_json(
                            layout, subject=subject,
                            session=session,
                            suffix="T2w",
                            acquisition="t2Ssfp3A12RF180",
                            part="mag",
                            extension="nii.gz",
                            run=run)

                        (t2w_a49rf0_file,
                         t2w_a49rf0_json_dict) = find_image_and_json(
                            layout, subject=subject,
                            session=session,
                            suffix="T2w",
                            acquisition="t2Ssfp3A49RF0",
                            part="mag",
                            extension="nii.gz",
                            run=run)

                        (t2w_a49rf180_file,
                         t2w_a49rf180_json_dict) = find_image_and_json(
                            layout, subject=subject,
                            session=session,
                            suffix="T2w",
                            acquisition="t2Ssfp3A49RF180",
                            part="mag",
                            extension="nii.gz",
                            run=run)

                        (b1_map_file,
                         b1_map_json_dict) = find_image_and_json(
                            layout, subject=subject,
                            session=session,
                            suffix="B1Map",
                            extension="nii.gz",
                            run=run)

                        (b1_anat_ref_file,
                         b1_anat_ref_json_dict) = find_image_and_json(
                            layout, subject=subject,
                            session=session,
                            acquisition="B1Ref",
                            suffix="magnitude",
                            extension="nii.gz",
                            run=run)

                    except ValueError as e:
                        # Print the error message
                        print(f"A ValueError occurred: {e}")
                        continue

                    t1w_files = [t1w_a2_file, t1w_a13_file]
                    t1w_json_dicts = [t1w_a2_json_dict, t1w_a13_json_dict]

                    for t1w_json_dict in t1w_json_dicts:
                        t1w_json_dict["RepetitionTimeExcitation"] = 0.0062

                    t2w_files = [t2w_a12rf180_file, t2w_a49rf0_file,
                                 t2w_a49rf180_file]

                    t2w_a12rf180_json_dict["RfPhaseIncrement"] = 180
                    t2w_a49rf0_json_dict["RfPhaseIncrement"] = 0
                    t2w_a49rf180_json_dict["RfPhaseIncrement"] = 180

                    t2w_json_dicts = [t2w_a12rf180_json_dict,
                                      t2w_a49rf0_json_dict,
                                      t2w_a49rf180_json_dict]

                    for t2w_json_dict in t2w_json_dicts:
                        t2w_json_dict["RepetitionTimeExcitation"] = 0.006
                        t2w_json_dict["RfPulseDuration"] = 0.0003

                    b1_normalization_factor = 1.0 / 100

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       t1w_files=t1w_files,
                                       t1w_json_dicts=t1w_json_dicts,
                                       t2w_files=t2w_files,
                                       t2w_json_dicts=t2w_json_dicts,
                                       b1_siemens_map_file=b1_map_file,
                                       b1_siemens_json_dict=b1_map_json_dict,
                                       b1_anat_ref_file=b1_anat_ref_file,
                                       b1_anat_json_dict=b1_anat_json_dict,
                                       b1_normalization_factor=b1_normalization_factor))

    print(inputs)

    # set up bids input node
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    wf = Workflow(name="prepare_kings_dataset")
    wf.base_dir = args.base_dir

    output_node = Node(IdentityInterface(fields=[
        "t2w_files",
        "t1w_files",
        "b1_relative_map_file"
    ]), name='output_node')

    # normalize B1 map
    normalize_b1 = pe.Node(
        fsl.BinaryMaths(operation='mul'),
        name="normalize_b1")
    wf.connect(input_node, "b1_siemens_map_file",
               normalize_b1, "in_file")
    wf.connect(input_node, "b1_normalization_factor",
               normalize_b1, "operand_value")

    # rescale T1w images
    t1w_scaling_factor = 1.0
    rescale_t1w = pe.MapNode(fsl.ImageMaths(
        op_string='-mul {}'.format(t1w_scaling_factor)),
        iterfield=['in_file'], name="rescale_t1w")
    wf.connect(input_node, "t1w_files", rescale_t1w, "in_file")

    t2w_scaling_factor = 1.0
    rescale_t2w = pe.MapNode(fsl.ImageMaths(
        op_string='-mul {}'.format(t2w_scaling_factor)),
        iterfield=['in_file'], name="rescale_t2w")
    wf.connect(input_node, "t2w_files", rescale_t2w, "in_file")

    wf.connect(rescale_t1w, "out_file", output_node, "t1w_files")
    wf.connect(rescale_t2w, "out_file", output_node, "t2w_files")
    wf.connect(normalize_b1, "out_file", output_node, "b1_relative_map_file")

    out_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                  'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                  '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'

    b1_map_file_writer = pe.Node(BidsOutputWriter(),
                                 name="b1_map_file_formatter")
    b1_map_file_writer.inputs.output_dir = args.output_derivative_dir
    b1_map_file_writer.inputs.pattern = out_pattern
    b1_map_file_writer.inputs.entity_overrides = dict(acquisition="B1",
                                                      suffix="B1Map")
    wf.connect(normalize_b1, "out_file",
               b1_map_file_writer, "in_file")
    wf.connect(input_node, "b1_siemens_json_dict",
               b1_map_file_writer, "json_dict")
    wf.connect(input_node, "b1_siemens_map_file",
               b1_map_file_writer, "template_file")

    b1_anat_ref_file_writer = pe.Node(BidsOutputWriter(),
                                      name="b1_anat_ref_file_writer")
    b1_anat_ref_file_writer.inputs.output_dir = args.output_derivative_dir
    b1_anat_ref_file_writer.inputs.pattern = out_pattern
    b1_anat_ref_file_writer.inputs.entity_overrides = dict(acquisition="B1Ref",
                                                           suffix="magnitude")
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "in_file")
    wf.connect(input_node, "b1_anat_ref_json_dict",
               b1_anat_ref_file_writer, "json_dict")
    wf.connect(input_node, "b1_anat_ref_file",
               b1_anat_ref_file_writer, "template_file")

    t1w_file_writer = pe.MapNode(BidsOutputWriter(),
                                 iterfield=['in_file', 'template_file',
                                            'json_dict'],
                                 name="t1w_file_writer")
    t1w_file_writer.inputs.output_dir = args.output_derivative_dir
    t1w_file_writer.inputs.pattern = out_pattern
    t1w_file_writer.inputs.entity_overrides = dict(part=None)
    wf.connect(rescale_t1w, "out_file",
               t1w_file_writer, "in_file")
    wf.connect(input_node, "t1w_files",
               t1w_file_writer, "template_file")
    wf.connect(input_node, "t1w_json_dicts",
               t1w_file_writer, "json_dict")

    t2w_file_writer = pe.MapNode(BidsOutputWriter(),
                                 iterfield=['in_file', 'template_file',
                                            'json_dict'],
                                 name="t2w_file_writer")
    t2w_file_writer.inputs.output_dir = args.output_derivative_dir
    t2w_file_writer.inputs.pattern = out_pattern
    t2w_file_writer.inputs.entity_overrides = dict(part=None)
    wf.connect(rescale_t2w, "out_file",
               t2w_file_writer, "in_file")
    wf.connect(input_node, "t2w_files",
               t2w_file_writer, "template_file")
    wf.connect(input_node, "t2w_json_dicts",
               t2w_file_writer, "json_dict")

    # wf.run()
    wf.run(**run_settings)


if __name__ == "__main__":
    main()
