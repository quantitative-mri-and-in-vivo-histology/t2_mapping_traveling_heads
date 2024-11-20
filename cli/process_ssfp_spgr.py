import argparse
import os
import json
import multiprocessing

import nipype.pipeline.engine as pe
from nipype import Node
from nipype.interfaces.utility import IdentityInterface

from workflows.parameter_estimation import \
    estimate_relaxation_ssfp
from workflows.processing import preprocess_ssfp_spgr, create_brain_mask
from nodes.io import ExplicitDataSink
from utils.io import get_nifti_fileparts
from utils.processing import create_qi_jsr_config

def main():
    parser = argparse.ArgumentParser(
        description="Process T2w (SSFP) and T1w (SPGR) images to estimate relaxation parameters.")

    # Lists of files (T1w and T2w images)
    parser.add_argument(
        '--t1w_files',
        nargs='+',
        required=True,
        help="List of T1-weighted (SPGR) image files. First image will be used " \
             "as reference for motion correction. There must be a JSON metadata " \
             "file with the same name for each T1w image. JSON metadata must " \
             "include tags: FlipAngle, RepetitionTimeExcitation, RfPulseDuration, RfPhaseIncrement."
    )
    parser.add_argument(
        '--t2w_files',
        nargs='+',
        required=True,
        help="List of T2-weighted (SSFP) image files. First image will be used " \
             "as reference for motion correction. There must be a JSON metadata " \
             "file with the same name for each T1w image. JSON metadata must " \
             "include tags: FlipAngle, RepetitionTimeExcitation, EchoTime."
    )
    parser.add_argument(
        '--b1_map_file',
        required=True,
        help="B1 map file. B1 map should be normalized to 1, with 1 indicating perfectly homogeneous field."
    )
    parser.add_argument(
        '--b1_anat_ref_file',
        required=True,
        help="B1 anatomical reference file."
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help="Directory to store the outputs."
    )
    parser.add_argument(
        '--temp_dir',
        required=False,
        default=os.getcwd(),
        help="Directory to store temporary outputs."
    )
    parser.add_argument(
        '--preprocess_only',
        action='store_true',
        default=False,
        help="If set, only preprocessing will be performed."
    )
    parser.add_argument('--n_procs', type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of processors to use (default: all available cores).')

    # Parse arguments
    args = parser.parse_args()

    # Define the reusable run settings in a dictionary
    run_settings = dict(plugin='MultiProc',
                        plugin_args={'n_procs': args.n_procs})

    # read json files for T1w images
    t1w_json_dicts = []
    for t1w_file in args.t1w_files:
        dirpath, basename, ext = get_nifti_fileparts(t1w_file)
    json_file = os.path.join(dirpath, f"{basename}.json")

    # Check if the corresponding JSON file exists
    if not os.path.exists(json_file):
        raise FileNotFoundError(
            f"Error: Expected JSON file '{json_file}' corresponding to NIfTI file '{t1w_file}' was not found.")

    # Load the JSON file into a dictionary
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

    t1w_json_dicts.append(json_dict)


    # read json files for T2w images
    t2w_json_dicts = []
    for t2w_file in args.t2w_files:
        dirpath, basename, ext = get_nifti_fileparts(t2w_file)
        json_file = os.path.join(dirpath, f"{basename}.json")

        # Check if the corresponding JSON file exists
        if not os.path.exists(json_file):
            raise FileNotFoundError(
                f"Error: Expected JSON file '{json_file}' corresponding to NIfTI file '{t2w_file}' was not found.")

        # Load the JSON file into a dictionary
        with open(json_file, 'r') as f:
            json_dict = json.load(f)

        t2w_json_dicts.append(json_dict)

    # create qi jsr fitting configuration file
    qi_jsr_config_dict = create_qi_jsr_config(t1w_json_dicts, t2w_json_dicts)

    input_dict = dict(
        t1w_files=args.t1w_files,
        t2w_files=args.t2w_files,
        t1w_reg_target_file=args.t1w_files[0],
        b1_map_file=args.b1_map_file,
        b1_anat_ref_file=args.b1_anat_ref_file,
        qi_jsr_config_dict=qi_jsr_config_dict)

    # set up workflow
    wf = pe.Workflow(name="process_ssfp_spgr")
    wf.base_dir = args.temp_dir

    # create input node using entries in input_dict
    input_node = Node(
        IdentityInterface(fields=list(input_dict.keys())),
        name='input_node')
    for key, value in input_dict.items():
        setattr(input_node.inputs, key, value)

    # preprocess images
    preprocess_ssfp_wf = preprocess_ssfp_spgr()
    wf.connect([(input_node, preprocess_ssfp_wf, [
        ('b1_map_file', 'input_node.b1_map_file'),
        ('b1_anat_ref_file', 'input_node.b1_anat_ref_file'),
        ('t1w_files', 'input_node.t1w_files'),
        ('t2w_files', 'input_node.t2w_files'),
        ('t1w_reg_target_file', 'input_node.reg_target_file')
    ])])

    # create brain mask
    create_brain_mask_wf = create_brain_mask()
    wf.connect(preprocess_ssfp_wf, "output_node.reg_target_file",
               create_brain_mask_wf, "input_node.in_file")

    # write registered b1 map
    b1_map_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="B1_map.nii.gz"),
        name="b1_map_writer")
    wf.connect(preprocess_ssfp_wf, "output_node.b1_map_file",
               b1_map_writer, "in_file")

    # write registered b1 anatomical reference
    b1_anat_ref_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="B1_anat_ref.nii.gz"),
        name="b1_anat_ref_writer")
    wf.connect(preprocess_ssfp_wf, "output_node.b1_anat_ref_file",
               b1_anat_ref_writer, "in_file")

    # write registered b1 map
    t1w_ref_image_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="T1w_ref.nii.gz"),
        name="t1w_ref_image_writer")
    wf.connect(preprocess_ssfp_wf, "output_node.reg_target_file",
               t1w_ref_image_writer, "in_file")

    # write registered b1 map
    brain_mask_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="brain_mask.nii.gz"),
        name="brain_mask_writer")
    wf.connect(create_brain_mask_wf, "output_node.out_file",
               brain_mask_writer, "in_file")

    # create T1w output filenames
    t1w_filenames = []
    for t1w_file in args.t1w_files:
        dirpath, basename, ext = get_nifti_fileparts(t1w_file)
        out_file = os.path.join(dirpath, f"{basename}_preprocessed{ext}")
        t1w_filenames.append(out_file)

    # write T1w output files
    t1w_file_writer = pe.MapNode(
        ExplicitDataSink(output_dir=args.output_dir),
        name="t1w_file_writer", iterfield=["in_file", "filename"])
    t1w_file_writer.inputs.filename = t1w_filenames
    wf.connect(preprocess_ssfp_wf, "output_node.t1w_files",
               t1w_file_writer, "in_file")

    # create T2w output filenames
    t2w_filenames = []
    for t2w_file in args.t2w_files:
        dirpath, basename, ext = get_nifti_fileparts(t2w_file)
        out_file = os.path.join(dirpath, f"{basename}_preprocessed{ext}")
        t2w_filenames.append(out_file)

    # write T2w output files
    t2w_file_writer = pe.MapNode(
        ExplicitDataSink(output_dir=args.output_dir),
        name="t2w_file_writer", iterfield=["in_file", "filename"])
    t2w_file_writer.inputs.filename = t2w_filenames
    wf.connect(preprocess_ssfp_wf, "output_node.t2w_files",
               t2w_file_writer, "in_file")

    if not args.preprocess_only:
        # estimate relaxation parameters
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

        # write R1 map
        r1_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="R1_map.nii.gz"),
            name="r1_map_writer")
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.r1_map_file",
                   r1_map_writer, "in_file")

        # write R2 map
        r2_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="R2_map.nii.gz"),
            name="r2_map_writer")
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.r2_map_file",
                   r2_map_writer, "in_file")

        # write T1 map
        t1_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="T1_map.nii.gz"),
            name="t1_map_writer")
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.t1_map_file",
                   t1_map_writer, "in_file")

        # write T2 map
        t2_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="T2_map.nii.gz"),
            name="t2_map_writer")
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.t2_map_file",
                   t2_map_writer, "in_file")

    wf.run(**run_settings)

if __name__ == "__main__":
    main()
