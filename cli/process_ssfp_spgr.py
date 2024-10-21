import argparse
import os
import multiprocessing

import nibabel as nib
import nipype.pipeline.engine as pe
from nipype import Node
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import IdentityInterface

from workflows.parameter_estimation import \
    estimate_relaxation_ssfp_multi_file
from workflows.processing import preprocess_ssfp_spgr, create_brain_mask


def main():
    parser = argparse.ArgumentParser(
        description="Process MRI images and parameters.")

    # Lists of files (T1w and T2w images)
    parser.add_argument(
        '--t1w_files',
        nargs='+',
        required=True,
        help="List of T1-weighted image files. First image will be used as reference for motion correction"
    )
    parser.add_argument(
        '--t2w_files',
        nargs='+',
        required=True,
        help="List of T2-weighted image files."
    )
    parser.add_argument(
        '--b1_map_file',
        required=True,
        help="B1 map file."
    )
    parser.add_argument(
        '--b1_anat_ref_file',
        required=True,
        help="B1 anatomical reference file."
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help="Directory to store the output files."
    )
    parser.add_argument(
        '--t1w_flip_angles',
        nargs='+',
        type=float,
        required=True,
        help="List of flip angles corresponding to T1-weighted images. Must match the length of t1w_files."
    )
    parser.add_argument(
        '--t2w_flip_angles',
        nargs='+',
        type=float,
        required=True,
        help="List of flip angles corresponding to T2-weighted images. Must match the length of t2w_files."
    )
    parser.add_argument(
        '--t1w_repetition_time',
        type=float,
        required=True,
        help="Repetition time for T1-weighted images (scalar)."
    )
    parser.add_argument(
        '--t2w_repetition_time',
        type=float,
        required=True,
        help="Repetition time for T2-weighted images (scalar)."
    )
    parser.add_argument(
        '--t1w_echo_time',
        type=float,
        required=True,
        help="Echo time for T1-weighted images (scalar)."
    )
    parser.add_argument(
        '--t2w_echo_time',
        type=float,
        required=True,
        help="Echo time for T2-weighted images (scalar)."
    )
    parser.add_argument(
        '--t2w_rf_pulse_duration',
        type=float,
        required=True,
        help="RF pulse duration for T2-weighted images (scalar)."
    )
    parser.add_argument(
        '--t2w_rf_phase_increments',
        nargs='+',
        type=float,
        required=True,
        help="List of phase increments for T2-weighted images (arbitrary length). Must match the length of t2w_files."
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

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define the reusable run settings in a dictionary
    run_settings = dict(plugin='MultiProc',
                        plugin_args={'n_procs': args.n_procs})

    # Validate that flip angle lists match the number of T1w and T2w files
    if len(args.t1w_flip_angles) != len(args.t1w_files):
        parser.error(
            "The number of T1w flip angles must match the number of T1w files.")
    if len(args.t2w_flip_angles) != len(args.t2w_files):
        parser.error(
            "The number of T2w flip angles must match the number of T2w files.")
    if len(args.t2w_rf_phase_increments) != len(args.t2w_files):
        parser.error(
            "The number of T2w RF phase incrments must match the number of T2w files.")

    # set up config dict for qi jsr fitting
    spgr_dict = dict(TR=args.t1w_repetition_time,
                     TE=args.t1w_echo_time,
                     FA=args.t1w_flip_angles)
    ssfp_dict = dict(TR=args.t2w_repetition_time,
                     Trf=args.t2w_echo_time,
                     FA=args.t2w_flip_angles,
                     PhaseInc=args.t2w_rf_phase_increments)
    qi_jsr_config_dict = dict(SPGR=spgr_dict, SSFP=ssfp_dict)

    input_dict = dict(
        t1w_files=args.t1w_files,
        t2w_files=args.t2w_files,
        t1w_reg_target_file=args.t1w_files[0],
        b1_map_file=args.b1_map_file,
        b1_anat_ref_file=args.b1_anat_ref_file,
        qi_jsr_config_dict=qi_jsr_config_dict)

    # set up workflow
    wf = pe.Workflow(name="process_ssfp_dataset")
    wf.base_dir = args.output_dir

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

    # write preprocessed images
    # datasink = pe.MapNode(DataSink(base_directory=args.output_dir),
    #                    name="datasink", iterfield=["in_file"])
    datasink = pe.Node(DataSink(base_directory=args.output_dir),
                       name="datasink")
    # Use MapNode for handling lists of files (T1w and T2w)
    # t1w_file_datasink = pe.MapNode(DataSink(base_directory=args.output_dir),
    #                                iterfield=['in_file'],
    #                                name="t1w_file_datasink")
    # wf.connect(preprocess_ssfp_wf, "output_node.t1w_files",
    #            t1w_file_datasink, "in_file")

    wf.connect(preprocess_ssfp_wf, "output_node.b1_map_file",
               datasink, "@b1_map_file")
    wf.connect(preprocess_ssfp_wf, "output_node.b1_anat_ref_file",
               datasink, "@b1_anat_ref_file")
    wf.connect(preprocess_ssfp_wf, "output_node.t1w_files",
               datasink, "@t1w_files")
    wf.connect(preprocess_ssfp_wf, "output_node.t2w_files",
               datasink, "@t2w_files")
    wf.connect(preprocess_ssfp_wf, "output_node.reg_target_file",
               datasink, "@reg_target_file")
    wf.connect(create_brain_mask_wf, "output_node.out_file",
               datasink, "@brain_mask")

    if not args.preprocess_only:

        # estimate relaxation parameters
        estimate_relaxation_ssfp_wf = estimate_relaxation_ssfp_multi_file()
        wf.connect([(preprocess_ssfp_wf, estimate_relaxation_ssfp_wf, [
            ('output_node.b1_map_file', 'input_node.b1_map_file'),
            ('output_node.t1w_files', 'input_node.t1w_files'),
            ('output_node.t2w_files', 'input_node.t2w_files'),
        ])])
        wf.connect(input_node, 'qi_jsr_config_dict',
                   estimate_relaxation_ssfp_wf, 'input_node.qi_jsr_config_dict')
        wf.connect(create_brain_mask_wf, 'output_node.out_file',
                   estimate_relaxation_ssfp_wf, 'input_node.brain_mask_file')

        # write relaxation parameter maps
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.r1_map_file",
                   datasink, "@R1Map")
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.r2_map_file",
                   datasink, "@R2Map")
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.t1_map_file",
                   datasink, "@T1Map")
        wf.connect(estimate_relaxation_ssfp_wf, "output_node.t2_map_file",
                   datasink, "@T2Map")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
