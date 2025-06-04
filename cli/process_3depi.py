import argparse
import json
import multiprocessing
import os

import nibabel as nib
import nipype.pipeline.engine as pe
from nipype import Node
from nipype.interfaces import fsl
from nipype.interfaces.utility import IdentityInterface

from nodes.io import ExplicitDataSink
from utils.io import get_nifti_fileparts
from workflows.parameter_estimation import estimate_relaxation_3d_epi
from workflows.processing import preprocess_3depi, create_brain_mask, \
    create_default_ants_rigid_registration_node


def main():
    parser = argparse.ArgumentParser(
        description="Process 3D-EPI images to estimate relaxation parameters.")
    parser.add_argument(
        '--t2w_mag',
        type=str,
        required=True,
        help="T2-weighted magnitude image."
    )
    parser.add_argument(
        '--t2w_phase',
        type=str,
        required=True,
        help="T2-weighted phase image in radian."
    )
    parser.add_argument(
        '--b1_map',
        type=str,
        required=True,
        help="B1 map normalized to 1, "
             "with 1 indicating perfectly homogeneous field."
    )
    parser.add_argument(
        '--b1_anat_ref',
        type=str,
        required=True,
        help="B1 anatomical reference. Should be a magnitude image aligned "
             "with B1 map."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory to store the outputs."
    )
    parser.add_argument(
        '--echo_time',
        type=float,
        required=False,
        default=None,
        help="Echo time in seconds. If not set, it will be read from the "
             "'EchoTime' field in the JSON file associated with the file "
             "provided in --t2w_mag."
    )
    parser.add_argument(
        '--repetition_time',
        type=float,
        required=False,
        default=None,
        help="Repetition time in seconds. If not set, it will be read from the "
             "'RepetitionTimeExcitation' field in the JSON file associated with the file "
             "provided in --t2w_mag."
    )
    parser.add_argument(
        '--flip_angle',
        type=float,
        required=False,
        default=None,
        help="Flip angle in degrees. If not set, it will be read from the "
             "'FlipAngle' field in the JSON file associated with the file "
             "provided in --t2w_mag."
    )
    parser.add_argument(
        '--rf_phase_increments',
        type=float,
        required=False,
        default=None,
        nargs='+',
        help="RF phase increments in degrees. If not set, it will be read from "
             "the 'RfPhaseIncrement' field in the JSON file associated with "
             "the file provided in --t2w_mag. To set values via command line, "
             "provide only positive RF phase increments, e.g., "
             "'--rf_phase_increments rf_phase_inc1 rf_phase_inc2 ... rf_phase_incN'. "
             "T2w magnitude image and files are expected to have "
             "2*len(rf_phase_increments) along the fourth dimension with "
             "elements ordered as: "
             "[+rf_phase_inc1, -rf_phase_inc1, +rf_phase_inc2, -rf_phase_inc2, ... "
             "+rf_phase_incN, -rf_phase_incN]"
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

    exclusive_group_masking = parser.add_mutually_exclusive_group()
    exclusive_group_masking.add_argument(
        '--t1w',
        type=str,
        help="T1w image for brain mask extraction (mutually exclusive with --mask). "
             "T1w image will first be registered to the T2w magnitude image. "
             "If not specified, no mask will be used for fitting."
    )

    exclusive_group_masking.add_argument(
        '--mask',
        type=str,
        help="Binary brain mask aligned to T2w magnitude image (mutually exclusive with --t1w). "
             "If set, it will be used directly for masking without T1w-based extraction."
    )

    # Parse arguments
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define the reusable run settings in a dictionary
    run_settings = dict(plugin='MultiProc',
                        plugin_args={'n_procs': args.n_procs})

    # read JSON metadata for T2w file if existent
    dirpath, basename, ext = get_nifti_fileparts(args.t2w_mag)
    t2w_mag_json_file = os.path.join(dirpath, f"{basename}.json")
    t2w_mag_json_dict = dict()
    if os.path.exists(t2w_mag_json_file):
        with open(t2w_mag_json_file, 'r') as f:
            t2w_mag_json_dict = json.load(f)

    # extract echo time from arguments or JSON metadata
    echo_time = args.echo_time
    if echo_time is None:
        print(f"--echo_time not specified. Trying to infer from metadata.")
        echo_time = t2w_mag_json_dict.get("EchoTime", None)
        if echo_time is not None:
            print(
                f"Found 'EchoTime' in JSON metadata. Value: {echo_time} seconds.")
        else:
            raise ValueError(
                f"Expected JSON metadata file '{t2w_mag_json_file}' with 'EchoTime' set.")

    # extract repetition time from arguments or JSON metadata
    repetition_time = args.repetition_time
    if repetition_time is None:
        print(
            f"--repetition_time not specified. Trying to infer from metadata.")
        repetition_time = t2w_mag_json_dict.get("RepetitionTimeExcitation",
                                                None)
        if repetition_time is not None:
            print(
                f"RepetitionTimeExcitation' in JSON metadata. Value: {repetition_time} seconds.")
        else:
            raise ValueError(
                f"Expected JSON metadata file '{t2w_mag_json_file}' with 'RepetitionTimeExcitation' set.")

    # extract flip angle from arguments or JSON metadata
    flip_angle = args.flip_angle
    if flip_angle is None:
        print(f"--flip_angle not specified. Trying to infer from metadata.")
        flip_angle = t2w_mag_json_dict.get("FlipAngle", None)
        if flip_angle is not None:
            print(
                f"Found 'FlipAngle' in JSON metadata. Value: {flip_angle} degrees.")
        else:
            raise ValueError(
                f"Expected JSON metadata file '{t2w_mag_json_file}' with 'FlipAngle' set.")

    # extract flip angle from arguments or JSON metadata
    rf_phase_increments = args.rf_phase_increments
    if rf_phase_increments is None:
        print(
            f"--rf_phase_increments not specified. Trying to infer from metadata.")
        rf_phase_increments = t2w_mag_json_dict.get("RfPhaseIncrement", None)
        if rf_phase_increments is not None:
            print(
                f"Found 'RfPhaseIncrement' in JSON metadata. Value: {rf_phase_increments} degrees.")
        else:
            raise ValueError(
                f"Expected JSON metadata file '{t2w_mag_json_file}' with 'RfPhaseIncrement' set.")

    if args.t1w is not None:
        print(
            f"--t1w specified. Will use {args.t1w} for brain mask extraction.")
    else:
        print(
            f"--t1w not specified for brain mask extraction. Will use no mask for fitting.")

    t2w_mag_nib = nib.load(args.t2w_mag)
    t2w_mag_image = t2w_mag_nib.get_fdata()

    if t2w_mag_image.shape[3] != 2 * len(rf_phase_increments):
        raise ValueError(
            f"Expected fourth dimension of T2w image to have 2 times the number "
            f"elements as in rf_phase_increments (call script with '--help' for "
            f"expected format). T2w shape: {t2w_mag_image.shape}, "
            f"rf_phase_increments: {len(args.rf_phase_increments)}.")

    input_dict = dict(
        t2w_mag_file=args.t2w_mag,
        t2w_phase_file=args.t2w_phase,
        b1_map_file=args.b1_map,
        b1_anat_ref_file=args.b1_anat_ref,
        echo_time=echo_time,
        repetition_time=repetition_time,
        flip_angle=flip_angle,
        rf_phase_increments=rf_phase_increments
    )

    use_t1w_brain_masking = args.t1w is not None
    use_explicit_mask = args.mask is not None

    if use_t1w_brain_masking:
        input_dict["t1w_file"] = args.t1w
    if use_explicit_mask:
        input_dict["mask_file"] = args.mask

    # set up workflow
    wf = pe.Workflow(name="process_3depi")
    wf.base_dir = os.path.join(args.output_dir, "temp")

    # create input node using entries in input_dict
    input_node = Node(
        IdentityInterface(fields=list(input_dict.keys())),
        name='input_node')
    for key, value in input_dict.items():
        setattr(input_node.inputs, key, value)

    # preprocess images
    preprocess_3depi_wf = preprocess_3depi()
    wf.connect([(input_node, preprocess_3depi_wf, [
        ('b1_map_file', 'input_node.b1_map_file'),
        ('b1_anat_ref_file', 'input_node.b1_anat_ref_file'),
        ('t2w_mag_file', 'input_node.magnitude_file'),
        ('t2w_phase_file', 'input_node.phase_file'),
    ])])

    # write registered b1 map
    b1_map_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="B1_map.nii.gz"),
        name="b1_map_writer")
    wf.connect(preprocess_3depi_wf, "output_node.b1_map_file",
               b1_map_writer, "in_file")

    # write registered b1 anatomical reference
    b1_anat_ref_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="B1_anat_ref.nii.gz"),
        name="b1_anat_ref_writer")
    wf.connect(preprocess_3depi_wf, "output_node.b1_anat_ref_file",
               b1_anat_ref_writer, "in_file")

    # write registered b1 map
    t2w_mag_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="T2w_mag_preprocessed.nii.gz"),
        name="t1w_ref_image_writer")
    wf.connect(preprocess_3depi_wf, "output_node.magnitude_file",
               t2w_mag_writer, "in_file")

    # write registered b1 map
    t2w_phase_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="T2w_phase_preprocessed.nii.gz"),
        name="t2w_phase_writer")
    wf.connect(preprocess_3depi_wf, "output_node.phase_file",
               t2w_phase_writer, "in_file")

    if use_t1w_brain_masking:
        # extract brain mask if T1w file is given
        create_brain_mask_wf = create_brain_mask()

        # register T1w image to T2w magnitde image
        mag_first_volume_extractor = Node(fsl.ExtractROI(),
                                          name="mag_first_volume_extractor")
        mag_first_volume_extractor.inputs.t_min = 0
        mag_first_volume_extractor.inputs.t_size = 1
        wf.connect(preprocess_3depi_wf, "output_node.magnitude_file",
                   mag_first_volume_extractor, "in_file")

        register_t1w_to_t2w = pe.Node(
            create_default_ants_rigid_registration_node(),
            name="register_t1w_to_t2w")
        wf.connect(mag_first_volume_extractor, "roi_file",
                   register_t1w_to_t2w, "fixed_image")
        wf.connect(input_node, "t1w_file",
                   register_t1w_to_t2w, "moving_image")

        # save registered T1w image
        t1w_registered_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="T1w_reg.nii.gz"),
            name="t1w_registered_writer")
        wf.connect(register_t1w_to_t2w, "warped_image",
                   t1w_registered_writer, "in_file")

        # extract brain mask from T1w image
        wf.connect(register_t1w_to_t2w, "warped_image",
                   create_brain_mask_wf, "input_node.in_file")

        # write brain mask
        brain_mask_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="brain_mask.nii.gz"),
            name="brain_mask_writer")
        wf.connect(create_brain_mask_wf, "output_node.out_file",
                   brain_mask_writer, "in_file")

    if not args.preprocess_only:

        # estimate relaxation parameter maps
        estimate_relaxation_3d_epi_wf = estimate_relaxation_3d_epi()
        wf.connect([(preprocess_3depi_wf, estimate_relaxation_3d_epi_wf, [
            ('output_node.b1_map_file', 'input_node.b1_map_file'),
            ('output_node.magnitude_file', 'input_node.t2w_magnitude_file'),
            ('output_node.phase_file', 'input_node.t2w_phase_file')
        ])])
        wf.connect([(input_node, estimate_relaxation_3d_epi_wf, [
            ('rf_phase_increments', 'input_node.rf_phase_increments'),
            ('repetition_time', 'input_node.repetition_time'),
            ('flip_angle', 'input_node.flip_angle')
        ])])

        if use_t1w_brain_masking:
            wf.connect(create_brain_mask_wf, 'output_node.out_file',
                       estimate_relaxation_3d_epi_wf,
                       'input_node.brain_mask_file')
        elif use_explicit_mask:
            wf.connect(input_node, 'mask_file',
                       estimate_relaxation_3d_epi_wf,
                       'input_node.brain_mask_file')

        # write R1 map
        r1_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="R1_map.nii.gz"),
            name="r1_map_writer")
        wf.connect(estimate_relaxation_3d_epi_wf, "output_node.r1_map_file",
                   r1_map_writer, "in_file")

        # write R2 map
        r2_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="R2_map.nii.gz"),
            name="r2_map_writer")
        wf.connect(estimate_relaxation_3d_epi_wf, "output_node.r2_map_file",
                   r2_map_writer, "in_file")

        # write T1 map
        t1_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="T1_map.nii.gz"),
            name="t1_map_writer")
        wf.connect(estimate_relaxation_3d_epi_wf, "output_node.t1_map_file",
                   t1_map_writer, "in_file")

        # write T2 map
        t2_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="T2_map.nii.gz"),
            name="t2_map_writer")
        wf.connect(estimate_relaxation_3d_epi_wf, "output_node.t2_map_file",
                   t2_map_writer, "in_file")

        # write AM map
        am_map_writer = pe.Node(
            ExplicitDataSink(output_dir=args.output_dir,
                             filename="AM_map.nii.gz"),
            name="am_map_writer")
        wf.connect(estimate_relaxation_3d_epi_wf, "output_node.am_map_file",
                   am_map_writer, "in_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
