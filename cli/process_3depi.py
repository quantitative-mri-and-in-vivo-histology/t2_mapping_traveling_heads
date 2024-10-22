import argparse
import json
import multiprocessing
import os

import nibabel as nib
import nipype.pipeline.engine as pe
import numpy as np
from nipype import Node
from nipype.interfaces import fsl
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, \
    TraitedSpec, File
from nipype.interfaces.base import isdefined
from nipype.interfaces.utility import IdentityInterface

from nodes.io import ExplicitDataSink
from utils.io import get_nifti_fileparts
from workflows.processing import preprocess_3depi, create_brain_mask, \
    create_default_ants_rigid_registration_node


class CreateDummyMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="Input NIfTI image", mandatory=True)
    out_file = File(desc="Output mask file",
                    mandatory=False)


class CreateDummyMaskOutputSpec(TraitedSpec):
    out_file = File(desc="Output mask file")


class CreateDummyMask(BaseInterface):
    input_spec = CreateDummyMaskInputSpec
    output_spec = CreateDummyMaskOutputSpec

    def _run_interface(self, runtime):
        # Load the input NIfTI image
        img = nib.load(self.inputs.in_file)

        # Create a mask of ones with the same shape as the input image
        mask_data = np.ones(img.shape[0:2], dtype=np.uint8)

        # If out_file is not provided, save it to the current working directory
        if not isdefined(self.inputs.out_file):
            basename = os.path.basename(self.inputs.in_file)
            self.inputs.out_file = os.path.join(os.getcwd(),
                                                f"{basename.split('.')[0]}_dummy_mask.nii.gz")

        # Save the mask to a new NIfTI file
        mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
        nib.save(mask_img, self.inputs.out_file)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.inputs.out_file
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Process 3D-EPI images to estimate relaxation parameters.")
    parser.add_argument(
        '--t2w_mag',
        required=True,
        help="T2-weighted magnitude image."
    )
    parser.add_argument(
        '--t2w_phase',
        required=True,
        help="T2-weighted phase image in radian."
    )
    parser.add_argument(
        '--b1_map',
        required=True,
        help="B1 map normalized to 1,\n"
             "with 1 indicating perfectly homogeneous field."
    )
    parser.add_argument(
        '--b1_anat_ref',
        required=True,
        help="B1 anatomical reference file."
    )
    parser.add_argument(
        '--t1w',
        required=False,
        default=None,
        help="T1w image for brain mask extraction. Otherwise, T2w magnitude image "
             "will be used for brain mask extraction."
    )
    parser.add_argument(
        '--echo_time',
        required=False,
        default=None,
        help="Echo time in seconds. If not set, it will be determined from\n"
             "'EchoTime' in JSON metadata."
    )
    parser.add_argument(
        '--repetition_time',
        required=False,
        default=None,
        help="Repetition time in seconds. If not set, it will be determined from\n"
             "'RepetitionTimeExcitation' in JSON metadata."
    )
    parser.add_argument(
        '--flip_angle',
        required=False,
        default=None,
        help="Flip angle in degrees. If not set, it will be determined from\n"
             "'FlipAngle' in JSON metadata."
    )
    parser.add_argument(
        '--rf_phase_increments',
        required=False,
        default=None,
        nargs='+',
        help="RF phase increments. Provide only positive RF phase increments, e.g.,\n"
             "'--rf_phase_increments rf_phase_inc1 rf_phase_inc2 ... rf_phase_incN'."
             "T2w magnitude image and files are expected to have\n"
             "2*len(rf_phase_increments) along the fourth dimension with\n"
             "elements ordered as:\n"
             "[+rf_phase_inc1, -rf_phase_inc1, +rf_phase_inc2, -rf_phase_inc2,\n"
             "+rf_phase_incN, -rf_phase_incN]"
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help="Directory to store the outputs."
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
                f"Error: Expected JSON metadata file '{t2w_mag_json_file}' with 'EchoTime' set.")

    # extract repetition time from arguments or JSON metadata
    repetition_time = args.repetition_time
    if repetition_time is None:
        print(
            f"--repetition_time not specified. Trying to infer from metadata.")
        repetition_time = t2w_mag_json_dict.get("RepetitionTimeExcitation",
                                                None)
        if repetition_time is not None:
            print(
                f"Found 'RepetitionTimeExcitation' in JSON metadata. Value: {repetition_time} seconds.")
        else:
            raise ValueError(
                f"Error: Expected JSON metadata file '{t2w_mag_json_file}' with 'RepetitionTimeExcitation' set.")

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
                f"Error: Expected JSON metadata file '{t2w_mag_json_file}' with 'FlipAngle' set.")

    # extract flip angle from arguments or JSON metadata
    rf_phase_increments = args.rf_phase_increments
    if rf_phase_increments is None:
        print(f"--rf_phase_increments not specified. Trying to infer from metadata.")
        rf_phase_increments = t2w_mag_json_dict.get("RfPhaseIncrement", None)
        if rf_phase_increments is not None:
            print(
                f"Found 'RfPhaseIncrement' in JSON metadata. Value: {rf_phase_increments} degrees.")
        else:
            raise ValueError(
                f"Error: Expected JSON metadata file '{t2w_mag_json_file}' with 'RfPhaseIncrement' set.")

    if args.t1w is not None:
        print(
            f"--t1w specified. Will use {args.t1w} for brain mask extraction.")
    else:
        print(f"--t1w not specified. Will use {args.t2w_mag} instead.")

    t2w_mag_nib = nib.load(args.t2w_mag)
    t2w_mag_image = t2w_mag_nib.get_fdata()

    if t2w_mag_image.shape[3] != 2 * len(rf_phase_increments):
        raise ValueError(
            f"Error: fourth dimension of T2w image to have 2 times the number "
            f"elements as in rf_phase_increments. T2w shape: {t2w_mag_image.shape}, "
            f"rf_phase_increments: {len(args.rf_phase_increments)}.")

    input_dict = dict(
        t2w_mag_file=args.t2w_mag,
        t2w_phase_file=args.t2w_phase,
        b1_map_file=args.b1_map,
        b1_anat_ref_file=args.b1_anat_ref,
        echo_time=echo_time,
        repetition_time=repetition_time,
        flip_angle=flip_angle,
        rf_phase_increments=args.rf_phase_increments
    )

    if args.t1w is not None:
        input_dict["t1w_file"] = args.t1w

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

    # create dummy node that holds brain mask
    brain_mask_node = pe.Node(IdentityInterface(fields=[
        "brain_mask_file"
    ]), name="phase_wrap_b1_node")
    brain_mask_writer = pe.Node(
        ExplicitDataSink(output_dir=args.output_dir,
                         filename="brain_mask.nii.gz"),
        name="brain_mask_writer")

    if "t1w_file" in input_dict:
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
        wf.connect(create_brain_mask_wf, "output_node.out_file",
                   brain_mask_writer, "in_file")
    else:
        # create dummy brain mask (all ones)
        create_dummy_mask = pe.Node(CreateDummyMask(), name="create_dummy_mask")
        wf.connect(preprocess_3depi_wf, "output_node.magnitude_file",
                   create_dummy_mask, "in_file")

        # write brain mask
        wf.connect(create_dummy_mask, "out_file",
                   brain_mask_writer, "in_file")

    # if not args.preprocess_only:
    #     # estimate relaxation parameters
    #     estimate_relaxation_ssfp_wf = estimate_relaxation_ssfp_multi_file()
    #     wf.connect([(preprocess_ssfp_wf, estimate_relaxation_ssfp_wf, [
    #         ('output_node.b1_map_file', 'input_node.b1_map_file'),
    #         ('output_node.t1w_files', 'input_node.t1w_files'),
    #         ('output_node.t2w_files', 'input_node.t2w_files'),
    #     ])])
    #     wf.connect(input_node, 'qi_jsr_config_dict',
    #                estimate_relaxation_ssfp_wf, 'input_node.qi_jsr_config_dict')
    #     wf.connect(create_brain_mask_wf, 'output_node.out_file',
    #                estimate_relaxation_ssfp_wf, 'input_node.brain_mask_file')
    #
    #     # write R1 map
    #     r1_map_writer = pe.Node(
    #         ExplicitDataSink(output_dir=args.output_dir,
    #                          filename="R1_map.nii.gz"),
    #         name="r1_map_writer")
    #     wf.connect(estimate_relaxation_ssfp_wf, "output_node.r1_map_file",
    #                r1_map_writer, "in_file")
    #
    #     # write R2 map
    #     r2_map_writer = pe.Node(
    #         ExplicitDataSink(output_dir=args.output_dir,
    #                          filename="R2_map.nii.gz"),
    #         name="r2_map_writer")
    #     wf.connect(estimate_relaxation_ssfp_wf, "output_node.r2_map_file",
    #                r2_map_writer, "in_file")
    #
    #     # write T1 map
    #     t1_map_writer = pe.Node(
    #         ExplicitDataSink(output_dir=args.output_dir,
    #                          filename="T1_map.nii.gz"),
    #         name="t1_map_writer")
    #     wf.connect(estimate_relaxation_ssfp_wf, "output_node.t1_map_file",
    #                t1_map_writer, "in_file")
    #
    #     # write T2 map
    #     t2_map_writer = pe.Node(
    #         ExplicitDataSink(output_dir=args.output_dir,
    #                          filename="T2_map.nii.gz"),
    #         name="t2_map_writer")
    #     wf.connect(estimate_relaxation_ssfp_wf, "output_node.t2_map_file",
    #                t2_map_writer, "in_file")
    #
    wf.run(**run_settings)

    # # Validate that flip angle lists match the number of T1w and T2w files
    # if len(args.t1w_flip_angles) != len(args.t1w_files):
    #     parser.error(
    #         "The number of T1w flip angles must match the number of T1w files.")
    # if len(args.t2w_flip_angles) != len(args.t2w_files):
    #     parser.error(
    #         "The number of T2w flip angles must match the number of T2w files.")
    # if len(args.t2w_rf_phase_increments) != len(args.t2w_files):
    #     parser.error(
    #         "The number of T2w RF phase incrments must match the number of T2w files.")
    #
    # # set up config dict for qi jsr fitting
    # spgr_dict = dict(TR=args.t1w_repetition_time,
    #                  TE=args.t1w_echo_time,
    #                  FA=args.t1w_flip_angles)
    # ssfp_dict = dict(TR=args.t2w_repetition_time,
    #                  Trf=args.t2w_echo_time,
    #                  FA=args.t2w_flip_angles,
    #                  PhaseInc=args.t2w_rf_phase_increments)
    # qi_jsr_config_dict = dict(SPGR=spgr_dict, SSFP=ssfp_dict)
    #
    # input_dict = dict(
    #     t1w_files=args.t1w_files,
    #     t2w_files=args.t2w_files,
    #     t1w_reg_target_file=args.t1w_files[0],
    #     b1_map_file=args.b1_map_file,
    #     b1_anat_ref_file=args.b1_anat_ref_file,
    #     qi_jsr_config_dict=qi_jsr_config_dict)
    #
    # # set up workflow
    # wf = pe.Workflow(name="process_ssfp_dataset")
    # wf.base_dir = args.output_dir
    #
    # # create input node using entries in input_dict
    # input_node = Node(
    #     IdentityInterface(fields=list(input_dict.keys())),
    #     name='input_node')
    # for key, value in input_dict.items():
    #     setattr(input_node.inputs, key, value)
    #
    # # write preprocessed images
    # datasink = pe.Node(DataSink(base_directory=args.output_dir),
    #                    name="datasink")
    #
    # # preprocess images
    # preprocess_3depi_wf = preprocess_3depi()
    # wf.connect([(input_node, preprocess_3depi_wf, [
    #     ('t2w_mag_file', 'input_node.magnitude_file'),
    #     ('t2w_phase_file', 'input_node.phase_file'),
    #     ('b1_map_file', 'input_node.b1_map_file'),
    #     ('b1_anat_ref_file', 'input_node.b1_anat_ref_file')
    # ])])
    #
    # mag_first_volume_extractor = Node(fsl.ExtractROI(),
    #                                   name="mag_first_volume_extractor")
    # mag_first_volume_extractor.inputs.t_min = 0
    # mag_first_volume_extractor.inputs.t_size = 1
    # wf.connect(preprocess_3depi_wf, "output_node.magnitude_file",
    #            mag_first_volume_extractor, "in_file")
    #
    # # create input node using entries in input_dict
    # brain_mask_input_node = Node(
    #     IdentityInterface(
    #         fields=["image_file"]),
    #     name='brain_mask_input_node')
    #
    # if args.t1w_file:
    #     # register T1w image to T2w mag image
    #     register_t1w_to_t2w = pe.Node(
    #         create_default_ants_rigid_registration_node(),
    #         name="register_t1w_to_t2w")
    #     wf.connect(mag_first_volume_extractor, "roi_file",
    #                register_t1w_to_t2w, "fixed_image")
    #     wf.connect(input_node, "t1w_file",
    #                register_t1w_to_t2w, "moving_image")
    #
    #     # use registered T1w image for brain mask estimation
    #     wf.connect(register_t1w_to_t2w, "warped_image",
    #                brain_mask_input_node, "image_file")
    #
    #     wf.connect(register_t1w_to_t2w, "warped_image",
    #                datasink, "@t1w_file")
    #
    # else:
    #
    #     # use registered T2w magnitude image for brain mask estimation
    #     wf.connect(input_node, "warped_image",
    #                brain_mask_input_node, "image_file")
    #
    # create_brain_mask_wf = create_brain_mask()
    # wf.connect(brain_mask_input_node, "image_file",
    #            create_brain_mask_wf, "input_node.in_file")
    #
    # # write images
    # wf.connect(preprocess_3depi_wf, "output_node.magnitude_file",
    #            datasink, "@t2w_mag_file")
    # wf.connect(preprocess_3depi_wf, "output_node.phase_file",
    #            datasink, "@t2w_phase_file")
    # wf.connect(preprocess_3depi_wf, "output_node.b1_map_file",
    #            datasink, "@b1_map_file")
    # wf.connect(preprocess_3depi_wf, "output_node.b1_anat_ref_file",
    #            datasink, "@b1_anat_ref_file")
    #
    # # if not args.preprocess_only:
    # #     # estimate relaxation parameters
    # #     estimate_relaxation_ssfp_wf = estimate_relaxation_ssfp_multi_file()
    # #     wf.connect([(preprocess_3depi_wf, estimate_relaxation_ssfp_wf, [
    # #         ('output_node.b1_map_file', 'input_node.b1_map_file'),
    # #         ('output_node.t1w_files', 'input_node.t1w_files'),
    # #         ('output_node.t2w_files', 'input_node.t2w_files'),
    # #     ])])
    # #     wf.connect(input_node, 'qi_jsr_config_dict',
    # #                estimate_relaxation_ssfp_wf, 'input_node.qi_jsr_config_dict')
    # #     wf.connect(create_brain_mask_wf, 'output_node.out_file',
    # #                estimate_relaxation_ssfp_wf, 'input_node.brain_mask_file')
    # #
    # #     # write relaxation parameter maps
    # #     wf.connect(estimate_relaxation_ssfp_wf, "output_node.r1_map_file",
    # #                datasink, "@R1Map")
    # #     wf.connect(estimate_relaxation_ssfp_wf, "output_node.r2_map_file",
    # #                datasink, "@R2Map")
    # #     wf.connect(estimate_relaxation_ssfp_wf, "output_node.t1_map_file",
    # #                datasink, "@T1Map")
    # #     wf.connect(estimate_relaxation_ssfp_wf, "output_node.t2_map_file",
    # #                datasink, "@T2Map")
    #
    # wf.run(**run_settings)


if __name__ == "__main__":
    main()
