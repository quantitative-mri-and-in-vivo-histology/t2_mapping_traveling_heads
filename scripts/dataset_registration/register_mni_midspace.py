import argparse
import multiprocessing
import os

import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from nipype import Workflow, Node
from nipype.interfaces.fsl import Info
from nipype.interfaces.utility import IdentityInterface, Select, Merge
from nipype.interfaces.ants import ApplyTransforms, N4BiasFieldCorrection

from nodes.registration import \
    create_default_ants_rigid_affine_syn_registration_node, \
    create_default_ants_rigid_double_affine_syn_registration_node, \
    create_default_ants_rigid_affine_registration_node
from nodes.io import BidsOutputWriter
from utils.bids_config import DEFAULT_NIFTI_READ_EXT_ENTITY, \
    DEFAULT_NIFTI_WRITE_EXT_ENTITY, \
    PROCESSED_ENTITY_OVERRIDES_R1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_R2_MAP, \
    PROCESSED_ENTITY_OVERRIDES_T1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_T2_MAP, \
    PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE, \
    PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK
from utils.io import write_minimal_bids_dataset_description, find_file


def main():
    parser = argparse.ArgumentParser(
        description="Process 3D-EPI dataset.")
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Path to the output derivatives folder.')
    parser.add_argument('-t', '--temp_dir', default=os.getcwd(),
                        help='Directory for intermediate outputs (default: current working directory).')
    parser.add_argument('--derivatives', required=False, default=None,
                        help='Path to the additional derivatives folder.')
    parser.add_argument('--n_procs', type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of processors to use (default: all available cores).')
    parser.add_argument('--subject', help='Process a specific subject.')
    parser.add_argument('--session', help='Process a specific session.')
    parser.add_argument('--run', help='Process a specific run.')
    parser.add_argument(
        '--reuse_registration', action='store_true', default=False,
        help="Reuse precomputed registration"
    )
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
    layout_derivatives = False if args.derivatives is None else [
        args.derivatives]
    layout = BIDSLayout(args.input_dir,
                        derivatives=layout_derivatives,
                        validate=False)

    # define pattern for output files
    REGISTRATION_BIDS_OUTPUT_PATTERN = 'sub-{subject}/ses-{session}/{datatype}/' \
                                       'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                                       '[_run-{run}][_space-{space}][_label-{label}]' \
                                       '[_desc-{desc}][_part-{part}]_{suffix}.{extension}'

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

                if len(runs) != 2:
                    continue


                input_dict = dict(
                    subject=subject,
                    session=session
                )

                input_dict["t1w_reg_target_file"] = find_file(
                    layout,
                    subject=subject,
                    session=session,
                    space="midspaceRuns",
                    desc="template",
                    suffix="T1w",
                    **DEFAULT_NIFTI_READ_EXT_ENTITY,
                )

                inputs.append(input_dict)

    # Create a workflow
    wf = Workflow(name='register_mni_midspace', base_dir=os.getcwd())
    wf.base_dir = args.temp_dir

    # create input node using entries in input_dict and
    # use independent subject-session-run combinations as iterables
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    mni_template = Info.standard_image(
        'MNI152_T1_1mm.nii.gz')  # Get MNI template path from FSL
    mni_template_mask = Info.standard_image(
        'MNI152_T1_1mm_brain_mask_dil.nii.gz')  # Get MNI template path from FSL
    # mni_template_mask_fine = Info.standard_image(
    #     'MNI152_T1_1mm_brain_mask_dil.nii.gz')  # Get MNI template path from FSL

    transforms_node = pe.Node(IdentityInterface(fields=[
        "subject_to_mni_transforms",
        "subject_to_mni_invert_flags",
        "mni_to_subject_transforms",
        "mni_to_subjectinvert_flags",
    ]), name="transforms_node")

    # remove low frequency bias
    n4_bias_field_correction = pe.Node(N4BiasFieldCorrection(
        dimension=3
    ),
        name="n4_bias_field_correction")
    wf.connect(input_node, "t1w_reg_target_file",
               n4_bias_field_correction, "input_image")

    # compute registration
    register_t1w = pe.Node(
        create_default_ants_rigid_affine_syn_registration_node(),
        name="register_t1w")
    register_t1w.inputs.fixed_image = mni_template
    register_t1w.inputs.fixed_image_masks = ["NULL", "NULL",
                                             mni_template_mask]
    # register_t1w.inputs.use_histogram_matching = True
    wf.connect(n4_bias_field_correction, "output_image",
               register_t1w, "moving_image")

    # get affine transform from list of transforms
    select_forward_affine_node = pe.Node(Select(index=1),
                                         name="select_forward_affine_node")
    wf.connect(register_t1w, "reverse_forward_transforms",
               select_forward_affine_node, "inlist")

    # write affine transform
    forward_affine_transform_writer = pe.Node(BidsOutputWriter(),
                                              name="forward_affine_transform_writer")
    forward_affine_transform_writer.inputs.output_dir = args.output_dir
    forward_affine_transform_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        desc="midspaceRunsToMNI152",
        suffix="transform",
        extension="mat")
    wf.connect(select_forward_affine_node, "out",
               forward_affine_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               forward_affine_transform_writer, "template_file")

    # extract forward warp from list of transforms
    select_forward_warp_node = pe.Node(Select(index=0),
                                       name="select_warp_node")
    wf.connect(register_t1w, "reverse_forward_transforms",
               select_forward_warp_node, "inlist")

    # write forward warp
    forward_warp_transform_writer = pe.Node(BidsOutputWriter(),
                                            name="forward_warp_transform_writer")
    forward_warp_transform_writer.inputs.output_dir = args.output_dir
    forward_warp_transform_writer.inputs.entity_overrides = dict(part=None,
                                                                 acquisition=None,
                                                                 desc="midspaceRunsToMNI152",
                                                                 suffix="warp",
                                                                 **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(select_forward_warp_node, "out",
               forward_warp_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               forward_warp_transform_writer, "template_file")

    # select reverse warp from list of transforms
    select_reverse_warp_node = pe.Node(Select(index=1),
                                       name="select_reverse_warp_node")
    wf.connect(register_t1w, "reverse_transforms",
               select_reverse_warp_node, "inlist")

    # write reverse warp
    reverse_warp_transform_writer = pe.Node(BidsOutputWriter(),
                                            name="reverse_warp_transform_writer")
    reverse_warp_transform_writer.inputs.output_dir = args.output_dir
    reverse_warp_transform_writer.inputs.entity_overrides = dict(part=None,
                                                                 acquisition=None,
                                                                 desc="MNI152ToMidspaceRuns",
                                                                 suffix="warp",
                                                                 **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(select_reverse_warp_node, "out",
               reverse_warp_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               reverse_warp_transform_writer, "template_file")

    # set forward and reverse transforms node using values from registration
    wf.connect(register_t1w, "reverse_forward_transforms",
               transforms_node, "subject_to_mni_transforms")
    wf.connect(register_t1w, "reverse_transforms",
               transforms_node, "mni_to_subject_transforms")
    wf.connect(register_t1w, "reverse_transforms",
               transforms_node, "forward_invert_flags")
    wf.connect(register_t1w, "reverse_invert_flags",
               transforms_node, "mni_to_subject_invert_flags")


    # Run the workflow
    wf.run(**run_settings)


if __name__ == "__main__":
    main()
