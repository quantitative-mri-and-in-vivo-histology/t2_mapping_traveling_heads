import argparse
import multiprocessing
import os

import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from nipype import Workflow, Node
from nipype.interfaces.fsl import Info
from nipype.interfaces.utility import IdentityInterface, Select, Merge
from nipype.interfaces.ants import ApplyTransforms

from nodes.registration import \
    create_default_ants_rigid_affine_syn_registration_node
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
    layout = BIDSLayout(args.input_dir,
                        derivatives=[args.derivatives],
                        validate=False)

    # define pattern for output files
    REGISTRATION_BIDS_OUTPUT_PATTERN = 'sub-{subject}/ses-{session}/{datatype}/' \
                                       'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                                       '[_run-{run}][_space-{space}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'

    mni_atlases_bids_entities = dict(space="subject", suffix="probseg", acquisition=None)
    mni_atlas_dir = os.path.abspath("./../../data/atlases")
    mni_atlas_dict = dict(
        subcortical=os.path.join(mni_atlas_dir, "HarvardOxford-sub-prob-1mm.nii.gz"),
        cortical=os.path.join(mni_atlas_dir, "HarvardOxford-cort-prob-1mm.nii.gz"),
        corticalLeft=os.path.join(mni_atlas_dir, "HarvardOxford-cort-left-prob-1mm.nii.gz"),
        corticalRight=os.path.join(mni_atlas_dir, "HarvardOxford-cort-right-prob-1mm.nii.gz"),
        wmPrior=os.path.join(mni_atlas_dir, "white_matter.nii.gz"),
        gmPrior=os.path.join(mni_atlas_dir, "gray_matter.nii.gz"),
        csfPrior=os.path.join(mni_atlas_dir, "csf.nii.gz"),
    )

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
                    input_dict = dict(
                        subject=subject,
                        session=session,
                        run=run
                    )

                    input_dict["r1_map_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_R1_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["r2_map_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_R2_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["t1_map_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_T1_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["t2_map_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_T2_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["t1w_reg_target_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["brain_mask_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    if args.reuse_registration:
                        print("Tying to load precomputed registration "
                              "from disk.")

                        input_dict["sub_to_mni_transform_file"] = \
                            find_file(layout,
                                      subject=subject,
                                      session=session,
                                      run=run,
                                      desc="SubToMni",
                                      suffix="transform",
                                      extension="mat")

                        input_dict["sub_to_mni_warp_file"] = \
                            find_file(layout,
                                      subject=subject,
                                      session=session,
                                      run=run,
                                      desc="SubToMni",
                                      suffix="warp",
                                      **DEFAULT_NIFTI_READ_EXT_ENTITY)

                        input_dict["mni_to_sub_warp_file"] = \
                            find_file(layout,
                                      subject=subject,
                                      session=session,
                                      run=run,
                                      desc="MniToSub",
                                      suffix="warp",
                                      **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["relaxation_maps"] = [
                        input_dict["r1_map_file"],
                        input_dict["r2_map_file"],
                        input_dict["t1_map_file"],
                        input_dict["t2_map_file"],
                    ]

                    inputs.append(input_dict)

    # Create a workflow
    wf = Workflow(name='register_maps_to_mni', base_dir=os.getcwd())
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
        'MNI152_T1_1mm_brain_mask.nii.gz')  # Get MNI template path from FSL

    # set up transforms (either computed of from files)
    subject_to_mni_transforms_node = pe.Node(IdentityInterface(fields=[
        "transforms"
    ]), name="subject_to_mni_transforms_node")
    mni_to_subject_transforms_node = pe.Node(IdentityInterface(fields=[
        "transforms"
    ]), name="mni_to_subject_transforms_node")

    if not args.reuse_registration:

        # compute registration
        register_t1w = pe.Node(
            create_default_ants_rigid_affine_syn_registration_node(),
            name="register_t1w")
        register_t1w.inputs.fixed_image = mni_template
        register_t1w.inputs.fixed_image_masks = ["NULL", mni_template_mask,
                                                 mni_template_mask]
        wf.connect(input_node, "t1w_reg_target_file", register_t1w,
                   "moving_image")

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
            desc="SubToMni",
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
                                                                     desc="SubToMni",
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
                                                                     desc="MniToSub",
                                                                     suffix="warp",
                                                                     **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
        wf.connect(select_reverse_warp_node, "out",
                   reverse_warp_transform_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   reverse_warp_transform_writer, "template_file")

        # set forward and reverse transforms node using values from registration
        wf.connect(register_t1w, "reverse_forward_transforms",
                   subject_to_mni_transforms_node, "transforms")
        wf.connect(register_t1w, "reverse_transforms",
                   mni_to_subject_transforms_node, "transforms")

    else:
        # use precomputed registration

        # merge forward transforms to list
        merge_subject_to_mni_transforms_node = (
            pe.Node(Merge(2), name="merge_subject_to_mni_transforms_node"))
        wf.connect(input_node, "sub_to_mni_warp_file",
                   merge_subject_to_mni_transforms_node, "in1")
        wf.connect(input_node, "sub_to_mni_transform_file",
                   merge_subject_to_mni_transforms_node, "in2")

        # merge reverse transforms to list
        merge_mni_to_subject_transforms_node = (
            pe.Node(Merge(2), name="merge_mni_to_subject_transforms_node"))
        wf.connect(input_node, "mni_to_sub_warp_file",
                   merge_mni_to_subject_transforms_node, "in1")
        wf.connect(input_node, "sub_to_mni_transform_file",
                   merge_mni_to_subject_transforms_node, "in2")

        # set forward and reverse transforms node from merged transforms
        wf.connect(merge_subject_to_mni_transforms_node, "out",
                   subject_to_mni_transforms_node, "transforms")
        wf.connect(merge_mni_to_subject_transforms_node, "out",
                   mni_to_subject_transforms_node, "transforms")

    subject_to_mni_transform_flags = [False, False]
    apply_transform_subject_to_mni_settings = dict(
        dimension=3,
        interpolation='Linear',
        invert_transform_flags=subject_to_mni_transform_flags,
        reference_image=mni_template
    )

    sub_to_mni_writer_settings = [
        ("t1w_reg_target_file", dict(
            space="mni", **PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE)),
        ("r1_map_file", dict(
            space="mni", **PROCESSED_ENTITY_OVERRIDES_R1_MAP)),
        ("r2_map_file", dict(
            space="mni", **PROCESSED_ENTITY_OVERRIDES_R2_MAP)),
        ("t1_map_file", dict(
            space="mni", **PROCESSED_ENTITY_OVERRIDES_T1_MAP)),
        ("t2_map_file", dict(
            space="mni", **PROCESSED_ENTITY_OVERRIDES_T2_MAP))
    ]

    # transform and save subject images/maps in MNI space
    for sub_to_mni_writer_setting in sub_to_mni_writer_settings:
        # transform image to MNI space
        apply_transform = pe.Node(ApplyTransforms(
            **apply_transform_subject_to_mni_settings),
            name=f"apply_transform_{sub_to_mni_writer_setting[0]}")
        wf.connect(subject_to_mni_transforms_node, 'transforms',
                   apply_transform, 'transforms')
        wf.connect(input_node, sub_to_mni_writer_setting[0],
                   apply_transform, 'input_image')

        # write image in MNI space
        file_writer = pe.Node(BidsOutputWriter(),
                                        name=f"file_writer_{sub_to_mni_writer_setting[0]}")
        file_writer.inputs.output_dir = args.output_dir
        file_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
        file_writer.inputs.entity_overrides = sub_to_mni_writer_setting[1]
        wf.connect(apply_transform, "output_image",
                   file_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   file_writer, "template_file")

    mni_to_subject_transform_flags = [False, True]
    apply_transform_subject_to_mni_settings = dict(
        dimension=3,
        interpolation='Linear',
        invert_transform_flags=mni_to_subject_transform_flags,
        input_image_type=3,
    )

    # transform and save atlases in subject space
    for description, filename in mni_atlas_dict.items():
        # transform image to subject space
        apply_transform = pe.Node(ApplyTransforms(
            **apply_transform_subject_to_mni_settings),
            name=f"apply_transform_{description}")
        apply_transform.inputs.input_image = filename
        wf.connect(mni_to_subject_transforms_node, 'transforms',
                   apply_transform, 'transforms')
        wf.connect(input_node, 't1w_reg_target_file',
                   apply_transform, 'reference_image')

        # write image in subject space
        file_writer = pe.Node(BidsOutputWriter(),
                                        name=f"file_writer_{description}")
        file_writer.inputs.output_dir = args.output_dir
        file_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
        file_writer.inputs.entity_overrides = dict(desc=description, **mni_atlases_bids_entities)
        wf.connect(apply_transform, "output_image",
                   file_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   file_writer, "template_file")

    # Run the workflow
    wf.run(**run_settings)


if __name__ == "__main__":
    main()
