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

    mni_atlases_bids_entities = dict(space="midspaceRuns", suffix="probseg",
                                     acquisition=None, desc=None)
    mni_atlas_dir = os.path.abspath("./../../data/atlases")
    mni_atlas_dict = dict(
        subcortical=os.path.join(mni_atlas_dir,
                                 "space-MNI152_label-subcortical_desc-HarvardOxford_probseg.nii.gz"),
        cortical=os.path.join(mni_atlas_dir,
                              "space-MNI152_label-cortical_desc-HarvardOxford_probseg.nii.gz"),
        wmPrior=os.path.join(mni_atlas_dir,
                             "space-MNI152_label-wm_desc-SPM_probseg.nii.gz"),
        gmPrior=os.path.join(mni_atlas_dir,
                             "space-MNI152_label-gm_desc-SPM_probseg.nii.gz"),
        csfPrior=os.path.join(mni_atlas_dir,
                              "space-MNI152_label-csf_desc-SPM_probseg.nii.gz"),
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

                input_dict["sub_to_mni_transform_file"] = \
                    find_file(layout,
                              subject=subject,
                              session=session,

                              desc="midspaceRunsToMNI152",
                              suffix="transform",
                              extension="mat")

                input_dict["sub_to_mni_warp_file"] = \
                    find_file(layout,
                              subject=subject,
                              session=session,
                              desc="midspaceRunsToMNI152",
                              suffix="warp",
                              **DEFAULT_NIFTI_READ_EXT_ENTITY)

                input_dict["mni_to_sub_warp_file"] = \
                    find_file(layout,
                              subject=subject,
                              session=session,
                              desc="MNI152ToMidspaceRuns",
                              suffix="warp",
                              **DEFAULT_NIFTI_READ_EXT_ENTITY)

                inputs.append(input_dict)

    # Create a workflow
    wf = Workflow(name='register_mni', base_dir=os.getcwd())
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

    transforms_node = pe.Node(IdentityInterface(fields=[
        "subject_to_mni_transforms",
        "subject_to_mni_invert_flags",
        "mni_to_subject_transforms",
        "mni_to_subjectinvert_flags",
    ]), name="transforms_node")


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
               transforms_node, "subject_to_mni_transforms")
    wf.connect(merge_mni_to_subject_transforms_node, "out",
               transforms_node, "mni_to_subject_transforms")
    transforms_node.inputs.subject_to_mni_invert_flags = [False, False]
    transforms_node.inputs.mni_to_subject_invert_flags = [False, True]

    apply_transform_subject_to_mni_settings = dict(
        dimension=3,
        interpolation='Linear',
        input_image_type=3,
    )

    # transform and save atlases in subject space
    for label, filename in mni_atlas_dict.items():
        # transform image to subject space
        apply_transform = pe.Node(ApplyTransforms(
            **apply_transform_subject_to_mni_settings),
            name=f"apply_transform_{label}")
        apply_transform.inputs.input_image = filename
        wf.connect(transforms_node, 'mni_to_subject_transforms',
                   apply_transform, 'transforms')
        wf.connect(transforms_node, 'mni_to_subject_invert_flags',
                   apply_transform, 'invert_transform_flags')
        wf.connect(input_node, 't1w_reg_target_file',
                   apply_transform, 'reference_image')

        # write image in subject space
        file_writer = pe.Node(BidsOutputWriter(),
                              name=f"file_writer_{label}")
        file_writer.inputs.output_dir = args.output_dir
        file_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
        file_writer.inputs.entity_overrides = dict(label=label,
                                                   **mni_atlases_bids_entities)
        wf.connect(apply_transform, "output_image",
                   file_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   file_writer, "template_file")

    # Run the workflow
    wf.run(**run_settings)


if __name__ == "__main__":
    main()
