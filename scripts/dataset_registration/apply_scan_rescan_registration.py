import argparse
import os
import multiprocessing
import ants
import antspynet
from nipype.interfaces.fsl import ApplyMask
from nipype import Workflow, Node, Function, Merge
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import Info
from nipype.interfaces.ants import ApplyTransforms
from bids.layout import BIDSLayout, BIDSFile
import nipype.pipeline.engine as pe
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    TraitedSpec, File, traits)
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    isdefined)
from nipype.interfaces.fsl import Threshold
from nodes.io import BidsOutputWriter
from utils.io import write_minimal_bids_dataset_description
from nipype.interfaces.utility import Select
from utils.bids_config import DEFAULT_NIFTI_READ_EXT_ENTITY, \
    DEFAULT_NIFTI_WRITE_EXT_ENTITY, \
    PROCESSED_ENTITY_OVERRIDES_R1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_R2_MAP, \
    PROCESSED_ENTITY_OVERRIDES_T1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_T2_MAP, \
    PROCESSED_ENTITY_OVERRIDES_MTSAT_MAP, \
    PROCESSED_ENTITY_OVERRIDES_PD_MAP, \
    PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE, \
    PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, InputMultiPath,
    traits, isdefined
)
from nipype.utils.filemanip import fname_presuffix
import ants
import os
from nodes.registration import \
    create_default_ants_rigid_affine_syn_registration_node
from utils.io import find_file


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
                runs = layout.get_runs(subject=subject, session=session)

                if len(runs) != 2:
                    continue

                for run in runs:
                    input_dict = dict(
                        run=run,
                        subject=subject,
                        session=session
                    )

                    input_dict["t1w_template_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  space="midspaceRuns",
                                  desc="template",
                                  suffix="T1w",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["subject_to_midspace_transform_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  desc="subjectToMidspaceRuns",
                                  suffix="transform",
                                  extension="mat")

                    input_dict["subject_to_midspace_warp_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  desc="subjectToMidspaceRuns",
                                  suffix="warp",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["brain_mask_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  space="subject",
                                  desc="brainTight",
                                  suffix="mask",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["csf_prior_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  label="csfPrior",
                                  space="subject",
                                  suffix="probseg",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["gm_prior_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  label="gmPrior",
                                  space="subject",
                                  suffix="probseg",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["wm_prior_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  label="wmPrior",
                                  space="subject",
                                  suffix="probseg",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["cortical_probseg_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  label="cortical",
                                  space="subject",
                                  suffix="probseg",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["cortical_left_probseg_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  label="corticalLeft",
                                  space="subject",
                                  suffix="probseg",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["cortical_right_probseg_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  label="corticalRight",
                                  space="subject",
                                  suffix="probseg",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["subcortical_probseg_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  label="subcortical",
                                  space="subject",
                                  suffix="probseg",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

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

                    t1_mpm_based_maps = layout.get(
                        subject=subject,
                        session=session,
                        run=run,
                        space="subject",
                        desc="mpmBased",
                        **PROCESSED_ENTITY_OVERRIDES_T1_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY
                    )
                    if len(t1_mpm_based_maps) == 1:
                        input_dict["t1_map_mpm_based_file"] = t1_mpm_based_maps[
                            0]

                    mt_sat_map_files = layout.get(
                        subject=subject,
                        session=session,
                        run=run,
                        space="subject",
                        desc="mpmBased",
                        **PROCESSED_ENTITY_OVERRIDES_MTSAT_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )
                    if len(mt_sat_map_files) == 1:
                        input_dict["mt_sat_map_file"] = mt_sat_map_files[
                            0]

                    pd_map_files = layout.get(
                        subject=subject,
                        session=session,
                        run=run,
                        space="subject",
                        desc="mpmBased",
                        **PROCESSED_ENTITY_OVERRIDES_PD_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )
                    if len(pd_map_files) == 1:
                        input_dict["pd_map_file"] = pd_map_files[
                            0]

                    inputs.append(input_dict)

    print(inputs)

    # Create a workflow
    wf = Workflow(name='register_scan_rescan', base_dir=os.getcwd())
    wf.base_dir = args.temp_dir

    # set up bids input node
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    inputs_to_register = [key for key in keys if key not in
                          ["subject", "session", "run", "t1w_template_file",
                           "subject_to_midspace_transform_file", "subject_to_midspace_warp_file"]]

    # merge forward transforms to list
    merge_transforms = (
        pe.Node(Merge(2), name="merge_transforms"))
    wf.connect(input_node, "subject_to_midspace_warp_file",
               merge_transforms, "in1")
    wf.connect(input_node, "subject_to_midspace_transform_file",
               merge_transforms, "in2")
    invert_transform_flags = [False, False]

    map_entity_overrides_common = dict(space="midspaceRuns")
    images_to_transform = [
        "csf_prior_file",
        "gm_prior_file",
        "wm_prior_file",
        "r1_map_file",
        "r2_map_file",
        "t1_map_file",
        "t2_map_file",
        "cortical_probseg_file",
        "cortical_left_probseg_file",
        "cortical_right_probseg_file",
        "subcortical_probseg_file",
        "brain_mask_file"
    ]

    if "t1_map_mpm_based_file" in input_dict:
        images_to_transform.append("t1_map_mpm_based_file")

    if "mt_sat_map_file" in input_dict:
        images_to_transform.append("mt_sat_map_file")

    if "pd_map_file" in input_dict:
        images_to_transform.append("pd_map_file")

    for image_to_transform in images_to_transform:

        apply_transform = pe.Node(ApplyTransforms(
            dimension=3,
            interpolation="Linear",
            invert_transform_flags=invert_transform_flags,
            input_image_type=3),
            name=f"apply_transform_{image_to_transform}")

        wf.connect(merge_transforms, 'out',
                   apply_transform, 'transforms')
        wf.connect(input_node, image_to_transform,
                   apply_transform, 'input_image')
        wf.connect(input_node, "t1w_template_file",
                   apply_transform, 'reference_image')

        # write image in MNI space
        file_writer = pe.Node(BidsOutputWriter(),
                              name=f"file_writer_{image_to_transform}")
        file_writer.inputs.output_dir = args.output_dir
        file_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
        file_writer.inputs.entity_overrides = map_entity_overrides_common
        wf.connect(apply_transform, "output_image",
                   file_writer, "in_file")
        wf.connect(input_node, image_to_transform,
                   file_writer, "template_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
