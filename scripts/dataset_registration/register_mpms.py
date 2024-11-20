import argparse
import multiprocessing
import os

import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from nipype import Workflow, Node
from nipype.interfaces.ants import ApplyTransforms, N4BiasFieldCorrection
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import Threshold
from nipype.interfaces.fsl import ApplyMask

from nodes.io import BidsOutputWriter
from nodes.processing import AntspynetBrainExtraction
from nodes.registration import create_default_ants_rigid_registration_node, \
    create_default_ants_rigid_affine_registration_node
from utils.bids_config import DEFAULT_NIFTI_READ_EXT_ENTITY, \
    DEFAULT_NIFTI_WRITE_EXT_ENTITY, \
    PROCESSED_ENTITY_OVERRIDES_R1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_T1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE, \
    PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK, \
    PROCESSED_ENTITY_OVERRIDES_MTSAT_MAP, \
    PROCESSED_ENTITY_OVERRIDES_PD_MAP
from utils.io import write_minimal_bids_dataset_description, find_file


def main():
    parser = argparse.ArgumentParser(
        description="Process 3D-EPI dataset.")
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-m', '--mpm_dir', required=True,
                        help='Path to the BIDS root of MPM  dataset.')
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
    processed_layout = BIDSLayout(args.input_dir,
                                  validate=False)

    mpm_layout = BIDSLayout(args.mpm_dir,
                            validate=False)

    # define pattern for output files
    REGISTRATION_BIDS_OUTPUT_PATTERN = 'sub-{subject}/ses-{session}/{datatype}/' \
                                       'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                                       '[_run-{run}][_space-{space}][_label-{label}]' \
                                       '[_desc-{desc}][_part-{part}]_{suffix}.{extension}'

    mni_atlases_bids_entities = dict(space="subject", suffix="probseg",
                                     acquisition=None)
    mni_atlas_dir = os.path.abspath("./../../data/atlases")
    mni_atlas_dict = dict(
        subcortical=os.path.join(mni_atlas_dir,
                                 "space-MNI152_label-subcortical_desc-HarvardOxford_probseg.nii.gz"),
        cortical=os.path.join(mni_atlas_dir,
                              "space-MNI152_label-cortical_desc-HarvardOxford_probseg.nii.gz"),
        corticalLeft=os.path.join(mni_atlas_dir,
                                  "space-MNI152_label-corticalLeft_desc-HarvardOxford_probseg.nii.gz"),
        corticalRight=os.path.join(mni_atlas_dir,
                                   "space-MNI152_label-corticalRight_desc-HarvardOxford_probseg.nii.gz"),
        wmPrior=os.path.join(mni_atlas_dir,
                             "space-MNI152_label-wm_desc-SPM_probseg.nii.gz"),
        gmPrior=os.path.join(mni_atlas_dir,
                             "space-MNI152_label-gm_desc-SPM_probseg.nii.gz"),
        csfPrior=os.path.join(mni_atlas_dir,
                              "space-MNI152_label-csf_desc-SPM_probseg.nii.gz"),
    )

    # collect data for each independent subject-session-run combination
    inputs = []
    subjects = [
        args.subject] if args.subject else processed_layout.get_subjects()
    for subject in subjects:
        sessions = [
            args.session] if args.session else processed_layout.get_sessions(
            subject=subject)
        if sessions:
            for session in sessions:
                runs = [args.run] if args.run else processed_layout.get_runs(
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
                        mpm_layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_R1_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["t1_map_file"] = find_file(
                        mpm_layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_T1_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["mt_sat_map_file"] = find_file(
                        mpm_layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_MTSAT_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["pd_map_file"] = find_file(
                        mpm_layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_PD_MAP,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["t1w_reg_target_file"] = find_file(
                        processed_layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["brain_mask_file"] = find_file(
                        processed_layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    inputs.append(input_dict)

    # Create a workflow
    wf = Workflow(name='register_mpms', base_dir=os.getcwd())
    wf.base_dir = args.temp_dir

    # create input node using entries in input_dict and
    # use independent subject-session-run combinations as iterables
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    # remove low frequency bias
    n4_bias_field_correction = pe.Node(N4BiasFieldCorrection(
        dimension=3
    ),
        name="n4_bias_field_correction")
    wf.connect(input_node, "t1w_reg_target_file",
               n4_bias_field_correction, "input_image")

    # create brain mask
    brain_extraction_t1w = pe.Node(AntspynetBrainExtraction(),
                                       name="brain_extraction_t1w")
    wf.connect(n4_bias_field_correction, "output_image",
               brain_extraction_t1w, "anatomical_image")
    brain_mask_threshold_t1w = pe.Node(Threshold(thresh=0.01, args="-bin"),
                                      name="brain_mask_threshold_t1w")
    wf.connect(brain_extraction_t1w, "output_image",
               brain_mask_threshold_t1w, "in_file")

    apply_mask_t1w = pe.Node(ApplyMask(),
                            name="apply_mask_t1w")
    wf.connect(brain_mask_threshold_t1w, "out_file",
               apply_mask_t1w, "mask_file")
    wf.connect(n4_bias_field_correction, "output_image",
               apply_mask_t1w, "in_file")

    apply_mask_t1_map = pe.Node(ApplyMask(),
                             name="apply_mask_t1_map")
    wf.connect(brain_mask_threshold_t1w, "out_file",
               apply_mask_t1_map, "mask_file")
    wf.connect(input_node, "t1_map_file",
               apply_mask_t1_map, "in_file")

    # register map to T1w
    register_t1_map_to_t1w = pe.Node(
        create_default_ants_rigid_registration_node(),
        name="register_t1_map_to_t1w")
    register_t1_map_to_t1w.inputs.sampling_percentage = [0.7]
    register_t1_map_to_t1w.winsorize_lower_quantile = 0.005
    register_t1_map_to_t1w.winsorize_upper_quantile = 0.995
    wf.connect(apply_mask_t1w, "out_file",
               register_t1_map_to_t1w, "fixed_image")
    wf.connect(apply_mask_t1_map, "out_file",
               register_t1_map_to_t1w, "moving_image")

    map_entity_overrides_common = dict(space="subject", desc="mpmBased")
    map_writer_settings = [
        ("mt_sat_map_file", dict(
            **map_entity_overrides_common,
            **PROCESSED_ENTITY_OVERRIDES_MTSAT_MAP
        )),
        ("pd_map_file", dict(
            **map_entity_overrides_common,
            **PROCESSED_ENTITY_OVERRIDES_PD_MAP
        )),
        ("r1_map_file", dict(
            **map_entity_overrides_common,
            **PROCESSED_ENTITY_OVERRIDES_R1_MAP
        )),
        ("t1_map_file", dict(
            **map_entity_overrides_common,
            **PROCESSED_ENTITY_OVERRIDES_T1_MAP
        )),
    ]

    # transform and save subject images/maps in MNI space
    for map_writer_setting in map_writer_settings:

        # transform map to subject space
        apply_transform = pe.Node(ApplyTransforms(
            dimension=3,
            interpolation="Linear"),
            name=f"apply_transform_{map_writer_setting[0]}")
        wf.connect(register_t1_map_to_t1w, 'reverse_forward_transforms',
                   apply_transform, 'transforms')
        wf.connect(register_t1_map_to_t1w, 'forward_invert_flags',
                   apply_transform, 'invert_transform_flags')
        wf.connect(input_node, map_writer_setting[0],
                   apply_transform, 'input_image')
        wf.connect(input_node, 't1w_reg_target_file',
                   apply_transform, 'reference_image')

        # write image in MNI space
        file_writer = pe.Node(BidsOutputWriter(),
                              name=f"file_writer_{map_writer_setting[0]}")
        file_writer.inputs.output_dir = args.output_dir
        file_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
        file_writer.inputs.entity_overrides = map_writer_setting[1]
        wf.connect(apply_transform, "output_image",
                   file_writer, "in_file")
        wf.connect(input_node, "t1w_reg_target_file",
                   file_writer, "template_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
