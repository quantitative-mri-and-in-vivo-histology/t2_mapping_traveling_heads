import argparse
import os
import multiprocessing
from nipype import Workflow, Node, Function
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import Info
from nipype.interfaces.ants import ApplyTransforms
from bids.layout import BIDSLayout
import nipype.pipeline.engine as pe
import nipype.interfaces.ants as ants
from nipype_utils import BidsOutputWriter
from utils.io import write_minimal_bids_dataset_description
from nipype.interfaces.utility import Select
from nipype.interfaces.utility import Merge


def main():
    parser = argparse.ArgumentParser(
        description="Process a dataset with optional steps.")
    parser.add_argument('-i', '--bids_root', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-d', '--derivatives', nargs='*', required=False,
                        help='One or more derivatives directories to use.')
    parser.add_argument('-o', '--output_derivative_dir', required=True,
                        help='Path to the output derivatives folder.')
    parser.add_argument('--base_dir', default=os.getcwd(),
                        help='Base directory for processing (default: current working directory).')
    parser.add_argument(
        '--preprocess_only', action='store_true', default=False,
        help="Preprocess the data only, without parameter estimation"
    )
    parser.add_argument('--subject', default=None,
                        help='Specify a subject to process (e.g., sub-01). If not provided, all subjects are processed.')
    parser.add_argument('--session', default=None,
                        help='Specify a session to process (e.g., ses-01). If not provided, all sessions are processed.')
    parser.add_argument('--run', default=None,
                        help='Specify a run to process (e.g., run-01). If not provided, all runs are processed.')
    parser.add_argument('--n_procs', type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of processors to use (default: all available cores).')
    args = parser.parse_args()

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
    # subjects = ["phy003"]
    for subject in subjects:
        sessions = layout.get_sessions(subject=subject)
        if sessions:  # Only add subjects with existing sessions
            for session in sessions:
                runs = layout.get_runs(subject=subject, session=session)

                if len(runs) == 0:
                    runs = [None]

                for run in runs:
                    t1_map_files = layout.get(subject=subject,
                                              session=session,
                                              suffix="T1map",
                                              extension="nii.gz",
                                              run=run)
                    assert (len(t1_map_files) == 1)
                    t1_map_file = t1_map_files[0]

                    t2_map_files = layout.get(subject=subject,
                                              session=session,
                                              suffix="T2map",
                                              extension="nii.gz",
                                              run=run)
                    assert (len(t2_map_files) == 1)
                    t2_map_file = t2_map_files[0]

                    r1_map_files = layout.get(subject=subject,
                                              session=session,
                                              suffix="R1map",
                                              extension="nii.gz",
                                              run=run)
                    assert (len(r1_map_files) == 1)
                    r1_map_file = r1_map_files[0]

                    r2_map_files = layout.get(subject=subject,
                                              session=session,
                                              suffix="R2map",
                                              extension="nii.gz",
                                              run=run)
                    assert (len(r2_map_files) == 1)
                    r2_map_file = r2_map_files[0]

                    t1w_reg_target_files = layout.get(subject=subject,
                                                      session=session,
                                                      acquisition="T1wRef",
                                                      suffix="T1w",
                                                      extension="nii.gz",
                                                      run=run)
                    assert (len(t1w_reg_target_files) == 1)
                    t1w_reg_target_file = t1w_reg_target_files[0]

                    brain_mask_files = layout.get(subject=subject,
                                                  session=session,
                                                  desc="brain",
                                                  suffix="mask",
                                                  extension="nii.gz",
                                                  run=run)
                    assert (len(brain_mask_files) == 1)
                    brain_mask_file = brain_mask_files[0]

                    sub_to_mni_transform_files = layout.get(subject=subject,
                                                      session=session,
                                                      suffix="transform",
                                                      desc="SubToMni",
                                                      extension="mat",
                                                      run=run)
                    assert (len(sub_to_mni_transform_files) == 1)
                    sub_to_mni_transform_file = sub_to_mni_transform_files[0]

                    sub_to_mni_warp_files = layout.get(subject=subject,
                                                      session=session,
                                                      suffix="warp",
                                                      desc="SubToMni",
                                                      extension="nii.gz",
                                                      run=run)
                    assert (len(sub_to_mni_warp_files) == 1)
                    sub_to_mni_warp_file = sub_to_mni_warp_files[0]

                    mni_to_sub_warp_files = layout.get(subject=subject,
                                                       session=session,
                                                       suffix="warp",
                                                       desc="SubToMni",
                                                       extension="nii.gz",
                                                       run=run)
                    assert (len(mni_to_sub_warp_files) == 1)
                    mni_to_sub_warp_file = mni_to_sub_warp_files[0]

                    relaxation_maps = [r1_map_file, r2_map_file, t1_map_file,
                                       t2_map_file]
                    relaxation_map_entities = [
                        dict(suffix="R1Map", desc=None, acquisition=None),
                        dict(suffix="R2Map", desc=None, acquisition=None),
                        dict(suffix="T1Map", desc=None, acquisition=None),
                        dict(suffix="T2Map", desc=None, acquisition=None)
                    ]

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       t1w_reg_target_file=t1w_reg_target_file,
                                       brain_mask_file=brain_mask_file,
                                       t1_map_file=t1_map_file,
                                       t2_map_file=t2_map_file,
                                       r1_map_file=r1_map_file,
                                       r2_map_file=r2_map_file,
                                       relaxation_maps=relaxation_maps,
                                       relaxation_map_entities=relaxation_map_entities,
                                       sub_to_mni_transform_file=sub_to_mni_transform_file,
                                       sub_to_mni_warp_file=sub_to_mni_warp_file,
                                       mni_to_sub_warp_file=mni_to_sub_warp_file))

    # Create a workflow
    wf = Workflow(name='register_maps_to_mni', base_dir=os.getcwd())
    wf.base_dir = args.base_dir

    # set up bids input node
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True


    mni_template = Info.standard_image(
        'MNI152_T1_2mm.nii.gz')  # Get MNI template path from FSL

    # set up transforms and flags
    merge_transforms_node = pe.Node(Merge(2), name="merge_transforms_node")
    wf.connect(input_node, "sub_to_mni_warp_file", merge_transforms_node, "in1")
    wf.connect(input_node, "sub_to_mni_transform_file", merge_transforms_node, "in2")

    # transform maps
    invert_transform_flags = [False, False]
    apply_transform_maps = pe.MapNode(ApplyTransforms(), name="apply_transform_maps",
                                  iterfield=["input_image"])
    apply_transform_maps.inputs.dimension = 3  # 3D image
    apply_transform_maps.inputs.reference_image = mni_template
    apply_transform_maps.inputs.interpolation = 'BSpline'
    apply_transform_maps.inputs.invert_transform_flags = invert_transform_flags
    wf.connect(merge_transforms_node, 'out',
               apply_transform_maps, 'transforms')
    wf.connect(input_node, 'relaxation_maps',
               apply_transform_maps, 'input_image')

    # transform T1w images
    apply_transform_t1w = pe.Node(ApplyTransforms(), name="apply_transform_t1w")
    apply_transform_t1w.inputs.dimension = 3  # 3D image
    apply_transform_t1w.inputs.reference_image = mni_template
    apply_transform_t1w.inputs.interpolation = 'BSpline'
    apply_transform_t1w.inputs.invert_transform_flags = invert_transform_flags
    wf.connect(merge_transforms_node, 'out',
               apply_transform_t1w, 'transforms')
    wf.connect(input_node, 't1w_reg_target_file',
               apply_transform_t1w, 'input_image')

    # write outputs
    out_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                  'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                  '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'

    t1w_reg_target_writer = pe.Node(BidsOutputWriter(),
                                    name="t1w_reg_target_writer")
    t1w_reg_target_writer.inputs.output_dir = args.output_derivative_dir
    t1w_reg_target_writer.inputs.pattern = out_pattern
    t1w_reg_target_writer.inputs.entity_overrides = dict(part=None,
                                                         acquisition="T1wRef")
    wf.connect(apply_transform_t1w, "output_image",
               t1w_reg_target_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               t1w_reg_target_writer, "template_file")

    map_writer = pe.MapNode(BidsOutputWriter(),
                            name="map_writer",
                            iterfield=["in_file", "entity_overrides"])
    map_writer.inputs.output_dir = args.output_derivative_dir
    map_writer.inputs.pattern = out_pattern
    wf.connect(apply_transform_maps, "output_image",
               map_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               map_writer, "template_file")
    wf.connect(input_node, "relaxation_map_entities",
               map_writer, "entity_overrides")



    run_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {'n_procs': args.n_procs}
    }

    # Run the workflow
    wf.run(**run_settings)
    # wf.run()


if __name__ == "__main__":
    main()
