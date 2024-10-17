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
    subjects = ["phy001"]
    for subject in subjects:
        sessions = layout.get_sessions(subject=subject)
        if sessions:  # Only add subjects with existing sessions
            for session in sessions:
                runs = layout.get_runs(subject=subject, session=session)

                if len(runs) != 2:
                    continue

                for run in runs:
                    t1_map_files = layout.get(subject=subject,
                                           session=session,
                                           suffix="T1map",
                                           extension="nii.gz")
                    assert(len(t1_map_files) == 2)

                    t2_map_files = layout.get(subject=subject,
                                           session=session,
                                           suffix="T2map",
                                           extension="nii.gz")
                    assert(len(t2_map_files) == 2)

                    r1_map_files = layout.get(subject=subject,
                                              session=session,
                                              suffix="R1map",
                                              extension="nii.gz")
                    assert (len(r1_map_files) == 2)

                    r2_map_files = layout.get(subject=subject,
                                              session=session,
                                              suffix="R2map",
                                              extension="nii.gz")
                    assert (len(r2_map_files) == 2)

                    t1w_reg_target_files = layout.get(subject=subject,
                                              session=session,
                                              acquisition="T1wRef",
                                              suffix="T1w",
                                              extension="nii.gz")
                    t1w_reg_target_files = [f for f in t1w_reg_target_files if
                                            "processed" in str(f)]
                    assert (len(t1w_reg_target_files) == 2)

                    # relaxation_maps = [r1_map_files, r2_map_files, t1_map_files, t2_map_files]
                    # # relaxation_map_suffixes = ["R1map", "R2map", "T1map", "T2map"]
                    # relaxation_map_entities = [
                    #     dict(suffix="R1Map", desc=None),
                    #     dict(suffix="R2Map", desc=None),
                    #     dict(suffix="T1Map", desc=None),
                    #     dict(suffix="T2Map", desc=None)
                    # ]

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       t1w_scan_file=t1w_reg_target_files[0],
                                       t1w_rescan_file=t1w_reg_target_files[1],
                                       t1_map_files=t1_map_files,
                                       t2_map_files=t2_map_files,
                                       r1_map_files=r1_map_files,
                                       r2_map_files=r2_map_files))
                                       # relaxation_maps=relaxation_maps,
                                       # relaxation_map_entities=relaxation_map_entities))

    print(inputs)

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
    mni_template_mask = Info.standard_image(
        'MNI152_T1_2mm_brain_mask.nii.gz')  # Get MNI template path from FSL

    # Adjust the ants.Registration node to perform symmetric registration
    ants_reg_params = dict(
        dimension=3,
        output_transform_prefix='run1_run2_midspace_',
        transforms=['SyN'],  # Only SyN to deform both to midspace
        transform_parameters=[(0.1, 3, 0)],  # Parameters for SyN
        metric=['CC'],  # Cross-correlation for both runs
        metric_weight=[1],  # Weight for metric
        radius_or_number_of_bins=[4],  # CC radius
        sampling_strategy=[None],  # No sampling for SyN
        convergence_threshold=[1e-6],  # Convergence threshold
        convergence_window_size=[10],  # Convergence window size
        number_of_iterations=[[100, 70, 50, 20]],  # Number of iterations
        shrink_factors=[[6, 4, 2, 1]],  # Shrink factors for multi-resolution
        smoothing_sigmas=[[3, 2, 1, 0]],
        # Smoothing sigmas for each resolution level
        interpolation='Linear',  # Linear interpolation
        output_warped_image='midspace_run1.nii.gz',
        # Midspace warped image for run1
        output_inverse_warped_image='midspace_run2.nii.gz',
        # Midspace warped image for run2
        use_histogram_matching=True,
        # Use histogram matching for multi-modal images
        # fixed_image=t1,  # Use run1 as the fixed image
        # moving_image=rescan_image,  # Use run2 as the moving image
        symmetric_forces=True  # Ensure symmetric registration for midspace
    )

    register_t1w = pe.Node(ants.Registration(**ants_reg_params),
                           name="register_t1w")
    wf.connect(input_node, "t1w_scan_file", register_t1w, "moving_image")
    wf.connect(input_node, "t1w_rescan_file", register_t1w, "fixed_image")

    # # Define the ApplyTransforms node
    # apply_transforms = pe.MapNode(ApplyTransforms(), name="apply_transforms",
    #                               iterfield=["input_image"])
    # apply_transforms.inputs.dimension = 3  # 3D image
    # apply_transforms.inputs.reference_image = mni_template
    # apply_transforms.inputs.interpolation = 'BSpline'
    #
    # wf.connect(register_t1w, 'reverse_forward_transforms',
    #            apply_transforms, 'transforms')
    # wf.connect(input_node, 'relaxation_maps',
    #            apply_transforms, 'input_image')
    #
    out_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                  'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                  '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'
    #
    # t1w_reg_target_writer = pe.Node(BidsOutputWriter(),
    #                                 name="t1w_reg_target_writer")
    # t1w_reg_target_writer.inputs.output_dir = args.output_derivative_dir
    # t1w_reg_target_writer.inputs.pattern = out_pattern
    # t1w_reg_target_writer.inputs.entity_overrides = dict(part=None,
    #                                                      acquisition="T1wRef")
    # wf.connect(register_t1w, "warped_image",
    #            t1w_reg_target_writer, "in_file")
    # wf.connect(input_node, "t1w_reg_target_file",
    #            t1w_reg_target_writer, "template_file")
    #
    # map_writer = pe.MapNode(BidsOutputWriter(),
    #                         name="map_writer",
    #                         iterfield=["in_file", "entity_overrides"])
    # map_writer.inputs.output_dir = args.output_derivative_dir
    # map_writer.inputs.pattern = out_pattern
    # wf.connect(apply_transforms, "output_image",
    #            map_writer, "in_file")
    # wf.connect(input_node, "t1w_reg_target_file",
    #            map_writer, "template_file")
    # wf.connect(input_node, "relaxation_map_entities",
    #            map_writer, "entity_overrides")

    # select_forward_affine_node = pe.Node(Select(index=1),
    #                                      name="select_forward_affine_node")
    # wf.connect(register_t1w, "reverse_forward_transforms",
    #            select_forward_affine_node, "inlist")
    # forward_affine_transform_writer = pe.Node(BidsOutputWriter(),
    #                                           name="forward_affine_transform_writer")
    # forward_affine_transform_writer.inputs.output_dir = args.output_derivative_dir
    # forward_affine_transform_writer.inputs.pattern = out_pattern
    # forward_affine_transform_writer.inputs.entity_overrides = dict(part=None,
    #                                                                acquisition=None,
    #                                                                desc="SubToMni",
    #                                                                suffix="transform",
    #                                                                extension="mat")
    # wf.connect(select_forward_affine_node, "out",
    #            forward_affine_transform_writer, "in_file")
    # wf.connect(input_node, "t1w_reg_target_file",
    #            forward_affine_transform_writer, "template_file")
    #
    # select_forward_warp_node = pe.Node(Select(index=0), name="select_warp_node")
    # wf.connect(register_t1w, "reverse_forward_transforms",
    #            select_forward_warp_node, "inlist")
    # forward_warp_transform_writer = pe.Node(BidsOutputWriter(),
    #                                         name="forward_warp_transform_writer")
    # forward_warp_transform_writer.inputs.output_dir = args.output_derivative_dir
    # forward_warp_transform_writer.inputs.pattern = out_pattern
    # forward_warp_transform_writer.inputs.entity_overrides = dict(part=None,
    #                                                              acquisition=None,
    #                                                              desc="SubToMni",
    #                                                              suffix="warp",
    #                                                              extension="nii.gz")
    # wf.connect(select_forward_warp_node, "out",
    #            forward_warp_transform_writer, "in_file")
    # wf.connect(input_node, "t1w_reg_target_file",
    #            forward_warp_transform_writer, "template_file")
    #
    # select_reverse_warp_node = pe.Node(Select(index=1),
    #                                    name="select_reverse_warp_node")
    # wf.connect(register_t1w, "reverse_transforms",
    #            select_reverse_warp_node, "inlist")
    # reverse_warp_transform_writer = pe.Node(BidsOutputWriter(),
    #                                         name="reverse_warp_transform_writer")
    # reverse_warp_transform_writer.inputs.output_dir = args.output_derivative_dir
    # reverse_warp_transform_writer.inputs.pattern = out_pattern
    # reverse_warp_transform_writer.inputs.entity_overrides = dict(part=None,
    #                                                              acquisition=None,
    #                                                              desc="MniToSub",
    #                                                              suffix="warp",
    #                                                              extension="nii.gz")
    # wf.connect(select_reverse_warp_node, "out",
    #            reverse_warp_transform_writer, "in_file")
    # wf.connect(input_node, "t1w_reg_target_file",
    #            reverse_warp_transform_writer, "template_file")


    run_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {'n_procs': args.n_procs}
    }

    # Run the workflow
    wf.run(**run_settings)
    # wf.run()


if __name__ == "__main__":
    main()
