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
from nodes.io import  BidsOutputWriter
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
    subjects = ["phy004"]
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

                    relaxation_maps = [r1_map_file, r2_map_file, t1_map_file,
                                       t2_map_file]
                    # relaxation_map_suffixes = ["R1map", "R2map", "T1map", "T2map"]
                    relaxation_map_entities = [
                        dict(suffix="R1Map", desc=None),
                        dict(suffix="R2Map", desc=None),
                        dict(suffix="T1Map", desc=None),
                        dict(suffix="T2Map", desc=None)
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
                                       relaxation_map_entities=relaxation_map_entities))

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


    # slow and good

    mni_template = Info.standard_image(
        'MNI152_T1_1mm.nii.gz')  # Get MNI template path from FSL
    mni_template_brain = Info.standard_image(
        'MNI152_T1_1mm_brain.nii.gz')  # Get MNI template path from FSL
    mni_template_mask = Info.standard_image(
        'MNI152_T1_1mm_brain_mask_dil.nii.gz')  # Get MNI template path from FSL

    ants_reg_params = dict(
        dimension=3,  # 3D registration
        output_transform_prefix='output_prefix_',  # Prefix for output files
        transforms=['Rigid', 'Affine', 'BSplineSyN'],  # Transformation types
        transform_parameters=[(0.1,), (0.1,), (0.1, 3, 0)],
        # Parameters for each transform
        metric=['MI', 'MI', 'CC'],
        # Metrics for each stage: MI for Rigid/Affine, CC for SyN
        metric_weight=[1, 1, 1],  # Weights for the metrics
        radius_or_number_of_bins=[32, 32, 4],
        # Number of bins for MI and radius for CC
        sampling_strategy=['Regular', 'Regular', None],
        # Sampling strategies for each stage
        sampling_percentage=[0.25, 0.25, None],  # Sampling percentages for MI
        convergence_threshold=[1e-6, 1e-6, 1e-6],  # Convergence thresholds
        convergence_window_size=[10, 10, 10],  # Convergence window sizes
        number_of_iterations=[[1000, 500, 250, 100], [1000, 500, 250, 100],
                              [100, 70, 50, 20]],
        # Iterations for each resolution level
        shrink_factors=[[8, 4, 2, 1], [8, 4, 2, 1], [6, 4, 2, 1]],
        # Shrink factors for the multi-resolution scheme
        smoothing_sigmas=[[3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0]],
        # Smoothing sigmas for the multi-resolution scheme
        interpolation='Linear',  # Linear interpolation
        output_warped_image='output_warped_image.nii.gz',  # Output warped image
        output_inverse_warped_image='output_inverse_warped_image.nii.gz',
        # Output inverse warped image
        use_histogram_matching=True,
        # Use histogram matching for multi-modal images
        winsorize_upper_quantile=0.995,
        # Winsorize image intensities (upper quantile)
        winsorize_lower_quantile=0.005,
        # Winsorize image intensities (lower quantile)
        initial_moving_transform_com=True,  # Align centers of mass
        fixed_image=mni_template
        # num_threads=1
    )


    # mni_template = Info.standard_image(
    #     'MNI152_T1_2mm.nii.gz')  # Get MNI template path from FSL
    # mni_template_brain = Info.standard_image(
    #     'MNI152_T1_2mm_brain.nii.gz')  # Get MNI template path from FSL
    # mni_template_mask = Info.standard_image(
    #     'MNI152_T1_2mm_brain_mask_dil.nii.gz')  # Get MNI template path from FSL

    # ants_reg_params = dict(
    #     dimension=3,  # 3D registration
    #     output_transform_prefix='output_prefix_',  # Prefix for output files
    #     transforms=['Rigid', 'Affine', 'SyN'],  # Rigid, affine, and SyN stages
    #     transform_parameters=[(0.1,), (0.1,), (0.1, 3, 0)],
    #     # Parameters for each transform
    #     metric=['MI', 'MI', 'CC'],  # MI for Rigid/Affine, CC for SyN
    #     metric_weight=[1, 1, 1],  # Equal weights for metrics
    #     radius_or_number_of_bins=[32, 32, 4],  # Bins for MI, radius for CC
    #     sampling_strategy=['Regular', 'Regular', None],
    #     # Regular sampling for MI, none for CC
    #     sampling_percentage=[0.1, 0.1, None],
    #     # Reduce sampling to 10% for faster MI
    #     convergence_threshold=[1e-4, 1e-4, 1e-4],
    #     # Relax convergence thresholds
    #     convergence_window_size=[5, 5, 5],
    #     # Smaller window sizes for convergence
    #     number_of_iterations=[[500, 250], [500, 250], [50, 20]],
    #     # Fewer iterations
    #     shrink_factors=[[4, 2], [4, 2], [2, 1]],  # Coarser shrink factors
    #     smoothing_sigmas=[[2, 1], [2, 1], [1, 0]],  # Reduced smoothing sigmas
    #     interpolation='Linear',  # Linear interpolation
    #     output_warped_image='output_warped_image_fast.nii.gz',
    #     # Output warped image
    #     output_inverse_warped_image='output_inverse_warped_image_fast.nii.gz',
    #     # Inverse warped image
    #     use_histogram_matching=True,
    #     # Histogram matching for multi-modal images
    #     winsorize_upper_quantile=0.995,  # Upper quantile for winsorization
    #     winsorize_lower_quantile=0.005,  # Lower quantile for winsorization
    #     initial_moving_transform_com=True,
    #     fixed_image=mni_template
    #     # Align centers of mass before registration
    # )

    # mni_template = Info.standard_image(
    #     'MNI152_T1_1mm.nii.gz')  # Get MNI template path from FSL
    # mni_template_brain = Info.standard_image(
    #     'MNI152_T1_1mm_brain.nii.gz')  # Get MNI template path from FSL
    # mni_template_mask = Info.standard_image(
    #     'MNI152_T1_1mm_brain_mask_dil.nii.gz')  # Get MNI template path from FSL

    # # quick
    #
    # mni_template = Info.standard_image(
    #     'MNI152_T1_2mm.nii.gz')  # Get MNI template path from FSL
    # mni_template_brain = Info.standard_image(
    #     'MNI152_T1_2mm_brain.nii.gz')  # Get MNI template path from FSL
    # mni_template_mask = Info.standard_image(
    #     'MNI152_T1_2mm_brain_mask_dil.nii.gz')  # Get MNI template path from FSL
    #
    # ants_reg_params = dict(
    #     dimension=3,  # 3D registration
    #     output_transform_prefix='output_prefix_',  # Prefix for output files
    #     transforms=['Rigid', 'Affine', 'SyN'],  # Rigid, affine, and SyN stages
    #     transform_parameters=[(0.1,), (0.1,), (0.1, 3, 0)],
    #     # Parameters for each transform
    #     metric=['MI', 'MI', 'CC'],  # MI for Rigid/Affine, CC for SyN
    #     metric_weight=[1, 1, 1],  # Equal weights for metrics
    #     radius_or_number_of_bins=[32, 32, 4],  # Bins for MI, radius for CC
    #     sampling_strategy=['Regular', 'Regular', None],
    #     # Regular sampling for MI, none for CC
    #     sampling_percentage=[0.1, 0.1, None],
    #     # Reduce sampling to 10% for faster MI
    #     convergence_threshold=[1e-3, 1e-3, 1e-3],
    #     # Relax convergence thresholds
    #     convergence_window_size=[5, 5, 5],
    #     # Smaller window sizes for convergence
    #     number_of_iterations=[[100, 50], [100, 50], [25, 10]],
    #     # Fewer iterations
    #     shrink_factors=[[4, 2], [4, 2], [2, 1]],  # Coarser shrink factors
    #     smoothing_sigmas=[[2, 1], [2, 1], [1, 0]],  # Reduced smoothing sigmas
    #     interpolation='Linear',  # Linear interpolation
    #     output_warped_image='output_warped_image_fast.nii.gz',
    #     # Output warped image
    #     output_inverse_warped_image='output_inverse_warped_image_fast.nii.gz',
    #     # Inverse warped image
    #     use_histogram_matching=True,
    #     # Histogram matching for multi-modal images
    #     winsorize_upper_quantile=0.995,  # Upper quantile for winsorization
    #     winsorize_lower_quantile=0.005,  # Lower quantile for winsorization
    #     initial_moving_transform_com=True,
    #     fixed_image=mni_template
    #     # fixed_image_masks=[mni_template_mask, mni_template_mask, mni_template_mask],
    #     # moving_image_masks=[brain_mask_file, brain_mask_file, brain_mask_file]
    #     # Align centers of mass before registration
    # )

    # custom
    # ants_reg_params = dict(
    #     dimension=3,  # 3D registration
    #     output_transform_prefix='output_prefix_',  # Prefix for output files
    #     transforms=['Rigid', 'Affine', 'SyN'],  # Transformation types
    #     transform_parameters=[(0.1,), (0.1,), (0.05, 3, 0)],
    #     # Parameters for each transform
    #     metric=['MI', 'MI', 'CC'],
    #     # Metrics for each stage: MI for Rigid/Affine, CC for SyN
    #     metric_weight=[1, 1, 0.5],  # Weights for the metrics
    #     radius_or_number_of_bins=[32, 32, 4],
    #     # Number of bins for MI and radius for CC
    #     sampling_strategy=['Regular', 'Regular', None],
    #     # Sampling strategies for each stage
    #     sampling_percentage=[0.25, 0.25, None],  # Sampling percentages for MI
    #     convergence_threshold=[1e-6, 1e-6, 1e-6],  # Convergence thresholds
    #     convergence_window_size=[10, 10, 10],  # Convergence window sizes
    #     number_of_iterations=[[1000, 500, 250, 100], [1000, 500, 250, 100],
    #                           [50, 30, 10, 5]],
    #     # Iterations for each resolution level
    #     shrink_factors=[[8, 4, 2, 1], [8, 4, 2, 1], [6, 4, 2, 1]],
    #     # Shrink factors for the multi-resolution scheme
    #     smoothing_sigmas=[[3, 2, 1, 0], [3, 2, 1, 0], [6, 4, 2, 1]],
    #     # Smoothing sigmas for the multi-resolution scheme
    #     interpolation='Linear',  # Linear interpolation
    #     output_warped_image='output_warped_image.nii.gz',  # Output warped image
    #     output_inverse_warped_image='output_inverse_warped_image.nii.gz',
    #     # Output inverse warped image
    #     use_histogram_matching=True,
    #     # Use histogram matching for multi-modal images
    #     winsorize_upper_quantile=0.995,
    #     # Winsorize image intensities (upper quantile)
    #     winsorize_lower_quantile=0.005,
    #     # Winsorize image intensities (lower quantile)
    #     initial_moving_transform_com=True,  # Align centers of mass
    #     fixed_image=mni_template,
    #     fixed_image_masks=["NULL", "NULL", mni_template_mask],
    #     moving_image_masks=["NULL", "NULL", brain_mask_file]
    # )

    register_t1w = pe.Node(ants.Registration(**ants_reg_params),
                           name="register_t1w")
    wf.connect(input_node, "t1w_reg_target_file", register_t1w, "moving_image")

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

    select_forward_affine_node = pe.Node(Select(index=1),
                                         name="select_forward_affine_node")
    wf.connect(register_t1w, "reverse_forward_transforms",
               select_forward_affine_node, "inlist")
    forward_affine_transform_writer = pe.Node(BidsOutputWriter(),
                                              name="forward_affine_transform_writer")
    forward_affine_transform_writer.inputs.output_dir = args.output_derivative_dir
    forward_affine_transform_writer.inputs.pattern = out_pattern
    forward_affine_transform_writer.inputs.entity_overrides = dict(part=None,
                                                                   acquisition=None,
                                                                   desc="SubToMni",
                                                                   suffix="transform",
                                                                   extension="mat")
    wf.connect(select_forward_affine_node, "out",
               forward_affine_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               forward_affine_transform_writer, "template_file")

    select_forward_warp_node = pe.Node(Select(index=0), name="select_warp_node")
    wf.connect(register_t1w, "reverse_forward_transforms",
               select_forward_warp_node, "inlist")
    forward_warp_transform_writer = pe.Node(BidsOutputWriter(),
                                            name="forward_warp_transform_writer")
    forward_warp_transform_writer.inputs.output_dir = args.output_derivative_dir
    forward_warp_transform_writer.inputs.pattern = out_pattern
    forward_warp_transform_writer.inputs.entity_overrides = dict(part=None,
                                                                 acquisition=None,
                                                                 desc="SubToMni",
                                                                 suffix="warp",
                                                                 extension="nii.gz")
    wf.connect(select_forward_warp_node, "out",
               forward_warp_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               forward_warp_transform_writer, "template_file")

    select_reverse_warp_node = pe.Node(Select(index=1),
                                       name="select_reverse_warp_node")
    wf.connect(register_t1w, "reverse_transforms",
               select_reverse_warp_node, "inlist")
    reverse_warp_transform_writer = pe.Node(BidsOutputWriter(),
                                            name="reverse_warp_transform_writer")
    reverse_warp_transform_writer.inputs.output_dir = args.output_derivative_dir
    reverse_warp_transform_writer.inputs.pattern = out_pattern
    reverse_warp_transform_writer.inputs.entity_overrides = dict(part=None,
                                                                 acquisition=None,
                                                                 desc="MniToSub",
                                                                 suffix="warp",
                                                                 extension="nii.gz")
    wf.connect(select_reverse_warp_node, "out",
               reverse_warp_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               reverse_warp_transform_writer, "template_file")

    run_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {'n_procs': args.n_procs}
    }

    # Run the workflow
    wf.run(**run_settings)
    # wf.run()


if __name__ == "__main__":
    main()
