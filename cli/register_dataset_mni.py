import argparse
import os
import multiprocessing
from nipype import Workflow
from nipype import Node, Workflow
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import FLIRT, FNIRT, ApplyWarp, Info
from datasets.dzne_three_dim_epi_dataset import DzneThreeDimEpiDataset
from datasets.kings_ssfp_dataset import KingsSsfpDataset
from datasets.uke_beat_ssfp_dataset import UkeBeatSsfpDataset
from datasets.uke_fibu_ssfp_dataset import UkeFibuSsfpDataset
from nipype.interfaces.ants import ApplyTransforms


def main():
    parser = argparse.ArgumentParser(
        description="Process a dataset with optional steps.")
    parser.add_argument('--dataset', required=True,
                        choices=['dzne', 'kings', 'ukebeat', 'ukefibu'],
                        help='Choose the dataset to process (e.g., dzne).')
    parser.add_argument('-i', '--bids_root', required=True,
                        help='Path to the BIDS root directory of the dataset.')
    parser.add_argument('-d', '--derivatives', nargs='+', required=True,
                        help='One or more derivatives directories to use.')
    parser.add_argument('-o', '--output_derivative_dir', required=True,
                        help='Path to the output derivatives folder.')
    parser.add_argument('--base_dir', default=os.getcwd(),
                        help='Base directory for processing (default: current working directory).')
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

    # Ensure `derivatives` is a list with one or more entries
    if not args.derivatives or len(args.derivatives) == 0:
        raise ValueError(
            "At least one derivatives directory must be specified with the -d option.")

    # Define the reusable run settings in a dictionary
    run_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {'n_procs': args.n_procs}
    }

    dataset_args = dict(
        bids_root=args.bids_root,
        derivatives_output_folder=args.output_derivative_dir,
        derivatives=args.derivatives
    )

    # Instantiate the selected dataset
    if args.dataset == 'dzne':
        dataset = DzneThreeDimEpiDataset(**dataset_args)
    elif args.dataset == 'kings':
        dataset = KingsSsfpDataset(**dataset_args)
    elif args.dataset == 'ukebeat':
        dataset = UkeBeatSsfpDataset(**dataset_args)
    elif args.dataset == 'ukefibu':
        dataset = UkeFibuSsfpDataset(**dataset_args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    sub_ses_run_combinations = dataset.get_subject_session_run_combinations(
        subject=args.subject, session=args.session, run=args.run)

    inputs = []
    for combination in sub_ses_run_combinations:
        input_dict = dict(
            subject=combination["subject"],
            session=combination["session"],
            run=combination["run"],
            t1_map_file=dataset.get_t1_map(**combination),
            t2_map_file=dataset.get_t2_map(**combination),
            t1w_image_file=dataset.get_t1w_fa_13_preprocessed(**combination)
        )
        inputs.append(input_dict)

    # Create a workflow
    wf = Workflow(name='register_t1_t2map_to_mni', base_dir=os.getcwd())
    wf.base_dir = args.base_dir

    # set up bids input node
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    # # Step 1: Register T1w image to MNI space using FLIRT
    # mni_template = Info.standard_image(
    #     'MNI152_T1_1mm.nii.gz')  # Get MNI template path from FSL
    #
    # flirt_t1 = Node(FLIRT(dof=12, reference=mni_template),
    #                 name='flirt_t1_to_mni')
    # wf.connect(input_node, 't1w_image_file', flirt_t1, 'in_file')
    #
    # # Step 2: Use FNIRT for non-linear registration (input is the FLIRT output)
    # fnirt = Node(FNIRT(ref_file=mni_template), name='fnirt')
    #
    # # Connect FLIRT's output to FNIRT's input
    # wf.connect(flirt_t1, 'out_file', fnirt, 'in_file')
    #
    # # Step 3: Apply warp to T2 map using the transformation from the T1w registration
    # apply_warp = Node(ApplyWarp(ref_file=mni_template), name='apply_warp')
    #
    # # Connect inputs
    # wf.connect(input_node, 't2_map_file', apply_warp, 'in_file')
    # wf.connect(fnirt, 'fieldcoeff_file', apply_warp, 'field_file')







    mni_template = Info.standard_image(
        'MNI152_T1_1mm.nii.gz')  # Get MNI template path from FSL

    # Step 1: ANTs Registration to MNI space (using defaults)
    ants_reg = Node(Registration(), name='ants_reg')
    ants_reg.inputs.fixed_image = mni_template# MNI Template
    # ants_reg.inputs.transforms = ['Affine','SyN']  # Affine and SyN for linear and non-linear registration
    ants_reg.inputs.output_transform_prefix = 'subject_to_mni_'  # Prefix for output transform files
    ants_reg.inputs.output_warped_image = 't1w_in_mni.nii.gz'  # Output T1w in MNI space
    ants_reg.inputs.collapse_output_transforms = True  # Single output transformation
    # Specify the transforms with their parameters
    ants_reg.inputs.transforms = ['Affine', 'SyN']  # Use both Affine and SyN
    ants_reg.inputs.transform_parameters = [(0.1,), (
    0.1, 3.0, 0.0)]  # Parameters for Affine and SyN
    ants_reg.inputs.num_threads = 11

    # Set up metrics for each transformation
    ants_reg.inputs.metric = ['MI',
                              'CC']  # Mutual Information (MI) for Affine, Correlation Coefficient (CC) for SyN
    ants_reg.inputs.metric_weight = [1, 1]  # Equal weighting
    ants_reg.inputs.radius_or_number_of_bins = [32,
                                                4]  # Number of bins for MI, radius for CC

    # Set smoothing sigmas for each resolution level
    ants_reg.inputs.smoothing_sigmas = [[4, 2, 1, 0], [3, 2, 1,
                                                       0]]  # Smoothing for affine and SyN

    # Set shrink factors for each resolution level
    ants_reg.inputs.shrink_factors = [[8, 4, 2, 1], [3, 2,
                                                     1]]  # Shrink factors for affine and SyN

    # Set number of iterations for each resolution level
    ants_reg.inputs.number_of_iterations = [[1000, 500, 250, 100], [100, 70,
                                                                    50]]  # Iterations for affine and SyN

    # Set convergence threshold
    ants_reg.inputs.convergence_threshold = [1e-6, 1e-6]
    # Set convergence threshold
    # ants_reg.inputs.convergence_threshold = [1e-6, 1e-6]

    # Connect the T1w image
    wf.connect(input_node, 't1w_image_file', ants_reg, 'moving_image')

    # Step 2: Apply forward transformation to T2 map
    apply_transforms = Node(ApplyTransforms(), name='apply_transforms')
    apply_transforms.inputs.reference_image = mni_template # MNI Template

    # Connect the forward transforms from ants_reg to apply_transforms
    wf.connect(ants_reg, 'forward_transforms', apply_transforms, 'transforms')

    # Connect the T2 map to be transformed to MNI space
    wf.connect(input_node, 't2_map_file', apply_transforms, 'input_image')

    # apply2con = MapNode(ApplyTransforms(args='--float',
    #                                     input_image_type=3,
    #                                     interpolation='BSpline',
    #                                     invert_transform_flags=[False],
    #                                     num_threads=1,
    #                                     reference_image=template,
    #                                     terminal_output='file'),
    #                     name='apply2con', iterfield=['input_image'])






    # run_settings = {
    #     'plugin': 'MultiProc',
    #     'plugin_args': {'n_procs': args.n_procs}
    # }

    # Run the workflow
    # wf.run(**run_settings)
    wf.run()


if __name__ == "__main__":
    main()
