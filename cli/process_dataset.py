import argparse
import os
import multiprocessing
from nipype import Workflow
from datasets.dzne_three_dim_epi_dataset import DzneThreeDimEpiDataset


def main():
    parser = argparse.ArgumentParser(description="Process a dataset with optional steps.")
    parser.add_argument('--dataset', required=True, choices=['dzne'],
                        help='Choose the dataset to process (e.g., dzne).')
    parser.add_argument('--prepare_data', action='store_true',
                        help='Prepare data for the selected dataset.')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess relaxation images for the selected dataset.')
    parser.add_argument('--estimate', action='store_true',
                        help='Estimate relaxation maps for the selected dataset.')
    parser.add_argument('--all', action='store_true',
                        help='Run all steps: prepare data, preprocess, and estimate.')
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
    parser.add_argument('--n_procs', type=int, default=multiprocessing.cpu_count(),
                        help='Number of processors to use (default: all available cores).')
    args = parser.parse_args()

    # Ensure `derivatives` is a list with one or more entries
    if not args.derivatives or len(args.derivatives) == 0:
        raise ValueError("At least one derivatives directory must be specified with the -d option.")

    # Instantiate the selected dataset
    if args.dataset == 'dzne':
        dataset = DzneThreeDimEpiDataset(
            bids_root=args.bids_root,
            derivatives_output_folder=args.output_derivative_dir,
            derivatives=args.derivatives
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Define a list to collect workflows to run
    workflows_to_run = []

    # Prepare the workflows based on the specified flags
    if args.all or args.prepare_data:
        print(f"Preparing data workflow for {args.dataset}...")
        prepare_workflow = dataset.prepare_data_workflow(
            base_dir=args.base_dir,
            subject=args.subject,
            session=args.session,
            run=args.run
        )
        workflows_to_run.append(prepare_workflow)

    if args.all or args.preprocess:
        print(f"Preparing relaxation images workflow for {args.dataset}...")
        preprocess_workflow = dataset.preprocess_relaxation_images_workflow(
            base_dir=args.base_dir,
            subject=args.subject,
            session=args.session,
            run=args.run
        )
        workflows_to_run.append(preprocess_workflow)

    if args.all or args.estimate:
        print(f"Preparing relaxation maps estimation workflow for {args.dataset}...")
        estimate_workflow = dataset.estimate_relaxation_maps_workflow(
            base_dir=args.base_dir,
            subject=args.subject,
            session=args.session,
            run=args.run
        )
        workflows_to_run.append(estimate_workflow)

    # Run the collected workflows sequentially using the specified number of processors
    for workflow in workflows_to_run:
        print(f"Running workflow: {workflow.name} with {args.n_procs} processors.")
        workflow.run('MultiProc', plugin_args={'n_procs': args.n_procs})

    print("Processing complete.")


if __name__ == "__main__":
    main()
