import argparse
import os
import multiprocessing
from nipype import Workflow
from datasets.dzne_three_dim_epi_dataset import DzneThreeDimEpiDataset
from datasets.kings_ssfp_dataset import KingsSsfpDataset
from datasets.uke_beat_ssfp_dataset import UkeBeatSsfpDataset
from datasets.uke_fibu_ssfp_dataset import UkeFibuSsfpDataset


def main():
    parser = argparse.ArgumentParser(
        description="Process a dataset with optional steps.")
    parser.add_argument('--dataset', required=True,
                        choices=['dzne', 'kings', 'ukebeat', 'ukefibu'],
                        help='Choose the dataset to process (e.g., dzne).')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare dataset for the selected dataset.')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess relaxation images for the selected dataset.')
    parser.add_argument('--estimate', action='store_true',
                        help='Estimate relaxation maps for the selected dataset.')
    parser.add_argument('--all', action='store_true',
                        help='Run all steps: prepare dataset, preprocess, and estimate.')
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

    # Run workflows in sequence but reuse the settings
    if args.all or args.prepare:
        print(f"Running data preparation workflow for {args.dataset}...")
        prepare_workflow = dataset.prepare_dataset_workflow(
            base_dir=args.base_dir,
            subject=args.subject,
            session=args.session,
            run=args.run
        )
        prepare_workflow.run(**run_settings)

    if args.all or args.preprocess:
        print(
            f"Running relaxation images preprocessing workflow for {args.dataset}...")
        preprocess_workflow = dataset.preprocess_workflow(
            base_dir=args.base_dir,
            subject=args.subject,
            session=args.session,
            run=args.run
        )
        preprocess_workflow.run(**run_settings)

    if args.all or args.estimate:
        print(
            f"Running relaxation maps estimation workflow for {args.dataset}...")
        estimate_workflow = dataset.estimate_workflow(
            base_dir=args.base_dir,
            subject=args.subject,
            session=args.session,
            run=args.run
        )
        estimate_workflow.run(**run_settings)

    print("Processing complete.")


if __name__ == "__main__":
    main()
