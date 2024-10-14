import argparse
import os
import multiprocessing
from nipype import Workflow
from nipype import Node, Workflow
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import FLIRT, FNIRT, ApplyWarp, Info
from datasets.dzne_three_dim_epi_dataset import DzneThreeDimEpiDataset
# from datasets.kings_ssfp_dataset import KingsSsfpDataset
# from datasets.uke_beat_ssfp_dataset import UkeBeatSsfpDataset
# from datasets.uke_fibu_ssfp_dataset import UkeFibuSsfpDataset
from nipype.interfaces.ants import ApplyTransforms
from bids.layout import BIDSLayout
import nipype.pipeline.engine as pe
import nipype.interfaces.mrtrix3 as mrtrix3
from nipype import Node, Function
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from nipype_utils import BidsRename, BidsOutputFormatter, BidsOutputWriter
from utils.io import write_minimal_bids_dataset_description


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
    subjects = ["phy002"]
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
                    assert(len(t1_map_files) == 1)
                    t1_map_file = t1_map_files[0]

                    t2_map_files = layout.get(subject=subject,
                                           session=session,
                                           suffix="T2map",
                                           extension="nii.gz",
                                           run=run)
                    assert(len(t2_map_files) == 1)
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

                    relaxation_maps = [r1_map_file, r2_map_file, t1_map_file, t2_map_file]
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
                                       t1_map_file=t1_map_file,
                                       t2_map_file=t2_map_file,
                                       r1_map_file=r1_map_file,
                                       r2_map_file=r2_map_file,
                                       relaxation_maps=relaxation_maps,
                                       relaxation_map_entities=relaxation_map_entities))

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

    ants_reg_params = dict(
        dimension=3,  # 3D images
        output_transform_prefix='output_prefix_',  # Prefix for outputs
        transforms=['Rigid', 'Affine', 'SyN'],
        # Start with rigid, then affine, then SyN (nonlinear)
        transform_parameters=[(0.1,), (0.1,), (0.1, 3, 0)],
        # Parameters for each transformation type
        metric=['MI', 'MI', 'CC'],
        # Mutual Information for rigid/affine, Cross Correlation for SyN
        metric_weight=[1, 1, 1],  # Weight for each metric
        radius_or_number_of_bins=[32, 32, 4],
        # Number of histogram bins for MI, patch radius for CC
        sampling_strategy=['Regular', 'Regular', None],
        # Regular sampling for MI, no sampling for CC
        sampling_percentage=[0.25, 0.25, None],  # Sampling percentage for MI
        convergence_threshold=[1e-6, 1e-6, 1e-6],
        # Convergence threshold for each stage
        convergence_window_size=[10, 10, 10],  # Window size for convergence
        number_of_iterations=[[500, 250, 100], [500, 250, 100], [200, 100, 50]],
        # More iterations for SyN
        shrink_factors=[[6, 4, 2], [6, 4, 2], [4, 2, 1]],
        # Shrink factors for each stage
        smoothing_sigmas=[[3, 2, 1], [3, 2, 1], [2, 1, 0]],
        # Smoothing sigmas for each stage
        interpolation='Linear',  # Linear interpolation
        output_warped_image='output_warped_image.nii.gz',
        fixed_image=mni_template  # Fixed image (e.g., MNI template)
    )

    register_t1w = pe.Node(ants.Registration(**ants_reg_params),
                              name="register_t1w",
                              iterfield=["moving_image"])
    wf.connect(input_node, "t1w_reg_target_file", register_t1w, "moving_image")

    # Define the ApplyTransforms node
    apply_transforms = pe.MapNode(ApplyTransforms(), name="apply_transforms", iterfield=["input_image"])
    apply_transforms.inputs.dimension = 3  # 3D image
    apply_transforms.inputs.reference_image = mni_template
    apply_transforms.inputs.interpolation = 'Linear'

    wf.connect(register_t1w, 'forward_transforms',
               apply_transforms, 'transforms')
    wf.connect(input_node, 'relaxation_maps',
               apply_transforms, 'input_image')


    out_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                  'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                  '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'

    t1w_reg_target_writer = pe.Node(BidsOutputWriter(),
                                     name="t1w_reg_target_writer")
    t1w_reg_target_writer.inputs.output_dir = args.output_derivative_dir
    t1w_reg_target_writer.inputs.pattern = out_pattern
    t1w_reg_target_writer.inputs.entity_overrides = dict(part=None, acquisition="T1wRef")
    wf.connect(register_t1w, "warped_image",
               t1w_reg_target_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_file",
               t1w_reg_target_writer, "template_file")


    map_writer = pe.MapNode(BidsOutputWriter(),
                                     name="map_writer",
                            iterfield=["in_file", "entity_overrides"])
    map_writer.inputs.output_dir = args.output_derivative_dir
    map_writer.inputs.pattern = out_pattern
    wf.connect(apply_transforms, "output_image",
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
