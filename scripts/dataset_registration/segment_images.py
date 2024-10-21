import argparse
import os
import shutil
import multiprocessing
import nibabel as nib
from nipype.interfaces.ants import Atropos
from nipype import Workflow, Node, Function
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import Info
from nipype.interfaces import fsl
from nipype.interfaces.ants import ApplyTransforms
from bids.layout import BIDSLayout
import nipype.pipeline.engine as pe
import nipype.interfaces.ants as ants
import nipype.interfaces.mrtrix3 as mrtrix3
from nipype_utils import BidsOutputWriter
from utils.io import write_minimal_bids_dataset_description
from nipype.interfaces.utility import Select
from nipype.interfaces.utility import Merge
from nipype.interfaces.base import TraitedSpec, CommandLineInputSpec, File, CommandLine, traits
from nipype import Node, Workflow
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    TraitedSpec, File, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.utility import Rename


# Define the InputSpec
class FslOrientSwapInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input file to swap orientation", position=0, argstr="%s")
    out_file = File("output_swaporient.nii.gz", usedefault=True, desc="Output file after orientation swap", position=1, argstr="%s")

# Define the OutputSpec
class FslOrientSwapOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output file after orientation swap")

# Define the custom CommandLine node for fslorient -swaporient
# Define the custom CommandLine node for fslorient -swaporient
class FslOrientSwap(CommandLine):
    _cmd = "fslorient -swaporient"
    input_spec = FslOrientSwapInputSpec
    output_spec = FslOrientSwapOutputSpec

    def _run_interface(self, runtime):
        # Copy the input file to the output file location
        out_file = self.inputs.out_file
        in_file = self.inputs.in_file

        if not isdefined(out_file):
            out_file = fname_presuffix(in_file, suffix='_swaporient',
                                       newpath=os.getcwd())

        # Copy the file to the output location
        shutil.copy(in_file, out_file)

        # Run fslorient -swaporient on the copied file
        self.inputs.in_file = out_file
        result = super(FslOrientSwap, self)._run_interface(runtime)

        return result

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


# Function to copy files and rename them, with automatic directory creation
def copy_and_rename_files(file_list, format_str, base_dir=None):
    import os
    import shutil

    if base_dir is None:
        base_dir = os.getcwd()

    # Automatically create an 'priors' subdirectory in the base directory
    output_dir = base_dir
    os.makedirs(output_dir,
                exist_ok=True)  # Create the directory if it doesn't exist

    output_format_str = os.path.join(output_dir, format_str)

    output_files = []
    for i, file_path in enumerate(file_list, start=1):
        # Generate the new filename based on the format_str
        new_filename = os.path.join(output_dir, format_str % i)
        # Copy the file to the new location with the new name
        shutil.copy(file_path, new_filename)
        output_files.append(new_filename)

    # Return the list of copied files and the output directory
    return output_files, output_format_str


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

    # entity_overrides = [
    #     dict(suffix="probseg", desc="subcortical", acquisition=None),
    #     dict(suffix="probseg", desc="cortical", acquisition=None),
    # ]

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

                    t1w_reg_target_files = layout.get(subject=subject,
                                                      session=session,
                                                      acquisition="T1wRef",
                                                      suffix="T1w",
                                                      extension="nii.gz",
                                                      run=run)
                    t1w_reg_target_files = [f for f in t1w_reg_target_files if
                                            "processed" in str(f)]
                    assert (len(t1w_reg_target_files) == 1)
                    t1w_reg_target_file = t1w_reg_target_files[0]

                    t1_map_files = layout.get(subject=subject,
                                              session=session,
                                              suffix="T1map",
                                              extension="nii.gz",
                                              run=run)
                    assert (len(t1_map_files) == 1)
                    t1_map_file = t1_map_files[0]

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

                    csf_probseg_files = layout.get(subject=subject,
                                                       session=session,
                                                       suffix="probseg",
                                                       desc="csf",
                                                       extension="nii.gz",
                                                       run=run)
                    assert (len(csf_probseg_files) == 1)
                    csf_probseg_file = csf_probseg_files[0]

                    gm_probseg_files = layout.get(subject=subject,
                                                   session=session,
                                                   suffix="probseg",
                                                   desc="gm",
                                                   extension="nii.gz",
                                                   run=run)
                    assert (len(gm_probseg_files) == 1)
                    gm_probseg_file = gm_probseg_files[0]

                    wm_probseg_files = layout.get(subject=subject,
                                                   session=session,
                                                   suffix="probseg",
                                                   desc="wm",
                                                   extension="nii.gz",
                                                   run=run)
                    assert (len(wm_probseg_files) == 1)
                    wm_probseg_file = wm_probseg_files[0]

                    entity_overrides = [
                        dict(suffix="probseg", desc="wmPosterior", acquisition=None),
                        dict(suffix="probseg", desc="gmPosterior", acquisition=None),
                        dict(suffix="probseg", desc="csfPosterior", acquisition=None)
                    ]

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       t1w_reg_target_file=t1w_reg_target_file,
                                       t1_map_file=t1_map_file,
                                       brain_mask_file=brain_mask_file,
                                       sub_to_mni_transform_file=sub_to_mni_transform_file,
                                       sub_to_mni_warp_file=sub_to_mni_warp_file,
                                       mni_to_sub_warp_file=mni_to_sub_warp_file,
                                       csf_probseg_file=csf_probseg_file,
                                       gm_probseg_file=gm_probseg_file,
                                       wm_probseg_file=wm_probseg_file,
                                       entity_overrides=entity_overrides))

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

    merge_priors_node = pe.Node(Merge(3), name="merge_priors_node")
    wf.connect(input_node, "wm_probseg_file", merge_priors_node, "in1")
    wf.connect(input_node, "gm_probseg_file", merge_priors_node, "in2")
    wf.connect(input_node, "csf_probseg_file", merge_priors_node, "in3")

    # Define the function node
    copy_prior_node = Node(Function(
        input_names=['file_list', 'format_str'],
        output_names=['output_files', 'output_format_str'],
        function=copy_and_rename_files
    ), name='copy_prior_node')
    copy_prior_node.inputs.format_str = 'prior_%02d.nii.gz'  # Format string
    wf.connect(merge_priors_node, "out", copy_prior_node, "file_list")

    dummy_prior_node = Node(IdentityInterface(fields=["prior_files"]),
                      name='dummy_prior_node')
    wf.connect(copy_prior_node, "output_files", dummy_prior_node, "prior_files")

    # Create the Atropos node
    atropos = Node(Atropos(), name='atropos')
    atropos.inputs.dimension = 3  # 3D image
    atropos.inputs.number_of_tissue_classes = 3  # Segment into 3 tissue types (CSF, gray matter, white matter)
    atropos.inputs.prior_weighting = 0.2  # Optional: Adjust if you have priors
    atropos.inputs.output_posteriors_name_template = 'posteriors%02d.nii.gz'  # Output template for the tissue class probability maps
    atropos.inputs.initialization = 'PriorProbabilityImages'
    atropos.inputs.posterior_formulation = 'Socrates'  # Try different formulations
    atropos.inputs.mrf_smoothing_factor = 0.1
    atropos.inputs.mrf_radius = [1, 1, 1]
    atropos.inputs.likelihood_model = "Gaussian"
    atropos.inputs.prior_probability_threshold = 0
    atropos.inputs.convergence_threshold = 0
    atropos.inputs.n_iterations = 5
    atropos.inputs.save_posteriors = True

    wf.connect(copy_prior_node, "output_format_str", atropos, "prior_image")
    wf.connect(input_node, "t1_map_file", atropos, "intensity_images")
    wf.connect(input_node, "brain_mask_file", atropos, "mask_image")


    # write outputs
    out_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                  'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                  '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'

    atlas_writer = pe.MapNode(BidsOutputWriter(),
                              name="atlas_writer",
                              iterfield=["in_file", "entity_overrides"])
    atlas_writer.inputs.output_dir = args.output_derivative_dir
    atlas_writer.inputs.pattern = out_pattern
    atlas_writer.inputs.entity_overrides = entity_overrides
    wf.connect(atropos, "posteriors",
               atlas_writer, "in_file")
    wf.connect(input_node, "brain_mask_file",
               atlas_writer, "template_file")

    run_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {'n_procs': args.n_procs}
    }

    # Run the workflow
    wf.run(**run_settings)


if __name__ == "__main__":
    main()
