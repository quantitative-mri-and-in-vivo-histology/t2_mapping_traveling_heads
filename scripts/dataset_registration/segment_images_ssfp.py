import argparse
import multiprocessing
import os
import shutil

import ants
import antspynet
import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from nipype import Function
from nipype import Node, Workflow
from nipype.interfaces.ants import Atropos, N4BiasFieldCorrection
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    TraitedSpec, File, traits)
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    isdefined)
from nipype.interfaces.fsl import Threshold
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.utility import Merge
from nipype.utils.filemanip import fname_presuffix

from nodes.io import BidsOutputWriter
from utils.bids_config import DEFAULT_NIFTI_READ_EXT_ENTITY, \
    DEFAULT_NIFTI_WRITE_EXT_ENTITY, \
    PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE, \
    PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK, \
    PROCESSED_ENTITY_OVERRIDES_T1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_T2W
from utils.io import write_minimal_bids_dataset_description, find_file
from nipype.interfaces.utility import Merge


class AntspynetBrainExtractionInputSpec(BaseInterfaceInputSpec):
    anatomical_image = File(exists=True,
                            desc="Input anatomical image (e.g., T1-weighted)",
                            mandatory=True)
    output_image = File(
        desc="Path to save the brain probability segmentation output",
        usedefault=True)
    modality = traits.Enum("t1", "t2", "flair",
                           desc="Modality of the anatomical image",
                           default="t1", usedefault=True)


class AntspynetBrainExtractionOutputSpec(TraitedSpec):
    output_image = File(exists=True,
                         desc="Path to the brain probability segmentation")


class AntspynetBrainExtraction(BaseInterface):
    input_spec = AntspynetBrainExtractionInputSpec
    output_spec = AntspynetBrainExtractionOutputSpec

    def _run_interface(self, runtime):
        # Load the input image using ANTs
        anatomical_image = ants.image_read(self.inputs.anatomical_image)

        # Set the default output path if not specified
        if not isdefined(self.inputs.output_image):
            self.inputs.output_image = os.path.join(os.getcwd(),
                                                    "brain_probseg.nii.gz")

        # Perform brain extraction using ANTsPyNet based on the specified modality
        brain_probseg = antspynet.brain_extraction(
            anatomical_image, modality=self.inputs.modality
        )

        # Save the output image
        brain_probseg.to_filename(self.inputs.output_image)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = self.inputs.output_image
        return outputs


# Define the InputSpec
class FslOrientSwapInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc="Input file to swap orientation", position=0,
                   argstr="%s")
    out_file = File("output_swaporient.nii.gz", usedefault=True,
                    desc="Output file after orientation swap", position=1,
                    argstr="%s")


# Define the OutputSpec
class FslOrientSwapOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output file after orientation swap")


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
    args = parser.parse_args()

    # write minimal dataset description for output derivatives
    os.makedirs(args.output_dir, exist_ok=True)
    write_minimal_bids_dataset_description(
        dataset_root=args.output_dir,
        dataset_name=os.path.dirname(args.output_dir)
    )

    # define the reusable run settings in a dictionary
    run_settings = dict(plugin='MultiProc',
                        plugin_args={'n_procs': args.n_procs})

    # collect inputs
    layout = BIDSLayout(args.input_dir,
                        derivatives=[args.derivatives],
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
                runs = [args.run] if args.run else layout.get_runs(
                    subject=subject, session=session)

                if len(runs) == 0:
                    runs = [None]

                for run in runs:
                    input_dict = dict(
                        subject=subject,
                        session=session,
                        run=run
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

                    # input_dict["t2w_file"] = find_file(
                    #     layout,
                    #     subject=subject,
                    #     session=session,
                    #     run=run,
                    #     space=None,
                    #     suffix="T2w",
                    #     acquisition="A1(2|3)RF180",
                    #     **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    #     regex_search=True,
                    # )

                    input_dict["t1w_reg_target_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["brain_mask_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space=None,
                        **PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK,
                        **DEFAULT_NIFTI_READ_EXT_ENTITY,
                    )

                    input_dict["sub_to_mni_transform_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  desc="SubjectToMNI152",
                                  suffix="transform",
                                  extension="mat")

                    input_dict["sub_to_mni_warp_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  desc="SubjectToMNI152",
                                  suffix="warp",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["mni_to_sub_warp_file"] = \
                        find_file(layout,
                                  subject=subject,
                                  session=session,
                                  run=run,
                                  desc="MNI152ToSubject",
                                  suffix="warp",
                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                    input_dict["wm_probseg_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space="subject",
                        label="wmPrior",
                        suffix="probseg",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY
                    )

                    input_dict["gm_probseg_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space="subject",
                        label="gmPrior",
                        suffix="probseg",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY
                    )

                    input_dict["csf_probseg_file"] = find_file(
                        layout,
                        subject=subject,
                        session=session,
                        run=run,
                        space="subject",
                        label="csfPrior",
                        suffix="probseg",
                        **DEFAULT_NIFTI_READ_EXT_ENTITY
                    )

                    inputs.append(input_dict)

    # set up workflow
    wf = Workflow(name='segment_images', base_dir=os.getcwd())
    wf.base_dir = args.temp_dir

    # set up bids input node
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    # collect tissue segmentation priors
    merge_priors_node = pe.Node(Merge(3), name="merge_priors_node")
    wf.connect(input_node, "wm_probseg_file", merge_priors_node, "in1")
    wf.connect(input_node, "gm_probseg_file", merge_priors_node, "in2")
    wf.connect(input_node, "csf_probseg_file", merge_priors_node, "in3")

    # write tissue segmentation priors in ants format i.e. 'prior_%02d.nii.gz'
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

    # create brain mask
    brain_extraction_node = pe.Node(AntspynetBrainExtraction(),
                                    name="brain_extraction")
    wf.connect(input_node, "t1w_reg_target_file",
               brain_extraction_node, "anatomical_image")
    brain_mask_threshold = pe.Node(Threshold(thresh=0.01),
                                   name="brain_mask_threshold")
    wf.connect(brain_extraction_node, "output_image",
               brain_mask_threshold, "in_file")

    # write brain_mask
    brain_mask_writer = pe.Node(BidsOutputWriter(),
                              name="brain_mask_writer")
    brain_mask_writer.inputs.output_dir = args.output_dir
    brain_mask_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
    brain_mask_writer.inputs.entity_overrides = dict(
        desc="brainForTissueSegmentation",
        space="subject",
        suffix="mask",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(brain_mask_threshold, "out_file",
               brain_mask_writer, "in_file")
    wf.connect(input_node, "brain_mask_file",
               brain_mask_writer, "template_file")

    # # remove low frequency bias
    # n4_bias_field_correction = pe.Node(N4BiasFieldCorrection(
    #     dimension=3
    # ),
    #     name="n4_bias_field_correction")
    # wf.connect(input_node, "t2w_file",
    #            n4_bias_field_correction, "input_image")
    #
    # merge_atropos_inputs = pe.Node(Merge(2), name="merge_atropos_inputs")
    # wf.connect(input_node, "t1_map_file",
    #            merge_atropos_inputs, "in1")
    # wf.connect(n4_bias_field_correction, "output_image",
    #            merge_atropos_inputs, "in2")

    # segment brain into wm, gm and csf
    atropos = Node(Atropos(), name='atropos')
    atropos.inputs.dimension = 3
    atropos.inputs.number_of_tissue_classes = 3
    atropos.inputs.prior_weighting = 0
    atropos.inputs.output_posteriors_name_template = 'posteriors%02d.nii.gz'
    atropos.inputs.initialization = 'PriorProbabilityImages'
    atropos.inputs.posterior_formulation = 'Socrates'
    atropos.inputs.mrf_smoothing_factor = 0.1
    atropos.inputs.mrf_radius = [1,1,1]
    atropos.inputs.likelihood_model = "Gaussian"
    atropos.inputs.prior_probability_threshold = 0
    atropos.inputs.convergence_threshold = 0
    atropos.inputs.n_iterations = 5
    atropos.inputs.save_posteriors = True
    wf.connect(copy_prior_node, "output_format_str", atropos, "prior_image")
    wf.connect(input_node, "t1_map_file", atropos, "intensity_images")
    wf.connect(brain_mask_threshold, "out_file", atropos, "mask_image")

    # write tissue segmentation probability maps
    tissue_probseg_entity_overrides_common = dict(
        suffix="probseg",
        acquisition=None,
        space="subject")
    tissue_probseg_entity_overrides = [
        dict(label="wm", **tissue_probseg_entity_overrides_common),
        dict(label="gm", **tissue_probseg_entity_overrides_common),
        dict(label="csf", **tissue_probseg_entity_overrides_common)
    ]
    tissue_segmentation_writer = pe.MapNode(BidsOutputWriter(),
                              name="tissue_segmentation_writer",
                              iterfield=["in_file", "entity_overrides"])
    tissue_segmentation_writer.inputs.output_dir = args.output_dir
    tissue_segmentation_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
    tissue_segmentation_writer.inputs.entity_overrides = tissue_probseg_entity_overrides
    wf.connect(atropos, "posteriors",
               tissue_segmentation_writer, "in_file")
    wf.connect(input_node, "brain_mask_file",
               tissue_segmentation_writer, "template_file")

    # Run the workflow
    wf.run(**run_settings)


if __name__ == "__main__":
    main()
