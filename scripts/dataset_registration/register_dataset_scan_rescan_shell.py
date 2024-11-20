import argparse
import os
import re
import multiprocessing
import ants
import antspynet
from nipype.interfaces.fsl import ApplyMask
from nipype import Workflow, Node, Function
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import Info
from nipype.interfaces.ants import ApplyTransforms
from bids.layout import BIDSLayout
import nipype.pipeline.engine as pe
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    TraitedSpec, File, traits)
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    isdefined)
from nipype.interfaces.fsl import Threshold, ImageMaths
from nodes.io import BidsOutputWriter
from utils.io import write_minimal_bids_dataset_description
from nipype.interfaces.utility import Select
from utils.bids_config import DEFAULT_NIFTI_READ_EXT_ENTITY, \
    DEFAULT_NIFTI_WRITE_EXT_ENTITY, \
    PROCESSED_ENTITY_OVERRIDES_R1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_R2_MAP, \
    PROCESSED_ENTITY_OVERRIDES_T1_MAP, \
    PROCESSED_ENTITY_OVERRIDES_T2_MAP, \
    PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE, \
    PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, InputMultiPath,
    traits, isdefined
)
from nipype.utils.filemanip import fname_presuffix
import ants
import os
from nodes.registration import \
    create_default_ants_rigid_affine_syn_registration_node
from nipype.interfaces.utility import Function


from nipype.interfaces.base import (
    TraitedSpec, CommandLineInputSpec, CommandLine, traits, File
)
import os


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


class AntsBuildTemplateInputSpec(BaseInterfaceInputSpec):
    input_images = InputMultiPath(
        File(exists=True),
        desc="List of input images to build the template from",
        mandatory=True
    )
    output_template = File(
        desc="Path to save the final template image",
        usedefault=True
    )
    iterations = traits.Int(
        desc="Number of iterations for template construction",
        default_value=5,
        usedefault=True
    )
    gradient_step = traits.Float(
        desc="Gradient step size for template construction",
        default_value=0.2,
        usedefault=True
    )


class AntsBuildTemplateOutputSpec(TraitedSpec):
    output_template = File(exists=True, desc="Path to the final template image")


class AntsBuildTemplate(BaseInterface):
    input_spec = AntsBuildTemplateInputSpec
    output_spec = AntsBuildTemplateOutputSpec

    def _run_interface(self, runtime):
        # Load input images using ANTs
        images = [ants.image_read(img) for img in self.inputs.input_images]

        # Set the default output path if not specified
        if not isdefined(self.inputs.output_template):
            self.inputs.output_template = os.path.join(
                os.getcwd(), "template_image.nii.gz"
            )

        # Perform template creation using ants.build_template
        template = ants.build_template(
            image_list=images,
            iterations=self.inputs.iterations,
            gradient_step=self.inputs.gradient_step
        )

        # Save the template image
        template.to_filename(self.inputs.output_template)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_template'] = self.inputs.output_template
        return outputs


class PairwiseRegistrationInputSpec(CommandLineInputSpec):
    moving_image_a = File(
        exists=True, mandatory=True, desc="First moving image", position=-2,
        argstr="%s"
    )
    moving_image_b = File(
        exists=True, mandatory=True, desc="Second moving image", position=-1,
        argstr="%s"
    )
    output_prefix = traits.Str(
        "out_", desc="Prefix for output files", argstr="-o %s", usedefault=True
    )
    iterations = traits.Int(
        4, desc="Number of iterations for template construction",
        argstr="-i %d", usedefault=True
    )
    dimensions = traits.Enum(
        2, 3, desc="Image dimension (2 or 3)", argstr="-d %d", mandatory=True
    )


class PairwiseRegistrationOutputSpec(TraitedSpec):
    template_image = File(desc="Generated template image")
    affine_transform_a = File(desc="Affine transform for moving image A")
    forward_warp_a = File(desc="Forward warp transform for moving image A")
    inverse_warp_a = File(desc="Inverse warp transform for moving image A")
    warped_to_template_a = File(desc="Warped image A to template")
    affine_transform_b = File(desc="Affine transform for moving image B")
    forward_warp_b = File(desc="Forward warp transform for moving image B")
    inverse_warp_b = File(desc="Inverse warp transform for moving image B")
    warped_to_template_b = File(desc="Warped image B to template")



class PairwiseRegistration(CommandLine):
    # _cmd = "antsMultivariateTemplateConstruction2.sh -n 0 -f 6x4x2x1 -q 50x30x20x10"  #
    _cmd = "antsMultivariateTemplateConstruction2.sh -n 0"  #
    input_spec = PairwiseRegistrationInputSpec
    output_spec = PairwiseRegistrationOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        output_prefix = self.inputs.output_prefix
        base_dir = os.getcwd()  # Use current working directory for outputs

        def find_file_with_pattern(pattern):
            """Helper function to search for a file matching a regex pattern."""
            for file in os.listdir(base_dir):
                if re.search(pattern, file):
                    return os.path.join(base_dir, file)
            return None

        # Generate regex patterns based on original filenames, ignoring prefixes
        filename_a = os.path.basename(self.inputs.moving_image_a).replace(".nii.gz", "")
        filename_b = os.path.basename(self.inputs.moving_image_b).replace(".nii.gz", "")

        # Define patterns for each output type
        outputs['template_image'] = find_file_with_pattern(rf"{output_prefix}template0\.nii\.gz")

        # Patterns for moving_image_a
        outputs['affine_transform_a'] = find_file_with_pattern(rf"{output_prefix}.*{filename_a}.*0GenericAffine\.mat")
        outputs['forward_warp_a'] = find_file_with_pattern(rf"{output_prefix}.*{filename_a}.*1Warp\.nii\.gz")
        outputs['inverse_warp_a'] = find_file_with_pattern(rf"{output_prefix}.*{filename_a}.*1InverseWarp\.nii\.gz")
        outputs['warped_to_template_a'] = find_file_with_pattern(rf"{output_prefix}.*{filename_a}.*WarpedToTemplate\.nii\.gz")

        # Patterns for moving_image_b
        outputs['affine_transform_b'] = find_file_with_pattern(rf"{output_prefix}.*{filename_b}.*0GenericAffine\.mat")
        outputs['forward_warp_b'] = find_file_with_pattern(rf"{output_prefix}.*{filename_b}.*1Warp\.nii\.gz")
        outputs['inverse_warp_b'] = find_file_with_pattern(rf"{output_prefix}.*{filename_b}.*1InverseWarp\.nii\.gz")
        outputs['warped_to_template_b'] = find_file_with_pattern(rf"{output_prefix}.*{filename_b}.*WarpedToTemplate\.nii\.gz")

        return outputs


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
                runs = layout.get_runs(subject=subject, session=session)

                if len(runs) != 2:
                    continue

                t1w_reg_target_files = layout.get(subject=subject,
                                                  session=session,
                                                  space=None,
                                                  **PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE,
                                                  **DEFAULT_NIFTI_READ_EXT_ENTITY)

                assert (len(t1w_reg_target_files) == 2)

                r2_map_files = layout.get(subject=subject,
                                          session=session,
                                          space=None,
                                          **PROCESSED_ENTITY_OVERRIDES_R2_MAP,
                                          **DEFAULT_NIFTI_READ_EXT_ENTITY)

                assert (len(r2_map_files) == 2)

                inputs.append(dict(subject=subject,
                                   session=session,
                                   t1w_scan_file=t1w_reg_target_files[0],
                                   t1w_rescan_file=t1w_reg_target_files[1],
                                   t1w_reg_target_files=t1w_reg_target_files,
                                   r2_map_files=r2_map_files
                                   ))

    # Create a workflow
    wf = Workflow(name='register_scan_rescan', base_dir=os.getcwd())
    wf.base_dir = args.temp_dir
    wf.config["remove_unnecessary_outputs"] = False

    # set up bids input node
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='bids_input_node')
    keys = inputs[0].keys()
    input_node.iterables = [
        (key, [input_dict[key] for input_dict in inputs]) for key in keys]
    input_node.synchronize = True

    # create brain mask
    brain_extraction_node = pe.MapNode(AntspynetBrainExtraction(),
                                       name="brain_extraction",
                                       iterfield=["anatomical_image"])
    wf.connect(input_node, "t1w_reg_target_files",
               brain_extraction_node, "anatomical_image")
    brain_mask_threshold = pe.MapNode(Threshold(thresh=0.01, args="-bin"),
                                      name="brain_mask_threshold",
                                      iterfield=["in_file"])
    wf.connect(brain_extraction_node, "output_image",
               brain_mask_threshold, "in_file")

    # write brain_mask
    brain_mask_writer = pe.MapNode(BidsOutputWriter(),
                                   name="brain_mask_writer",
                                   iterfield=["in_file", "template_file"])
    brain_mask_writer.inputs.output_dir = args.output_dir
    brain_mask_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
    brain_mask_writer.inputs.entity_overrides = dict(
        acquisition=None,
        desc="brainTight",
        space="subject",
        suffix="mask",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(brain_mask_threshold, "out_file",
               brain_mask_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_files",
               brain_mask_writer, "template_file")

    # apply mask
    apply_mask = pe.MapNode(ApplyMask(),
                            name="apply_mask",
                            iterfield=["in_file", "mask_file"])
    wf.connect(brain_mask_threshold, "out_file",
               apply_mask, "mask_file")
    wf.connect(input_node, "r2_map_files",
               apply_mask, "in_file")

    # # Define the node
    # clip_outliers_node = pe.MapNode(
    #     Function(
    #         input_names=["in_file", "lower_percentile", "upper_percentile"],
    #         output_names=["out_file"],
    #         function=clip_outliers
    #     ),
    #     name="clip_outliers_node",
    #     iterfield=["in_file"]
    # )
    # clip_outliers_node.inputs.lower_percentile = 1
    # clip_outliers_node.inputs.lower_percentile = 98
    # wf.connect(apply_mask, "out_file",
    #            clip_outliers_node, "in_file")

    erosion_node = pe.MapNode(ImageMaths(op_string="-ero -ero"),
                        name="erosion_node",
                              iterfield=["in_file"])
    wf.connect(apply_mask, "out_file",
               erosion_node, "in_file")

    select_r2_map_scan = pe.Node(Select(index=0),
                                 name="select_r2_map_scan")
    wf.connect(erosion_node, "out_file",
               select_r2_map_scan, "inlist")

    select_r2_map_rescan = pe.Node(Select(index=1),
                                   name="select_r2_map_rescan")
    wf.connect(erosion_node, "out_file",
               select_r2_map_rescan, "inlist")



    pairwise_registration = pe.Node(PairwiseRegistration(
        dimensions=3,
        iterations=1
    ),
        name="pairwise_registration")
    wf.connect(select_r2_map_scan, "out",
               pairwise_registration, "moving_image_a")
    wf.connect(select_r2_map_rescan, "out",
               pairwise_registration, "moving_image_b")

    # write scan affine transform
    template_writer = pe.Node(BidsOutputWriter(),
                                                   name="template_writer")
    template_writer.inputs.output_dir = args.output_dir
    template_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
    template_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        run=None,
        space="midspaceRuns",
        desc="template",
        suffix="R2map",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(pairwise_registration, "template_image",
               template_writer, "in_file")
    wf.connect(input_node, "t1w_scan_file",
               template_writer, "template_file")

    # write scan affine transform
    scan_forward_affine_transform_writer = pe.Node(BidsOutputWriter(),
                                                   name="scan_forward_affine_transform_writer")
    scan_forward_affine_transform_writer.inputs.output_dir = args.output_dir
    scan_forward_affine_transform_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        run=1,
        desc="subjectToMidspaceRuns",
        suffix="transform",
        extension="mat")
    wf.connect(pairwise_registration, "affine_transform_a",
               scan_forward_affine_transform_writer, "in_file")
    wf.connect(input_node, "t1w_scan_file",
               scan_forward_affine_transform_writer, "template_file")

    # write rescan affine transform
    rescan_forward_affine_transform_writer = pe.Node(BidsOutputWriter(),
                                                     name="rescan_forward_affine_transform_writer")
    rescan_forward_affine_transform_writer.inputs.output_dir = args.output_dir
    rescan_forward_affine_transform_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        run=2,
        desc="subjectToMidspaceRuns",
        suffix="transform",
        extension="mat")
    wf.connect(pairwise_registration, "affine_transform_b",
               rescan_forward_affine_transform_writer, "in_file")
    wf.connect(input_node, "t1w_rescan_file",
               rescan_forward_affine_transform_writer, "template_file")

    # write forward warp
    scan_forward_warp_writer = pe.Node(BidsOutputWriter(),
                                       name="scan_forward_warp_writer")
    scan_forward_warp_writer.inputs.output_dir = args.output_dir
    scan_forward_warp_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        run=1,
        desc="subjectToMidspaceRuns",
        suffix="warp",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(pairwise_registration, "forward_warp_a",
               scan_forward_warp_writer, "in_file")
    wf.connect(input_node, "t1w_scan_file",
               scan_forward_warp_writer, "template_file")

    # write rescan forward warp
    rescan_forward_warp_writer = pe.Node(BidsOutputWriter(),
                                       name="rescan_forward_warp_writer")
    rescan_forward_warp_writer.inputs.output_dir = args.output_dir
    rescan_forward_warp_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        run=2,
        desc="subjectToMidspaceRuns",
        suffix="warp",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(pairwise_registration, "forward_warp_b",
               rescan_forward_warp_writer, "in_file")
    wf.connect(input_node, "t1w_scan_file",
               rescan_forward_warp_writer, "template_file")

    # write scan inverse warp
    scan_inverse_warp_writer = pe.Node(BidsOutputWriter(),
                                       name="scan_inverse_warp_writer")
    scan_inverse_warp_writer.inputs.output_dir = args.output_dir
    scan_inverse_warp_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        run=1,
        desc="midspaceRunsToSubject",
        suffix="warp",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(pairwise_registration, "inverse_warp_a",
               scan_inverse_warp_writer, "in_file")
    wf.connect(input_node, "t1w_scan_file",
               scan_inverse_warp_writer, "template_file")

    # write rescan inverse warp
    rescan_inverse_warp_writer = pe.Node(BidsOutputWriter(),
                                       name="rescan_inverse_warp_writer")
    rescan_inverse_warp_writer.inputs.output_dir = args.output_dir
    rescan_inverse_warp_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        run=2,
        desc="midspaceRunsToSubject",
        suffix="warp",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(pairwise_registration, "inverse_warp_b",
               rescan_inverse_warp_writer, "in_file")
    wf.connect(input_node, "t1w_scan_file",
               rescan_inverse_warp_writer, "template_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
