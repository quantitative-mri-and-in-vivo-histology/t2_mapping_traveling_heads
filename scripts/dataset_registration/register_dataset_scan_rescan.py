import argparse
import os
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
from nipype.interfaces.fsl import Threshold
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

                inputs.append(dict(subject=subject,
                                   session=session,
                                   t1w_scan_file=t1w_reg_target_files[0],
                                   t1w_rescan_file=t1w_reg_target_files[1],
                                   t1w_reg_target_files=t1w_reg_target_files))

    # Create a workflow
    wf = Workflow(name='register_scan_rescan', base_dir=os.getcwd())
    wf.base_dir = args.temp_dir

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
    wf.connect(input_node, "t1w_reg_target_files",
               apply_mask, "in_file")

    # generate template
    build_template = pe.Node(AntsBuildTemplate(),
                             name="build_template")
    build_template.inputs.iterations = 3
    wf.connect(apply_mask, "out_file",
               build_template, "input_images")

    # write brain_mask
    template_writer = pe.Node(BidsOutputWriter(),
                              name="template_writer",
                              iterfield=["in_file", "template_file"])
    template_writer.inputs.output_dir = args.output_dir
    template_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
    template_writer.inputs.entity_overrides = dict(
        run=None,
        acquisition=None,
        desc="template",
        space="midspaceRuns",
        suffix="T1w",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(build_template, "output_template",
               template_writer, "in_file")
    wf.connect(input_node, "t1w_scan_file",
               template_writer, "template_file")

    # mni_template = Info.standard_image(
    #     'MNI152_T1_1mm.nii.gz')  # Get MNI template path from FSL
    # mni_template_mask = Info.standard_image(
    #     'MNI152_T1_1mm_brain_mask_dil.nii.gz')  # Get MNI template path from FSL

    # compute registration
    register_t1w = pe.MapNode(
        create_default_ants_rigid_affine_syn_registration_node(),
        name="register_t1w", iterfield=["moving_image"])
    # register_t1w.inputs.fixed_image = mni_template
    # register_t1w.inputs.fixed_image_masks = ["NULL", "NULL",
    #                                          mni_template_mask]
    # register_t1w.inputs.use_histogram_matching = True
    wf.connect(apply_mask, "out_file",
               register_t1w, "moving_image")

    wf.connect(build_template, "output_template",
               register_t1w, "fixed_image")

    # get affine transform from list of transforms
    select_forward_affine_node = pe.MapNode(Select(index=1),
                                            name="select_forward_affine_node",
                                            iterfield=["inlist"])
    wf.connect(register_t1w, "reverse_forward_transforms",
               select_forward_affine_node, "inlist")

    # write affine transform
    forward_affine_transform_writer = pe.MapNode(BidsOutputWriter(),
                                                 name="forward_affine_transform_writer",
                                                 iterfield=["in_file",
                                                            "template_file"])
    forward_affine_transform_writer.inputs.output_dir = args.output_dir
    forward_affine_transform_writer.inputs.entity_overrides = dict(
        part=None,
        acquisition=None,
        desc="subjectToMidspaceRuns",
        suffix="transform",
        extension="mat")
    wf.connect(select_forward_affine_node, "out",
               forward_affine_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_files",
               forward_affine_transform_writer, "template_file")

    # extract forward warp from list of transforms
    select_forward_warp_node = pe.MapNode(Select(index=0),
                                          name="select_warp_node",
                                          iterfield=["inlist"])
    wf.connect(register_t1w, "reverse_forward_transforms",
               select_forward_warp_node, "inlist")

    # write forward warp
    forward_warp_transform_writer = pe.MapNode(BidsOutputWriter(),
                                               name="forward_warp_transform_writer",
                                               iterfield=["in_file",
                                                          "template_file"])
    forward_warp_transform_writer.inputs.output_dir = args.output_dir
    forward_warp_transform_writer.inputs.entity_overrides = dict(part=None,
                                                                 acquisition=None,
                                                                 desc="subjectToMidspaceRuns",
                                                                 suffix="warp",
                                                                 **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(select_forward_warp_node, "out",
               forward_warp_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_files",
               forward_warp_transform_writer, "template_file")

    # select reverse warp from list of transforms
    select_reverse_warp_node = pe.MapNode(Select(index=1),
                                       name="select_reverse_warp_node",
                                          iterfield=["inlist"])
    wf.connect(register_t1w, "reverse_transforms",
               select_reverse_warp_node, "inlist")

    # write reverse warp
    reverse_warp_transform_writer = pe.MapNode(BidsOutputWriter(),
                                            name="reverse_warp_transform_writer",
                                            iterfield=["in_file",
                                                       "template_file"])
    reverse_warp_transform_writer.inputs.output_dir = args.output_dir
    reverse_warp_transform_writer.inputs.entity_overrides = dict(part=None,
                                                                 acquisition=None,
                                                                 desc="midspaceRunsToSubject",
                                                                 suffix="warp",
                                                                 **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(select_reverse_warp_node, "out",
               reverse_warp_transform_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_files",
               reverse_warp_transform_writer, "template_file")


    # write brain_mask
    t1w_reg_target_writer = pe.MapNode(BidsOutputWriter(),
                              name="t1w_reg_target_writer",
                              iterfield=["in_file", "template_file"])
    t1w_reg_target_writer.inputs.output_dir = args.output_dir
    t1w_reg_target_writer.inputs.pattern = REGISTRATION_BIDS_OUTPUT_PATTERN
    t1w_reg_target_writer.inputs.entity_overrides = dict(
        acquisition=None,
        space="midspaceRuns",
        suffix="T1w",
        **DEFAULT_NIFTI_WRITE_EXT_ENTITY)
    wf.connect(register_t1w, "warped_image",
               t1w_reg_target_writer, "in_file")
    wf.connect(input_node, "t1w_reg_target_files",
               t1w_reg_target_writer, "template_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
