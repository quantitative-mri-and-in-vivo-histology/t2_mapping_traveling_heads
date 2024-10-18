import os
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nipype.interfaces.mrtrix3 as mrtrix3
from nipype import Node, Function
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from utils.processing import QiTgv, subtract_background_phase
from nipype_utils import ApplyXfm4D, get_common_parent_directory


from nipype.interfaces.ants.segmentation import BrainExtraction, BrainExtractionInputSpec
from nipype.interfaces.base import File, traits

class CustomBrainExtractionInputSpec(BrainExtractionInputSpec):
    initial_transform = File(
        exists=True,
        argstr="-r %s",
        desc="Initial transform to guide brain extraction.",
        mandatory=False,
    )

class CustomBrainExtraction(BrainExtraction):
    input_spec = CustomBrainExtractionInputSpec



def denoise_mag_and_phase_in_complex_domain_workflow(base_dir=os.getcwd(),
                                                     name="denoise_mag_phase_complex"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir
    input_node = pe.Node(interface=util.IdentityInterface(
        fields=['magnitude_file', 'phase_file']), name='input_node')
    output_node = pe.Node(interface=util.IdentityInterface(
        fields=['magnitude_file', 'phase_file']), name='output_node')

    convert_mag_and_phase_to_complex = pe.Node(fsl.Complex(),
                                               "convert_mag_and_phase_to_complex")
    convert_mag_and_phase_to_complex.interface.inputs.complex_polar = True

    denoise = pe.Node(interface=mrtrix3.DWIDenoise(), name='dwidenoise')
    denoise.inputs.extent = (3, 3, 3)

    convert_complex_to_mag_and_phase = pe.Node(fsl.Complex(),
                                               "convert_complex_to_mag_and_phase")
    convert_complex_to_mag_and_phase.interface.inputs.real_polar = True

    workflow.connect(input_node, 'magnitude_file',
                     convert_mag_and_phase_to_complex, 'magnitude_in_file')
    workflow.connect(input_node, 'phase_file', convert_mag_and_phase_to_complex,
                     'phase_in_file')
    workflow.connect(convert_mag_and_phase_to_complex, 'complex_out_file',
                     denoise, 'in_file')
    workflow.connect(denoise, 'out_file', convert_complex_to_mag_and_phase,
                     'complex_in_file')
    workflow.connect(convert_complex_to_mag_and_phase, 'phase_out_file',
                     output_node, 'phase_file')
    workflow.connect(convert_complex_to_mag_and_phase, 'magnitude_out_file',
                     output_node, 'magnitude_file')

    return workflow


def motion_correction_mag_and_phase_workflow(base_dir=os.getcwd(),
                                             name="motion_correction_mag_and_phase"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir
    input_node = pe.Node(interface=util.IdentityInterface(
        fields=['magnitude_file', 'phase_file']), name='input_node')
    output_node = pe.Node(interface=util.IdentityInterface(
        fields=['magnitude_file', 'phase_file']), name='output_node')

    mcflirt = pe.Node(fsl.preprocess.MCFLIRT(), name='mcflirt')
    mcflirt.inputs.ref_vol = 0
    mcflirt.inputs.save_mats = True
    mcflirt.inputs.cost = 'mutualinfo'

    convert_mag_and_phase_to_complex = pe.Node(fsl.Complex(),
                                               "convert_mag_and_phase_to_complex")
    convert_mag_and_phase_to_complex.interface.inputs.complex_polar = True

    convert_complex_to_real_cartesian = pe.Node(fsl.Complex(),
                                                "convert_complex_to_real_cartesian")
    convert_complex_to_real_cartesian.interface.inputs.real_cartesian = True

    get_mcflirt_trans_dir = Node(
        Function(input_names=["file_list"], output_names=["trans_dir"],
                 function=get_common_parent_directory),
        name="get_mcflirt_trans_dir")

    applyxfm4d_to_real = pe.Node(ApplyXfm4D(), "applyxfm4d_to_real")
    applyxfm4d_to_real.inputs.four_digit = True
    applyxfm4d_to_imag = pe.Node(ApplyXfm4D(), "applyxfm4d_to_imag")
    applyxfm4d_to_imag.inputs.four_digit = True

    complex_conv_moco = pe.Node(fsl.Complex(), "complex_conv_moco")
    complex_conv_moco.interface.inputs.complex_cartesian = True

    convert_mag_and_phase_to_complex_post_moco = pe.Node(fsl.Complex(),
                                                         "convert_mag_and_phase_to_complex_post_moco")
    convert_mag_and_phase_to_complex_post_moco.interface.inputs.real_polar = True

    copy_geometry_mag = pe.Node(fsl.CopyGeom(), name="copy_geometry_mag")
    copy_geometry_phase = pe.Node(fsl.CopyGeom(), name="copy_geometry_phase")

    workflow.connect(input_node, 'magnitude_file', mcflirt, 'in_file')
    workflow.connect(input_node, 'magnitude_file',
                     convert_mag_and_phase_to_complex, 'magnitude_in_file')
    workflow.connect(input_node, 'phase_file', convert_mag_and_phase_to_complex,
                     'phase_in_file')
    workflow.connect(convert_mag_and_phase_to_complex, 'complex_out_file',
                     convert_complex_to_real_cartesian, 'complex_in_file')
    workflow.connect(mcflirt, 'mat_file', get_mcflirt_trans_dir, 'file_list')
    workflow.connect(convert_complex_to_real_cartesian, 'real_out_file',
                     applyxfm4d_to_real, 'in_file')
    workflow.connect(input_node, 'magnitude_file', applyxfm4d_to_real,
                     'ref_vol')
    workflow.connect(get_mcflirt_trans_dir, 'trans_dir', applyxfm4d_to_real,
                     'trans_dir')
    workflow.connect(convert_complex_to_real_cartesian, 'imaginary_out_file',
                     applyxfm4d_to_imag, 'in_file')
    workflow.connect(input_node, 'magnitude_file', applyxfm4d_to_imag,
                     'ref_vol')
    workflow.connect(get_mcflirt_trans_dir, 'trans_dir', applyxfm4d_to_imag,
                     'trans_dir')
    workflow.connect(applyxfm4d_to_real, 'out_file', complex_conv_moco,
                     'real_in_file')
    workflow.connect(applyxfm4d_to_imag, 'out_file', complex_conv_moco,
                     'imaginary_in_file')
    workflow.connect(complex_conv_moco, 'complex_out_file',
                     convert_mag_and_phase_to_complex_post_moco,
                     'complex_in_file')
    workflow.connect(input_node, 'magnitude_file', copy_geometry_mag, 'in_file')
    workflow.connect(convert_mag_and_phase_to_complex_post_moco,
                     'magnitude_out_file', copy_geometry_mag, 'dest_file')
    workflow.connect(input_node, 'phase_file', copy_geometry_phase, 'in_file')
    workflow.connect(convert_mag_and_phase_to_complex_post_moco,
                     'phase_out_file', copy_geometry_phase, 'dest_file')
    workflow.connect(copy_geometry_mag, 'out_file', output_node,
                     'magnitude_file')
    workflow.connect(copy_geometry_phase, 'out_file', output_node, 'phase_file')

    return workflow


def create_brain_mask_workflow(base_dir=os.getcwd(), name="create_brain_mask"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir
    input_node = pe.Node(interface=util.IdentityInterface(
        fields=['in_file']),
        name='input_node')
    output_node = pe.Node(interface=util.IdentityInterface(
        fields=['out_file']),
        name='output_node')

    first_volume_extractor = Node(fsl.ExtractROI(),
                                  name="first_volume_extractor")
    first_volume_extractor.inputs.t_min = 0
    first_volume_extractor.inputs.t_size = 1
    bet_node = Node(fsl.BET(), name="bet")
    bet_node.inputs.robust = True
    bet_node.inputs.mask = True

    workflow.connect(input_node, "in_file",
                     first_volume_extractor, "in_file")
    workflow.connect(first_volume_extractor, "roi_file",
                     bet_node, "in_file")
    workflow.connect(bet_node, "mask_file",
                     output_node, "out_file")
    return workflow


def register_image_workflow(base_dir=os.getcwd(), name="register_image",
                            apply_brain_masking=False):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir

    input_node = pe.Node(util.IdentityInterface(
        fields=['moving_file', 'reference_file', 'target_file']),
        name='input_node')

    output_node = pe.Node(util.IdentityInterface(fields=['out_file']),
                          name='output_node')

    flirt_estimate = pe.Node(fsl.FLIRT(uses_qform=True, dof=6),
                             "flirt_estimate")
    flirt_apply = pe.Node(fsl.FLIRT(apply_xfm=True, uses_qform=True, dof=6),
                          "flirt_apply")

    first_volume_extractor = Node(fsl.ExtractROI(),
                                  name="first_volume_extractor")
    first_volume_extractor.inputs.t_min = 0
    first_volume_extractor.inputs.t_size = 1

    if apply_brain_masking:
        bet_target = Node(fsl.BET(), name="bet_target")
        bet_target.inputs.robust = True

        bet_reference = Node(fsl.BET(), name="bet_reference")
        bet_reference.inputs.robust = True

        workflow.connect(input_node, "reference_file", bet_reference, "in_file")
        workflow.connect(input_node, "target_file", bet_target, "in_file")

        workflow.connect(bet_target, "out_file", first_volume_extractor,
                         "in_file")
        workflow.connect(bet_reference, "out_file", flirt_estimate, "in_file")
        workflow.connect(first_volume_extractor, "roi_file", flirt_estimate,
                         "reference")

        workflow.connect(input_node, "moving_file", flirt_apply, "in_file")
        workflow.connect(first_volume_extractor, "roi_file", flirt_apply,
                         "reference")
        workflow.connect(flirt_estimate, "out_matrix_file", flirt_apply,
                         "in_matrix_file")

        workflow.connect(flirt_apply, "out_file", output_node, "out_file")
    else:
        workflow.connect(input_node, "target_file", first_volume_extractor,
                         "in_file")
        workflow.connect(input_node, "reference_file", flirt_estimate,
                         "in_file")
        workflow.connect(first_volume_extractor, "roi_file", flirt_estimate,
                         "reference")

        workflow.connect(input_node, "moving_file", flirt_apply, "in_file")
        workflow.connect(first_volume_extractor, "roi_file", flirt_apply,
                         "reference")
        workflow.connect(flirt_estimate, "out_matrix_file", flirt_apply,
                         "in_matrix_file")

        workflow.connect(flirt_apply, "out_file", output_node, "out_file")

    return workflow


def register_b1_map_to_t2w(base_dir=os.getcwd(), name="register_b1_map_to_t2w"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir
    input_node = pe.Node(interface=util.IdentityInterface(
        fields=['b1_map_file', 'b1_anat_ref_file', 'target_file']),
        name='input_node')
    output_node = pe.Node(
        interface=util.IdentityInterface(fields=['out_file']),
        name='output_node')

    flirt_estimate = pe.Node(fsl.FLIRT(uses_qform=True, dof=6),
                             "flirt_estimate")
    flirt_apply = pe.Node(fsl.FLIRT(apply_xfm=True, uses_qform=True, dof=6),
                          "flirt_apply")

    first_volume_extractor = Node(fsl.ExtractROI(),
                                  name="first_volume_extractor")
    first_volume_extractor.inputs.t_min = 0
    first_volume_extractor.inputs.t_size = 1

    workflow.connect(input_node, "target_file",
                     first_volume_extractor, "in_file")
    workflow.connect(input_node, "b1_anat_ref_file",
                     flirt_estimate, "in_file")
    workflow.connect(first_volume_extractor, "roi_file",
                     flirt_estimate, "reference")

    workflow.connect(input_node, "b1_map_file",
                     flirt_apply, "in_file")
    workflow.connect(first_volume_extractor, "roi_file",
                     flirt_apply, "reference")
    workflow.connect(flirt_estimate, "out_matrix_file",
                     flirt_apply, "in_matrix_file")

    workflow.connect(flirt_apply, "out_file",
                     output_node, "out_file")

    return workflow


def compute_t2_t1_amplitude_maps(magnitude_file,
                                 phase_file,
                                 mask_file,
                                 b1_map_file,
                                 repetition_time,
                                 flip_angle,
                                 delta_phi
                                 ):
    from T2T1AM import cal_T2T1AM
    import os

    base_dir = os.getcwd()
    output_dir = base_dir

    cal_T2T1AM(magnitude_file, phase_file, mask_file, b1_map_file,
               repetition_time, flip_angle, delta_phi, outputdir=output_dir)

    t2_map_file = os.path.join(base_dir, "T2_.nii.gz")
    t1_map_file = os.path.join(base_dir, "T1_.nii.gz")
    am_map_file = os.path.join(base_dir, "Am_.nii.gz")

    return t2_map_file, t1_map_file, am_map_file


def preprocess_ssfp(base_dir=os.getcwd(), name="preprocess_ssfp"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(
        IdentityInterface(fields=[
            "b1_map_file",
            "b1_anat_ref_file",
            "t1w_fa_13_file",
            "t1w_fa_2_file",
            "t2w_fa_12_rf_180_file",
            "t2w_fa_49_rf_0_file",
            "t2w_fa_49_rf_180_file"
        ]),
        name="input_node"
    )

    output_node = pe.Node(
        IdentityInterface(fields=[
            "b1_map_file",
            "t1w_fa_13_file",
            "t1w_fa_2_file",
            "t2w_fa_12_rf_180_file",
            "t2w_fa_49_rf_0_file",
            "t2w_fa_49_rf_180_file",
            "brain_mask_file"
        ]),
        name="output_node"
    )

    anat_target_file = "t1w_fa_13_file"
    anat_moving_files = [
        "t1w_fa_2_file",
        "t2w_fa_12_rf_180_file",
        "t2w_fa_49_rf_0_file",
        "t2w_fa_49_rf_180_file"
    ]

    # denoise target image
    tgv_alpha = 1e-5
    denoise_target_node = pe.Node(QiTgv(alpha=tgv_alpha),
                                  name="denoise_{}".format(anat_target_file))
    wf.connect(input_node, anat_target_file,
               denoise_target_node, "in_file")
    wf.connect(denoise_target_node, "out_file",
               output_node, anat_target_file)

    # denoise each moving image, register to target image, and store as output
    for moving_file in anat_moving_files:
        denoise_node = pe.Node(QiTgv(alpha=tgv_alpha),
                               name="denoise_{}".format(moving_file))
        wf.connect(input_node, moving_file, denoise_node, "in_file")

        register_wf = register_image_workflow(
            name="register_image_wf_{}".format(moving_file))
        wf.connect(input_node, moving_file,
                   register_wf, "input_node.moving_file")
        wf.connect(input_node, moving_file,
                   register_wf, "input_node.reference_file")
        wf.connect(denoise_target_node, "out_file",
                   register_wf, "input_node.target_file")
        wf.connect(register_wf, "output_node.out_file",
                   output_node, moving_file)

    # register b1 map
    register_b1_wf = register_image_workflow(
        name="register_b1_wf")
    wf.connect(input_node, "b1_map_file",
               register_b1_wf, "input_node.moving_file")
    wf.connect(input_node, "b1_anat_ref_file",
               register_b1_wf, "input_node.reference_file")
    wf.connect(denoise_target_node, "out_file",
               register_b1_wf, "input_node.target_file")
    wf.connect(register_b1_wf, "output_node.out_file",
               output_node, "b1_map_file")

    # create brain mask for t2w images
    create_brain_mask_wf = create_brain_mask_workflow()
    wf.connect(denoise_target_node, "out_file",
               create_brain_mask_wf, "input_node.in_file")
    wf.connect(create_brain_mask_wf, "output_node.out_file",
               output_node, "brain_mask_file")

    return wf


def preprocess_ssfp_multi_file(base_dir=os.getcwd(),
                               name="preprocess_ssfp_multi_file"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(
        IdentityInterface(fields=[
            "b1_map_file",
            "b1_anat_ref_file",
            "t1w_files",
            "t2w_files",
            "reg_target_file"
        ]),
        name="input_node"
    )

    output_node = pe.Node(
        IdentityInterface(fields=[
            "b1_map_file",
            "b1_anat_ref_file",
            "t1w_files",
            "t2w_files",
            "reg_target_file",
            "brain_mask_file"
        ]),
        name="output_node"
    )

    tgv_alpha = 1e-5
    denoise_t1w = pe.MapNode(QiTgv(alpha=tgv_alpha),
                             name="denoise_t1w", iterfield=["in_file"])
    wf.connect(input_node, "t1w_files", denoise_t1w, "in_file")

    denoise_t2w = pe.MapNode(QiTgv(alpha=tgv_alpha),
                             name="denoise_t2w", iterfield=["in_file"])
    wf.connect(input_node, "t2w_files", denoise_t2w, "in_file")

    denoise_reg_target = pe.Node(QiTgv(alpha=tgv_alpha),
                                 name="denoise_reg_target",
                                 iterfield=["in_file"])
    wf.connect(input_node, "reg_target_file", denoise_reg_target, "in_file")

    ants_reg_params = dict(
        dimension=3,  # 3D images
        output_transform_prefix='output_prefix_',  # Prefix for outputs
        transforms=['Rigid'],  # Rigid transformation
        transform_parameters=[(0.1,)],  # Step size for rigid transformation
        metric=['MI'],  # Mutual Information
        metric_weight=[1],  # Weight of the metric
        radius_or_number_of_bins=[32],  # Number of histogram bins for MI
        sampling_strategy=['Regular'],  # Regular sampling
        sampling_percentage=[0.25],  # Sampling percentage
        convergence_threshold=[1e-6],  # Convergence threshold
        convergence_window_size=[10],  # Convergence window size
        number_of_iterations=[[500, 250, 100]],
        # Reduced iterations at each level
        shrink_factors=[[6, 3, 1]],  # Shrink factors for multi-resolution
        smoothing_sigmas=[[2, 1, 0]],  # Smoothing sigmas for multi-resolution
        interpolation='Linear',  # Linear interpolation
        output_warped_image='output_warped_image.nii.gz'
    )

    register_t1w = pe.MapNode(ants.Registration(**ants_reg_params),
                              name="register_t1w",
                              iterfield=["moving_image"])
    wf.connect(denoise_reg_target, "out_file", register_t1w, "fixed_image")
    wf.connect(denoise_t1w, "out_file", register_t1w, "moving_image")

    register_t2w = pe.MapNode(ants.Registration(**ants_reg_params),
                              name="register_t2w",
                              iterfield=["moving_image"])
    wf.connect(denoise_reg_target, "out_file", register_t2w, "fixed_image")
    wf.connect(denoise_t2w, "out_file", register_t2w, "moving_image")

    register_b1_anat_ref = pe.Node(ants.Registration(**ants_reg_params),
                                   name="register_b1_anat_ref")
    wf.connect(denoise_reg_target, "out_file", register_b1_anat_ref,
               "fixed_image")
    wf.connect(input_node, "b1_anat_ref_file", register_b1_anat_ref,
               "moving_image")

    apply_trans_to_b1_map = pe.Node(ants.ApplyTransforms(),
                                    name="apply_trans_to_b1_map")
    apply_trans_to_b1_map.inputs.dimension = 3  # 3D images
    apply_trans_to_b1_map.inputs.interpolation = 'Linear'
    wf.connect(register_b1_anat_ref, "forward_transforms",
               apply_trans_to_b1_map, "transforms")
    wf.connect(input_node, "b1_map_file",
               apply_trans_to_b1_map, "input_image")
    wf.connect(denoise_reg_target, "out_file",
               apply_trans_to_b1_map, "reference_image")

    # extract brain
    mni_template_2mm = fsl.Info.standard_image('MNI152_T1_2mm.nii.gz')
    mni_template_2mm_mask = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask_dil.nii.gz')
    extract_brain = Node(ants.BrainExtraction(), name='extract_brain')
    extract_brain.inputs.dimension = 3  # 3D brain extraction
    extract_brain.inputs.brain_template = mni_template_2mm
    extract_brain.inputs.brain_probability_mask = mni_template_2mm_mask
    extract_brain.inputs.out_prefix = 'output_prefix_'
    wf.connect(denoise_reg_target, "out_file",
               extract_brain, "anatomical_image")

    wf.connect(apply_trans_to_b1_map, "output_image", output_node,
               "b1_map_file")
    wf.connect(register_b1_anat_ref, "warped_image", output_node,
               "b1_anat_ref_file")
    wf.connect(denoise_reg_target, "out_file", output_node, "reg_target_file")
    wf.connect(register_t1w, "warped_image", output_node, "t1w_files")
    wf.connect(register_t2w, "warped_image", output_node, "t2w_files")
    wf.connect(extract_brain, "BrainExtractionMask", output_node, "brain_mask_file")

    return wf


def preprocess_3depi_workflow(base_dir=os.getcwd(),
                                              name="preprocess_3depi_workflow"):

    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(IdentityInterface(fields=[
        "phase_file",
        "magnitude_file",
        "b1_map_file",
        "b1_anat_ref_file",
        "t1w_file"
    ]), name="input_node")

    output_node = pe.Node(IdentityInterface(fields=[
        "phase_file",
        "magnitude_file",
        "b1_map_file",
        "b1_anat_ref_file",
        "brain_mask_file",
        "t1w_file"
    ]), name="output_node")

    # denoise in T2w images
    denoise_wf = denoise_mag_and_phase_in_complex_domain_workflow()
    wf.connect(input_node, "phase_file",
               denoise_wf, "input_node.phase_file")
    wf.connect(input_node, "magnitude_file",
               denoise_wf, "input_node.magnitude_file")

    # correct motion in T2w images
    motion_correction_wf = motion_correction_mag_and_phase_workflow()
    wf.connect(denoise_wf, "output_node.magnitude_file",
               motion_correction_wf, "input_node.magnitude_file")
    wf.connect(denoise_wf, "output_node.phase_file",
               motion_correction_wf, "input_node.phase_file")

    # subtract background phase in T2w images
    subtract_background_phase_node = Node(Function(
        input_names=['magnitude_file', 'phase_file'],
        output_names=['magnitude_file', 'phase_file'],
        function=subtract_background_phase),
        name='subtract_background_phase')
    wf.connect(motion_correction_wf, "output_node.magnitude_file",
               subtract_background_phase_node, "magnitude_file")
    wf.connect(motion_correction_wf, "output_node.phase_file",
               subtract_background_phase_node, "phase_file")

    mag_first_volume_extractor = Node(fsl.ExtractROI(),
                                  name="mag_first_volume_extractor")
    mag_first_volume_extractor.inputs.t_min = 0
    mag_first_volume_extractor.inputs.t_size = 1
    wf.connect(subtract_background_phase_node, "magnitude_file",
               mag_first_volume_extractor, "in_file")

    t1w_first_volume_extractor = Node(fsl.ExtractROI(),
                                  name="t1w_first_volume_extractor")
    t1w_first_volume_extractor.inputs.t_min = 0
    t1w_first_volume_extractor.inputs.t_size = 1
    wf.connect(input_node, "t1w_file",
               t1w_first_volume_extractor, "in_file")

    ants_reg_params = dict(
        dimension=3,  # 3D images
        output_transform_prefix='output_prefix_',  # Prefix for outputs
        transforms=['Rigid'],  # Rigid transformation
        transform_parameters=[(0.1,)],  # Step size for rigid transformation
        metric=['MI'],  # Mutual Information
        metric_weight=[1],  # Weight of the metric
        radius_or_number_of_bins=[32],  # Number of histogram bins for MI
        sampling_strategy=['Regular'],  # Regular sampling
        sampling_percentage=[0.25],  # Sampling percentage
        convergence_threshold=[1e-6],  # Convergence threshold
        convergence_window_size=[10],  # Convergence window size
        number_of_iterations=[[500, 250, 100]],
        # Reduced iterations at each level
        shrink_factors=[[6, 3, 1]],  # Shrink factors for multi-resolution
        smoothing_sigmas=[[2, 1, 0]],  # Smoothing sigmas for multi-resolution
        interpolation='Linear',  # Linear interpolation
        output_warped_image='output_warped_image.nii.gz'
    )

    register_b1_anat_ref_to_t2w = pe.Node(ants.Registration(**ants_reg_params),
                              name="register_b1_anat_ref_to_t2w")
    wf.connect(mag_first_volume_extractor, "roi_file",
               register_b1_anat_ref_to_t2w, "fixed_image")
    wf.connect(input_node, "b1_anat_ref_file",
               register_b1_anat_ref_to_t2w, "moving_image")

    register_t1w_to_t2w = pe.Node(ants.Registration(**ants_reg_params),
                              name="register_t1w_to_t2w")
    wf.connect(mag_first_volume_extractor, "roi_file",
               register_t1w_to_t2w, "fixed_image")
    wf.connect(t1w_first_volume_extractor, "roi_file",
               register_t1w_to_t2w, "moving_image")

    apply_trans_to_b1_map = pe.Node(ants.ApplyTransforms(),
                                    name="apply_trans_to_b1_map")
    apply_trans_to_b1_map.inputs.dimension = 3  # 3D images
    apply_trans_to_b1_map.inputs.interpolation = 'Linear'
    wf.connect(register_b1_anat_ref_to_t2w, "forward_transforms",
               apply_trans_to_b1_map, "transforms")
    wf.connect(input_node, "b1_map_file",
               apply_trans_to_b1_map, "input_image")
    wf.connect(mag_first_volume_extractor, "roi_file",
               apply_trans_to_b1_map, "reference_image")

    # # extract brain
    # mni_template_1mm = fsl.Info.standard_image('MNI152_T1_1mm.nii.gz')
    # register_t1w_to_mni = pe.Node(ants.Registration(**ants_reg_params),
    #                           name="register_t1w_to_mni")
    # register_t1w_to_mni.inputs.fixed_image = mni_template_1mm
    # wf.connect(register_t1w_to_t2w, "warped_image",
    #            register_t1w_to_mni, "moving_image")

    # extract brain
    mni_template = fsl.Info.standard_image('MNI152_T1_2mm.nii.gz')
    # mni_template_mask = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')
    mni_template_mask = "/home/laurin/workspace/t2_mapping_traveling_heads/data/atlases/brain_probseg_2mm.nii.gz"
    extract_brain = Node(CustomBrainExtraction(), name='extract_brain')
    extract_brain.inputs.dimension = 3  # 3D brain extraction
    extract_brain.inputs.brain_template = mni_template
    extract_brain.inputs.brain_probability_mask = mni_template_mask
    extract_brain.inputs.out_prefix = 'output_prefix_'
    wf.connect(register_t1w_to_t2w, "warped_image",
               extract_brain, "anatomical_image")
    # wf.connect(register_t1w_to_t2w, "composite_transform",
    #            extract_brain, "initial_transform")

    wf.connect(subtract_background_phase_node, "magnitude_file",
               output_node, "magnitude_file")
    wf.connect(subtract_background_phase_node, "phase_file",
               output_node, "phase_file")
    wf.connect(extract_brain, "BrainExtractionMask",
               output_node, "brain_mask_file")
    wf.connect(apply_trans_to_b1_map, "output_image",
               output_node, "b1_map_file")
    wf.connect(register_b1_anat_ref_to_t2w, "warped_image",
               output_node, "b1_anat_ref_file")
    wf.connect(register_t1w_to_t2w, "warped_image",
               output_node, "t1w_file")

    return wf
