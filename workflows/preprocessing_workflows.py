from ast import iter_fields

# from bids_validator.test_bids_validator import files
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nipype.interfaces.mrtrix3 as mrtrix3
from nipype import Node, Function
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import os
from utils.processing import QiTgv, QiJsr
from nipype_utils import ApplyXfm4D, get_common_parent_directory


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
                             name="denoise_reg_target", iterfield=["in_file"])
    wf.connect(input_node, "reg_target_file", denoise_reg_target, "in_file")


    flirt_settings = dict(uses_qform=True, dof=6)

    # t1w registration
    flirt_estimate_t1w = pe.MapNode(fsl.FLIRT(**flirt_settings),
                                    name="flirt_estimate_t1w",
                                    iterfield=["in_file"])
    wf.connect(denoise_t1w, "out_file", flirt_estimate_t1w, "in_file")
    wf.connect(input_node, "reg_target_file", flirt_estimate_t1w, "reference")
    flirt_apply_t1w = pe.MapNode(
        fsl.FLIRT(apply_xfm=True, **flirt_settings),
        name="flirt_apply_t1w", iterfield=["in_file", "in_matrix_file"])
    wf.connect(denoise_t1w, "out_file", flirt_apply_t1w, "in_file")
    wf.connect(denoise_reg_target, "out_file", flirt_apply_t1w, "reference")
    wf.connect(flirt_estimate_t1w, "out_matrix_file", flirt_apply_t1w,
               "in_matrix_file")

    # t2w registration
    flirt_estimate_t2w = pe.MapNode(fsl.FLIRT(**flirt_settings),
                                    name="flirt_estimate_t2w",
                                    iterfield=["in_file"])
    wf.connect(denoise_t2w, "out_file", flirt_estimate_t2w, "in_file")
    wf.connect(denoise_reg_target, "out_file", flirt_estimate_t2w, "reference")

    flirt_apply_t2w = pe.MapNode(
        fsl.FLIRT(apply_xfm=True, **flirt_settings),
        name="flirt_apply_t2w", iterfield=["in_file", "in_matrix_file"])
    wf.connect(denoise_t2w, "out_file", flirt_apply_t2w, "in_file")
    wf.connect(denoise_reg_target, "out_file", flirt_apply_t2w, "reference")
    wf.connect(flirt_estimate_t2w, "out_matrix_file", flirt_apply_t2w,
               "in_matrix_file")

    # b1 map registration
    flirt_estimate_b1 = pe.Node(fsl.FLIRT(**flirt_settings),
                                    name="flirt_estimate_b1")
    wf.connect(input_node, "b1_anat_ref_file", flirt_estimate_b1, "in_file")
    wf.connect(denoise_reg_target, "out_file", flirt_estimate_b1, "reference")

    flirt_apply_b1_map = pe.Node(
        fsl.FLIRT(apply_xfm=True, **flirt_settings),
        name="flirt_apply_b1_map")
    wf.connect(input_node, "b1_map_file", flirt_apply_b1_map, "in_file")
    wf.connect(denoise_reg_target, "out_file", flirt_apply_b1_map, "reference")
    wf.connect(flirt_estimate_b1, "out_matrix_file", flirt_apply_b1_map,
               "in_matrix_file")

    flirt_apply_b1_anat_ref = pe.Node(
        fsl.FLIRT(apply_xfm=True, **flirt_settings),
        name="flirt_apply_b1_anat_ref")
    wf.connect(input_node, "b1_anat_ref_file", flirt_apply_b1_anat_ref, "in_file")
    wf.connect(denoise_reg_target, "out_file", flirt_apply_b1_anat_ref, "reference")
    wf.connect(flirt_estimate_b1, "out_matrix_file", flirt_apply_b1_anat_ref,
               "in_matrix_file")

    bet_node = Node(fsl.BET(), name="bet")
    bet_node.inputs.robust = True
    bet_node.inputs.mask = True
    bet_node.inputs.frac = 0.5

    wf.connect(input_node, "reg_target_file",
                     bet_node, "in_file")

    wf.connect(flirt_apply_b1_map, "out_file", output_node, "b1_map_file")
    wf.connect(flirt_apply_b1_anat_ref, "out_file", output_node, "b1_anat_ref_file")
    wf.connect(denoise_reg_target, "out_file", output_node, "reg_target_file")
    wf.connect(flirt_apply_t1w, "out_file", output_node, "t1w_files")
    wf.connect(flirt_apply_t2w, "out_file", output_node, "t2w_files")
    wf.connect(bet_node, "mask_file", output_node, "brain_mask_file")

    return wf
