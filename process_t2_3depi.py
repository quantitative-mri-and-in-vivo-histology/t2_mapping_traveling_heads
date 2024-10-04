import sys
import argparse
import math
import multiprocessing
from nipype.pipeline import Workflow
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nipype.interfaces.mrtrix3 as mrtrix3
from nipype import Node, Function
import nipype.interfaces.fsl as fsl
import os
from pathlib import Path
from bids.layout import BIDSLayout
from nipype_utils import ApplyXfm4D, BidsRename, get_common_parent_directory, \
    create_output_folder

num_cores = multiprocessing.cpu_count()


def subtract_background_phase(magnitude_file, phase_file):
    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()
    mag_nib = nib.load(magnitude_file)
    phase_nib = nib.load(phase_file)
    mag = mag_nib.get_fdata()
    phase = phase_nib.get_fdata()

    com = mag * np.exp(1.0j * phase)
    hip = com[..., 1::2] * np.conj(com[..., 0::2])

    phase_bg_sub = np.angle(hip) / 2.0
    mag_bg_sub = np.sqrt(np.abs(hip))

    phase_bg_sub = phase_bg_sub[..., (0, 3, 1, 4, 2, 5)]
    mag_bg_sub = mag_bg_sub[..., (0, 3, 1, 4, 2, 5)]

    phase_bg_sub_nii = nib.Nifti1Image(phase_bg_sub, phase_nib.affine,
                                       phase_nib.header)
    mag_bg_sub_nii = nib.Nifti1Image(mag_bg_sub, mag_nib.affine, mag_nib.header)

    magnitude_out_file = os.path.join(base_dir,
                                      "{}{}".format("magnitude", ".nii.gz"))
    phase_out_file = os.path.join(base_dir, "{}{}".format("phase", ".nii.gz"))

    nib.save(mag_bg_sub_nii, magnitude_out_file)
    nib.save(phase_bg_sub_nii, phase_out_file)

    return magnitude_out_file, phase_out_file


def denoise_workflow(base_dir=os.getcwd(), name="denoise"):
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


def motion_correction_workflow(base_dir=os.getcwd(), name="motion_correction"):
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


def create_brain_mask(base_dir=os.getcwd(), name="create_brain_mask"):
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


def scale_b1_map_to_percent(base_dir=os.getcwd(), name="scale_b1_map_to_percent"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir
    input_node = pe.Node(interface=util.IdentityInterface(
        fields=['in_file']),
        name='input_node')
    output_node = pe.Node(interface=util.IdentityInterface(
        fields=['out_file']),
        name='output_node')

    scaling_factor = 100 / (60 * 10)  # fa * 10 [percent]
    scale_b1_to_percent = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor)),
        name="scale_b1_to_percent")

    workflow.connect(input_node, "in_file",
                     scale_b1_to_percent, "in_file")
    workflow.connect(scale_b1_to_percent, "out_file",
                     output_node, "out_file")
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

    flirt_estimate = pe.Node(fsl.FLIRT(uses_qform = True, dof=6),
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


def preprocess_workflow(base_dir=os.getcwd(), name="preprocess"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir
    input_node = pe.Node(interface=util.IdentityInterface(
        fields=['magnitude_file',
                'phase_file',
                'b1_map_file',
                'b1_anat_ref_file']),
        name='input_node')
    output_node = pe.Node(interface=util.IdentityInterface(
        fields=['magnitude_file',
                'phase_file',
                'b1_map_file',
                'brain_mask_file']),
        name='output_node')

    # scale phase to radian
    scaling_factor = math.pi / 4096.0
    scale_phase_from_siemens_to_rad = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor)),
        name="scale_phase_from_siemens_to_rad")
    workflow.connect(input_node, "phase_file", scale_phase_from_siemens_to_rad,
                     "in_file")

    # denoise in T2w images
    denoise_wf = denoise_workflow()
    workflow.connect(scale_phase_from_siemens_to_rad, "out_file", denoise_wf,
                     "input_node.phase_file")
    workflow.connect(input_node, "magnitude_file", denoise_wf,
                     "input_node.magnitude_file")

    # correct motion in T2w images
    motion_correction_wf = motion_correction_workflow()
    workflow.connect(denoise_wf, "output_node.magnitude_file",
                     motion_correction_wf, "input_node.magnitude_file")
    workflow.connect(denoise_wf, "output_node.phase_file",
                     motion_correction_wf, "input_node.phase_file")

    # subtract background phase in T2w images
    subtract_background_phase_node = Node(Function(
        input_names=['magnitude_file', 'phase_file'],
        output_names=['magnitude_file', 'phase_file'],
        function=subtract_background_phase), name='subtract_background_phase')
    workflow.connect(motion_correction_wf, "output_node.magnitude_file",
                     subtract_background_phase_node, "magnitude_file")
    workflow.connect(motion_correction_wf, "output_node.phase_file",
                     subtract_background_phase_node, "phase_file")

    # create brain mask for T2w images
    create_brain_mask_wf = create_brain_mask()
    workflow.connect(subtract_background_phase_node, "magnitude_file",
                     create_brain_mask_wf, "input_node.in_file")

    # scale b1 map to percent
    scale_b1_map_to_percent_wf = scale_b1_map_to_percent()
    workflow.connect(input_node, "b1_map_file",
                     scale_b1_map_to_percent_wf, "input_node.in_file")

    # compute b1 map in percent
    register_b1_map_to_t2w_wf = register_b1_map_to_t2w()
    workflow.connect(scale_b1_map_to_percent_wf, "output_node.out_file",
                     register_b1_map_to_t2w_wf, "input_node.b1_map_file")
    workflow.connect(input_node, "b1_anat_ref_file",
                     register_b1_map_to_t2w_wf, "input_node.b1_anat_ref_file")
    workflow.connect(subtract_background_phase_node, "magnitude_file",
                     register_b1_map_to_t2w_wf, "input_node.target_file")

    # set outputs
    workflow.connect(subtract_background_phase_node, "magnitude_file",
                     output_node, "magnitude_file")
    workflow.connect(subtract_background_phase_node, "phase_file",
                     output_node, "phase_file")
    workflow.connect(create_brain_mask_wf, "output_node.out_file",
                     output_node, "brain_mask_file")
    workflow.connect(register_b1_map_to_t2w_wf, "output_node.out_file",
                     output_node, "b1_map_file")

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


def main():
    parser = argparse.ArgumentParser(
        description='Perform complex-valued preprocessing for T2 data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_directory", '-i',
                        help='bids input dataset root', type=str)
    parser.add_argument("--input_derivatives", '-d',
                        help='bids input dataset root', type=str, nargs='*')
    parser.add_argument("--output_directory", '-o',
                        help='output directory',
                        type=str)
    parser.add_argument("--subject_id", '-s',
                        help='subject to process; leave empty for all subjects',
                        type=str, default=None)
    parser.add_argument("--session_id", '-t',
                        help='session to process; leave empty for all sessions',
                        type=str, default=None)
    parser.add_argument("--run_id", '-r',
                        help='run id to process; leave empty for all runs',
                        type=str, default=None)
    parser.add_argument('--n_procs', '-n', type=int, default=num_cores,
                        help='number of cores for parallel processing. '
                             'default: number of available processors.')
    args = parser.parse_args()

    # collect inputs
    layout = BIDSLayout(args.input_directory,
                        derivatives=args.input_derivatives,
                        validate=False)
    inputs = []
    subjects = [args.subject_id] if isinstance(args.subject_id,
                                               str) else layout.get_subjects()
    for subject in subjects:
        sessions = [args.session_id] if isinstance(args.session_id,
                                                   str) else layout.get_sessions(
            subject=subject)
        if sessions:  # Only add subjects with existing sessions
            for session in sessions:
                valid_runs = layout.get(return_type='id', subject=subject,
                                        session=session, target='run',
                                        suffix='T2w',
                                        part="phase", extension="nii.gz")
                runs = [args.run_id] if isinstance(args.run_id,
                                                   str) else valid_runs

                if len(runs) == 0:
                    runs = [None]
                for run in runs:
                    t2w_phase_file = layout.get(subject=subject,
                                                session=session, suffix="T2w",
                                                extension="nii.gz",
                                                part="phase", run=run)

                    t2w_magnitude_file = layout.get(subject=subject,
                                                    session=session,
                                                    suffix="T2w",
                                                    extension="nii.gz",
                                                    part="mag", run=run)

                    b1_map_files = layout.get(subject=subject, session=session,
                                              suffix="TB1map",
                                              extension="nii.gz",
                                              run=run)

                    b1_map_file = b1_map_files[0]
                    for file in b1_map_files:
                        if "wrapCorrected" in file.entities.get('desc', ''):
                            b1_map_file = file
                            break

                    b1_mag1_files = layout.get(subject=subject, session=session,
                                               suffix="magnitude1",
                                               extension="nii.gz",
                                               acquisition="dznebnB1",
                                               run=run)
                    b1_mag1_file = b1_mag1_files[0]
                    for file in b1_mag1_files:
                        if "wrapCorrected" in file.entities.get('desc', ''):
                            b1_mag1_file = file
                            break

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       t2w_magnitude_file=t2w_magnitude_file[0],
                                       t2w_phase_file=t2w_phase_file[0],
                                       b1_map_file=b1_map_file,
                                       b1_anat_ref_file=b1_mag1_file))

    # generate input node from collected inputs
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='input_node')
    keys = inputs[0].keys()
    input_node.iterables = [(key, [input_dict[key] for input_dict in inputs])
                            for key in keys]
    input_node.synchronize = True

    # set up worfklow
    wf = Workflow(name="bids_workflow",
                  base_dir=Path(args.output_directory).joinpath("nipype"))
    # Set the execution mode to sequential
    wf.config['execution'] = {
        'remove_unnecessary_outputs': False,
        'sequential': True  # This is key to process one subject at a time
    }

    # preprocess
    preprocess_wf = preprocess_workflow()
    wf.connect(input_node, "t2w_magnitude_file",
               preprocess_wf, "input_node.magnitude_file")
    wf.connect(input_node, "t2w_phase_file",
               preprocess_wf, "input_node.phase_file")
    wf.connect(input_node, "b1_map_file",
               preprocess_wf, "input_node.b1_map_file")
    wf.connect(input_node, "b1_anat_ref_file",
               preprocess_wf, "input_node.b1_anat_ref_file")

    # map t1, t2 and am
    compute_t2_t1_am_node = Node(
        Function(input_names=["magnitude_file", "phase_file", "mask_file",
                              "b1_map_file", "repetition_time", "flip_angle",
                              "delta_phi"],
                 output_names=["t2_map_file", "t1_map_file", "am_map_file"],
                 function=compute_t2_t1_amplitude_maps),
        name="compute_t2_t1_am")
    compute_t2_t1_am_node.inputs.repetition_time = 13.5
    compute_t2_t1_am_node.inputs.flip_angle = 20
    compute_t2_t1_am_node.inputs.delta_phi = [1, 1.5, 2, 3, 4, 5]
    wf.connect(preprocess_wf, "output_node.brain_mask_file",
               compute_t2_t1_am_node, "mask_file")
    wf.connect(preprocess_wf, "output_node.magnitude_file",
               compute_t2_t1_am_node, "magnitude_file")
    wf.connect(preprocess_wf, "output_node.phase_file",
               compute_t2_t1_am_node, "phase_file")
    wf.connect(preprocess_wf, "output_node.b1_map_file",
               compute_t2_t1_am_node, "b1_map_file")

    # set up output directories
    Path(args.output_directory).mkdir(exist_ok=True, parents=True)
    data_sink = pe.Node(nio.DataSink(), name='data_sink')
    data_sink.inputs.base_directory = args.output_directory
    output_folder_node = Node(Function(input_names=['subject', 'session'],
                                       output_names=['output_folder'],
                                       function=create_output_folder),
                              name='output_folder_node')
    wf.connect(input_node, 'subject', output_folder_node, 'subject')
    wf.connect(input_node, 'session', output_folder_node, 'session')
    wf.connect(output_folder_node, 'output_folder', data_sink, 'container')

    # save output: magnitude image (preprocessed)
    t2w_preproc_pattern = ("sub-{subject}_ses-{session}_acq-{acquisition}["
                           "_run-{run}][_part-{part}]_desc-preproc_{"
                           "suffix}.nii.gz")
    bids_rename_mag_preproc = pe.Node(BidsRename(), "bids_rename_mag_preproc")
    bids_rename_mag_preproc.inputs.pattern = t2w_preproc_pattern
    wf.connect(preprocess_wf, 'output_node.magnitude_file',
               bids_rename_mag_preproc, 'in_file')
    wf.connect(input_node, 't2w_magnitude_file',
               bids_rename_mag_preproc, 'template_file')
    wf.connect(bids_rename_mag_preproc, 'out_file',
               data_sink, '@magnitude_preprocessed_file')

    # save output: magnitude image (preprocessed)
    bids_rename_phase_preproc = pe.Node(BidsRename(),
                                        "bids_rename_phase_preproc")
    bids_rename_phase_preproc.inputs.pattern = t2w_preproc_pattern
    wf.connect(preprocess_wf, 'output_node.phase_file',
               bids_rename_phase_preproc, 'in_file')
    wf.connect(input_node, 't2w_phase_file',
               bids_rename_phase_preproc, 'template_file')
    wf.connect(bids_rename_phase_preproc, 'out_file',
               data_sink, '@phase_preprocessed_file')

    # save output: brain mask
    t2w_brain_mask_pattern = ("sub-{subject}_ses-{session}_acq-{acquisition}["
                              "_run-{run}]_desc-brain_mask.nii.gz")
    bids_rename_brain_mask = pe.Node(BidsRename(), "bids_rename_brain_mask")
    bids_rename_brain_mask.inputs.pattern = t2w_brain_mask_pattern
    wf.connect(preprocess_wf, 'output_node.brain_mask_file',
               bids_rename_brain_mask, 'in_file')
    wf.connect(input_node, 't2w_phase_file',
               bids_rename_brain_mask, 'template_file')
    wf.connect(bids_rename_brain_mask, 'out_file',
               data_sink, '@brain_mask_file')

    # save output: b1 map in percent
    b1_map_pattern = ("sub-{subject}_ses-{session}_acq-{acquisition}[_run-{"
                      "run}]_desc-percent_TB1map.nii.gz")
    bids_rename_b1_map = pe.Node(BidsRename(), "bids_rename_b1_map")
    bids_rename_b1_map.inputs.pattern = b1_map_pattern
    wf.connect(preprocess_wf, 'output_node.b1_map_file',
               bids_rename_b1_map, 'in_file')
    wf.connect(input_node, 'b1_map_file',
               bids_rename_b1_map, 'template_file')
    wf.connect(bids_rename_b1_map, 'out_file',
               data_sink, '@b1_map_file')

    # save output: T1 map
    t1_map_pattern = ("sub-{subject}_ses-{session}_acq-{acquisition}[_run-{"
                      "run}]_desc-MagPhsT2_T1map.nii.gz")
    bids_rename_t1_map = pe.Node(BidsRename(), "bids_rename_t1_map")
    bids_rename_t1_map.inputs.pattern = t1_map_pattern
    wf.connect(compute_t2_t1_am_node, 't1_map_file',
               bids_rename_t1_map, 'in_file')
    wf.connect(input_node, 't2w_magnitude_file',
               bids_rename_t1_map, 'template_file')
    wf.connect(bids_rename_t1_map, 'out_file',
               data_sink, '@t1_map_file')

    # save output: T2 map
    t2_map_pattern = ("sub-{subject}_ses-{session}_acq-{acquisition}[_run-{"
                      "run}]_desc-MagPhsT2_T2map.nii.gz")
    bids_rename_t2_map = pe.Node(BidsRename(), "bids_rename_t2_map")
    bids_rename_t2_map.inputs.pattern = t2_map_pattern
    wf.connect(compute_t2_t1_am_node, 't2_map_file',
               bids_rename_t2_map, 'in_file')
    wf.connect(input_node, 't2w_magnitude_file',
               bids_rename_t2_map, 'template_file')
    wf.connect(bids_rename_t2_map, 'out_file',
               data_sink, '@t2_map_file')

    # save output: T2 map
    am_map_pattern = ("sub-{subject}_ses-{session}_acq-{acquisition}[_run-{"
                      "run}]_desc-MagPhsT2_Ammap.nii.gz")
    bids_rename_am_map = pe.Node(BidsRename(), "bids_rename_am_map")
    bids_rename_am_map.inputs.pattern = am_map_pattern
    wf.connect(compute_t2_t1_am_node, 'am_map_file',
               bids_rename_am_map, 'in_file')
    wf.connect(input_node, 't2w_magnitude_file',
               bids_rename_am_map, 'template_file')
    wf.connect(bids_rename_am_map, 'out_file',
               data_sink, '@am_map_file')

    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


# Main
if __name__ == "__main__":
    sys.exit(main())
