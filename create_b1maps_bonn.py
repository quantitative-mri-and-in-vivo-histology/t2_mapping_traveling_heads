import sys
import argparse
import math
import nibabel as nib
import numpy as np
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


def reslice_image_to_target(base_dir=os.getcwd(), name="reslice_image_to_target"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir
    input_node = pe.Node(interface=util.IdentityInterface(
        fields=['in_file', 'target_file']),
        name='input_node')
    output_node = pe.Node(
        interface=util.IdentityInterface(fields=['out_file']),
        name='output_node')

    flirt = pe.Node(fsl.FLIRT(uses_qform = True, dof=6, apply_xfm=True),
                             "flirt")
    workflow.connect(input_node, "target_file",
                     flirt, "reference")
    workflow.connect(input_node, "in_file",
                     flirt, "in_file")

    workflow.connect(flirt, "out_file",
                     output_node, "out_file")

    return workflow


def register_image(base_dir=os.getcwd(), name="register_image"):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir
    input_node = pe.Node(interface=util.IdentityInterface(
        fields=['moving_file', 'reference_file', 'target_file']),
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

    bet_target = Node(fsl.BET(), name="bet_target")
    bet_target.inputs.robust = True

    bet_reference = Node(fsl.BET(), name="bet_reference")
    bet_reference.inputs.robust = True

    workflow.connect(input_node, "reference_file",
                     bet_reference, "in_file")
    workflow.connect(input_node, "target_file",
                     bet_target, "in_file")

    workflow.connect(bet_target, "out_file",
                     first_volume_extractor, "in_file")
    workflow.connect(bet_reference, "out_file",
                     flirt_estimate, "in_file")
    workflow.connect(first_volume_extractor, "roi_file",
                     flirt_estimate, "reference")

    workflow.connect(input_node, "moving_file",
                     flirt_apply, "in_file")
    workflow.connect(first_volume_extractor, "roi_file",
                     flirt_apply, "reference")
    workflow.connect(flirt_estimate, "out_matrix_file",
                     flirt_apply, "in_matrix_file")

    workflow.connect(flirt_apply, "out_file",
                     output_node, "out_file")

    return workflow


# Core function to compute the B1 map
def compute_b0_map(b0_phase2_file):

    import nibabel as nib
    import os

    base_dir = os.getcwd()

    # read ste and fid b1 magnitude images
    b0_phase2_file_nib = nib.load(b0_phase2_file)
    b0_phase2_image = b0_phase2_file_nib.get_fdata()

    # Compute the B0 map
    t_e = 0.002
    b0_scaling_factor = 1.0/(4096*t_e*2)
    b0_map = b0_scaling_factor*b0_phase2_image

    # write b1 map
    b0_output_filename = os.path.join(base_dir, 'b0map.nii.gz')
    b0_image_nib = nib.Nifti1Image(b0_map, b0_phase2_file_nib.affine, b0_phase2_file_nib.header)
    nib.save(b0_image_nib, b0_output_filename)

    return b0_output_filename


def compute_b1_map(b1_ste_file, b1_fid_file):

    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()

    # read ste and fid b1 magnitude images
    b1_ste_file_nib = nib.load(b1_ste_file)
    b1_fid_image_nib = nib.load(b1_fid_file)
    b1_ste_image = b1_ste_file_nib.get_fdata()
    b1_fid_image = b1_fid_image_nib.get_fdata()

    # Compute the B1 map
    fa = 60
    siemens_scaling_factor = 10
    b1_scaling_factor = fa*siemens_scaling_factor
    b1_map = b1_scaling_factor*np.arctan(np.sqrt(np.divide(2*np.abs(b1_ste_image),np.abs(b1_fid_image))))

    # write b1 map
    b1_output_filename = os.path.join(base_dir, 'b1map.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_map, b1_ste_file_nib.affine, b1_ste_file_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename


# Core function to compute the B1 map
def compute_b1_ref(b1_ste_file, b1_fid_file):

    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()

    # read ste and fid b1 magnitude images
    b1_ste_file_nib = nib.load(b1_ste_file)
    b1_fid_image_nib = nib.load(b1_fid_file)
    b1_ste_image = b1_ste_file_nib.get_fdata()
    b1_fid_image = b1_fid_image_nib.get_fdata()

    # Compute the B1 ref
    fa = 60
    siemens_scaling_factor = 10
    b1_scaling_factor = 1
    b1_ref = b1_scaling_factor*(2*b1_ste_image+b1_fid_image)

    # write b1 map
    b1_output_filename = os.path.join(base_dir, 'b1ref.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_ref, b1_ste_file_nib.affine, b1_ste_file_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename


def correct_b1_map(b1_map_file, b0_map_file):
    # Load input images
    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()
    b1_map_file_nib = nib.load(b1_map_file)
    b0_map_file_nib = nib.load(b0_map_file)

    # Get the data arrays
    b1_map_image = b1_map_file_nib.get_fdata()
    b0_map_image = b0_map_file_nib.get_fdata()

    def cos_omega_eff(omega_eff, delta_omega, tau=2.445e-3):
        return np.cos(omega_eff * tau) + ((delta_omega / omega_eff) ** 2) * (1 - np.cos(omega_eff * tau))

    def gamma_b1(alpha, tau=2.445e-3):
        return alpha / tau

    ImgPulDur = 2.450e-3  # imaging pulse duration of 3DREAM at 3T
    ImgPulFA = 60  # imaging pulse flip angle of 3DREAM at 3T
    tau = ImgPulDur
    fa_nominal = np.deg2rad(10.0)
    fa_actual = fa_nominal * b1_map_image / (ImgPulFA * 10.0)
    delta_omega = b0_map_image * 2.0 * np.pi

    omega_eff = np.sqrt((delta_omega) ** 2 + (gamma_b1(fa_actual, tau=tau)) ** 2)  # nutation frequency
    cosine = cos_omega_eff(omega_eff=omega_eff, delta_omega=delta_omega, tau=tau)
    fa_actual_with_offresonance = np.arccos(cosine) #[rad]
    b1_map = 100*fa_actual_with_offresonance/fa_nominal

    b1_output_filename = os.path.join(base_dir, 'b1_map_corr.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_map, b0_map_file_nib.affine, b0_map_file_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename



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
                    b1_map_files = layout.get(subject=subject, session=session,
                                              suffix="TB1map",
                                              extension="nii.gz",
                                              run=run)

                    b0_mag1_files = layout.get(subject=subject, session=session,
                                               suffix="magnitude1",
                                               extension="nii.gz",
                                               acquisition="dznebnB0",
                                               run='2')

                    b0_mag2_files = layout.get(subject=subject, session=session,
                                               suffix="magnitude2",
                                               extension="nii.gz",
                                               acquisition="dznebnB0",
                                               run='2')

                    b0_phase2_files = layout.get(subject=subject, session=session,
                                               suffix="phase2",
                                               extension="nii.gz",
                                               acquisition="dznebnB0",
                                               run='2')

                    b1_ste_files = layout.get(subject=subject, session=session,
                                               suffix="magnitude1",
                                               extension="nii.gz",
                                               acquisition="dznebnB1",
                                               run=run)

                    b1_fid_files = layout.get(subject=subject, session=session,
                                               suffix="magnitude2",
                                               extension="nii.gz",
                                               acquisition="dznebnB1",
                                               run=run)

                    inputs.append(dict(subject=subject,
                                       session=session,
                                       run=run,
                                       b1_map_file=b1_map_files[0],
                                       b0_mag1_file=b0_mag1_files[0],
                                       b0_mag2_file=b0_mag2_files[0],
                                       b0_phase2_file=b0_phase2_files[0],
                                       b1_ste_file=b1_ste_files[0],
                                       b1_fid_file=b1_fid_files[0]))

    # generate input node from collected inputs
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='input_node')
    keys = inputs[0].keys()
    input_node.iterables = [(key, [input_dict[key] for input_dict in inputs])
                            for key in keys]
    input_node.synchronize = True

    # set up worfklow
    wf = Workflow(name="b0_b1_mapping",
                  base_dir=Path(args.output_directory).joinpath("nipype"))
    # Set the execution mode to sequential
    wf.config['execution'] = {
        'remove_unnecessary_outputs': False,
        'sequential': True  # This is key to process one subject at a time
    }

    # compute b0
    compute_b0_map_node = pe.Node(interface=util.Function(
        input_names=['b0_phase2_file'],
        output_names=['out_file'],
        function=compute_b0_map),
        name='compute_b0_map')
    wf.connect(input_node, 'b0_phase2_file', compute_b0_map_node, 'b0_phase2_file')

    # compute b1 map
    compute_b1_map_node = pe.Node(interface=util.Function(
        input_names=['b1_ste_file', 'b1_fid_file'],
        output_names=['out_file'],
        function=compute_b1_map),
        name='compute_b1_map')
    wf.connect(input_node, 'b1_ste_file', compute_b1_map_node, 'b1_ste_file')
    wf.connect(input_node, 'b1_fid_file', compute_b1_map_node, 'b1_fid_file')


    # compute b1 anat ref
    compute_b1_ref_node = pe.Node(interface=util.Function(
        input_names=['b1_ste_file', 'b1_fid_file'],
        output_names=['out_file'],
        function=compute_b1_ref),
        name='compute_b1_ref')
    wf.connect(input_node, 'b1_ste_file', compute_b1_ref_node, 'b1_ste_file')
    wf.connect(input_node, 'b1_fid_file', compute_b1_ref_node, 'b1_fid_file')

    # reslice b1 map to b0
    register_b1_map_to_b0_map = reslice_image_to_target(name="register_b1_map_to_b0_map")
    wf.connect(compute_b1_map_node, "out_file",
               register_b1_map_to_b0_map, "input_node.in_file")
    wf.connect(compute_b0_map_node, "out_file",
               register_b1_map_to_b0_map, "input_node.target_file")

    # reslice b1 anat ref to b0
    register_b1_ref_to_b0_map = reslice_image_to_target(name="register_b1_ref_to_b0_map")
    wf.connect(compute_b1_ref_node, "out_file",
               register_b1_ref_to_b0_map, "input_node.in_file")
    wf.connect(compute_b1_ref_node, "out_file",
               register_b1_ref_to_b0_map, "input_node.target_file")

    # correct b1 map
    correct_b1_map_node = pe.Node(interface=util.Function(
        input_names=['b1_map_file', 'b0_map_file'],
        output_names=['b1_output_file'],
        function=correct_b1_map),
        name='correct_b1_map')
    wf.connect(register_b1_map_to_b0_map, "output_node.out_file",
               correct_b1_map_node, "b1_map_file")
    wf.connect(compute_b0_map_node, "out_file",
               correct_b1_map_node, "b0_map_file")

    # scale b1 map to percent
    scaling_factor = 100 / (60 * 10)  # fa * 10 [percent]
    scale_b1_to_percent = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor)),
        name="scale_b1_to_percent")
    wf.connect(compute_b1_map_node, "out_file",
                     scale_b1_to_percent, "in_file")

    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


# Main
if __name__ == "__main__":
    sys.exit(main())
