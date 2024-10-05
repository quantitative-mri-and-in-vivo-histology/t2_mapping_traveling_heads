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


# Core function to compute the B1 map
def compute_b1_map_initial(b1_mag1_file, b1_mag2_file):
    # Load input images
    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()
    b1_mag1_image_nib = nib.load(b1_mag1_file)
    b1_mag2_image_nib = nib.load(b1_mag2_file)

    # Get the data arrays
    b1_mag1_image = b1_mag1_image_nib.get_fdata()
    b1_mag2_image = b1_mag2_image_nib.get_fdata()

    # Compute the B1 map
    b1_map = 600*np.arctan(np.sqrt(np.divide(2*np.abs(b1_mag1_image),np.abs(b1_mag2_image))))
    b1_output_filename = os.path.join(base_dir, 'b1_map_mag_init.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_map, b1_mag1_image_nib.affine, b1_mag1_image_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename


# Core function to compute the B1 map
def compute_b1_map(b1_map_file, b0_mag1_file, b0_mag2_file, b1_mag1_file, b1_mag2_file):
    # Load input images
    import nibabel as nib
    import numpy as np
    import os

    base_dir = os.getcwd()
    b0_mag1_image_nib = nib.load(b0_mag1_file)
    b0_mag2_image_nib = nib.load(b0_mag2_file)
    b1_mag1_image_nib = nib.load(b1_mag1_file)
    b1_mag2_image_nib = nib.load(b1_mag2_file)
    b1_map_image_nib = nib.load(b1_map_file)

    # Get the data arrays
    b0_mag1_image = b0_mag1_image_nib.get_fdata()
    b0_mag2_image = b0_mag2_image_nib.get_fdata()
    b1_mag1_image = b1_mag1_image_nib.get_fdata()
    b1_mag2_image = b1_mag2_image_nib.get_fdata()
    b1_map_image = b1_map_image_nib.get_fdata()

    # # Compute the B1 map
    # b1_map_mag = 2 * b1_mag1_image + b1_mag2_image
    # b1_output_filename = os.path.join(base_dir, 'b1_map_mag.nii.gz')
    # b1_image_nib = nib.Nifti1Image(b1_map_mag, b0_mag1_image_nib.affine, b0_mag1_image_nib.header)
    # nib.save(b1_image_nib, b1_output_filename)

    # b1_map_mag = np.sqrt(2*np.divide(np.abs(b1_mag1_image),np.abs(b1_mag2_image)))
    # b1_output_filename = os.path.join(base_dir, 'b1_map.nii.gz')
    # b1_image_nib = nib.Nifti1Image(b1_map_mag, b0_mag1_image_nib.affine, b0_mag1_image_nib.header)
    # nib.save(b1_image_nib, b1_output_filename)


    def cos_omega_eff(omega_eff, delta_omega, tau=2.445e-3):
        return np.cos(omega_eff * tau) + ((delta_omega / omega_eff) ** 2) * (1 - np.cos(omega_eff * tau))

    def gamma_b1(alpha, tau=2.445e-3):
        return alpha / tau

    ImgPulDur = 2.450e-3  # imaging pulse duration of 3DREAM at 3T
    ImgPulFA = 60  # imaging pulse flip angle of 3DREAM at 3T
    tau = ImgPulDur
    fa_nominal = np.deg2rad(10.0)
    fa_actual = fa_nominal * b1_map_image / (ImgPulFA * 10.0)
    delta_omega = b0_mag2_image * 2.0 * np.pi

    omega_eff = np.sqrt((delta_omega) ** 2 + (gamma_b1(fa_actual, tau=tau)) ** 2)  # nutation frequency
    cosine = cos_omega_eff(omega_eff=omega_eff, delta_omega=delta_omega, tau=tau)
    fa_actual_with_offresonance = np.arccos(cosine) #[rad]
    b1_map = 100*fa_actual_with_offresonance/fa_nominal

    b1_output_filename = os.path.join(base_dir, 'b1_map_corr.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_map, b0_mag1_image_nib.affine, b0_mag1_image_nib.header)
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
                                               run='1')

                    b0_mag2_files = layout.get(subject=subject, session=session,
                                               suffix="magnitude2",
                                               extension="nii.gz",
                                               acquisition="dznebnB0",
                                               run='1')

                    b1_mag1_files = layout.get(subject=subject, session=session,
                                               suffix="magnitude1",
                                               extension="nii.gz",
                                               acquisition="dznebnB1",
                                               run=run)

                    b1_mag2_files = layout.get(subject=subject, session=session,
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
                                       b1_mag1_file=b1_mag1_files[0],
                                       b1_mag2_file=b1_mag2_files[0]))

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

    reslice_b1_mag1_to_b0_mag1 = reslice_image_to_target(name="reslice_b1_mag1_to_b0_mag1")
    wf.connect(input_node, "b1_mag1_file",
               reslice_b1_mag1_to_b0_mag1, "input_node.in_file")
    wf.connect(input_node, "b0_mag1_file",
               reslice_b1_mag1_to_b0_mag1, "input_node.target_file")

    reslice_b1_mag2_to_b0_mag1 = reslice_image_to_target(name="reslice_b1_mag2_to_b0_mag1")
    wf.connect(input_node, "b1_mag2_file",
               reslice_b1_mag2_to_b0_mag1, "input_node.in_file")
    wf.connect(input_node, "b0_mag1_file",
               reslice_b1_mag2_to_b0_mag1, "input_node.target_file")

    reslice_b1_map_to_b0_mag1 = reslice_image_to_target(name="reslice_b1_map_to_b0_mag1")
    wf.connect(input_node, "b1_map_file",
               reslice_b1_map_to_b0_mag1, "input_node.in_file")
    wf.connect(input_node, "b0_mag1_file",
               reslice_b1_map_to_b0_mag1, "input_node.target_file")

    compute_b1_node = pe.Node(interface=util.Function(
        input_names=['b1_map_file', 'b0_mag1_file', 'b0_mag2_file', 'b1_mag1_file', 'b1_mag2_file'],
        output_names=['b1_output_file'],
        function=compute_b1_map),
        name='compute_b1_node')

    compute_b1_initial_node = pe.Node(interface=util.Function(
        input_names=[ 'b1_mag1_file', 'b1_mag2_file'],
        output_names=['b1_output_file'],
        function=compute_b1_map_initial),
        name='compute_b1_node_initial')

    # Connect the nodes

    wf.connect(input_node, 'b1_mag1_file', compute_b1_initial_node, 'b1_mag1_file')
    wf.connect(input_node, 'b1_mag2_file', compute_b1_initial_node, 'b1_mag2_file')

    # wf.connect(input_node, 'b1_map_file', compute_b1_node, 'b1_map_file')
    wf.connect(input_node, 'b0_mag1_file', compute_b1_node, 'b0_mag1_file')
    wf.connect(input_node, 'b0_mag2_file', compute_b1_node, 'b0_mag2_file')

    wf.connect(reslice_b1_mag1_to_b0_mag1, 'output_node.out_file', compute_b1_node, 'b1_mag1_file')
    wf.connect(reslice_b1_mag2_to_b0_mag1, 'output_node.out_file', compute_b1_node, 'b1_mag2_file')
    wf.connect(reslice_b1_map_to_b0_mag1, 'output_node.out_file', compute_b1_node, 'b1_map_file')

    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


# Main
if __name__ == "__main__":
    sys.exit(main())
