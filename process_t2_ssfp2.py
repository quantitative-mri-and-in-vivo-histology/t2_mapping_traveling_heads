import sys
import argparse
import multiprocessing
import os
from nipype.pipeline import Workflow
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from nipype import Node
import nipype.interfaces.fsl as fsl
from pathlib import Path
from bids.layout import BIDSLayout
from nipype.interfaces.utility import Merge as ListMerge, IdentityInterface
from nipype.interfaces.fsl import BET, Merge
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    TraitedSpec, File, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix


num_cores = multiprocessing.cpu_count()


# Define the input specification for QiTgv
class QiTgvInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, desc='Input file', mandatory=True, position=0, argstr='%s')
    out_file = File(desc='Output file', position=1, argstr='--out=%s')  # Optional
    alpha = traits.Float(desc='Alpha parameter', position=2, argstr='--alpha=%f', usedefault=False)  # Optional

# Define the output specification for QiTgv
class QiTgvOutputSpec(TraitedSpec):
    out_file = File(desc='Output file', exists=True)

# Define the custom command-line wrapper for QiTgv
class QiTgv(CommandLine):
    _cmd = 'qi tgv'  # The command should map to "qi tgv"
    input_spec = QiTgvInputSpec
    output_spec = QiTgvOutputSpec

    # Override _list_outputs to auto-generate the out_file in the current working directory
    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)  # Ensure full path
        else:
            # Use fname_presuffix to add '_tgv' to the basename, ensure it's in the current working directory
            outputs['out_file'] = fname_presuffix(self.inputs.in_file, suffix='_tgv', newpath=os.getcwd())
        return outputs


# Define the input specification for QiJsr
class QiJsrInputSpec(CommandLineInputSpec):
    spgr_file = File(exists=True, desc='SPGR input file', mandatory=True, position=0, argstr='%s')
    ssfp_file = File(exists=True, desc='SSFP input file', mandatory=True, position=1, argstr='%s')
    b1_file = File(exists=True, desc='B1 map file', mandatory=True, position=2, argstr='--B1=%s')
    mask_file = File(exists=True, desc='Mask file', position=3, argstr='--mask=%s', mandatory=False)  # Optional
    npsi = traits.Int(6, desc='Number of PSI components', usedefault=True, position=4, argstr='--npsi=%d')
    json_file = File(exists=True, desc='Input JSON file', position=5, argstr='--json=%s')  # Use --json flag now

# Define the output specification for QiJsr
class QiJsrOutputSpec(TraitedSpec):
    out_dir = traits.Str(desc='Output directory', exists=True)  # The result will be saved in cwd by default

# Define the custom command-line wrapper for QiJsr
class QiJsr(CommandLine):
    _cmd = 'qi jsr'  # The command should map to "qi jsr"
    input_spec = QiJsrInputSpec
    output_spec = QiJsrOutputSpec

    # Override _list_outputs to ensure the output is generated with a fixed suffix "jsr" in the cwd
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_dir'] = os.path.join(os.getcwd(), 'jsr')  # Set the output path with fixed suffix "jsr"
        return outputs


def all_files_length_one(*files):
    return all(len(file) == 1 for file in files)


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
                valid_runs = layout.get(return_type='id',
                                        target='run',
                                        subject=subject,
                                        session=session,
                                        suffix='T1w',
                                        extension="nii.gz")
                runs = [args.run_id] if isinstance(args.run_id,
                                                   str) else valid_runs

                if len(runs) == 0:
                    runs = [None]
                for run in runs:
                    t1w_a2_file = layout.get(subject=subject,
                                             session=session,
                                             suffix="T1w",
                                             acquisition="t2Ssfp2A2",
                                             part='mag',
                                             extension="nii.gz",
                                             run=run)

                    t1w_a13_file = layout.get(subject=subject,
                                              session=session,
                                              suffix="T1w",
                                              acquisition="t2Ssfp2A13",
                                              part='mag',
                                              extension="nii.gz",
                                              run=run)

                    t2w_a12rf180_file = layout.get(subject=subject,
                                                   session=session,
                                                   suffix="T2w",
                                                   acquisition="t2Ssfp2A12RF180",
                                                   part='mag',
                                                   extension="nii.gz",
                                                   run=run)

                    t2w_a49rf0_file = layout.get(subject=subject,
                                                 session=session,
                                                 suffix="T2w",
                                                 acquisition="t2Ssfp2A49RF0",
                                                 part='mag',
                                                 extension="nii.gz",
                                                 run=run)

                    t2w_a49rf180_file = layout.get(subject=subject,
                                                   session=session,
                                                   suffix="T2w",
                                                   acquisition="t2Ssfp2A49RF180",
                                                   part='mag',
                                                   extension="nii.gz",
                                                   run=run)

                    b1_map_file = layout.get(subject=subject,
                                             session=session,
                                             suffix="B1map",
                                             extension="nii.gz",
                                             run=run)

                    b1_anat_ref_file = layout.get(subject=subject,
                                                  session=session,
                                                  suffix="B1ref",
                                                  extension="nii.gz",
                                                  run=run)

                    if all_files_length_one(t1w_a2_file, t1w_a13_file, t2w_a12rf180_file, t2w_a49rf0_file,
                                            t2w_a49rf180_file, b1_map_file, b1_anat_ref_file):
                        inputs.append(dict(subject=subject,
                                           session=session,
                                           run=run,
                                           t1w_a2_file=t1w_a2_file[0],
                                           t1w_a13_file=t1w_a13_file[0],
                                           t2w_a12rf180_file=t2w_a12rf180_file[0],
                                           t2w_a49rf0_file=t2w_a49rf0_file[0],
                                           t2w_a49rf180_file=t2w_a49rf180_file[0],
                                           b1_map_file=b1_map_file[0],
                                           b1_anat_ref_file=b1_anat_ref_file[0]))

    # generate input node from collected inputs
    input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                      name='input_node')
    keys = inputs[0].keys()
    input_node.iterables = [(key, [input_dict[key] for input_dict in inputs])
                            for key in keys]
    input_node.synchronize = True

    # set up worfklow
    wf = Workflow(name="t2_ssfp2_workflow",
                  base_dir=Path(args.output_directory).joinpath("nipype"))
    # Set the execution mode to sequential
    wf.config['execution'] = {
        'remove_unnecessary_outputs': False,
        'sequential': True  # This is key to process one subject at a time
    }

    # register b1 to t1w_a13
    register_image_wf_b1 = register_image(name="register_image_wf_b1")
    wf.connect(input_node, "b1_map_file",
               register_image_wf_b1, "input_node.moving_file")
    wf.connect(input_node, "b1_anat_ref_file",
               register_image_wf_b1, "input_node.reference_file")
    wf.connect(input_node, "t1w_a13_file",
               register_image_wf_b1, "input_node.target_file")

    # register t1w_a2 to t1w_a13
    register_image_wf_t1w_a2 = register_image(name="register_image_wf_t1w_a2")
    wf.connect(input_node, "t1w_a2_file",
               register_image_wf_t1w_a2, "input_node.moving_file")
    wf.connect(input_node, "t1w_a2_file",
               register_image_wf_t1w_a2, "input_node.reference_file")
    wf.connect(input_node, "t1w_a13_file",
               register_image_wf_t1w_a2, "input_node.target_file")

    # register t2w_a12rf180 to t1w_a13
    register_image_wf_t2w_a12rf180 = register_image(name="register_image_wf_t2w_a12rf180")
    wf.connect(input_node, "t2w_a12rf180_file",
               register_image_wf_t2w_a12rf180, "input_node.moving_file")
    wf.connect(input_node, "t2w_a12rf180_file",
               register_image_wf_t2w_a12rf180, "input_node.reference_file")
    wf.connect(input_node, "t1w_a13_file",
               register_image_wf_t2w_a12rf180, "input_node.target_file")

    # register t2w_a49rf0 to t1w_a13
    register_image_wf_t2w_a49rf0  = register_image(name="register_image_wf_t2w_a49rf0")
    wf.connect(input_node, "t2w_a49rf0_file",
               register_image_wf_t2w_a49rf0, "input_node.moving_file")
    wf.connect(input_node, "t2w_a49rf0_file",
               register_image_wf_t2w_a49rf0, "input_node.reference_file")
    wf.connect(input_node, "t1w_a13_file",
               register_image_wf_t2w_a49rf0, "input_node.target_file")

    # register t2w_a49rf180 to t1w_a13
    register_image_wf_t2w_a49rf180  = register_image(name="register_image_wf_t2w_a49rf180")
    wf.connect(input_node, "t2w_a49rf180_file",
               register_image_wf_t2w_a49rf180, "input_node.moving_file")
    wf.connect(input_node, "t2w_a49rf180_file",
               register_image_wf_t2w_a49rf180, "input_node.reference_file")
    wf.connect(input_node, "t1w_a13_file",
               register_image_wf_t2w_a49rf180, "input_node.target_file")

    # denoise t1w and t2w images
    tgv_alpha = 1e-5
    tgv_t1w_a13 = pe.Node(QiTgv(alpha=tgv_alpha), name="tgv_t1w_a13")
    tgv_t1w_a2 =  pe.Node(QiTgv(alpha=tgv_alpha), name="tgv_t1w_a2")
    tgv_t2w_a12rf180 = pe.Node(QiTgv(alpha=tgv_alpha), name="tgv_t2w_a12rf180")
    tgv_t2w_a49rf0 = pe.Node(QiTgv(alpha=tgv_alpha), name="tgv_t2w_a49rf0")
    tgv_t2w_a49rf180 = pe.Node(QiTgv(alpha=tgv_alpha), name="tgv_t2w_a49rf180")
    wf.connect(input_node, "t1w_a13_file",
               tgv_t1w_a13, "in_file")
    wf.connect(register_image_wf_t1w_a2, "output_node.out_file",
               tgv_t1w_a2, "in_file")
    wf.connect(register_image_wf_t2w_a12rf180, "output_node.out_file",
               tgv_t2w_a12rf180, "in_file")
    wf.connect(register_image_wf_t2w_a49rf0, "output_node.out_file",
               tgv_t2w_a49rf0, "in_file")
    wf.connect(register_image_wf_t2w_a49rf180, "output_node.out_file",
               tgv_t2w_a49rf180, "in_file")

    # scale t1w images
    scaling_factor_t1w = 1./3.0
    scale_t1w_a13 = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor_t1w)),
        name="scale_t1w_a13")
    wf.connect(tgv_t1w_a13, "out_file",
                     scale_t1w_a13, "in_file")
    scale_t1w_a2 = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor_t1w)),
        name="scale_t1w_a2")
    wf.connect(tgv_t1w_a2, "out_file",
                     scale_t1w_a2, "in_file")

    # scale t2w images
    scaling_factor_t2w = 1./7.0
    scale_t2w_a12rf180 = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor_t2w)),
        name="scale_t2w_a12rf180")
    wf.connect(tgv_t2w_a12rf180, "out_file",
                     scale_t2w_a12rf180, "in_file")
    scale_t2w_a49rf0 = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor_t2w)),
        name="scale_t2w_a49rf0")
    wf.connect(tgv_t2w_a49rf0, "out_file",
                     scale_t2w_a49rf0, "in_file")
    scale_t2w_a49rf180 = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor_t2w)),
        name="scale_t2w_a49rf180")
    wf.connect(tgv_t2w_a49rf180, "out_file",
                     scale_t2w_a49rf180, "in_file")

    # scale b1 map
    scaling_factor_b1 = 1./(100)
    scale_b1_map = pe.Node(
        fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor_b1)),
        name="scale_b1_map")
    wf.connect(register_image_wf_b1, "output_node.out_file",
                     scale_b1_map, "in_file")

    # merge t1w images
    t1w_list_merge = Node(ListMerge(2), name='t1w_list_merge')  # Merge will take 3 inputs
    t1w_image_merge = Node(Merge(dimension='t'), name='t1w_image_merge')
    wf.connect(scale_t1w_a2, 'out_file', t1w_list_merge, 'in1')
    wf.connect(scale_t1w_a13, 'out_file', t1w_list_merge, 'in2')
    wf.connect(t1w_list_merge, 'out', t1w_image_merge, 'in_files')

    # merge t2w images
    t2w_list_merge = Node(ListMerge(3), name='t2w_list_merge')  # Merge will take 3 inputs
    t2w_image_merge = Node(Merge(dimension='t'), name='t2w_image_merge')
    wf.connect(scale_t2w_a49rf180, 'out_file', t2w_list_merge, 'in1')
    wf.connect(scale_t2w_a12rf180, 'out_file', t2w_list_merge, 'in2')
    wf.connect(scale_t2w_a49rf0, 'out_file', t2w_list_merge, 'in3')
    wf.connect(t2w_list_merge, 'out', t2w_image_merge, 'in_files')

    # create brain mask for t2w images
    create_brain_mask_wf = create_brain_mask()
    wf.connect(scale_t1w_a13, "out_file",
                     create_brain_mask_wf, "input_node.in_file")

    # run iq jsr
    qi_jsr = pe.Node(QiJsr(npsi=6), name="qi_jsr")
    qi_jsr.inputs.json_file = Path("qi_jsr_config.json").absolute().as_posix()
    wf.connect(t1w_image_merge, "merged_file", qi_jsr, "spgr_file")
    wf.connect(t2w_image_merge, "merged_file", qi_jsr, "ssfp_file")
    wf.connect(scale_b1_map, "out_file", qi_jsr, "b1_file")
    wf.connect(create_brain_mask_wf, "output_node.out_file", qi_jsr, "mask_file")

    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


# Main
if __name__ == "__main__":
    sys.exit(main())
