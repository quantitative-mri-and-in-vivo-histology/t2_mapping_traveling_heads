import os
import json
from os import path
from pathlib import Path
from bids.layout import BIDSLayout
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.utils.filemanip import copyfile
from nipype.interfaces.base import File, TraitedSpec, traits, isdefined
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
import os
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    TraitedSpec, File, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix



class ApplyXfm4DInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, position=0, argstr='%s',
                   mandatory=True, desc="timeseries to motion-correct")
    ref_vol = File(exists=True, position=1, argstr='%s',
                   mandatory=True, desc="volume with final FOV and resolution")
    out_file = File(exists=True, position=2, argstr='%s',
                    genfile=True, desc="file to write", hash_files=False)
    trans_file = File(argstr='%s', position=3,
                      desc="single tranformation matrix", xor=[
            "trans_dir"], requires=["single_matrix"])
    trans_dir = File(argstr='%s', position=3,
                     desc="folder of transformation matricies",
                     xor=["tans_file"])
    single_matrix = traits.Bool(
        argstr='-singlematrix',
        desc="true if applying one volume to all timepoints")
    four_digit = traits.Bool(
        argstr='-fourdigit', desc="true mat names have four digits not five")
    user_prefix = traits.Str(
        argstr='-userprefix %s',
        desc="supplied prefix if mats don't start with 'MAT_'")


class ApplyXfm4DOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="transform applied timeseries")


class ApplyXfm4D(FSLCommand):
    """
    Wraps the applyxfm4D command line tool for applying one 3D transform to every volume in a 4D image OR
    a directory of 3D tansforms to a 4D image of the same length.

    Examples
    ---------
    >>> import nipype.interfaces.fsl as fsl
    >>> from nipype.testing import example_data
    >>> applyxfm4d = fsl.ApplyXfm4D()
    >>> applyxfm4d.inputs.in_file = example_data('functional.nii')
    >>> applyxfm4d.inputs.in_matrix_file = example_data('functional_mcf.mat')
    >>> applyxfm4d.inputs.out_file = 'newfile.nii'
    >>> applyxfm4d.inputs.reference = example_data('functional_mcf.nii')
    >>> result = applyxfm.run() # doctest: +SKIP

    """

    _cmd = 'applyxfm4D'
    input_spec = ApplyXfm4DInputSpec
    output_spec = ApplyXfm4DOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if isdefined(out_file):
            out_file = path.realpath(out_file)
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = self._gen_fname(
                self.inputs.in_file, suffix='_warp4D')
        return path.abspath(out_file)



# Define the input specification for QiTgv
class QiTgvInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, desc='Input file', mandatory=True, position=0,
                   argstr='%s')
    out_file = File(desc='Output file', position=1,
                    argstr='--out=%s')  # Optional
    alpha = traits.Float(desc='Alpha parameter', position=2,
                         argstr='--alpha=%f', usedefault=False)  # Optional


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
            outputs['out_file'] = os.path.abspath(
                self.inputs.out_file)  # Ensure full path
        else:
            # Use fname_presuffix to add '_tgv' to the basename, ensure it's in the current working directory
            outputs['out_file'] = fname_presuffix(self.inputs.in_file,
                                                  suffix='_tgv',
                                                  newpath=os.getcwd())
        return outputs


# Define the input specification for QiJsr
class QiJsrInputSpec(CommandLineInputSpec):
    spgr_file = File(exists=True, desc='SPGR input file', mandatory=True,
                     position=0, argstr='%s')
    ssfp_file = File(exists=True, desc='SSFP input file', mandatory=True,
                     position=1, argstr='%s')
    b1_file = File(exists=True, desc='B1 map file', mandatory=True, position=2,
                   argstr='--B1=%s')
    mask_file = File(exists=True, desc='Mask file', position=3,
                     argstr='--mask=%s', mandatory=False)  # Optional
    npsi = traits.Int(6, desc='Number of PSI components', usedefault=True,
                      position=4, argstr='--npsi=%d')
    json_file = File(exists=True, desc='Input JSON file', position=5,
                     argstr='--json=%s')


# Define the output specification for QiJsr
class QiJsrOutputSpec(TraitedSpec):
    t2_map_file = File(desc='Path to the generated T2 map file', exists=True)
    t1_map_file = File(desc='Path to the generated T1 map file', exists=True)


# Define the custom command-line wrapper for QiJsr
class QiJsr(CommandLine):
    _cmd = 'qi jsr'  # The command should map to "qi jsr"
    input_spec = QiJsrInputSpec
    output_spec = QiJsrOutputSpec

    # Override _list_outputs to specify the expected output files.
    def _list_outputs(self):
        outputs = self.output_spec().get()

        # Use the working directory where the node is executed.
        output_dir = os.path.abspath(self.inputs.cwd) \
            if hasattr(self, 'inputs') and hasattr(
            self.inputs, 'cwd') else os.getcwd()

        # Define the paths to the output files based on the output directory.
        outputs['t2_map_file'] = os.path.join(output_dir, 'JSR_T2.nii.gz')
        outputs['t1_map_file'] = os.path.join(output_dir, 'JSR_T1.nii.gz')

        return outputs
