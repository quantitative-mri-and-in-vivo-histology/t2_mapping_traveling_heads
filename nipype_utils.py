import os
from os import path
from pathlib import Path
from bids.layout import BIDSLayout
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.utils.filemanip import copyfile
from nipype.interfaces.base import File, TraitedSpec, traits, isdefined
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec


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


def get_common_parent_directory(file_list):
    from pathlib import Path
    return Path(file_list[0]).parent.as_posix()


def select_first_file(file_list):
    return file_list[0] if isinstance(file_list, list) else file_list


def create_output_dir(base_dir, subject, session, run):
    import os
    if run is not None:
        output_dir = os.path.join(base_dir,
                                  f"sub-{subject}_ses-{session}_run-{run}")
    else:
        output_dir = os.path.join(base_dir, f"sub-{subject}_ses-{session}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def create_output_folder(subject, session):
    return f'sub-{subject}/ses-{session}'


class BidsRenameInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="Input BIDS file to rename")
    template_file = File(exists=True, desc="Template file")
    pattern = traits.Str(mandatory=True, desc="Pattern for output filename")


class BidsRenameOutputSpec(TraitedSpec):
    out_file = File(desc="Generated output filename")


class BidsRename(BaseInterface):
    input_spec = BidsRenameInputSpec
    output_spec = BidsRenameOutputSpec

    def __init__(self, **inputs):
        # Call the super class constructor to properly initialize the interface
        super(BidsRename, self).__init__(**inputs)

    def _run_interface(self, runtime):
        # This function handles the core functionality, which we run in the workflow
        layout = BIDSLayout(self.inputs.template_file, validate=False)

        # Parse the input file's entities
        file_entities = layout.parse_file_entities(self.inputs.template_file)

        # Build the output filename using the provided pattern
        # out_file = Path(self.inputs.in_file).parent.joinpath(Path(layout.build_path(file_entities, self.inputs.pattern, validate=False)).name)
        out_file = Path(runtime.cwd).joinpath(
            Path(layout.build_path(file_entities, self.inputs.pattern,
                                   validate=False)).name)

        copyfile(self.inputs.in_file, out_file, copy=True)

        # Store the result for use in _list_outputs
        self._out_file = out_file

        return runtime

    def _list_outputs(self):
        # This function returns the outputs of the interface
        outputs = self._outputs().get()
        outputs[
            'out_file'] = self._out_file  # Return the filename generated in _run_interface
        return outputs


from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits, isdefined
)
import os

# Define the input spec with optional output_dir
class BidsOutputFormatterInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="Input BIDS file to rename")
    subject = traits.Str(mandatory=True, desc="Subject id")
    session = traits.Str(mandatory=True, desc="Session id")
    run = traits.Either(None, traits.Int, desc="Run id (optional, can be None)")
    pattern = traits.Str(mandatory=True, desc="Pattern for output filename")
    output_dir = traits.Directory(desc="Directory where the renamed file will be saved (optional)")

# Define the output spec with the output filename
class BidsOutputFormatterOutputSpec(TraitedSpec):
    out_file = File(desc="Generated output filename")

# Define the interface for renaming the file based on BIDS metadata
class BidsOutputFormatter(BaseInterface):
    input_spec = BidsOutputFormatterInputSpec
    output_spec = BidsOutputFormatterOutputSpec

    def _run_interface(self, runtime):
        # Get the input values
        in_file = self.inputs.in_file
        subject = self.inputs.subject
        session = self.inputs.session
        run = self.inputs.run
        pattern = self.inputs.pattern

        # Determine the output directory; use runtime.cwd if output_dir is not provided
        output_dir = self.inputs.output_dir if isdefined(self.inputs.output_dir) else runtime.cwd

        # Construct the filename based on the pattern
        # Example pattern: "sub-{subject}_ses-{session}_run-{run}_desc-preproc.nii.gz"
        if run is not None and isdefined(run):
            # Replace placeholders with the provided values
            filename = pattern.format(subject=subject, session=session, run=run)
        else:
            # If run is None or not defined, remove the "run" part from the pattern
            filename = pattern.replace('_run-{run}', '')
            filename = filename.format(subject=subject, session=session)

        # Define the full output path
        out_file = os.path.join(output_dir, filename)

        # Perform the renaming operation (copying or renaming in place)
        copyfile(in_file, out_file, copy=True)

        # Store the output filename
        self._out_file = out_file

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs[
            'out_file'] = self._out_file  # Return the filename generated in _run_interface
        return outputs

