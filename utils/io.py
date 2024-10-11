import os
import shutil
from nipype.interfaces.io import DataSink
from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, File


class ExplicitPathDataSinkInputSpec(BaseInterfaceInputSpec):
    in_file = File(desc="Input file to be stored", exists=True, mandatory=True)
    out_file = File(desc="Explicit absolute path for the output file",
                    mandatory=True)


class ExplicitPathDataSinkOutputSpec(TraitedSpec):
    out_file = File(desc="Path to the saved output file")


class ExplicitPathDataSink(DataSink):
    input_spec = ExplicitPathDataSinkInputSpec
    output_spec = ExplicitPathDataSinkOutputSpec

    def _run_interface(self, runtime):
        """
        Overrides the run method to copy the input file to the specified output path.
        """
        in_file = self.inputs.in_file
        out_file = self.inputs.out_file

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # Copy the file to the specified location
        shutil.copy2(in_file, out_file)

        return runtime

    def _list_outputs(self):
        """
        Overrides the _list_outputs method to ensure the output file path is returned.
        """
        outputs = self._outputs().get()
        outputs['out_file'] = self.inputs.out_file
        return outputs