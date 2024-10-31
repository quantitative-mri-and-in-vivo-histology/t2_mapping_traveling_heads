import json
import os
import shutil

from bids.layout import BIDSLayout, BIDSFile
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, traits, Directory
)
from nipype.interfaces.io import DataSink
from nipype.utils.filemanip import copyfile
from nipype.interfaces.base import isdefined
from utils.bids_config import DEFAULT_BIDS_OUTPUT_PATTERN


class BidsOutputWriterInputSpec(BaseInterfaceInputSpec):

    def __init__(self, *args, **kwargs):
        if 'pattern' not in kwargs:
            kwargs['pattern'] = DEFAULT_BIDS_OUTPUT_PATTERN
        super(BidsOutputWriterInputSpec, self).__init__(*args, **kwargs)

    in_file = File(exists=True, desc="Input BIDS file to rename",
                   mandatory=True)
    template_file = File(exists=True, desc="Template file", mandatory=True)
    json_dict = traits.Dict(desc="Dictionary to be saved as a JSON file",
                            mandatory=False)
    entity_overrides = traits.Dict(
        desc="Dictionary of tags to override in file entities", mandatory=False)
    pattern = traits.Str(mandatory=False, desc="Pattern for output filename",
                         default_value=DEFAULT_BIDS_OUTPUT_PATTERN)
    output_dir = traits.Directory(
        desc="Directory where the renamed file will be saved", mandatory=True)


# Define the output spec with the output filename
class BidsOutputWriterOutputSpec(TraitedSpec):
    out_file = File(desc="Generated output filename")
    out_json = File(desc="Generated JSON filename", mandatory=False)


# Define the interface for renaming the file based on BIDS metadata
class BidsOutputWriter(DataSink):
    input_spec = BidsOutputWriterInputSpec
    output_spec = BidsOutputWriterOutputSpec

    def __init__(self, **kwargs):
        super(BidsOutputWriter, self).__init__(**kwargs)

    def _run_interface(self, runtime):
        # Load BIDS layout and parse file entities
        # layout = BIDSLayout(self.inputs.template_file, validate=False)

        layout = BIDSLayout(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(self.inputs.template_file)))),
                   validate=False)
        file_entities = layout.parse_file_entities(self.inputs.template_file)

        # Apply entity_overrides if provided
        if self.inputs.entity_overrides:
            file_entities.update(self.inputs.entity_overrides)

        # Build the output filename based on the pattern and updated entities
        filename = layout.build_path(file_entities, validate=False,
                                     path_patterns=[self.inputs.pattern],
                                     absolute_paths=False)

        # Create output directory if it doesn't exist
        out_file = os.path.join(self.inputs.output_dir, filename)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # Perform the renaming operation (copying or renaming in place)
        copyfile(self.inputs.in_file, out_file, copy=True)

        # Store the output image filename
        self._out_file = out_file

        # If json_dict is provided, save it as a JSON file with the same name as the image file
        if self.inputs.json_dict:
            json_filename = out_file.replace('.nii.gz', '.json')
            with open(json_filename, 'w') as json_file:
                json.dump(self.inputs.json_dict, json_file, indent=4)
            self._out_json = json_filename
        else:
            self._out_json = None

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs[
            'out_file'] = self._out_file  # Return the filename generated in _run_interface
        if self.inputs.json_dict:
            outputs[
                'out_json'] = self._out_json  # Return the JSON filename if generated
        return outputs


class ExplicitDataSinkInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="Input BIDS file to rename", mandatory=True)
    filename = File(desc="Filename for the output file", mandatory=False)  # Add filename to input spec
    output_dir = Directory(desc="Output directory", mandatory=True)  # Add filename to input spec

class ExplicitDataSink(DataSink):

    input_spec = ExplicitDataSinkInputSpec

    def _run_interface(self, runtime):
        # Set directory and filename from input spec or default to the original file
        out_file = os.path.join(self.inputs.output_dir, self.inputs.filename)

        # Ensure the target directory exists
        os.makedirs(self.inputs.output_dir, exist_ok=True)

        # Move or copy the file to the desired location
        shutil.copy(self.inputs.in_file, out_file)

        # Store the output image filename
        self._out_file = out_file

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._out_file
        return outputs
