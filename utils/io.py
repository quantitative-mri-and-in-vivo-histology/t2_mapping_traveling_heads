import os
import shutil
import json
from nipype.interfaces.io import DataSink
from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, File


def write_minimal_bids_dataset_description(dataset_root, dataset_name):
    input_dict = dict(
        Name=dataset_name,
        BIDSVersion="1.6.0",
        DatasetType="derivative"
    )
    out_file = os.path.join(dataset_root, "dataset_description.json")
    if not os.path.exists(out_file):
        with open(out_file, 'w') as f:
            json.dump(input_dict, f, indent=4)

def write_json(data, filename, base_dir=None):
    import os
    import json

    if base_dir is None:
        base_dir = os.getcwd()
    output_dir = os.path.abspath(base_dir)
    out_file = os.path.join(output_dir, filename)

    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)

    return out_file


def find_image_and_json(layout, **query_dict):
    """
    Given a BIDS layout and query parameters, find the NIfTI image and its associated JSON sidecar.

    Parameters:
    - layout: A BIDSLayout object.
    - query_dict: A dictionary of query parameters for searching the BIDS dataset (e.g., subject, session, suffix, etc.).

    Returns:
    - A tuple (nifti_file, json_dict):
      - nifti_file: The path to the NIfTI image file.
      - json_dict: The contents of the associated JSON sidecar file as a dictionary.
    """
    # Search for the NIfTI file
    nifti_files = layout.get(**query_dict)
    if len(nifti_files) != 1:
        raise ValueError(
            f"Expected one NIfTI file, found {len(nifti_files)} for query {query_dict}")

    nifti_file = nifti_files[0]

    # Find the associated JSON file
    json_files = nifti_file.get_associations()
    if len(json_files) != 1:
        raise ValueError(
            f"Expected one JSON file, found {len(json_files)} for {nifti_file.path}")

    json_file = json_files[0].path

    # Load the JSON data
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

    return nifti_file, json_dict


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
