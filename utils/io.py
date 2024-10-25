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
            f"Expected one NIfTI file, found {len(nifti_files)} for query {query_dict}. NifTI files are: {nifti_files}")

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


def find_file(layout, **query_dict):
    """
    Given a BIDS layout and query parameters, find the NIfTI image.

    Parameters:
    - layout: A BIDSLayout object.
    - query_dict: A dictionary of query parameters for searching the BIDS dataset (e.g., subject, session, suffix, etc.).

    Returns:
    - A tuple (nifti_file, json_dict):
      - nifti_file: The path to the NIfTI image file.
    """
    # Search for the NIfTI file
    files = layout.get(**query_dict)
    if len(files) != 1:
        raise ValueError(
            f"Expected one file, found {len(files)} for query {query_dict}. Files are: {files}")

    file = files[0]

    return file



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


def create_output_folder(subject, session, datatype):
    folder = f'sub-{subject}/ses-{session}'
    if datatype is not None:
        folder = "{}/{}".format(folder, datatype)
    return folder


def get_nifti_fileparts(file_path):
    """
    Splits a NIfTI file path into its directory, basename (filename without extension), and extension (.nii or .nii.gz).
    Throws an error if the file does not have a valid NIfTI extension.

    Parameters:
    - file_path: Full path to the NIfTI file (.nii or .nii.gz)

    Returns:
    - directory: The directory of the file
    - basename: The filename without the extension
    - extension: The extension of the file (.nii or .nii.gz)

    Raises:
    - ValueError: If the file does not have a .nii or .nii.gz extension
    """
    directory = os.path.dirname(file_path)

    if file_path.endswith('.nii.gz'):
        basename, _ = os.path.splitext(
            os.path.splitext(os.path.basename(file_path))[0])
        extension = '.nii.gz'
    elif file_path.endswith('.nii'):
        basename, extension = os.path.splitext(os.path.basename(file_path))
    else:
        raise ValueError(
            f"Error: {file_path} is not a valid NIfTI file (must end with .nii or .nii.gz).")

    return directory, basename, extension



