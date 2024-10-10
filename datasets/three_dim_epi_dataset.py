import os
import sys
import argparse
import multiprocessing
from threading import Thread

from nipype.pipeline import Workflow
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from nipype import Node, Function
from pathlib import Path
from abc import ABC, abstractmethod
from bids.layout import BIDSLayout
from nipype_utils import BidsRename, BidsOutputFormatter, create_output_folder


class ThreeDimEpiDataset:

    def __init__(self, bids_root, derivatives_output_folder, derivatives=None):
        self.bids_root = bids_root
        self.layout = BIDSLayout(bids_root, derivatives=derivatives,
                                 validate=False)
        self.derivatives_output_folder = derivatives_output_folder
        self.default_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                               'sub-{subject}_ses-{session}[_acquisition-{acquisition}]' \
                               '[_run-{run}][_desc-{desc}]_{suffix}.{extension}'
        self.patterns = [self.default_pattern]

    def get_subjects(self):
        return self.layout.get_subjects()

    def get_sessions(self, subject):
        return self.layout.get_sessions(subject=subject)

    def get_subject_session_run_combinations(self, subject=None, session=None,
                                             run=None):
        """
        Generates a list of combinations of subjects, sessions, and runs.
        If `subject`, `session`, or `run` is None, retrieves all available options from the BIDS layout.
        """
        # Get a list of subjects, or all subjects if `subject` is None.
        subjects = [subject] if subject else self.layout.get_subjects()
        combinations = []

        for subject_id in subjects:
            # Get a list of sessions for the subject, or all sessions if `session` is None.
            sessions = [session] if session else self.layout.get_sessions(
                subject=subject_id)

            if sessions:
                for session_id in sessions:
                    # Retrieve valid runs for the subject and session based on the data available in the layout.
                    valid_runs = self.layout.get(
                        return_type='id',
                        subject=subject_id,
                        session=session_id,
                        target='run',
                        suffix='T2w',
                        part="phase",
                        extension="nii.gz"
                    )
                    # Use the specified `run` if provided, otherwise use valid runs from the layout.
                    runs = [run] if isinstance(run, str) else valid_runs

                    # If no valid runs exist, use [None] to indicate no run-specific processing.
                    if len(runs) == 0:
                        runs = [None]

                    # Create combinations of subject, session, and run.
                    for run_id in runs:
                        combinations.append({
                            "subject": subject_id,
                            "session": session_id,
                            "run": run_id
                        })

        return combinations

    def get_file(self, allow_multiple_files=False, **query):
        """
        Retrieve a single file or multiple files based on the query parameters.

        Parameters:
            allow_multiple_files (bool): If False (default), raisession an error if more than one file is found.
                                         If True, allows multiple files and returns the list.
            **query: Arguments to pass to the BIDSLayout.get() method.

        Returns:
            str: Path to the single file found if `allow_multiple_files` is False.
            list of str: Paths to multiple files if `allow_multiple_files` is True.

        Raisession:
            FileNotFoundError: If no files match the query.
            ValueError: If more than one file matches the query and `allow_multiple_files` is False.
        """
        files = self.layout.get(**query)

        if len(files) == 0:
            raise FileNotFoundError(f"No files found for query: {query}")

        if allow_multiple_files:
            return [f for f in files]

        if len(files) > 1:
            raise ValueError(
                f"Multiple files found for query: {query}. Expected exactly one.")

        return files[0]

    def get_t2w_magnitude_raw(self, subject, session, run=None,
                              allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            suffix='T2w',
            part='mag',
            extension=['nii', 'nii.gz'],
            run=run
        )

    def get_t2w_magnitude_preprocessed(self, subject, session, run=None,
                                       allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            suffix='T2w',
            part='mag',
            extension=['nii', 'nii.gz'],
            run=run
        )

    def get_t2w_phase_raw(self, subject, session, run=None,
                          allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            suffix='T2w',
            part='phase',
            extension=['nii', 'nii.gz'],
            run=run
        )

    def get_t2w_phase_preprocessed(self, subject, session, run=None,
                                   allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            suffix='T2w',
            part='phase',
            extension=['nii', 'nii.gz'],
            run=run
        )

    def get_b0_anat_ref(self, subject, session, run=None, generate=False,
                        extension='nii.gz'):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B0Anat",
            suffix="magnitude", extension=extension, datatype='fmap')

        if generate:
            relative_path = self.layout.build_path(
                entities, path_patterns=self.patterns,
                validate=False, absolute_paths=False)
            absolute_path = os.path.join(self.derivatives_output_folder,
                                         relative_path)
            return absolute_path
        else:
            return self.get_file(**entities)

    def get_b0_map_radian(self, subject, session, run=None, generate=False,
                        extension='nii.gz'):
        entities = dict(
            subject=subject, session=session, run=run, desc="radian",
            suffix="B0map", extension=extension, datatype='fmap')

        if generate:
            relative_path = self.layout.build_path(
                entities, path_patterns=self.patterns,
                validate=False, absolute_paths=False)
            absolute_path = os.path.join(self.derivatives_output_folder,
                                         relative_path)
            return absolute_path
        else:
            return self.get_file(**entities)

    def get_b1_anat_ref(self, subject, session, run=None, generate=False,
                        extension='nii.gz'):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B1Anat",
            suffix="magnitude", extension=extension, datatype='fmap')

        if generate:
            relative_path = self.layout.build_path(
                entities, path_patterns=self.patterns,
                validate=False, absolute_paths=False)
            absolute_path = os.path.join(self.derivatives_output_folder,
                                         relative_path)
            return absolute_path
        else:
            return self.get_file(**entities)

    def get_b1_in_percent(self, subject, session, run=None, generate=False,
                        extension='nii.gz'):
        entities = dict(
            subject=subject, session=session, run=run, desc="percent",
            suffix="B1map", extension=extension, datatype='fmap')

        if generate:
            relative_path = self.layout.build_path(
                entities, path_patterns=self.patterns,
                validate=False, absolute_paths=False)
            absolute_path = os.path.join(self.derivatives_output_folder,
                                         relative_path)
            return absolute_path
        else:
            return self.get_file(**entities)

    def get_b1_in_percent_for_t2w(self, subject, session, run=None,
                                  allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            suffix='T2w',
            part='phase',
            extension=['nii', 'nii.gz'],
            run=run
        )

    def get_brain_mask_for_t2w(self, subject, session, run=None,
                               allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            suffix='T2w',
            part='phase',
            extension=['nii', 'nii.gz'],
            run=run
        )

    def get_corrected_b1_map(self, subject, session, run=None, generate=False,
                             extension='nii.gz'):
        entities = {
            'subject': subject,
            'session': session,
            'suffix': 'B1map',
            'run': run,
            'extension': extension,
            'desc': 'corrected',
            'datatype': 'fmap'
        }

        if generate:
            # Use build_path to generate the output path for saving
            pattern = 'subject-{subject}/session-{session}/{datatype}/subject-{subject}_session-{session}[_run-{run}]_desc-{desc}_b1map.{extension}'
            relative_path = self.layout.build_path(entities,
                                                   path_patterns=[pattern],
                                                   validate=False,
                                                   absolute_paths=False)
            absolute_path = os.path.join(self.derivatives_output_folder,
                                         relative_path)
            return absolute_path
        else:
            return self.get_file(**entities)

    @abstractmethod
    def prepare_data_workflow(self, base_dir=os.getcwd(), name="prepare_data",
                              subject=None, session=None, run=None):
        pass

    @abstractmethod
    def preprocess_relaxation_images_workflow(self, base_dir=os.getcwd(),
                                              name="preprocess_relaxation_images",
                                              subject=None, session=None,
                                              run=None):
        pass

    @abstractmethod
    def estimate_relaxation_maps_workflow(self, base_dir=os.getcwd(),
                                          name="estimate_relaxation_maps",
                                          subject=None, session=None, run=None):
        pass
