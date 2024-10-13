import os
from abc import ABC, abstractmethod
from bids.layout import BIDSLayout


class BidsDataset(ABC):

    def __init__(self, bids_root, derivatives_output_folder, derivatives=None):
        self.bids_root = bids_root
        self.derivatives = derivatives
        self.derivatives_output_folder = derivatives_output_folder
        if self.derivatives is None:
            self.derivatives = []
        if isinstance(self.derivatives, list):
            self.derivatives.append(self.derivatives_output_folder)
        else:
            self.derivatives = [self.derivatives,
                                self.derivatives_output_folder]
        self.layout = BIDSLayout(bids_root, derivatives=self.derivatives,
                                 validate=False, is_derivative=False)
        self.default_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                               'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                               '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'
        self.patterns = [self.default_pattern]
        self._nifti_read_ext = ['.nii', '.nii.gz']
        self._nifti_write_ext = '.nii.gz'

    @abstractmethod
    def prepare_dataset_workflow(self, base_dir=os.getcwd(), name="prepare_dataset",
                                 subject=None, session=None, run=None):
        pass

    @abstractmethod
    def preprocess_workflow(self, base_dir=os.getcwd(),
                            name="preprocess",
                            subject=None, session=None,
                            run=None):
        pass

    @abstractmethod
    def estimate_workflow(self, base_dir=os.getcwd(),
                          name="estimate",
                          subject=None, session=None, run=None):
        pass

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
                    runs = self.layout.get_runs(subject=subject_id, session=session_id)
                    # Create combinations of subject, session, and run.
                    for run_id in runs:
                        combinations.append({
                            "subject": subject_id,
                            "session": session_id,
                            "run": run_id
                        })
        return combinations

    def get_r1_map(self, subject, session, run=None,
                   allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            suffix="R1Map", extension=self._nifti_read_ext, datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_r2_map(self, subject, session, run=None,
                   allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            suffix="R2Map", extension=self._nifti_read_ext, datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_t1_map(self, subject, session, run=None,
                   allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            suffix="T1Map", extension=self._nifti_read_ext, datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_t2_map(self, subject, session, run=None,
                   allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            suffix="T2Map", extension=self._nifti_read_ext, datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def _query_file(self, allow_multiple_files=False, **query):
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
        files = self.layout.get(**query, invalid_filters=True)

        if len(files) == 0:
            raise FileNotFoundError(f"No files found for query: {query}")

        if allow_multiple_files:
            return [f for f in files]

        if len(files) > 1:
            raise ValueError(
                f"Multiple files found for query: {query}. Expected exactly one.")

        return files[0]

    def _build_path(self, entities, patterns=None):
        if patterns is None:
            patterns = self.patterns

        if isinstance(entities["extension"], list) and self._nifti_write_ext in \
                entities["extension"]:
            entities["extension"] = self._nifti_write_ext

        relative_path = self.layout.build_path(
            entities, path_patterns=patterns,
            validate=False, absolute_paths=False)
        absolute_path = os.path.join(self.derivatives_output_folder,
                                     relative_path)
        return absolute_path

    def _get_file(self, entities, allow_multiple_files=False,
                  generate=False):
        if generate:
            return self._build_path(entities)
        else:
            return self._query_file(**entities,
                                    allow_multiple_files=allow_multiple_files)
