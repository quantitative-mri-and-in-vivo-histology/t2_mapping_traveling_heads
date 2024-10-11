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
from workflows.preprocessing_workflows import \
    denoise_mag_and_phase_in_complex_domain_workflow, \
    motion_correction_mag_and_phase_workflow, create_brain_mask_workflow, \
    register_image_workflow
from utils.io import ExplicitPathDataSink
import nipype.interfaces.fsl as fsl


class ThreeDimEpiDataset:

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
                                 validate=False)
        self.default_pattern = 'sub-{subject}/ses-{session}/{datatype}/' \
                               'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                               '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'
        self.patterns = [self.default_pattern]
        self._nifti_read_ext = ['.nii', '.nii.gz']
        self._nifti_write_ext = '.nii.gz'

    @abstractmethod
    def prepare_data_workflow(self, base_dir=os.getcwd(), name="prepare_data",
                              subject=None, session=None, run=None):
        pass

    @staticmethod
    def subtract_background_phase(magnitude_file, phase_file):
        import nibabel as nib
        import numpy as np
        import os

        base_dir = os.getcwd()
        mag_nib = nib.load(magnitude_file)
        phase_nib = nib.load(phase_file)
        mag = mag_nib.get_fdata()
        phase = phase_nib.get_fdata()

        com = mag * np.exp(1.0j * phase)
        hip = com[..., 1::2] * np.conj(com[..., 0::2])

        phase_bg_sub = np.angle(hip) / 2.0
        mag_bg_sub = np.sqrt(np.abs(hip))

        phase_bg_sub = phase_bg_sub[..., (0, 3, 1, 4, 2, 5)]
        mag_bg_sub = mag_bg_sub[..., (0, 3, 1, 4, 2, 5)]

        phase_bg_sub_nii = nib.Nifti1Image(phase_bg_sub, phase_nib.affine,
                                           phase_nib.header)
        mag_bg_sub_nii = nib.Nifti1Image(mag_bg_sub, mag_nib.affine,
                                         mag_nib.header)

        magnitude_out_file = os.path.join(base_dir,
                                          "{}{}".format("magnitude", ".nii.gz"))
        phase_out_file = os.path.join(base_dir,
                                      "{}{}".format("phase", ".nii.gz"))

        nib.save(mag_bg_sub_nii, magnitude_out_file)
        nib.save(phase_bg_sub_nii, phase_out_file)

        return magnitude_out_file, phase_out_file

    @staticmethod
    def compute_t2_t1_amplitude_maps(magnitude_file,
                                      phase_file,
                                      mask_file,
                                      b1_map_file,
                                      repetition_time,
                                      flip_angle,
                                      delta_phi
                                      ):
        from T2T1AM import cal_T2T1AM
        import os

        base_dir = os.getcwd()
        output_dir = base_dir

        cal_T2T1AM(magnitude_file, phase_file, mask_file, b1_map_file,
                   repetition_time, flip_angle, delta_phi, outputdir=output_dir)

        t2_map_file = os.path.join(base_dir, "T2_.nii.gz")
        t1_map_file = os.path.join(base_dir, "T1_.nii.gz")
        am_map_file = os.path.join(base_dir, "Am_.nii.gz")

        return t2_map_file, t1_map_file, am_map_file

    def preprocess_relaxation_images_workflow(self, base_dir=os.getcwd(),
                                              name="preprocess_relaxation_images",
                                              subject=None, session=None,
                                              run=None):
        sub_ses_run_combinations = self.get_subject_session_run_combinations(
            subject=subject, session=session, run=run)

        inputs = []
        for combination in sub_ses_run_combinations:
            input_dict = dict(
                subject=combination["subject"],
                session=combination["session"],
                run=combination["run"],
                b1_map_file=self.get_b1_plus_relative_t2w_adjusted_map(
                    **combination, generate=True),
                b1_map_t2w_registered_file=self.get_b1_plus_relative_t2w_adjusted_registered_map(
                    **combination, generate=True),
                b1_anat_ref_file=self.get_b1_anat_ref(
                    **combination, generate=True),
                t2w_phase_radian_raw_file=self.get_t2w_phase_radian_raw(
                    **combination, generate=True),
                t2w_phase_radian_preprocessed_file=self.get_t2w_phase_radian_preprocessed(
                    **combination,
                    generate=True),
                t2w_magnitude_raw_file=self.get_t2w_magnitude_raw(
                    **combination),
                t2w_magnitude_preprocessed_file=self.get_t2w_magnitude_preprocessed(
                    **combination,
                    generate=True),
                t2w_brain_mask_file=self.get_brain_mask_for_t2w(**combination,
                                                                generate=True)
            )
            inputs.append(input_dict)

        wf = pe.Workflow(name=name)
        wf.base_dir = base_dir

        # set up bids input node
        input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                          name='bids_input_node')
        keys = inputs[0].keys()
        input_node.iterables = [
            (key, [input_dict[key] for input_dict in inputs]) for key in keys]
        input_node.synchronize = True

        # denoise in T2w images
        denoise_wf = denoise_mag_and_phase_in_complex_domain_workflow()
        wf.connect(input_node, "t2w_phase_radian_raw_file",
                   denoise_wf, "input_node.phase_file")
        wf.connect(input_node, "t2w_magnitude_raw_file",
                   denoise_wf, "input_node.magnitude_file")

        # correct motion in T2w images
        motion_correction_wf = motion_correction_mag_and_phase_workflow()
        wf.connect(denoise_wf, "output_node.magnitude_file",
                   motion_correction_wf, "input_node.magnitude_file")
        wf.connect(denoise_wf, "output_node.phase_file",
                   motion_correction_wf, "input_node.phase_file")

        # subtract background phase in T2w images
        subtract_background_phase_node = Node(Function(
            input_names=['magnitude_file', 'phase_file'],
            output_names=['magnitude_file', 'phase_file'],
            function=ThreeDimEpiDataset.subtract_background_phase),
            name='subtract_background_phase')
        wf.connect(motion_correction_wf, "output_node.magnitude_file",
                   subtract_background_phase_node, "magnitude_file")
        wf.connect(motion_correction_wf, "output_node.phase_file",
                   subtract_background_phase_node, "phase_file")

        # create brain mask for T2w images
        create_brain_mask_wf = create_brain_mask_workflow()
        wf.connect(subtract_background_phase_node, "magnitude_file",
                   create_brain_mask_wf, "input_node.in_file")

        # register B1 map to T2w magnitude image
        register_b1_map_to_t2w_wf = register_image_workflow()
        wf.connect(input_node, "b1_map_file",
                   register_b1_map_to_t2w_wf, "input_node.moving_file")
        wf.connect(input_node, "b1_anat_ref_file",
                   register_b1_map_to_t2w_wf, "input_node.reference_file")
        wf.connect(subtract_background_phase_node, "magnitude_file",
                   register_b1_map_to_t2w_wf, "input_node.target_file")

        # write preprocessed T2w magnitude image
        t2w_magnitude_preproc_data_sink = Node(
            ExplicitPathDataSink(), name='t2w_magnitude_preproc_data_sink')
        wf.connect(subtract_background_phase_node, "magnitude_file",
                   t2w_magnitude_preproc_data_sink, "in_file")
        wf.connect(input_node, "t2w_magnitude_preprocessed_file",
                   t2w_magnitude_preproc_data_sink, "out_file")

        # write preprocessed T2w phase image
        t2w_phase_preproc_data_sink = Node(
            ExplicitPathDataSink(), name='t2w_phase_preproc_data_sink')
        wf.connect(subtract_background_phase_node, "phase_file",
                   t2w_phase_preproc_data_sink, "in_file")
        wf.connect(input_node, "t2w_phase_radian_preprocessed_file",
                   t2w_phase_preproc_data_sink, "out_file")

        # write T2w brain mask image
        t2w_brain_mask_data_sink = Node(
            ExplicitPathDataSink(), name='brain_mask_data_sink')
        wf.connect(create_brain_mask_wf, "output_node.out_file",
                   t2w_brain_mask_data_sink, "in_file")
        wf.connect(input_node, "t2w_brain_mask_file",
                   t2w_brain_mask_data_sink, "out_file")

        # write B1 map registered to T2w image
        b1_map_registered_data_sink = Node(
            ExplicitPathDataSink(), name='b1_map_registered_data_sink')
        wf.connect(register_b1_map_to_t2w_wf, "output_node.out_file",
                   b1_map_registered_data_sink, "in_file")
        wf.connect(input_node, "b1_map_t2w_registered_file",
                   b1_map_registered_data_sink, "out_file")

        return wf

    def estimate_relaxation_maps_workflow(self, base_dir=os.getcwd(),
                                          name="estimate_relaxation_maps",
                                          subject=None, session=None, run=None):
        sub_ses_run_combinations = self.get_subject_session_run_combinations(
            subject=subject, session=session, run=run)

        inputs = []
        for combination in sub_ses_run_combinations:
            t2w_magnitude_raw_file = self.get_t2w_magnitude_raw(
                **combination)
            fa_nominal_in_degrees = t2w_magnitude_raw_file.entities["FlipAngle"]
            repetition_time_in_sec = 1000.0*t2w_magnitude_raw_file.entities["RepetitionTimeExcitation"]

            input_dict = dict(
                subject=combination["subject"],
                session=combination["session"],
                run=combination["run"],
                b1_map_t2w_registered_file=self.get_b1_plus_relative_t2w_adjusted_registered_map(
                    **combination),
                t2w_phase_radian_preprocessed_file=self.get_t2w_phase_radian_preprocessed(
                    **combination),
                t2w_magnitude_preprocessed_file=self.get_t2w_magnitude_preprocessed(
                    **combination),
                t2w_brain_mask_file=self.get_brain_mask_for_t2w(**combination),
                r1_map_file=self.get_r1_map(**combination, generate=True),
                r2_map_file=self.get_r2_map(**combination, generate=True),
                t1_map_file=self.get_t1_map(**combination, generate=True),
                t2_map_file=self.get_t2_map(**combination, generate=True),
                am_map_file=self.get_am_map(**combination, generate=True),
                fa_nominal_in_degrees=fa_nominal_in_degrees,
                repetition_time_in_sec=repetition_time_in_sec
            )
            inputs.append(input_dict)

        wf = pe.Workflow(name=name)
        wf.base_dir = base_dir

        # set up bids input node
        input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                          name='bids_input_node')
        keys = inputs[0].keys()
        input_node.iterables = [
            (key, [input_dict[key] for input_dict in inputs]) for key in keys]
        input_node.synchronize = True

        # scale B1 map to percent
        scale_b1_to_percent = pe.Node(
            fsl.ImageMaths(op_string='-mul 100.0'),
            name="scale_b1_to_percent")
        wf.connect(input_node, "b1_map_t2w_registered_file",
                         scale_b1_to_percent, "in_file")

        # compute T1, T2, AM
        compute_t2_t1_am_node = Node(
            Function(input_names=["magnitude_file", "phase_file", "mask_file",
                                  "b1_map_file", "repetition_time",
                                  "flip_angle",
                                  "delta_phi"],
                     output_names=["t2_map_file", "t1_map_file", "am_map_file"],
                     function=ThreeDimEpiDataset.compute_t2_t1_amplitude_maps),
            name="compute_t2_t1_am")
        compute_t2_t1_am_node.inputs.delta_phi = [1, 1.5, 2, 3, 4, 5]
        wf.connect(input_node, "t2w_brain_mask_file",
                   compute_t2_t1_am_node, "mask_file")
        wf.connect(input_node, "t2w_magnitude_preprocessed_file",
                   compute_t2_t1_am_node, "magnitude_file")
        wf.connect(input_node, "t2w_phase_radian_preprocessed_file",
                   compute_t2_t1_am_node, "phase_file")
        wf.connect(scale_b1_to_percent, "out_file",
                   compute_t2_t1_am_node, "b1_map_file")
        wf.connect(input_node, "fa_nominal_in_degrees",
                   compute_t2_t1_am_node, "flip_angle")
        wf.connect(input_node, "repetition_time_in_sec",
                   compute_t2_t1_am_node, "repetition_time")

        # compute R1 map
        compute_r1 = pe.Node(
            fsl.ImageMaths(op_string='-recip'),
            name="compute_r1")
        wf.connect(compute_t2_t1_am_node, "t1_map_file",
                   compute_r1, "in_file")

        # compute R2 map
        compute_r2 = pe.Node(
            fsl.ImageMaths(op_string='-recip'),
            name="compute_r2")
        wf.connect(compute_t2_t1_am_node, "t2_map_file",
                   compute_r2, "in_file")

        # write R1 map
        r1_map_data_sink = Node(
            ExplicitPathDataSink(), name='r1_map_data_sink')
        wf.connect(compute_r1, "out_file",
                   r1_map_data_sink, "in_file")
        wf.connect(input_node, "r1_map_file",
                   r1_map_data_sink, "out_file")

        # write R2 map
        r2_map_data_sink = Node(
            ExplicitPathDataSink(), name='r2_map_data_sink')
        wf.connect(compute_r2, "out_file",
                   r2_map_data_sink, "in_file")
        wf.connect(input_node, "r2_map_file",
                   r2_map_data_sink, "out_file")

        # write T1 map
        t1_map_data_sink = Node(
            ExplicitPathDataSink(), name='t1_map_data_sink')
        wf.connect(compute_t2_t1_am_node, "t1_map_file",
                   t1_map_data_sink, "in_file")
        wf.connect(input_node, "t1_map_file",
                   t1_map_data_sink, "out_file")

        # write T2 map
        t2_map_data_sink = Node(
            ExplicitPathDataSink(), name='t2_map_data_sink')
        wf.connect(compute_t2_t1_am_node, "t2_map_file",
                   t2_map_data_sink, "in_file")
        wf.connect(input_node, "t2_map_file",
                   t2_map_data_sink, "out_file")

        # write T2 map
        am_map_data_sink = Node(
            ExplicitPathDataSink(), name='am_map_data_sink')
        wf.connect(compute_t2_t1_am_node, "am_map_file",
                   am_map_data_sink, "in_file")
        wf.connect(input_node, "am_map_file",
                   am_map_data_sink, "out_file")

        return wf

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

    def get_t2w_magnitude_raw(self, subject, session, run=None,
                              allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, desc=None,
            suffix='T2w', part='mag', extension=self._nifti_read_ext,
            datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_t2w_magnitude_preprocessed(self, subject, session, run=None,
                                       allow_multiple_files=False,
                                       generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            suffix='T2w', part='mag', extension=self._nifti_read_ext,
            datatype='anat', desc='preproc')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_t2w_phase_radian_raw(self, subject, session, run=None,
                                 allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, desc="radian",
            suffix='T2w', part='phase', extension=self._nifti_read_ext,
            datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_t2w_phase_radian_preprocessed(self, subject, session, run=None,
                                          allow_multiple_files=False,
                                          generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            suffix='T2w', part='phase', extension=self._nifti_read_ext,
            datatype='anat', desc='radianPreproc')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_am_map(self, subject, session, run=None,
                   allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            suffix="AmMap", extension=self._nifti_read_ext, datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

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

    def get_b0_anat_ref(self, subject, session, run=None,
                        allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B0",
            suffix="magnitude", extension=self._nifti_read_ext, datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b0_map_radian(self, subject, session, run=None,
                          allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B0",
            suffix="phasediff", extension=self._nifti_read_ext, datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_anat_ref(self, subject, session, run=None,
                        allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B1",
            suffix="magnitude", extension=self._nifti_read_ext, datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_plus_relative_unadjusted_map(self, subject, session, run=None,
                                            allow_multiple_files=False,
                                            generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B1",
            desc="unadjusted",
            suffix="B1map", extension=self._nifti_read_ext, datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_plus_relative_t2w_adjusted_map(self, subject, session, run=None,
                                              allow_multiple_files=False,
                                              generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B1",
            desc="T2wAdjusted",
            suffix="B1map", extension=self._nifti_read_ext, datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_plus_relative_t2w_adjusted_registered_map(self, subject, session, run=None,
                                              allow_multiple_files=False,
                                              generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B1",
            desc="T2wAdjustedRegistered",
            suffix="B1map", extension=self._nifti_read_ext, datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_brain_mask_for_t2w(self, subject, session, run=None,
                               allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            desc="brain", suffix="mask", extension=self._nifti_read_ext,
            datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)
