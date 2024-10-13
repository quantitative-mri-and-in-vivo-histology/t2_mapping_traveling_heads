import os

from PyQt6.uic.pyuic import generate
from nipype.interfaces.utility import IdentityInterface
import nipype.pipeline.engine as pe
from nipype import Node, Function
from abc import abstractmethod, ABC
from bids.layout import BIDSLayout
from datasets.bids_dataset import BidsDataset
from workflows.preprocessing_workflows import \
    denoise_mag_and_phase_in_complex_domain_workflow, \
    motion_correction_mag_and_phase_workflow, create_brain_mask_workflow, \
    register_image_workflow
from utils.io import ExplicitPathDataSink
import nipype.interfaces.fsl as fsl
from workflows.preprocessing_workflows import preprocess_ssfp
from workflows.parameter_estimation_workflows import estimate_relaxation_ssfp


class SsfpDataset(BidsDataset, ABC):

    def __init__(self, bids_root, derivatives_output_folder, derivatives=None):
        super().__init__(bids_root=bids_root,
                         derivatives_output_folder=derivatives_output_folder,
                         derivatives=derivatives)

    def preprocess_workflow(self, base_dir=os.getcwd(),
                            name="preprocess",
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
                t1w_fa_2_raw_file=self.get_t1w_fa_2_raw(
                    **combination),
                t1w_fa_2_preprocessed_file=self.get_t1w_fa_2_preprocessed(
                    **combination, generate=True),
                t1w_fa_13_raw_file=self.get_t1w_fa_13_raw(
                    **combination),
                t1w_fa_13_preprocessed_file=self.get_t1w_fa_13_preprocessed(
                    **combination, generate=True),
                t2w_fa_12_rf_180_raw_file=self.get_t2w_fa_12_rf_180_raw(
                    **combination),
                t2w_fa_12_rf_180_preprocessed_file=self.get_t2w_fa_12_rf_180_preprocessed(
                    **combination, generate=True),
                t2w_fa_49_rf_0_raw_file=self.get_t2w_fa_49_rf_0_raw(
                    **combination),
                t2w_fa_49_rf_0_preprocessed_file=self.get_t2w_fa_49_rf_0_preprocessed(
                    **combination, generate=True),
                t2w_fa_49_rf_180_raw_file=self.get_t2w_fa_49_rf_180_raw(
                    **combination),
                t2w_fa_49_rf_180_preprocessed_file=self.get_t2w_fa_49_rf_180_preprocessed(
                    **combination, generate=True),
                b1_anat_ref_file=self.get_b1_anat_ref(**combination),
                b1_map_file=self.get_b1_plus_relative_map(**combination),
                b1_map_registered_file=self.get_b1_plus_relative_registered_map(
                    **combination, generate=True),
                brain_mask_file = self.get_brain_mask_file(
                    **combination, generate=True)
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

        preprocess_ssfp_wf = preprocess_ssfp()

        wf.connect([(input_node, preprocess_ssfp_wf, [
            ('b1_map_file', 'input_node.b1_map_file'),
            ('b1_anat_ref_file', 'input_node.b1_anat_ref_file'),
            ('t1w_fa_2_raw_file', 'input_node.t1w_fa_2_file'),
            ('t1w_fa_13_raw_file', 'input_node.t1w_fa_13_file'),
            ('t2w_fa_12_rf_180_raw_file', 'input_node.t2w_fa_12_rf_180_file'),
            ('t2w_fa_49_rf_0_raw_file', 'input_node.t2w_fa_49_rf_0_file'),
            ('t2w_fa_49_rf_180_raw_file', 'input_node.t2w_fa_49_rf_180_file'),

        ])])

        # write output files
        in_out_pairs = [
            ("output_node.b1_map_file", "b1_map_registered_file"),
            ("output_node.t1w_fa_2_file", "t1w_fa_2_preprocessed_file"),
            ("output_node.t1w_fa_13_file", "t1w_fa_13_preprocessed_file"),
            ("output_node.t2w_fa_12_rf_180_file",
             "t2w_fa_12_rf_180_preprocessed_file"),
            ("output_node.t2w_fa_49_rf_0_file",
             "t2w_fa_49_rf_0_preprocessed_file"),
            ("output_node.t2w_fa_49_rf_180_file",
             "t2w_fa_49_rf_180_preprocessed_file"),
            ("output_node.brain_mask_file",
             "brain_mask_file")
        ]
        for (preproc_name, out_file) in in_out_pairs:
            data_sink = Node(ExplicitPathDataSink(),
                             name='data_sink_{}'.format(out_file))
            wf.connect(preprocess_ssfp_wf, preproc_name,
                       data_sink, "in_file")
            wf.connect(input_node, out_file,
                       data_sink, "out_file")

        return wf

    def estimate_workflow(self, base_dir=os.getcwd(),
                          name="estimate",
                          subject=None, session=None, run=None):

        sub_ses_run_combinations = self.get_subject_session_run_combinations(
            subject=subject, session=session, run=run)

        inputs = []
        for combination in sub_ses_run_combinations:
            input_dict = dict(
                subject=combination["subject"],
                session=combination["session"],
                run=combination["run"],
                t1w_fa_2_file=self.get_t1w_fa_2_preprocessed(
                    **combination),
                t1w_fa_13_file=self.get_t1w_fa_13_preprocessed(
                    **combination),
                t2w_fa_12_rf_180_file=self.get_t2w_fa_12_rf_180_preprocessed(
                    **combination),
                t2w_fa_49_rf_0_file=self.get_t2w_fa_49_rf_0_preprocessed(
                    **combination),
                t2w_fa_49_rf_180_file=self.get_t2w_fa_49_rf_180_preprocessed(
                    **combination),
                b1_map_file=self.get_b1_plus_relative_registered_map(
                    **combination),
                brain_mask_file=self.get_brain_mask_file(**combination),
                t1_map_file=self.get_t1_map(**combination, generate=True),
                t2_map_file=self.get_t2_map(**combination, generate=True),
                r1_map_file=self.get_r1_map(**combination, generate=True),
                r2_map_file=self.get_r2_map(**combination, generate=True)
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

        estimate_relaxation_ssfp_wf = estimate_relaxation_ssfp()

        wf.connect([(input_node, estimate_relaxation_ssfp_wf, [
            ('b1_map_file', 'input_node.b1_map_file'),
            ('t1w_fa_2_file', 'input_node.t1w_fa_2_file'),
            ('t1w_fa_13_file', 'input_node.t1w_fa_13_file'),
            ('t2w_fa_12_rf_180_file', 'input_node.t2w_fa_12_rf_180_file'),
            ('t2w_fa_49_rf_0_file', 'input_node.t2w_fa_49_rf_0_file'),
            ('t2w_fa_49_rf_180_file', 'input_node.t2w_fa_49_rf_180_file'),
            ('brain_mask_file', 'input_node.brain_mask_file')
        ])])

        # write output files
        in_out_pairs = [
            ("output_node.r1_map_file", "r1_map_file"),
            ("output_node.r2_map_file", "r2_map_file"),
            ("output_node.t1_map_file", "t1_map_file"),
            ("output_node.t2_map_file", "t2_map_file"),
        ]
        for (in_file, out_file) in in_out_pairs:
            data_sink = Node(ExplicitPathDataSink(),
                             name='data_sink_{}'.format(out_file))
            wf.connect(estimate_relaxation_ssfp_wf, in_file,
                       data_sink, "in_file")
            wf.connect(input_node, out_file,
                       data_sink, "out_file")

        return wf

    def get_t1w_raw_images(self, subject, session, run=None,
                           allow_multiple_files=True, generate=False):
        pass

    def get_t1w_preprocessed_images(self, subject, session, run=None,
                                    allow_multiple_files=True, generate=False):
        pass

    def get_t2w_raw_images(self, subject, session, run=None,
                           allow_multiple_files=True, generate=False):
        pass

    def get_t2w_preprocessed_images(self, subject, session, run=None,
                                    allow_multiple_files=True, generate=False):
        pass

    def get_brain_mask_file(self, subject, session, run=None,
                            allow_multiple_files=False,
                            generate=False):
        entities = dict(
            subject=subject, session=session, run=run,
            desc='brain', suffix='mask', extension=self._nifti_read_ext,
            datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_plus_raw_map(self, subject, session, run=None,
                            allow_multiple_files=False,
                            generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition='B1',
            suffix='B1map', extension=self._nifti_read_ext, desc=None,
            datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_plus_preprocessed_map(self, subject, session, run=None,
                                     allow_multiple_files=False,
                                     generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition='B1',
            suffix='B1map', extension=self._nifti_read_ext, desc="registered",
            datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_anat_ref(self, subject, session, run=None,
                        allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition='B1ref',
            suffix='magnitude', extension=self._nifti_read_ext, desc=None,
            datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)


