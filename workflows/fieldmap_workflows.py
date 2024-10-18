import os
import argparse
import sys
import nipype.interfaces.io as nio
import multiprocessing
from pathlib import Path
from bids.layout import BIDSLayout
from nipype import Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype_utils import BidsRename, BidsOutputFormatter, create_output_folder
from utils.processing import correct_b1_map
from workflows.preprocessing_workflows import register_image_workflow


def correct_b1_with_b0(base_dir=os.getcwd(), name="correct_b1_with_b0"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(util.IdentityInterface(
        fields=['b0_map_file',
                'b1_map_file',
                'b0_anat_ref_file',
                'b1_anat_ref_file',
                'fa_nominal_in_degrees',
                'pulse_duration_in_seconds']),
        name='input_node')
    output_node = pe.Node(util.IdentityInterface(fields=['out_file']),
                          name='output_node')

    register_b0_map_to_b1_map = register_image_workflow(
        name="register_b0_map_to_b1_map")
    wf.connect(input_node, "b0_map_file", register_b0_map_to_b1_map,
               "input_node.moving_file")
    wf.connect(input_node, "b0_anat_ref_file", register_b0_map_to_b1_map,
               "input_node.reference_file")
    wf.connect(input_node, "b1_anat_ref_file", register_b0_map_to_b1_map,
               "input_node.target_file")

    # Correct B1 map
    correct_b1_map_node = Node(Function(
        input_names=['b1_map_file', 'b0_map_file', 'fa_nominal_in_degrees',
                     'pulse_duration_in_seconds'],
        output_names=['b1_output_file'],
        function=correct_b1_map),
        name='correct_b1_map')

    wf.connect(register_b0_map_to_b1_map, "output_node.out_file",
               correct_b1_map_node, "b0_map_file")
    wf.connect(input_node, "b1_map_file", correct_b1_map_node, "b1_map_file")
    wf.connect(input_node, "fa_nominal_in_degrees", correct_b1_map_node,
               "fa_nominal_in_degrees")
    wf.connect(input_node, "pulse_duration_in_seconds", correct_b1_map_node,
               "pulse_duration_in_seconds")
    wf.connect(correct_b1_map_node, "b1_output_file", output_node, "out_file")

    return wf
