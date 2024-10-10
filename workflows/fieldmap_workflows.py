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


def register_image(base_dir=os.getcwd(), name="register_image",
                   apply_brain_masking=False):

    workflow = pe.Workflow(name=name)
    workflow.base_dir = base_dir

    input_node = pe.Node(util.IdentityInterface(
        fields=['moving_file', 'reference_file', 'target_file']),
        name='input_node')

    output_node = pe.Node(util.IdentityInterface(fields=['out_file']),
                          name='output_node')

    flirt_estimate = pe.Node(fsl.FLIRT(uses_qform=True, dof=6),
                             "flirt_estimate")
    flirt_apply = pe.Node(fsl.FLIRT(apply_xfm=True, uses_qform=True, dof=6),
                          "flirt_apply")

    first_volume_extractor = Node(fsl.ExtractROI(),
                                  name="first_volume_extractor")
    first_volume_extractor.inputs.t_min = 0
    first_volume_extractor.inputs.t_size = 1

    if apply_brain_masking:
        bet_target = Node(fsl.BET(), name="bet_target")
        bet_target.inputs.robust = True

        bet_reference = Node(fsl.BET(), name="bet_reference")
        bet_reference.inputs.robust = True

        workflow.connect(input_node, "reference_file", bet_reference, "in_file")
        workflow.connect(input_node, "target_file", bet_target, "in_file")

        workflow.connect(bet_target, "out_file", first_volume_extractor,
                         "in_file")
        workflow.connect(bet_reference, "out_file", flirt_estimate, "in_file")
        workflow.connect(first_volume_extractor, "roi_file", flirt_estimate,
                         "reference")

        workflow.connect(input_node, "moving_file", flirt_apply, "in_file")
        workflow.connect(first_volume_extractor, "roi_file", flirt_apply,
                         "reference")
        workflow.connect(flirt_estimate, "out_matrix_file", flirt_apply,
                         "in_matrix_file")

        workflow.connect(flirt_apply, "out_file", output_node, "out_file")
    else:
        workflow.connect(input_node, "target_file", first_volume_extractor,
                         "in_file")
        workflow.connect(input_node, "reference_file", flirt_estimate,
                         "in_file")
        workflow.connect(first_volume_extractor, "roi_file", flirt_estimate,
                         "reference")

        workflow.connect(input_node, "moving_file", flirt_apply, "in_file")
        workflow.connect(first_volume_extractor, "roi_file", flirt_apply,
                         "reference")
        workflow.connect(flirt_estimate, "out_matrix_file", flirt_apply,
                         "in_matrix_file")

        workflow.connect(flirt_apply, "out_file", output_node, "out_file")

    return workflow


def correct_b1_with_b0(base_dir=os.getcwd(), name="correct_b1_with_b0"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(util.IdentityInterface(
        fields=['b0_map_file',
                'b1_map_file',
                'b0_anat_ref_file',
                'b1_anat_ref_file',
                'fa_b1_in_degrees',
                'fa_nominal_in_degrees',
                'pulse_duration_in_seconds']),
        name='input_node')
    output_node = pe.Node(util.IdentityInterface(fields=['out_file']),
                          name='output_node')

    # Register B1 map to B0 map
    register_b1_map_to_b0_map = register_image(name="register_b1_map_to_b0_map")
    wf.connect(input_node, "b1_map_file", register_b1_map_to_b0_map,
               "input_node.moving_file")
    wf.connect(input_node, "b1_anat_ref_file", register_b1_map_to_b0_map,
               "input_node.reference_file")
    wf.connect(input_node, "b0_anat_ref_file", register_b1_map_to_b0_map,
               "input_node.target_file")

    # Define the FLIRT node for reslicing
    reslice = pe.Node(fsl.FLIRT(), name="reslice")
    reslice.inputs.apply_xfm = True  # Use the transformation directly without searching
    reslice.inputs.interp = 'trilinear'  # Optional: Specify interpolation method
    # reslice.inputs.in_matrix_file = 'identity_matrix.mat'
    reslice.inputs.uses_qform = True

    wf.connect(input_node, "b0_map_file",
               reslice, "in_file")
    wf.connect(register_b1_map_to_b0_map, "output_node.out_file",
               reslice, "reference")

    # Correct B1 map
    correct_b1_map_node = Node(Function(
        input_names=['b1_map_file', 'b0_map_file', 'fa_b1_in_degrees',
                     'fa_nominal_in_degrees', 'pulse_duration_in_seconds'],
        output_names=['b1_output_file'],
        function=correct_b1_map),
        name='correct_b1_map')

    wf.connect(register_b1_map_to_b0_map, "output_node.out_file",
               correct_b1_map_node, "b1_map_file")
    wf.connect(reslice, "out_file", correct_b1_map_node, "b0_map_file")
    wf.connect(input_node, "fa_b1_in_degrees", correct_b1_map_node,
               "fa_b1_in_degrees")
    wf.connect(input_node, "fa_nominal_in_degrees", correct_b1_map_node,
               "fa_nominal_in_degrees")
    wf.connect(input_node, "pulse_duration_in_seconds", correct_b1_map_node,
               "pulse_duration_in_seconds")
    wf.connect(correct_b1_map_node, "b1_output_file", output_node, "out_file")

    return wf
