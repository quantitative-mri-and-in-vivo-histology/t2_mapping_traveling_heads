import argparse
import multiprocessing
import os

import nipype.pipeline.engine as pe
from nipype import Node
from nipype.interfaces.utility import IdentityInterface

from nodes.io import ExplicitDataSink
from workflows.processing import correct_b1_with_b0


def main():
    parser = argparse.ArgumentParser(
        description="Adjust B1 map to compensate for off-resonance spin frequencies.\n\n"
                    "Magnetic susceptibility variations near the sphenoid and ethmoid sinuses\n"
                    "lead to off-resonance spin frequencies, causing a mismatch between the\n"
                    "local spin frequency and the nominal Larmor frequency. This script calculates\n"
                    "an effective B1 map that accounts for this mismatch. See details here:\n"
                    "https://doi.org/10.1002/mrm.29383.")
    parser.add_argument(
        '--out_b1_map',
        type=str,
        required=True,
        help="Path to store adjusted B1 map."
    )
    parser.add_argument(
        '--b1_map',
        type=str,
        required=True,
        help="B1 map to be adjusted. B1 map should be normalized to 1, "
             " with 1 indicating perfectly homogeneous field."
    )
    parser.add_argument(
        '--b0_map',
        type=str,
        required=True,
        help="B0 map file. B0 map should be provided in Hertz."
    )
    parser.add_argument(
        '--b1_anat_ref',
        type=str,
        required=True,
        help="B1 anatomical reference image file. Should be a magnitude image."
    )
    parser.add_argument(
        '--b0_anat_ref',
        type=str,
        required=True,
        help="B0 anatomical reference image file. Should be a magnitude image."
    )
    parser.add_argument(
        '--flip_angle_target',
        type=float,
        required=True,
        help="Flip angle of target image in degrees."
    )
    parser.add_argument(
        '--rf_pulse_duration_target',
        type=float,
        required=True,
        help="RF pulse duration of target image in seconds."
    )
    parser.add_argument(
        '--n_procs',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Number of processors to use (default: all available cores).')

    # Parse arguments
    args = parser.parse_args()

    # Define the reusable run settings in a dictionary
    run_settings = dict(plugin='MultiProc',
                        plugin_args={'n_procs': args.n_procs})

    # set up workflow
    wf = pe.Workflow(name="adjust_b1_map")
    wf.base_dir = os.path.join(os.path.dirname(args.out_b1_map), "temp")

    # set up inputs
    input_dict = dict(
        b1_map_file=args.b1_map,
        b1_anat_ref_file=args.b1_anat_ref,
        b0_map_file=args.b0_map,
        b0_anat_ref_file=args.b0_anat_ref,
        flip_angle_target=args.flip_angle_target,
        rf_pulse_duration_target=args.rf_pulse_duration_target
    )
    input_node = Node(
        IdentityInterface(fields=list(input_dict.keys())),
        name='input_node')
    for key, value in input_dict.items():
        setattr(input_node.inputs, key, value)

    # adjust B1 map
    correct_b1_with_b0_wf = correct_b1_with_b0()
    wf.connect([(input_node, correct_b1_with_b0_wf, [
        ('b1_map_file', 'input_node.b1_map_file'),
        ('b1_anat_ref_file', 'input_node.b1_anat_ref_file'),
        ('b0_map_file', 'input_node.b0_map_file'),
        ('b0_anat_ref_file', 'input_node.b0_anat_ref_file'),
        ('flip_angle_target', 'input_node.fa_nominal_in_degrees'),
        ('rf_pulse_duration_target', 'input_node.pulse_duration_in_seconds')
    ])])

    # write adjusted B1 map
    b1_map_writer = pe.Node(
        ExplicitDataSink(output_dir=os.path.dirname(args.out_b1_map),
                         filename=os.path.basename(args.out_b1_map)),
        name="b1_map_writer")
    wf.connect(correct_b1_with_b0_wf, "output_node.out_file",
               b1_map_writer, "in_file")

    wf.run(**run_settings)


if __name__ == "__main__":
    main()
