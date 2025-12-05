from bids import BIDSLayout
import os
import subprocess  # Or import your custom CLI function directly

# Path to your BIDS dataset
bids_root = '/media/frank/SHD_01/Epilepsy/Patients/derivatives/InSitu/T2mapping'
bids_derivative_mask = '/media/frank/SHD_01/Epilepsy/Patients/derivatives/InSitu/ITKsnap'

# Initialize the BIDS layout
layout = BIDSLayout(bids_root, validate=False)
layout_mask = BIDSLayout(bids_derivative_mask, validate=False)

# Get all subjects and sessions
subjects = layout.get_subjects()
sessions = layout.get_sessions()

# Loop through each subject/session and collect files
for subject in subjects:
    subject_sessions = layout.get_sessions(subject=subject)  # Handle no sessions

    for session in subject_sessions:
        # Collect input files
        t2w_mag_file = layout.get(subject=subject, session=session, datatype='anat', part='mag', suffix='T2w', return_type='file')
        t2w_phase_file = layout.get(subject=subject, session=session, datatype='anat', part='phase', suffix='T2w', return_type='file')
        b1_map_file = layout.get(subject=subject, session=session, datatype='fmap', suffix='B1map', return_type='file')
        b1_ref_file = layout.get(subject=subject, session=session, datatype='fmap', suffix='B1ref', return_type='file')
        mask_file = layout_mask.get(subject=subject, session=session, datatype='anat', suffix='T2w', return_type='file')

        output_dir = os.path.join(
            bids_root, f"sub-{subject}", f"ses-{session}", "anat"
        )

        # Make sure the output dir exists
        os.makedirs(output_dir, exist_ok=True)

        # Pass to your CLI function or subprocess
        # Example using subprocess to call CLI-scripted Python function
        cmd = [
            '/home/frank/anaconda3/envs/t2mapping/bin/python', '/home/frank/t2_mapping_traveling_heads/cli/process_3depi.py',
            '--t2w_mag', *t2w_mag_file,
            '--t2w_phase', *t2w_phase_file,
            '--b1_map', *b1_map_file,
            '--b1_anat_ref', *b1_ref_file,
            '--mask', *mask_file,
            '--output_dir', output_dir,
            '--echo_time', '0.00414',
            '--repetition_time', '0.0108',
            '--flip_angle', '20',
            '--rf_phase_increments', '1', '1.5', '2', '3', '4', '5'
        ]

        subprocess.run(cmd)