from tokenize import endpats

from bids import BIDSLayout
import os
import subprocess  # Or import your custom CLI function directly

# Path to your BIDS dataset
bids_root = '/media/frank/SHD_01/Epilepsy/Patients/derivatives/ExVivo/HighRes/Hamburg_3T_Prisma/T2mapping'
bids_derivative_mask = '/media/frank/SHD_01/Epilepsy/Patients/derivatives/ExVivo/HighRes/Hamburg_3T_Prisma/ITKsnap'

# Initialize the BIDS layout
layout = BIDSLayout(bids_root, validate=False)
layout_mask = BIDSLayout(bids_derivative_mask, validate=False)

# Get all subjects and sessions
subjects = layout.get_subjects()

# Loop through each subject/session and collect files
for subject in subjects:
    if subject != 'tle003':
        continue

    subject_sessions = layout.get_sessions(subject=subject)  # Handle no sessions

    for session in subject_sessions:
        subject_sessions_recons = layout.get_reconstructions(subject=subject, session=session)

        for recon in subject_sessions_recons:
            # Collect input files
            t2w_mag_file = layout.get(subject=subject, session=session, reconstruction=recon, datatype='anat', part='mag', suffix='T2w', return_type='file')
            t2w_phase_file = layout.get(subject=subject, session=session, reconstruction=recon, datatype='anat', part='phase', suffix='T2w', return_type='file')
            b1_map_file = layout.get(subject=subject, session=session, reconstruction=recon, datatype='fmap', suffix='B1map', return_type='file')
            b1_ref_file = layout.get(subject=subject, session=session, reconstruction=recon, datatype='fmap', suffix='B1ref', return_type='file')
            mask_file = layout_mask.get(subject=subject, session=session, reconstruction=recon, datatype='anat', suffix='T2w', return_type='file')

            output_dir = os.path.join(
                bids_root, f"sub-{subject}", f"ses-{session}", "anat", f"recon-{recon}"
            )

            # Make sure the output dir exists
            os.makedirs(output_dir, exist_ok=True)

            if recon == '0p3':
                # Pass to your CLI function or subprocess
                # Example using subprocess to call CLI-scripted Python function
                cmd = [
                    '/home/frank/anaconda3/envs/t2mapping/bin/python',
                    '/home/frank/t2_mapping_traveling_heads/cli/process_3depi.py',
                    '--t2w_mag', *t2w_mag_file,
                    '--t2w_phase', *t2w_phase_file,
                    '--b1_map', *b1_map_file,
                    '--b1_anat_ref', *b1_ref_file,
                    '--mask', *mask_file,
                    '--output_dir', output_dir,
                    '--echo_time', '0.00864',
                    '--repetition_time', '0.0247',
                    '--flip_angle', '20',
                    '--rf_phase_increments', '1', '1.5', '2', '3', '4', '5', '1', '1.5', '2', '3', '4', '5', '1', '1.5', '2', '3', '4', '5'
                ]
            elif recon == '0p45':
                # Pass to your CLI function or subprocess
                # Example using subprocess to call CLI-scripted Python function
                cmd = [
                    '/home/frank/anaconda3/envs/t2mapping/bin/python',
                    '/home/frank/t2_mapping_traveling_heads/cli/process_3depi.py',
                    '--t2w_mag', *t2w_mag_file,
                    '--t2w_phase', *t2w_phase_file,
                    '--b1_map', *b1_map_file,
                    '--b1_anat_ref', *b1_ref_file,
                    '--mask', *mask_file,
                    '--output_dir', output_dir,
                    '--echo_time', '0.00657',
                    '--repetition_time', '0.0165',
                    '--flip_angle', '20',
                    '--rf_phase_increments', '1', '1.5', '2', '3', '4', '5'
                ]
            else:
                continue
                # Pass to your CLI function or subprocess
                # Example using subprocess to call CLI-scripted Python function
                cmd = [
                    '/home/frank/anaconda3/envs/t2mapping/bin/python',
                    '/home/frank/t2_mapping_traveling_heads/cli/process_3depi.py',
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