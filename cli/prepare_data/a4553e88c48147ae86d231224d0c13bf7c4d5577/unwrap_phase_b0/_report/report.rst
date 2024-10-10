Node: unwrap_phase_b0 (utility)
===============================


 Hierarchy : prepare_data.unwrap_phase_b0
 Exec ID : unwrap_phase_b0.a0


Original Inputs
---------------


* b0_phase_diff_file : <BIDSImageFile filename='/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/sub-phy003/ses-001/anat/sub-phy003_ses-001_acq-dznebnep3d_part-phase_T2w.nii.gz'>
* b0_te_delta : 0.0429
* function_str : def unwrap_phase_b0_siemens(b0_phase_diff_file, b0_te_delta):
    import nibabel as nib
    import os

    base_dir = os.getcwd()

    # read b0 image
    b0_phase_diff_file_nib = nib.load(b0_phase_diff_file)
    b0_phase_diff_image = b0_phase_diff_file_nib.get_fdata()

    # Compute the B0 map
    phase_unwrap_factor = 1.0 / (4096 * b0_te_delta * 2)
    b0_map = phase_unwrap_factor * b0_phase_diff_image

    # write b1 map
    b0_output_filename = os.path.join(base_dir, 'b0map.nii.gz')
    b0_image_nib = nib.Nifti1Image(b0_map, b0_phase_diff_file_nib.affine,
                                   b0_phase_diff_file_nib.header)
    nib.save(b0_image_nib, b0_output_filename)

    return b0_output_filename



Execution Inputs
----------------


* b0_phase_diff_file : <BIDSImageFile filename='/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/sub-phy003/ses-001/anat/sub-phy003_ses-001_acq-dznebnep3d_part-phase_T2w.nii.gz'>
* b0_te_delta : 0.0429
* function_str : def unwrap_phase_b0_siemens(b0_phase_diff_file, b0_te_delta):
    import nibabel as nib
    import os

    base_dir = os.getcwd()

    # read b0 image
    b0_phase_diff_file_nib = nib.load(b0_phase_diff_file)
    b0_phase_diff_image = b0_phase_diff_file_nib.get_fdata()

    # Compute the B0 map
    phase_unwrap_factor = 1.0 / (4096 * b0_te_delta * 2)
    b0_map = phase_unwrap_factor * b0_phase_diff_image

    # write b1 map
    b0_output_filename = os.path.join(base_dir, 'b0map.nii.gz')
    b0_image_nib = nib.Nifti1Image(b0_map, b0_phase_diff_file_nib.affine,
                                   b0_phase_diff_file_nib.header)
    nib.save(b0_image_nib, b0_output_filename)

    return b0_output_filename



Execution Outputs
-----------------


* out_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/a4553e88c48147ae86d231224d0c13bf7c4d5577/unwrap_phase_b0/b0map.nii.gz


Runtime info
------------


* duration : 7.979135
* hostname : laurin-dlm
* prev_wd : /home/laurin/workspace/t2_mapping_traveling_heads/cli
* working_dir : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/a4553e88c48147ae86d231224d0c13bf7c4d5577/unwrap_phase_b0


Environment
~~~~~~~~~~~


* CONDA_DEFAULT_ENV : bonnt2env
* CONDA_PREFIX : /home/laurin/anaconda3/envs/bonnt2env
* CONDA_PROMPT_MODIFIER : (bonnt2env) 
* CONDA_SHLVL : 1
* DBUS_SESSION_BUS_ADDRESS : unix:path=/run/user/1000/bus
* DESKTOP_SESSION : ubuntu
* DISPLAY : :1
* FSLDIR : /usr/local/fsl
* FSLMULTIFILEQUIT : TRUE
* FSLOUTPUTTYPE : NIFTI_GZ
* FSLTCLSH : /usr/local/fsl/bin/fsltclsh
* FSLWISH : /usr/local/fsl/bin/fslwish
* FSL_LOAD_NIFTI_EXTENSIONS : 0
* FSL_SKIP_GLOBAL : 0
* GDMSESSION : ubuntu
* GIO_LAUNCHED_DESKTOP_FILE : /home/laurin/.local/share/applications/jetbrains-pycharm-ce.desktop
* GIO_LAUNCHED_DESKTOP_FILE_PID : 5723
* GJS_DEBUG_OUTPUT : stderr
* GJS_DEBUG_TOPICS : JS ERROR;JS LOG
* GNOME_DESKTOP_SESSION_ID : this-is-deprecated
* GNOME_SHELL_SESSION_MODE : ubuntu
* GPG_AGENT_INFO : /run/user/1000/gnupg/S.gpg-agent:0:1
* GTK_IM_MODULE : ibus
* GTK_MODULES : gail:atk-bridge
* HOME : /home/laurin
* INVOCATION_ID : 5a2a2254b9ad4788b70d7dda6c36297a
* JOURNAL_STREAM : 8:17940
* LANG : en_US.UTF-8
* LC_ADDRESS : de_DE.UTF-8
* LC_IDENTIFICATION : de_DE.UTF-8
* LC_MEASUREMENT : de_DE.UTF-8
* LC_MONETARY : de_DE.UTF-8
* LC_NAME : de_DE.UTF-8
* LC_NUMERIC : de_DE.UTF-8
* LC_PAPER : de_DE.UTF-8
* LC_TELEPHONE : de_DE.UTF-8
* LC_TIME : de_DE.UTF-8
* LM_LICENSE_FILE : /usr/local/flexlm/Bruker/licenses/license.dat
* LOGNAME : laurin
* MANAGERPID : 2340
* MATHEMATICA_HOME : /usr/local/Wolfram/Mathematica/13.2
* NIPYPE_NO_ET : 1
* PATH : /home/laurin/anaconda3/envs/bonnt2env/bin:/home/laurin/anaconda3/condabin:/usr/local/fsl/share/fsl/bin:/usr/local/fsl/share/fsl/bin:/home/laurin/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/laurin/.local/share/JetBrains/Toolbox/scripts
* PWD : /home/laurin/workspace/t2_mapping_traveling_heads/cli
* PYCHARM_HOSTED : 1
* PYTHONIOENCODING : UTF-8
* PYTHONPATH : /home/laurin/workspace/t2_mapping_traveling_heads:/home/laurin/workspace/t2_mapping_traveling_heads/external/MagPhsT2
* PYTHONUNBUFFERED : 1
* QT_ACCESSIBILITY : 1
* QT_IM_MODULE : ibus
* SESSION_MANAGER : local/laurin-dlm:@/tmp/.ICE-unix/2617,unix/laurin-dlm:/tmp/.ICE-unix/2617
* SHELL : /bin/bash
* SHLVL : 0
* SSH_AGENT_LAUNCHER : gnome-keyring
* SSH_AUTH_SOCK : /run/user/1000/keyring/ssh
* SYSTEMD_EXEC_PID : 2640
* USER : laurin
* USERNAME : laurin
* WINDOWPATH : 2
* XAUTHORITY : /run/user/1000/gdm/Xauthority
* XDG_CONFIG_DIRS : /etc/xdg/xdg-ubuntu:/etc/xdg
* XDG_CURRENT_DESKTOP : ubuntu:GNOME
* XDG_DATA_DIRS : /usr/share/ubuntu:/usr/share/gnome:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop
* XDG_MENU_PREFIX : gnome-
* XDG_RUNTIME_DIR : /run/user/1000
* XDG_SESSION_CLASS : user
* XDG_SESSION_DESKTOP : ubuntu
* XDG_SESSION_TYPE : x11
* XMODIFIERS : @im=ibus

