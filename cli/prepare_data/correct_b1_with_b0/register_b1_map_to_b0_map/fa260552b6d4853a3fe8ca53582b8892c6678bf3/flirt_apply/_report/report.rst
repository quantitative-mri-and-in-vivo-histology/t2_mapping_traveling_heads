Node: correct_b1_with_b0 (register_b1_map_to_b0_map (flirt_apply (fsl)
======================================================================


 Hierarchy : prepare_data.correct_b1_with_b0.register_b1_map_to_b0_map.flirt_apply
 Exec ID : flirt_apply.a0


Original Inputs
---------------


* angle_rep : <undefined>
* apply_isoxfm : <undefined>
* apply_xfm : True
* args : <undefined>
* bbrslope : <undefined>
* bbrtype : <undefined>
* bgvalue : <undefined>
* bins : <undefined>
* coarse_search : <undefined>
* cost : <undefined>
* cost_func : <undefined>
* datatype : <undefined>
* display_init : <undefined>
* dof : 6
* echospacing : <undefined>
* environ : {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
* fieldmap : <undefined>
* fieldmapmask : <undefined>
* fine_search : <undefined>
* force_scaling : <undefined>
* in_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_phase_wrap_around/fa260552b6d4853a3fe8ca53582b8892c6678bf3/impaint_b1_map/image_smoothed.nii.gz
* in_matrix_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/flirt_estimate/image_smoothed_flirt.mat
* in_weight : <undefined>
* interp : <undefined>
* min_sampling : <undefined>
* no_clamp : <undefined>
* no_resample : <undefined>
* no_resample_blur : <undefined>
* no_search : <undefined>
* out_file : <undefined>
* out_log : <undefined>
* out_matrix_file : <undefined>
* output_type : NIFTI_GZ
* padding_size : <undefined>
* pedir : <undefined>
* ref_weight : <undefined>
* reference : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/first_volume_extractor/sub-phy003_ses-001_acq-dznebnB0_run-1_magnitude1_roi.nii.gz
* rigid2D : <undefined>
* save_log : <undefined>
* schedule : <undefined>
* searchr_x : <undefined>
* searchr_y : <undefined>
* searchr_z : <undefined>
* sinc_width : <undefined>
* sinc_window : <undefined>
* uses_qform : True
* verbose : <undefined>
* wm_seg : <undefined>
* wmcoords : <undefined>
* wmnorms : <undefined>


Execution Inputs
----------------


* angle_rep : <undefined>
* apply_isoxfm : <undefined>
* apply_xfm : True
* args : <undefined>
* bbrslope : <undefined>
* bbrtype : <undefined>
* bgvalue : <undefined>
* bins : <undefined>
* coarse_search : <undefined>
* cost : <undefined>
* cost_func : <undefined>
* datatype : <undefined>
* display_init : <undefined>
* dof : 6
* echospacing : <undefined>
* environ : {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
* fieldmap : <undefined>
* fieldmapmask : <undefined>
* fine_search : <undefined>
* force_scaling : <undefined>
* in_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_phase_wrap_around/fa260552b6d4853a3fe8ca53582b8892c6678bf3/impaint_b1_map/image_smoothed.nii.gz
* in_matrix_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/flirt_estimate/image_smoothed_flirt.mat
* in_weight : <undefined>
* interp : <undefined>
* min_sampling : <undefined>
* no_clamp : <undefined>
* no_resample : <undefined>
* no_resample_blur : <undefined>
* no_search : <undefined>
* out_file : <undefined>
* out_log : <undefined>
* out_matrix_file : <undefined>
* output_type : NIFTI_GZ
* padding_size : <undefined>
* pedir : <undefined>
* ref_weight : <undefined>
* reference : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/first_volume_extractor/sub-phy003_ses-001_acq-dznebnB0_run-1_magnitude1_roi.nii.gz
* rigid2D : <undefined>
* save_log : <undefined>
* schedule : <undefined>
* searchr_x : <undefined>
* searchr_y : <undefined>
* searchr_z : <undefined>
* sinc_width : <undefined>
* sinc_window : <undefined>
* uses_qform : True
* verbose : <undefined>
* wm_seg : <undefined>
* wmcoords : <undefined>
* wmnorms : <undefined>


Execution Outputs
-----------------


* out_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/flirt_apply/image_smoothed_flirt.nii.gz
* out_log : <undefined>
* out_matrix_file : <undefined>


Runtime info
------------


* cmdline : flirt -in /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_phase_wrap_around/fa260552b6d4853a3fe8ca53582b8892c6678bf3/impaint_b1_map/image_smoothed.nii.gz -ref /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/first_volume_extractor/sub-phy003_ses-001_acq-dznebnB0_run-1_magnitude1_roi.nii.gz -out image_smoothed_flirt.nii.gz -omat image_smoothed_flirt.mat -applyxfm -dof 6 -init /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/flirt_estimate/image_smoothed_flirt.mat -usesqform
* duration : 0.236225
* hostname : laurin-dlm
* prev_wd : /home/laurin/workspace/t2_mapping_traveling_heads/cli
* working_dir : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_b1_with_b0/register_b1_map_to_b0_map/fa260552b6d4853a3fe8ca53582b8892c6678bf3/flirt_apply


Terminal output
~~~~~~~~~~~~~~~


 


Terminal - standard output
~~~~~~~~~~~~~~~~~~~~~~~~~~


 


Terminal - standard error
~~~~~~~~~~~~~~~~~~~~~~~~~


 


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

