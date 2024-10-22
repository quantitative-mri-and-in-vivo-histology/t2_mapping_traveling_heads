Node: preprocess_3depi (register_b1_anat_ref_to_t2w (ants)
==========================================================


 Hierarchy : process_3depi.preprocess_3depi.register_b1_anat_ref_to_t2w
 Exec ID : register_b1_anat_ref_to_t2w


Original Inputs
---------------


* args : <undefined>
* collapse_output_transforms : True
* convergence_threshold : [1e-06]
* convergence_window_size : [10]
* dimension : 3
* environ : {'NSLOTS': '1'}
* fixed_image : ['/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/mag_first_volume_extractor/magnitude_roi.nii.gz']
* fixed_image_mask : <undefined>
* fixed_image_masks : <undefined>
* float : <undefined>
* initial_moving_transform : <undefined>
* initial_moving_transform_com : <undefined>
* initialize_transforms_per_stage : False
* interpolation : Linear
* interpolation_parameters : <undefined>
* invert_initial_moving_transform : <undefined>
* metric : ['MI']
* metric_item_trait : <undefined>
* metric_stage_trait : <undefined>
* metric_weight : [1.0]
* metric_weight_item_trait : 1.0
* metric_weight_stage_trait : <undefined>
* moving_image : ['/home/laurin/workspace/t2_cli_tests/process_3depi/sub-phy002_ses-001_acq-B1Ref_magnitude.nii.gz']
* moving_image_mask : <undefined>
* moving_image_masks : <undefined>
* num_threads : 1
* number_of_iterations : [[500, 250, 100]]
* output_inverse_warped_image : <undefined>
* output_transform_prefix : output_prefix_
* output_warped_image : output_warped_image.nii.gz
* radius_bins_item_trait : 5
* radius_bins_stage_trait : <undefined>
* radius_or_number_of_bins : [32]
* random_seed : <undefined>
* restore_state : <undefined>
* restrict_deformation : <undefined>
* sampling_percentage : [0.25]
* sampling_percentage_item_trait : <undefined>
* sampling_percentage_stage_trait : <undefined>
* sampling_strategy : ['Regular']
* sampling_strategy_item_trait : <undefined>
* sampling_strategy_stage_trait : <undefined>
* save_state : <undefined>
* shrink_factors : [[6, 3, 1]]
* sigma_units : <undefined>
* smoothing_sigmas : [[2.0, 1.0, 0.0]]
* transform_parameters : [(0.1,)]
* transforms : ['Rigid']
* use_estimate_learning_rate_once : <undefined>
* use_histogram_matching : True
* verbose : False
* winsorize_lower_quantile : 0.0
* winsorize_upper_quantile : 1.0
* write_composite_transform : False


Execution Inputs
----------------


* args : <undefined>
* collapse_output_transforms : True
* convergence_threshold : [1e-06]
* convergence_window_size : [10]
* dimension : 3
* environ : {'NSLOTS': '1'}
* fixed_image : ['/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/mag_first_volume_extractor/magnitude_roi.nii.gz']
* fixed_image_mask : <undefined>
* fixed_image_masks : <undefined>
* float : <undefined>
* initial_moving_transform : <undefined>
* initial_moving_transform_com : <undefined>
* initialize_transforms_per_stage : False
* interpolation : Linear
* interpolation_parameters : <undefined>
* invert_initial_moving_transform : <undefined>
* metric : ['MI']
* metric_item_trait : <undefined>
* metric_stage_trait : <undefined>
* metric_weight : [1.0]
* metric_weight_item_trait : 1.0
* metric_weight_stage_trait : <undefined>
* moving_image : ['/home/laurin/workspace/t2_cli_tests/process_3depi/sub-phy002_ses-001_acq-B1Ref_magnitude.nii.gz']
* moving_image_mask : <undefined>
* moving_image_masks : <undefined>
* num_threads : 1
* number_of_iterations : [[500, 250, 100]]
* output_inverse_warped_image : <undefined>
* output_transform_prefix : output_prefix_
* output_warped_image : output_warped_image.nii.gz
* radius_bins_item_trait : 5
* radius_bins_stage_trait : <undefined>
* radius_or_number_of_bins : [32]
* random_seed : <undefined>
* restore_state : <undefined>
* restrict_deformation : <undefined>
* sampling_percentage : [0.25]
* sampling_percentage_item_trait : <undefined>
* sampling_percentage_stage_trait : <undefined>
* sampling_strategy : ['Regular']
* sampling_strategy_item_trait : <undefined>
* sampling_strategy_stage_trait : <undefined>
* save_state : <undefined>
* shrink_factors : [[6, 3, 1]]
* sigma_units : <undefined>
* smoothing_sigmas : [[2.0, 1.0, 0.0]]
* transform_parameters : [(0.1,)]
* transforms : ['Rigid']
* use_estimate_learning_rate_once : <undefined>
* use_histogram_matching : True
* verbose : False
* winsorize_lower_quantile : 0.0
* winsorize_upper_quantile : 1.0
* write_composite_transform : False


Execution Outputs
-----------------


* composite_transform : <undefined>
* elapsed_time : <undefined>
* forward_invert_flags : <undefined>
* forward_transforms : <undefined>
* inverse_composite_transform : <undefined>
* inverse_warped_image : <undefined>
* metric_value : <undefined>
* reverse_forward_invert_flags : <undefined>
* reverse_forward_transforms : ['/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/register_b1_anat_ref_to_t2w/output_prefix_0GenericAffine.mat']
* reverse_invert_flags : <undefined>
* reverse_transforms : <undefined>
* save_state : <undefined>
* warped_image : /home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/register_b1_anat_ref_to_t2w/output_warped_image.nii.gz


Runtime info
------------


* cmdline : antsRegistration --collapse-output-transforms 1 --dimensionality 3 --initialize-transforms-per-stage 0 --interpolation Linear --output [ output_prefix_, output_warped_image.nii.gz ] --transform Rigid[ 0.1 ] --metric MI[ /home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/mag_first_volume_extractor/magnitude_roi.nii.gz, /home/laurin/workspace/t2_cli_tests/process_3depi/sub-phy002_ses-001_acq-B1Ref_magnitude.nii.gz, 1, 32, Regular, 0.25 ] --convergence [ 500x250x100, 1e-06, 10 ] --smoothing-sigmas 2.0x1.0x0.0 --shrink-factors 6x3x1 --use-histogram-matching 1 --winsorize-image-intensities [ 0.0, 1.0 ]  --write-composite-transform 0
* duration : 21.322305
* hostname : laurin-ThinkPad
* prev_wd : /home/laurin/workspace/t2_mapping_traveling_heads/cli
* working_dir : /home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/register_b1_anat_ref_to_t2w


Terminal output
~~~~~~~~~~~~~~~


 


Terminal - standard output
~~~~~~~~~~~~~~~~~~~~~~~~~~


 


Terminal - standard error
~~~~~~~~~~~~~~~~~~~~~~~~~


 


Environment
~~~~~~~~~~~


* ARV_ROOT_PATH : /home/laurin/workspace/axon_radius_validation
* COLORTERM : truecolor
* CONDA_DEFAULT_ENV : t2mappingenv
* CONDA_EXE : /home/laurin/anaconda3/bin/conda
* CONDA_PREFIX : /home/laurin/anaconda3/envs/t2mappingenv
* CONDA_PROMPT_MODIFIER : (t2mappingenv) 
* CONDA_PYTHON_EXE : /home/laurin/anaconda3/bin/python
* CONDA_SHLVL : 1
* DBUS_SESSION_BUS_ADDRESS : unix:path=/run/user/1000/bus
* DEBUGINFOD_URLS : https://debuginfod.ubuntu.com 
* DESKTOP_SESSION : ubuntu
* DISPLAY : :0
* FSLDIR : /home/laurin/fsl
* FSLMULTIFILEQUIT : TRUE
* FSLOUTPUTTYPE : NIFTI_GZ
* FSLTCLSH : /home/laurin/fsl/bin/fsltclsh
* FSLWISH : /home/laurin/fsl/bin/fslwish
* FSL_LOAD_NIFTI_EXTENSIONS : 0
* FSL_SKIP_GLOBAL : 0
* GDMSESSION : ubuntu
* GNOME_DESKTOP_SESSION_ID : this-is-deprecated
* GNOME_KEYRING_CONTROL : /run/user/1000/keyring
* GNOME_SETUP_DISPLAY : :1
* GNOME_SHELL_SESSION_MODE : ubuntu
* GNOME_TERMINAL_SCREEN : /org/gnome/Terminal/screen/b21c3ea9_22fe_4b13_936a_d63564ac26e1
* GNOME_TERMINAL_SERVICE : :1.132
* GSM_SKIP_SSH_AGENT_WORKAROUND : true
* GTK_MODULES : gail:atk-bridge
* HOME : /home/laurin
* IM_CONFIG_PHASE : 1
* ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS : 0
* LANG : en_US.UTF-8
* LESSCLOSE : /usr/bin/lesspipe %s %s
* LESSOPEN : | /usr/bin/lesspipe %s
* LOGNAME : laurin
* LS_COLORS : rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.avif=01;35:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.webp=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:*~=00;90:*#=00;90:*.bak=00;90:*.crdownload=00;90:*.dpkg-dist=00;90:*.dpkg-new=00;90:*.dpkg-old=00;90:*.dpkg-tmp=00;90:*.old=00;90:*.orig=00;90:*.part=00;90:*.rej=00;90:*.rpmnew=00;90:*.rpmorig=00;90:*.rpmsave=00;90:*.swp=00;90:*.tmp=00;90:*.ucf-dist=00;90:*.ucf-new=00;90:*.ucf-old=00;90:
* MEMORY_PRESSURE_WATCH : /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/session.slice/org.gnome.SettingsDaemon.MediaKeys.service/memory.pressure
* MEMORY_PRESSURE_WRITE : c29tZSAyMDAwMDAgMjAwMDAwMAA=
* NIPYPE_NO_ET : 1
* NSLOTS : 1
* PATH : /home/laurin/anaconda3/envs/t2mappingenv/bin:/opt/ants/bin:/opt/qi:/opt/pycharm/bin:/opt/qi:/opt/pycharm/bin:/home/laurin/anaconda3/condabin:/home/laurin/fsl/share/fsl/bin:/home/laurin/fsl/share/fsl/bin:/home/laurin/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
* PWD : /home/laurin/workspace/t2_mapping_traveling_heads/cli
* PYCHARM_DISPLAY_PORT : 63342
* PYCHARM_HOSTED : 1
* PYCHARM_INTERACTIVE_PLOTS : 1
* PYCHARM_PROJECT_ID : ecae3fb
* PYTHONIOENCODING : UTF-8
* PYTHONPATH : /home/laurin/workspace/t2_mapping_traveling_heads:/home/laurin/workspace/t2_mapping_traveling_heads/external/MagPhsT2:/opt/pycharm/plugins/python-ce/helpers/pycharm_plotly_backend:/opt/pycharm/plugins/python-ce/helpers/pycharm_matplotlib_backend:/opt/pycharm/plugins/python-ce/helpers/pycharm_display
* PYTHONUNBUFFERED : 1
* QTWEBENGINE_DICTIONARIES_PATH : /usr/share/hunspell-bdic/
* QT_ACCESSIBILITY : 1
* QT_IM_MODULE : ibus
* SESSION_MANAGER : local/laurin-ThinkPad:@/tmp/.ICE-unix/2741,unix/laurin-ThinkPad:/tmp/.ICE-unix/2741
* SHELL : /bin/bash
* SHLVL : 1
* SSH_AUTH_SOCK : /run/user/1000/keyring/ssh
* SYSTEMD_EXEC_PID : 2952
* TERM : xterm-256color
* USER : laurin
* USERNAME : laurin
* VTE_VERSION : 7600
* WAYLAND_DISPLAY : wayland-0
* XAUTHORITY : /run/user/1000/.mutter-Xwaylandauth.LFTCW2
* XDG_CONFIG_DIRS : /etc/xdg/xdg-ubuntu:/etc/xdg
* XDG_CURRENT_DESKTOP : ubuntu:GNOME
* XDG_DATA_DIRS : /usr/share/ubuntu:/usr/share/gnome:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop
* XDG_MENU_PREFIX : gnome-
* XDG_RUNTIME_DIR : /run/user/1000
* XDG_SESSION_CLASS : user
* XDG_SESSION_DESKTOP : ubuntu
* XDG_SESSION_TYPE : wayland
* XMODIFIERS : @im=ibus
* _ : /opt/pycharm/bin/pycharm
* _CE_CONDA : 
* _CE_M : 

