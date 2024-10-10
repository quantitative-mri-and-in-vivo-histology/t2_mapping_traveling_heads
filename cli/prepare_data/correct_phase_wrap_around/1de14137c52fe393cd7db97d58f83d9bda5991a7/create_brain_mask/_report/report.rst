Node: correct_phase_wrap_around (create_brain_mask (utility)
============================================================


 Hierarchy : prepare_data.correct_phase_wrap_around.create_brain_mask
 Exec ID : create_brain_mask.a0


Original Inputs
---------------


* function_str : def create_brain_mask_from_anatomical_b1(in_file, threshold=200, fwhm=8):
    """
    Shift a specified number of voxels along a given dimension and merge them to the other side.

    Parameters:
    in_file (str): Path to the input NIfTI file.
    n_voxels (int): Number of voxels to shift.
    axis (int): Dimension along which to shift (0 for x, 1 for y, 2 for z).
    out_file (str): Path for the output NIfTI file.

    Returns:
    out_file (str): Path to the modified NIfTI file.
    """
    import os
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter

    base_dir = os.getcwd()

    # Load the NIfTI file
    image_nib = nib.load(in_file)
    image = image_nib.get_fdata()

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    smoothed_data = gaussian_filter(image, sigma=sigma)
    brain_mask = smoothed_data > threshold

    brain_mask_nib = nib.Nifti1Image(brain_mask, image_nib.affine,
                                     image_nib.header)
    out_file = os.path.join(base_dir, 'brain_mask.nii.gz')
    nib.save(brain_mask_nib, out_file)

    return out_file

* fwhm : 8
* in_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_phase_wrap_around/1de14137c52fe393cd7db97d58f83d9bda5991a7/cut_and_merge_b1_anat_ref/image_shifted.nii.gz


Execution Inputs
----------------


* function_str : def create_brain_mask_from_anatomical_b1(in_file, threshold=200, fwhm=8):
    """
    Shift a specified number of voxels along a given dimension and merge them to the other side.

    Parameters:
    in_file (str): Path to the input NIfTI file.
    n_voxels (int): Number of voxels to shift.
    axis (int): Dimension along which to shift (0 for x, 1 for y, 2 for z).
    out_file (str): Path for the output NIfTI file.

    Returns:
    out_file (str): Path to the modified NIfTI file.
    """
    import os
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter

    base_dir = os.getcwd()

    # Load the NIfTI file
    image_nib = nib.load(in_file)
    image = image_nib.get_fdata()

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    smoothed_data = gaussian_filter(image, sigma=sigma)
    brain_mask = smoothed_data > threshold

    brain_mask_nib = nib.Nifti1Image(brain_mask, image_nib.affine,
                                     image_nib.header)
    out_file = os.path.join(base_dir, 'brain_mask.nii.gz')
    nib.save(brain_mask_nib, out_file)

    return out_file

* fwhm : 8
* in_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_phase_wrap_around/1de14137c52fe393cd7db97d58f83d9bda5991a7/cut_and_merge_b1_anat_ref/image_shifted.nii.gz


Execution Outputs
-----------------


* out_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_phase_wrap_around/1de14137c52fe393cd7db97d58f83d9bda5991a7/create_brain_mask/brain_mask.nii.gz


Runtime info
------------


* duration : 0.023643
* hostname : laurin-dlm
* prev_wd : /home/laurin/workspace/t2_mapping_traveling_heads/cli
* working_dir : /home/laurin/workspace/t2_mapping_traveling_heads/cli/prepare_data/correct_phase_wrap_around/1de14137c52fe393cd7db97d58f83d9bda5991a7/create_brain_mask


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

