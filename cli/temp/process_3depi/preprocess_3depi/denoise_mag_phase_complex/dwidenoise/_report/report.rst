Node: preprocess_3depi (denoise_mag_phase_complex (dwidenoise (mrtrix3)
=======================================================================


 Hierarchy : process_3depi.preprocess_3depi.denoise_mag_phase_complex.dwidenoise
 Exec ID : dwidenoise


Original Inputs
---------------


* args : <undefined>
* bval_scale : <undefined>
* environ : {}
* extent : (3, 3, 3)
* grad_file : <undefined>
* grad_fsl : <undefined>
* in_bval : <undefined>
* in_bvec : <undefined>
* in_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz
* mask : <undefined>
* noise : <undefined>
* nthreads : <undefined>
* out_bval : <undefined>
* out_bvec : <undefined>
* out_file : <undefined>


Execution Inputs
----------------


* args : <undefined>
* bval_scale : <undefined>
* environ : {}
* extent : (3, 3, 3)
* grad_file : <undefined>
* grad_fsl : <undefined>
* in_bval : <undefined>
* in_bvec : <undefined>
* in_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz
* mask : <undefined>
* noise : <undefined>
* nthreads : <undefined>
* out_bval : <undefined>
* out_bvec : <undefined>
* out_file : <undefined>


Execution Outputs
-----------------


* noise : <undefined>
* out_file : /home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/dwidenoise/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz


Runtime info
------------


* cmdline : dwidenoise -extent 3,3,3 -noise sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz /home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz
* duration : 56.92076
* hostname : laurin-ThinkPad
* prev_wd : /home/laurin/workspace/t2_mapping_traveling_heads/cli
* working_dir : /home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/dwidenoise


Terminal output
~~~~~~~~~~~~~~~


 


Terminal - standard output
~~~~~~~~~~~~~~~~~~~~~~~~~~


 


Terminal - standard error
~~~~~~~~~~~~~~~~~~~~~~~~~


 [?7l
dwidenoise: [  0%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  1%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  2%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  3%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  4%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  5%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  6%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  7%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  8%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  9%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 10%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 11%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 12%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 13%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 14%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 15%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 16%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 17%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 18%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 19%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 20%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 21%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 22%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 23%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 24%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 25%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 26%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 27%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 28%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 29%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 30%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 31%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 32%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 33%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 34%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 35%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 36%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 37%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 38%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 39%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 40%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 41%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 42%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 43%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 44%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 45%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 46%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 47%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 48%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 49%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 50%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 51%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 52%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 53%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 54%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 55%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 56%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 57%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 58%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 59%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 60%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 61%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 62%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 63%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 64%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 65%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 66%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 67%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 68%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 69%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 70%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 71%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 72%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 73%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 74%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 75%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 76%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 77%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 78%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 79%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 80%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 81%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 82%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 83%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 84%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 85%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 86%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 87%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 88%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 89%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 90%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 91%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 92%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 93%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 94%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 95%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 96%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 97%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 98%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 99%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h
dwidenoise: [100%] uncompressing image "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"[0K
[?7l
dwidenoise: [  0%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  1%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  2%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  3%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  4%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  5%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  6%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  7%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  8%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [  9%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 10%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 11%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 12%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 13%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 14%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 15%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 16%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 17%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 18%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 19%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 20%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 21%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 22%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 23%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 24%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 25%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 26%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 27%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 28%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 29%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 30%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 31%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 32%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 33%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 34%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 35%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 36%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 37%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 38%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 39%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 40%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 41%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 42%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 43%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 44%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 45%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 46%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 47%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 48%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 49%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 50%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 51%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 52%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 53%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 54%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 55%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 56%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 57%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 58%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 59%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 60%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 61%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 62%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 63%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 64%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 65%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 66%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 67%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 68%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 69%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 70%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 71%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 72%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 73%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 74%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 75%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 76%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 77%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 78%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 79%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 80%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 81%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 82%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 83%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 84%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 85%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 86%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 87%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 88%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 89%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 90%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 91%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 92%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 93%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 94%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 95%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 96%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 97%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 98%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 99%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h[?7l
dwidenoise: [100%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"...[0K[?7h
dwidenoise: [100%] preloading data for "/home/laurin/workspace/t2_mapping_traveling_heads/cli/temp/process_3depi/preprocess_3depi/denoise_mag_phase_complex/convert_mag_and_phase_to_complex/sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx.nii.gz"[0K
[?7l
dwidenoise: [  0%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  1%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  2%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  3%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  4%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  5%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  6%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  7%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  8%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [  9%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 10%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 11%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 12%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 13%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 14%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 15%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 16%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 17%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 18%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 19%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 20%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 21%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 22%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 23%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 24%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 25%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 26%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 27%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 28%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 29%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 30%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 31%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 32%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 33%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 34%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 35%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 36%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 37%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 38%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 39%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 40%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 41%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 42%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 43%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 44%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 45%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 46%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 47%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 48%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 49%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 50%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 51%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 52%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 53%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 54%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 55%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 56%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 57%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 58%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 59%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 60%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 61%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 62%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 63%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 64%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 65%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 66%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 67%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 68%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 69%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 70%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 71%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 72%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 73%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 74%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 75%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 76%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 77%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 78%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 79%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 80%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 81%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 82%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 83%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 84%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 85%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 86%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 87%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 88%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 89%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 90%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 91%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 92%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 93%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 94%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 95%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 96%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 97%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 98%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [ 99%] running MP-PCA denoising...[0K[?7h[?7l
dwidenoise: [100%] running MP-PCA denoising...[0K[?7h
dwidenoise: [100%] running MP-PCA denoising[0K
[?7l
dwidenoise: [  0%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  1%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  2%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  3%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  4%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  5%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  6%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  7%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  8%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [  9%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 10%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 11%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 12%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 13%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 14%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 15%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 16%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 17%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 18%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 19%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 20%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 21%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 22%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 23%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 24%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 25%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 26%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 27%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 28%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 29%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 30%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 31%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 32%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 33%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 34%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 35%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 36%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 37%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 38%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 39%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 40%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 41%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 42%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 43%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 44%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 45%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 46%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 47%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 48%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 49%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 50%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 51%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 52%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 53%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 54%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 55%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 56%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 57%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 58%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 59%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 60%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 61%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 62%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 63%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 64%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 65%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 66%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 67%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 68%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 69%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 70%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 71%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 72%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 73%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 74%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 75%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 76%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 77%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 78%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 79%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 80%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 81%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 82%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 83%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 84%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 85%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 86%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 87%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 88%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 89%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 90%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 91%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 92%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 93%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 94%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 95%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 96%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 97%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 98%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 99%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"...[0K[?7h
dwidenoise: [100%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_denoised.nii.gz"[0K
[?7l
dwidenoise: [  1%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [  3%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [  4%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [  6%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [  7%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [  9%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 10%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 12%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 13%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 15%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 16%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 18%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 19%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 21%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 22%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 24%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 25%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 27%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 28%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 30%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 31%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 33%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 34%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 36%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 37%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 39%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 40%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 42%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 43%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 45%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 46%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 48%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 49%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 51%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 52%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 54%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 55%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 57%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 58%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 60%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 61%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 63%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 64%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 66%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 67%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 69%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 70%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 72%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 73%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 75%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 76%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 78%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 79%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 81%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 82%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 84%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 85%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 87%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 88%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 90%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 91%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 93%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 94%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 96%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 97%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [ 99%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h[?7l
dwidenoise: [100%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"...[0K[?7h
dwidenoise: [100%] compressing image "sub-phy002_ses-001_acq-dznebnep3d_part-mag_T2w_cplx_noise.nii.gz"[0K


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

