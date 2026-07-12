@echo off
setlocal EnableExtensions
set ISAACLAB=A:\AllIsaac\IsaacLab
set ABL=A:\AllIsaac\flow_matching_project\scripts\ablation
set LOG=A:\AllIsaac\flow_matching_project\_pushes_47.log
set CKPT_NOREG=A:\AllIsaac\IsaacLab\logs\rsl_rl\unitree_go1_flat_noreg\2026-05-04_12-34-51\model_299.pt
set FLOW=A:\AllIsaac\IsaacLab\flow_model_balanced_lp_noreg.pt

echo === RE-RUN PUSHES 4 and 7 with noreg-trained flow > %LOG%
cd /d %ISAACLAB%

echo. >> %LOG%
echo [4/7] noreg_flow (with noreg-trained Bal-LP) >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 --variant noreg_flow --mode flow --checkpoint %CKPT_NOREG% --flow_model %FLOW% --t_end 1.0 --headless >> %LOG% 2>&1

echo. >> %LOG%
echo [7/7] noreg_flow_lp (with noreg-trained Bal-LP) >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 --variant noreg_flow_lp --mode flow_lp --checkpoint %CKPT_NOREG% --flow_model %FLOW% --t_end 1.0 --headless >> %LOG% 2>&1

echo. >> %LOG%
echo === DONE === >> %LOG%
type nul > A:\AllIsaac\flow_matching_project\_pushes_47_done.flag
