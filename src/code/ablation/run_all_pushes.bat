@echo off
setlocal EnableExtensions EnableDelayedExpansion

set ISAACLAB=A:\AllIsaac\IsaacLab
set ABL=A:\AllIsaac\flow_matching_project\scripts\ablation
set OUT=A:\AllIsaac\flow_matching_project\data\ablation
set LOG=A:\AllIsaac\flow_matching_project\_pushes.log

set CKPT_NOREG=A:\AllIsaac\IsaacLab\logs\rsl_rl\unitree_go1_flat_noreg\2026-05-04_12-34-51\model_299.pt
set CKPT_SOMEREG=A:\AllIsaac\IsaacLab\logs\rsl_rl\unitree_go1_flat\2026-04-06_12-42-26\model_299.pt
set CKPT_EXTREMEREG=A:\AllIsaac\IsaacLab\logs\rsl_rl\unitree_go1_flat_extremereg\2026-05-04_12-43-47\model_299.pt
set FLOW=A:\AllIsaac\IsaacLab\flow_model_balanced_lp.pt

echo === RUNNING 7-WAY PUSH RECOVERY BATCH > %LOG%
echo Started: %date% %time% >> %LOG%

cd /d %ISAACLAB%

:: variant 1: noreg raw
echo. >> %LOG%
echo [1/7] noreg (raw) >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 --variant noreg --mode raw --checkpoint %CKPT_NOREG% --headless >> %LOG% 2>&1

:: variant 2: somereg raw  (use existing PPO + stock task)
echo. >> %LOG%
echo [2/7] somereg (raw) >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --variant somereg --mode raw --checkpoint %CKPT_SOMEREG% --headless >> %LOG% 2>&1

:: variant 3: extremereg raw
echo. >> %LOG%
echo [3/7] extremereg (raw) >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-ExtremeReg-v0 --variant extremereg --mode raw --checkpoint %CKPT_EXTREMEREG% --headless >> %LOG% 2>&1

:: variant 4: noreg + flow (Bal-LP, t_end=1.0)
echo. >> %LOG%
echo [4/7] noreg_flow >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 --variant noreg_flow --mode flow --checkpoint %CKPT_NOREG% --flow_model %FLOW% --t_end 1.0 --headless >> %LOG% 2>&1

:: variant 5: noreg + lp (causal IIR 15Hz)
echo. >> %LOG%
echo [5/7] noreg_lp >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 --variant noreg_lp --mode lp --checkpoint %CKPT_NOREG% --headless >> %LOG% 2>&1

:: variant 6: somereg + lp (causal IIR 15Hz)
echo. >> %LOG%
echo [6/7] somereg_lp >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --variant somereg_lp --mode lp --checkpoint %CKPT_SOMEREG% --headless >> %LOG% 2>&1

:: variant 7: noreg + flow + lp
echo. >> %LOG%
echo [7/7] noreg_flow_lp >> %LOG%
call isaaclab.bat -p %ABL%\push_recovery_runner.py --task Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 --variant noreg_flow_lp --mode flow_lp --checkpoint %CKPT_NOREG% --flow_model %FLOW% --t_end 1.0 --headless >> %LOG% 2>&1

echo. >> %LOG%
echo === DONE === >> %LOG%
echo Finished: %date% %time% >> %LOG%
type nul > A:\AllIsaac\flow_matching_project\_pushes_done.flag
