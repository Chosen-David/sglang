#!/bin/bash
# E67 升 τ e2e 等待器：等 E72 任务链 CHAIN_DONE（双卡全量结束）后，
# 双卡并行跑 tau01/tau02 两臂（musique+qasper far-heavy 多跳判决）。
# 幂等：E67_DONE marker 存在则退出。
CD=/tmp/tli_chain
LOG=/tmp/tli_e67_tau.log
[ -f $CD/E67_DONE ] && exit 0
while [ ! -f $CD/CHAIN_DONE ]; do
  echo "[$(date '+%m-%d %H:%M:%S')] wait CHAIN_DONE" >> $LOG
  sleep 300
done
echo "[$(date '+%m-%d %H:%M:%S')] CHAIN_DONE detected, launching E67 tau arms" >> $LOG
cd /home/wangyuanshuo02/sglang
ENV3="SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SGLANG_IS_FLASHINFER_AVAILABLE=false SGLANG_ENABLE_JIT_DEEPGEMM=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:/home/wangyuanshuo02/sglang/python"
env CUDA_VISIBLE_DEVICES=0 ARM=tau01 $ENV3 nohup python3 test_tli_e67_tau.py > /tmp/e67_tau01.log 2>&1 &
env CUDA_VISIBLE_DEVICES=1 ARM=tau02 $ENV3 nohup python3 test_tli_e67_tau.py > /tmp/e67_tau02.log 2>&1 &
wait
grep -q "E67 tau01 DONE" /tmp/e67_tau01.log && grep -q "E67 tau02 DONE" /tmp/e67_tau02.log && touch $CD/E67_DONE
echo "[$(date '+%m-%d %H:%M:%S')] E67 tau arms done" >> $LOG
