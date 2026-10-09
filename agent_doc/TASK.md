
## E117 逐层配置 e2e 小试判决【进行中】
- **两臂**：perlayer（8 层覆写最优配置，L4/24/28=单池）vs uniform（全层冠军 (.25,.125,.625)）
- **机制**：#191 harness wrapper（pred.py + register_patch 逐层覆写）已验证，两臂输出已产生 7/200 真实差异
- **判决规则**：perlayer 13 任务平均分显著优于 uniform（配对 bootstrap CI 不含 0）→ GO 并入生产；否则 NO-GO 关闭
- **状态**：3/13 任务完成，预计 ~06:30 收官
- **数据落盘**：/tmp/e117_trial/pred_e117{per,uni}/，13 任务×200 行×2 臂
