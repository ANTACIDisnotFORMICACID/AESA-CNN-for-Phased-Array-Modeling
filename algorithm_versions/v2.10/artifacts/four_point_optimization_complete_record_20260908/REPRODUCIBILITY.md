# 复现与完整性检查

## 环境

- Python与NumPy用于矩阵复算和统计；
- PyTorch用于O3理想阵列控制；
- CVXPY/半正定求解器用于既有SDP上界；
- HFSS 2025 R2与PyAEDT 0.9.6用于既有直接交叉验证；
- 频率47 GHz。

## 快速整包检查

在本目录执行：

```powershell
python tools/verify_complete_record.py
```

检查内容包括案例数、四类平衡、CNN为空、恒模误差、HFSS矩阵增益复算、SDP逐案例不低于O3、理想阵列结果验证和32400方向谱记录。

## 重建谱上界与随机四点概率

```powershell
python tools/build_spectral_probability_record.py
```

脚本读取同一V2.10目录中 `../front_hemisphere_20260906/datasets/` 的四个冻结HFSS方向数据文件，重新生成 `analysis/` 下三个文件。

## 重跑理想阵列控制

进入 `evidence/ideal_array_control/` 后运行：

```powershell
python run_ideal_array_control.py
python build_report.py
```

脚本使用相对于V2.10目录的位置解析O3结果、四点套件和完整数据。移动时应整体移动V2.10目录；算法参数不得改变，否则不能称为与HFSS O3相同的控制实验。

## HFSS复核限制

AEDT工程快照已保存，但约752 MB的 `.aedtresults` 未复制进本包。现有四点矩阵、HFSS直接复核输出和自动化脚本足以复核报告中的数值；若要重新求解电磁模型，需要原结果目录或重新执行Setup2。

记录包的文件路径和大小列入 `FILE_INDEX.csv`。实验协议中原有的来源标识继续作为历史溯源信息；目录管理不再额外生成整包哈希清单。
