# AESA-CNN 算法版本演进

本仓库整理 47 GHz 有源相控阵项目中由本人完成的算法版本。提交历史从 V2.5 开始，
每个版本对应一个独立提交和同名 Git 标签；接手项目前的原始主线代码不在本仓库中。

当前已纳入：**V2.5–V2.10**。

## 研究主线

项目围绕物理约束的远场建模和波束控制展开：将阵列几何、传播相位和相干叠加写入
解析主干，再用轻量神经网络学习解析模型未覆盖的残差。后续版本逐步增加 HFSS 数据合同、
幅度监督、复算子审计、结构消融和跨阵列验证。

## 当前版本

| 版本 | 内容 | 状态 |
|---|---|---|
| [V2.5](algorithm_versions/v2.5/README.md) | 轻量控制器与完整复方向图观察器 | 已冻结 |
| [V2.6](algorithm_versions/v2.6/README.md) | 47 GHz、1×4 HFSS 固定阵列控制器 | 主机实现与物理审计完成 |
| [V2.7](algorithm_versions/v2.7/README.md) | 二维可达边界与停止门禁审计 | 边界审计完成，未发布新模型 |
| [V2.8](algorithm_versions/v2.8/README.md) | 4×4 纯幅度数据合同与前向基线 | 数据合同和离线基线完成 |
| [V2.9](algorithm_versions/v2.9/README.md) | 物理分层建模与四目标优化 | 分层建模和受控实验完成 |
| [V2.10](algorithm_versions/v2.10/README.md) | 严格正半球纯幅度监督基线 | 协议、基线和证据已冻结 |

## 环境与检查

```powershell
python -m pip install -r requirements.txt
python algorithm_versions/run_all_checks.py
```

代码测试通过只说明当前仓库中的实现和紧凑证据可读取，不等同于重新完成 HFSS 求解、
模型训练、暗室测试或硬件部署。
