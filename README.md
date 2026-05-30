# RM OpenPI MuJoCo

这是一个面向 RM65B 机械臂的 OpenPI + MuJoCo 项目仓库。仓库以最新版 OpenPI 代码为基础，保留模型、策略、训练和客户端能力，同时把 RM65B 仿真相关内容整理到独立项目目录，方便后续做仿真验证、策略接入和 sim-to-real 实验。

> 说明：当前代码是从 2025 年 6 月份的一个基于 pi0 的小项目整理出来的。原项目里混有实验脚本、旧版 MuJoCo 包、机器人 SDK、截图和本地配置；本仓库已经做了清理和结构整理，但仍可能存在兼容性问题、脚本细节 bug 或未覆盖的边界情况，后续使用时建议先从仿真和小范围测试开始。

## 这个仓库包含什么

- `src/openpi/`：同步后的 OpenPI 核心代码，包括模型、策略、数据处理和训练逻辑。
- `packages/openpi-client/`：轻量级 websocket 客户端，用于连接外部策略推理服务。
- `projects/rm65b/`：RM65B 机械臂项目目录，包含 MuJoCo 场景、最小资产和运行脚本。
- `pyproject.toml` / `uv.lock`：统一依赖管理，MuJoCo 使用 Python 包 `mujoco==3.9.0`，不再提交旧版二进制 SDK 目录。

## 已经清理掉的内容

- 硬编码的私有服务器 IP、机器人 IP 默认值和项目脚本里的 API key 参数。
- OpenPI 服务端入口、远程推理文档和服务端 Docker 示例。
- 旧版 MuJoCo 二进制包、机器人 SDK 二进制/示例工程、运行截图、日志和压缩包。
- 大量与 RM65B 项目无关的官方 MuJoCo 示例和临时实验文件。

## 目录结构

```text
src/openpi/                         OpenPI 模型、策略、训练代码
packages/openpi-client/             OpenPI 客户端工具
projects/rm65b/simulation/assets/   RM65B MuJoCo 场景和必要网格资产
projects/rm65b/simulation/scripts/  RM65B 仿真、策略客户端、sim-to-real 脚本
docs/                               保留的 OpenPI 通用文档
scripts/                            OpenPI 训练和工具脚本
```

## 环境安装

推荐使用 `uv`：

```bash
uv sync
```

如果要控制真实 RM 机械臂，请单独安装 RealMan 官方 `Robotic_Arm` Python 包。这个仓库不再内置机器人 SDK，避免把二进制文件和本地示例工程混进项目代码。

## RM65B 仿真

先运行关节和夹爪的基础测试：

```bash
uv run python projects/rm65b/simulation/scripts/test_actuators.py
```

连接本地 OpenPI 兼容策略服务进行仿真控制：

```bash
uv run python projects/rm65b/simulation/scripts/simulate_policy_client.py --host localhost --port 8000
```

如果要做 sim-to-real，需要显式传入真实机器人 IP：

```bash
uv run python projects/rm65b/simulation/scripts/sim_to_real_client.py --host localhost --port 8000 --run-robot --robot-ip <机器人IP>
```

注意：仓库里不会再给真实机器人地址写默认值，避免误连设备。

## 推荐仓库信息

建议把 GitHub 仓库名改为：

```text
rm-openpi-mujoco
```

建议 About 描述写成：

```text
RM65B 机械臂的 MuJoCo 仿真与 OpenPI 策略接入项目。
```
