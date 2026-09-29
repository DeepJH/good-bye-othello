# good-bye-othello

将强大的黑白棋（Othello / Reversi）AlphaZero AI 应用到实战中。

本项目包含两个核心子系统：
1. **AlphaZero 核心引擎**：基于深度卷积神经网络（PyTorch）与蒙特卡洛树搜索（MCTS）的高水平黑白棋 AI，内置官方 6x6 与 8x8 预训练模型。
2. **多端自动化控制架构 (`othello_agent`)**：采用**双向解耦适配器（Adapter）**架构，将 AI 决策、纯视觉棋盘识别（Vision）、被控端交互（ADB / Mock / Web）彻底分离，支持通过 Android ADB 自动化操控任意黑白棋 App 进行实战对弈。

---

## 总体系统架构

系统采用高内聚、低耦合的标准转化层设计：

```text
┌────────────────────────────────────────────────────────┐
│                   Game Orchestrator                    │
│           （调度中枢：协调感知、决策、轮次与执行）             │
└───────────┬────────────────────────────────┬───────────┘
            │                                │
    标准棋盘状态 (BoardState)           标准动作 Action (row, col)
            │                                │
            ▼                                ▼
┌───────────────────────┐        ┌───────────────────────┐
│     Model Adapter     │        │  Controller Adapter   │
│   （模型输入输出转化层）  │        │    （被控端执行转化层）   │
│ 包装 AlphaZero/MCTS   │        │ 包装 ADB / Web / PC   │
└───────────────────────┘        └───────────┬───────────┘
                                             │
                                   获取屏幕图像 / 发送物理点击
                                             │
                                             ▼
                                 ┌───────────────────────┐
                                 │     Vision Module     │
                                 │   （通用可复用棋盘视觉）  │
                                 │ 图像 -> 标准 BoardState│
                                 └───────────────────────┘
```

- **标准数据协议 (`othello_agent/protocols.py`)**：统一标准 `BoardState`、`Player`、`CellState`、`GridGeometry`。
- **模型转化层 (`othello_agent/model_adapter.py`)**：负责自动视角色彩反转（白棋转为标准视角）、合法步过滤、MCTS 策略概率推理与动作解码。
- **可复用视觉识别 (`othello_agent/vision/`)**：纯算法模块，不绑定 ADB。接收图像与棋盘区域配置（ROI），毫秒级提取 8x8 棋盘状态并计算每个格子的物理中心点坐标。
- **被控端抽象 (`othello_agent/controllers/`)**：定义统一控制协议，包含 `AdbController`（安卓高速截屏与模拟点击）和 `MockController`（离线单元测试）。

---

## 目录结构

```text
good-bye-othello/
├── othello_agent/               # 外挂自动化核心架构包
│   ├── protocols.py             # 核心数据模型与标准接口定义
│   ├── model_adapter.py         # AlphaZero 模型的标准接口转化层
│   ├── orchestrator.py          # 调度中枢（感知 -> 推断 -> 决策 -> 点击循环）
│   ├── vision/                  # 通用可复用视觉模块
│   │   ├── calibration.py       # 棋盘 ROI 标定与配置存取
│   │   └── recognizer.py        # 图像色彩空间采样与棋盘矩阵识别
│   └── controllers/             # 被控端转化层
│       ├── base.py              # 控制器抽象基类
│       ├── mock.py              # 离线虚拟控制器
│       └── adb.py               # 安卓原生 ADB 控制器
├── alpha-zero-general/          # AlphaZero 核心算法与预训练权重
│   ├── Arena.py                 # 对弈调度
│   ├── Game.py                  # 游戏基类
│   ├── MCTS.py                  # MCTS 核心实现
│   ├── NeuralNet.py             # 神经网络基类
│   ├── utils.py                 # 基础工具
│   ├── othello/                 # 游戏规则与 PyTorch 卷积网络
│   └── pretrained_models/       # 预训练模型 (6x6 与 8x8)
├── config/                      # 标定配置文件目录
│   └── default_8x8_config.json  # 默认 8x8 屏幕配置示例
├── calibrate.py                 # 一键截屏标定工具（生成视觉覆盖预览图）
├── run_bot.py                   # ADB 自动对弈外挂主入口
├── play.py                      # 本地终端对弈与自对弈演示入口
├── test_alpha_zero.py           # AlphaZero 引擎基础测试
├── tests/                       # 自动化测试套件（100% 覆盖）
├── requirements.txt             # 依赖声明
├── LICENSE                      # 开源许可证（保留原作者 MIT 版权）
└── README.md                    # 本说明文件
```

---

## 环境准备与快速上手

### 1. 依赖安装

推荐使用 `uv` 创建 Python 3.12 环境：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. 运行自动化测试（14 项测试全绿）

```bash
.venv/bin/pytest -v
```

---

## 安卓 ADB 自动化控制实战指南

### 步骤 1：连接安卓手机或模拟器

确保安卓设备开启了“USB 调试”，并通过数据线连接电脑（或启动雷电、MuMu、Android Studio 模拟器）：

```bash
adb devices
```
终端显示设备序列号即说明连接成功。

### 步骤 2：对目标 App 进行一键棋盘标定

打开小众黑白棋 App 进入对局界面，运行标定脚本截屏：

```bash
python calibrate.py --capture app_screen.png
```
脚本会自动在当前目录下保存 `app_screen.png`。如果您知道棋盘坐标范围（左,上,右,下），可直接传入；如果不传则默认居中：

```bash
# 示例：指定棋盘左上角 (40, 600) 到右下角 (1040, 1600)
python calibrate.py --image app_screen.png --roi 40,600,1040,1600 --save config/my_app.json
```
标定工具会生成 `config/calibration_preview.png`，打开该图片即可直观看到棋盘边界红框与每个格子的青色采样点击中心点是否对齐。

### 步骤 3：启动 AI 外挂自动下棋

```bash
python run_bot.py --config config/my_app.json --sims 25
```
- `--color auto`（默认）：AI 自动根据盘面棋子总数的奇偶性推断轮次与阵营；也可以显式指定 `--color black` 或 `--color white`。
- `--sims`：每步 MCTS 模拟次数（默认 25，步耗时仅约 0.2 秒）。
- 外挂将全自动检测屏幕、等待对手走棋、在轮到我方时毫秒级计算最优步并发送触控点击！

---

## 本地交互与预训练模型验证

如果不连接手机，也可以直接在终端与预训练模型下棋：

```bash
# 终端人机对弈（玩家输入坐标与 AI 对战）
python play.py --board-size 8 --player1 human --player2 alpha

# 本地自动对弈演示（AI vs 随机对手）
python play.py --board-size 8 --sims 25
```

---

## 许可证说明（License & Attribution）

- 本项目采用 **MIT License** 开源。
- 核心算法引擎模块（`alpha-zero-general/`）派生自 [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)，遵循原作者的 MIT 许可，其完整的著作权声明与原许可证已严格保留在根目录 [LICENSE](./LICENSE) 文件中。

---

## 鸣谢与致谢（Acknowledgments & Credits）

本项目底层算法与预训练模型源自开源社区与先驱学者的杰出成果，特别向以下项目与研究者致以诚挚的感谢：

1. **上游开源项目**：
   - 感谢 [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) 提供的通用 AlphaZero 算法实现与预训练权重。
2. **核心贡献者团队**：
   - 感谢原作者 **Surag Nair**，以及核心作者 **Shantanu Thakoor** 和 **Megha Jhunjhunwala**。
3. **学术报告引用**：
   ```bibtex
   @misc{thakoor2016learning,
     title={Learning to play othello without human knowledge},
     author={Thakoor, Shantanu and Nair, Surag and Jhunjhunwala, Megha},
     year={2016},
     publisher={Stanford University, Final Project Report}
   }
   ```
4. **理论奠基**：
   - 致敬 DeepMind 团队关于 AlphaGo Zero 的里程碑论文（Silver et al., Nature 2017）。
