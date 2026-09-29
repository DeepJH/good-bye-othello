# good-bye-othello

将强大的黑白棋（Othello / Reversi）AlphaZero AI 应用到实战中。

本项目集成了基于蒙特卡洛树搜索（MCTS）与深度卷积神经网络（PyTorch）的 AlphaZero 引擎，并内置预训练好的 6x6 与 8x8 黑白棋高水平模型权重，可直接用于对弈与后续实战系统开发。

---

## 运行配置需求与资源消耗

本项目在**普通个人电脑、轻量云服务器或嵌入式设备（如树莓派 4B/5）**上均可流畅运行推理：

| 硬件维度 | 最低配置要求 | 推荐配置 | 实测运行时开销 |
| :--- | :--- | :--- | :--- |
| **处理器 (CPU)** | 64位 双核 CPU (x86_64 / ARM64) | 4核以上主流处理器 | 单局推理单核占用，单步耗时约 0.05 ~ 0.2 秒 |
| **显卡 (GPU)** | **无需独立显卡**（纯 CPU 即可运行） | 支持 CUDA 的 NVIDIA 显卡（可选加速） | 默认纯 CPU 模式运行极快 |
| **内存 (RAM)** | 至少 **1 GB** 空闲可用内存 | 2 GB 以上 | **实测峰值内存（RSS）仅约 300 MB ~ 350 MB** |
| **磁盘空间** | **1 GB** 可用空间 | 2 GB 可用空间 | 模型权重约 100 MB，CPU 版虚拟环境约 500 MB |
| **软件环境** | Python 3.8 ~ 3.12，Linux / macOS / Windows | Python 3.12 (推荐搭配 uv) | 依赖仅为 `torch`、`numpy`、`tqdm` |

---

## 目录结构

```text
good-bye-othello/
├── alpha-zero-general/          # AlphaZero 核心算法与黑白棋引擎模块
│   ├── Arena.py                 # 对弈竞技场调度逻辑
│   ├── Game.py                  # 棋类环境基类规范
│   ├── MCTS.py                  # 蒙特卡洛树搜索核心实现
│   ├── NeuralNet.py             # 神经网络基类定义
│   ├── utils.py                 # 工具函数与字典结构
│   ├── othello/                 # 黑白棋游戏逻辑与 PyTorch 网络
│   │   ├── OthelloGame.py       # 棋盘规则、有效步判定与胜负结算
│   │   ├── OthelloLogic.py      # 黑白棋底层走子翻子算法
│   │   ├── OthelloPlayers.py    # 玩家策略（Random, Greedy, Human）
│   │   └── pytorch/             # OthelloNNet 卷积网络与包装器
│   └── pretrained_models/       # 预训练模型权重
│       └── othello/pytorch/
│           ├── 6x100x25_best.pth.tar       (6x6 模型，约 38MB)
│           └── 8x8_100checkpoints_best.pth.tar (8x8 模型，约 62MB)
├── play.py                      # 统一对弈与演示入口（支持人机、机机、单局与循环赛）
├── test_alpha_zero.py           # 自动化测试与模型走子验证脚本
├── requirements.txt             # 项目依赖清单
├── LICENSE                      # 开源许可证（含原作者 MIT 许可保留）
└── README.md                    # 本说明文件
```

---

## 环境准备与快速上手

### 1. 依赖安装

推荐使用 `uv` 创建轻量 Python 3.12 环境（也可使用普通 `venv`）：

```bash
# 使用 uv 创建虚拟环境并安装依赖
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **提示**：如果使用 pip 直接安装且无需 GPU，可安装 CPU 版 PyTorch 以大幅节省下载体积与磁盘占用：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install numpy tqdm
> ```

### 2. 运行预训练模型对弈

#### 快速自对弈演示（默认 8x8 棋盘，AI vs 随机对手）：
```bash
python play.py --board-size 8 --sims 25
```

#### 快速验证 6x6 棋盘：
```bash
python play.py --board-size 6 --sims 15
```

#### 亲自与 AI 对战（人机交互模式）：
```bash
python play.py --board-size 8 --player1 human --player2 alpha
```
在控制台中，根据提示输入坐标（例如 `2 3`）即可落子。

### 3. 运行自动化验证测试

```bash
python test_alpha_zero.py
```

---

## 许可证说明（License & Attribution）

- 本项目采用 **MIT License** 开源。
- 核心算法引擎模块（`alpha-zero-general/`）派生自 [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)，遵循原作者的 MIT 许可，其完整的著作权声明与原许可证已严格保留在根目录 [LICENSE](./LICENSE) 文件中。

---

## 鸣谢与致谢（Acknowledgments & Credits）

本项目底层算法与预训练模型源自开源社区与先驱学者的杰出成果，特别向以下项目与研究者致以诚挚的感谢：

1. **上游开源项目**：
   - 感谢 [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) 提供的清晰、优雅且模块化的通用 AlphaZero 算法实现与预训练权重。
2. **核心贡献者团队**：
   - 感谢原作者 **Surag Nair**，以及核心作者 **Shantanu Thakoor** 和 **Megha Jhunjhunwala** 的算法设计与训练工作。
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
   - 致敬 DeepMind 团队关于 AlphaGo Zero 的里程碑论文：
     *Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017).*
