# NeurIPS 2021: MineRL BASALT 行为克隆基线

[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/BT9uegr)

本仓库提供了一个示例，展示如何将基于**行为克隆(Behavioral Cloning)**的解决方案集成到2021年MineRL BASALT竞赛的提交工具包中。

MineRL BASALT是一个专注于解决人类评判任务的竞赛。该竞赛中的任务没有预定义的奖励函数：目标是产生被真实人类评判为能有效解决给定任务的轨迹。

更多详情参见[竞赛主页](https://minerl.io/basalt/)。

## 📦 本仓库包含内容

- **文档**：如何向排行榜提交你的智能体
- **流程**：第一轮和第二轮的程序
- **入门代码**：使用[imitation](https://github.com/HumanCompatibleAI/imitation/)实现的行为克隆来训练简单智能体

## 🔗 其他资源

- [AICrowd竞赛页面](https://www.aicrowd.com/challenges/neurips-2021-minerl-basalt-competition) - 主要注册页面和排行榜
- [MineRL文档](http://minerl.io/docs) - `minerl`包的文档！
- [Imitation文档](https://imitation.readthedocs.io/en/latest/) - `imitation`包的文档，以Stable Baselines 3风格训练基于模仿的模型
- [Sacred文档](https://sacred.readthedocs.io/en/stable/) - `sacred`包的文档，用于结构化和定义实验配置

# 🏗️ 代码结构

## basalt_utils
这部分仓库被构建为一个小型实用程序包，包含包装器、工具和兼容性包装器，使我们能够更轻松地在BASALT环境上进行训练。通过遵循设置说明并从`environment.yml`创建conda环境，它应该会自动安装，但也可以通过导航到目录并调用`pip install .`来手动安装。

## basalt_baselines
这部分仓库是行为克隆训练过程实际逻辑所在的位置，具体在`basalt_baselines/bc.py`中；其他基线是当前正在进行的工作。

训练过程被构建为一个Sacred实验。关于这一点最需要注意的事情是：
1. 在`@bc_baseline.config`装饰的方法中指定的配置值（在本例中为`default_config`），会自动提供给任何用`@bc_baseline.capture`、`@bc_baseline.main`或`bc_baseline.automain`装饰的函数。
2. 如果你想直接运行测试或训练代码，你可以用`basalt_baselines/bc.py with mode='train'`或`with mode='test'`来调用。你还可以通过在命令行上指定配置方法中定义的任何内容的新值来实验不同的配置参数。例如，你可以调用`basalt_baselines.bc.py with batch_size=16`

这个BC基线旨在简单和最小化，因此，它尝试做出最简单的设计选择，使其能够处理Minecraft环境的结构。这些包括：
- 仅提取像素POV观察，并在该观察上使用CNN作为BC模型的输入
- 将连续相机动作转换为离散分块的左/右和上/下移动，因为否则连续空间的对数似然规模会淹没离散空间
- 为构成联合Minecraft动作的每个动作构建单独的动作分布（可以是Discrete和Box的混合），并将这些动作分布组合成一个MultiModalActionDistribution，用于预测BC预测的动作（通过从每个动作空间的分布中独立采样）
- 这种动作分布架构通过学习单个潜在向量，然后将该表示馈送到将其映射到每个动作分布所需参数的头部来工作

# 🚀 如何在AICrowd上提交模型

简要说明：你使用Anaconda环境文件定义你的Python环境，AICrowd系统将构建一个Docker镜像并使用`utility`目录中的docker脚本运行你的代码。

你提交预训练模型、评估代码和训练代码。训练代码应该产生与你作为提交一部分上传的相同模型。

你的评估代码（`test_submission_code.py`）只需要控制智能体并完成环境的任务。评估服务器将处理视频录制。

你使用`aicrowd.json`文件、`tags`字段指定要提交智能体的任务（见下文）。

## 🔧 设置

### 1. 克隆仓库
```bash
git clone https://github.com/minerllabs/basalt_competition_baseline_submissions.git
```

### 2. 安装Java JDK！
**确保你首先安装了[JDK 8](http://minerl.io/docs/tutorials/getting_started.html)！**
-> 访问 http://minerl.io/docs/tutorials/getting_started.html

### 3. 指定你的特定提交依赖项（PyTorch、Tensorflow等）

#### Anaconda环境
要在本地机器上运行此基线代码，你需要在本地机器上创建具有正确依赖项的环境。我们为此目的推荐`anaconda`，并包含了指定运行我们的BC基线所需依赖项的`environment.yml`文件。确保安装了至少版本`4.5.11`的`anaconda`（通过遵循[此处](https://www.anaconda.com/download)的说明）。

如果你的机器没有可以支持`cudatoolkit=10.2`的NVIDIA驱动程序，请在尝试安装之前从`environment.yml`文件中删除该依赖项。然后：

**创建新的conda环境**
使用以下命令：
```bash
conda-env create -f environment.yml
conda activate basalt
```

这将安装`minerl`环境（包含所有竞赛环境），以及用于基线训练本身的依赖项。

**你的代码特定依赖项**
将你自己的依赖项添加到`environment.yml`文件中。**记得添加任何额外的通道**。例如，PyTorch需要`pytorch`通道。
你也可以使用以下命令在本地安装它们：
```bash
conda install <your-package>
```

#### Pip包
如果你需要pip包（不在conda上），你可以将它们添加到`environment.yml`文件中（参见当前填充的版本）：

#### Apt包
如果你的训练过程或智能体依赖于特定的Debian（Ubuntu等）包，将它们添加到`apt.txt`。

这些文件用于构建**本地和AICrowd docker容器**，你的智能体将在其中训练。

如果上述内容对于定义你的环境过于限制，请参见[此Discourse主题以获取更多信息](https://discourse.aicrowd.com/t/how-to-specify-runtime-environment-for-your-submission/2274)。

### 常见设置问题
- 一些用户报告在Mac上安装这组依赖项时遇到问题，并遇到了[此错误](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial)的某些变体。我们目前的看法是，这是一个系统级设置问题，并且没有一个适用于所有Mac OS版本和CUDA版本的单一解决方案，这就是为什么我们在这里不提供具体建议的解决方法。
- 如果你在没有原生显示的机器上运行测试代码（比如通过SSH连接的headless linux服务器），我们建议安装`xvfb`并按照`xvfb-run -a python test_framework.py`的模式运行代码。如果你遇到类似于以下内容的错误，我们建议按照[此博客文章](https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html)中的说明安装没有GL选项的CUDA。
```
There was an error with Malmo"/"No OpenGL context found"/"Couldn't set pixel formal"
```

## 📁 代码结构应该是什么样的？

请遵循入门工具包中共享的示例结构进行代码结构。

不同的文件和目录具有以下含义：
```
.
├── aicrowd.json             # 提交元信息，如你的用户名
├── apt.txt                  # 要在docker镜像内安装的包
├── data                     # 下载的数据，目录路径也可作为`MINERL_DATA_ROOT`环境变量
├── test_submission_code.py  # 重要：你的测试/推理阶段代码。注意：这不是测试阶段的入口点！
├── train                    # 你的训练模型必须保存在此目录中
├── train_submission_code.py # 重要：你的训练代码。运行这应该产生与你作为智能体一部分上传的相同智能体。
├── test_framework.py        # 测试阶段的入口点，设置环境。你的代码不放在这里。
└── utility                  # 为你提供更流畅体验的实用程序脚本。
    ├── debug_build.sh
    ├── docker_run.sh
    ├── environ.sh
    ├── evaluation_locally.sh
    ├── parser.py
    ├── train_locally.sh
    └── verify_or_download_data.sh
```

最后，**你必须在`aicrowd.json`中指定AIcrowd提交JSON才能被评分！**

每个提交的`aicrowd.json`应包含以下内容：
```json
{
  "challenge_id": "neurips-2021-minerl-basalt-competition",
  "authors": ["your-aicrowd-username"],
  "description": "关于你的优秀智能体的示例描述",
  "tags": "FindCave",
  "license": "MIT",
  "gpu": false
}
```

此JSON用于将你的提交映射到所述挑战，所以请记住使用如上指定的正确`challenge_id`。

你需要使用`tags`字段指定提交的任务，使用以下之一：`{"FindCave", "MakeWaterfall", "CreateVillageAnimalPen", "BuildVillageHouse"}`。你需要为每个任务创建一个提交以覆盖所有任务。

请指定你的代码是否将使用GPU进行模型评估。如果你在GPU中指定`true`，将提供并使用**NVIDIA Tesla K80 GPU**进行评估。

### 数据集位置

你**不需要**在提交中上传MineRL数据集，它将在在线提交中在`MINERL_DATA_ROOT`路径提供，如果你需要的话。对于本地训练和评估，你可以通过`python ./utility/verify_or_download_data.py`在你的系统中下载一次，或手动放置到`./data/`文件夹中。

## 📤 如何提交！

要进行提交，你需要在[https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/)上创建一个私有仓库。

你需要按照[此处](https://docs.gitlab.com/ee/gitlab-basics/create-your-ssh-keys.html)的说明将SSH密钥添加到GitLab账户。
如果你没有SSH密钥，你首先需要[生成一个](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair)。

然后你可以通过在[https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/)上的仓库进行_tag push_来创建提交。
**任何标签推送（标签名称以"submission-"开头）到你的私有仓库都被视为提交**

然后你可以添加正确的git远程端点，最后通过以下方式提交：
```bash
cd competition_submission_starter_template
# 添加AIcrowd git远程端点
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USER_NAME>/basalt_competition_submission_template.git
git push aicrowd master

# 为你的提交创建标签并推送
git tag submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# 注意：如果你的仓库内容（最新提交哈希）没有改变，
# 那么推送新标签将**不会**触发新的评估。
```

你现在应该能够在以下位置看到你的提交详情：`https://gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/basalt_competition_submission_template/issues/`

**祝你好运** :tada: :tada:

## ✅ 确保你的代码有效

你可以使用本目录中共享的实用程序脚本执行本地训练和评估。要模拟在线训练阶段，你可以从仓库根目录运行`./utility/train_locally.sh`，你可以指定`--verbose`以查看完整日志。

对于代码的本地评估，你可以使用`./utility/evaluation_locally.sh`，如果你想查看完整日志，请添加`--verbose`。**注意**你不需要在你的代码中录制视频！AICrowd服务器将处理这个问题。你的代码只需要玩游戏。

要在docker环境（与在线提交相同）中运行/测试你的提交，你可以使用`./utility/docker_train_locally.sh`和`./utility/docker_evaluation_locally.sh`。你还可以使用`./utility/docker_run.sh`的帮助，使用bash入口点运行docker镜像进行调试。这些脚本尊重以下参数：

* `--no-build`: 跳过docker镜像构建并使用上次构建的镜像
* `--nvidia`: 使用`nvidia-docker`而不是`docker`，这会在docker镜像中包含你的nvidia相关驱动程序

# 👥 团队

快速入门工具包由[Anssi Kanervisto](https://github.com/Miffyli)和[Shivam Khandelwal](https://twitter.com/skbly7)编写，[William H. Guss](http://wguss.ml)提供帮助。

BASALT竞赛由以下团队组织：

* [Rohin Shah](https://rohinshah.com)（加州大学伯克利分校）
* Cody Wild（加州大学伯克利分校）
* Steven H. Wang（加州大学伯克利分校）
* Neel Alex（加州大学伯克利分校）
* Brandon Houghton（OpenAI和卡内基梅隆大学）
* [William H. Guss](http://wguss.ml)（OpenAI和卡内基梅隆大学）
* Sharada Mohanty（AIcrowd）
* Anssi Kanervisto（东芬兰大学）
* [Stephanie Milani](https://stephmilani.github.io/)（卡内基梅隆大学）
* Nicholay Topin（卡内基梅隆大学）
* Pieter Abbeel（加州大学伯克利分校）
* Stuart Russell（加州大学伯克利分校）
* Anca Dragan（加州大学伯克利分校）