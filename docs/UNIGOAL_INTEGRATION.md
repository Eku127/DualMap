# DualMap-UniGoal 融合使用指南

本文档介绍如何使用DualMap与UniGoal的融合系统进行具身导航任务。

## 目录

- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [安装依赖](#安装依赖)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [API参考](#api参考)
- [常见问题](#常见问题)

## 快速开始

### 1. 克隆UniGoal仓库

```bash
cd /path/to/your/workspace
git clone https://github.com/bagh2178/UniGoal.git
```

### 2. 安装UniGoal依赖

```bash
cd UniGoal
# 按照UniGoal的README安装依赖
pip install -r requirements.txt
```

### 3. 运行融合系统

```bash
cd /path/to/DualMap
conda activate dualmap

# 使用数据集模式
python applications/runner_unigoal.py

# 或指定自定义配置
python applications/runner_unigoal.py \
    navigation.goal_mode=llm_inquiry \
    unigoal.llm_backend=ollama
```

### 4. 运行示例

```bash
# 查看所有示例
python examples/unigoal_example.py
```

## 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                   DualMap-UniGoal系统                    │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼────────┐                 ┌───────▼────────┐
│  DualMap       │                 │ UniGoal        │
│  语义映射      │◄───────────────►│ 零样本导航     │
└────────────────┘                 └────────────────┘
        │                                   │
        │         DualMapUniGoalBridge      │
        │              (桥接器)              │
        └───────────────┬───────────────────┘
                        │
                ┌───────▼────────┐
                │  ROS/Habitat   │
                │  机器人接口     │
                └────────────────┘
```

### 核心组件

#### 1. **DualMapInterface**
- 从DualMap获取语义地图数据
- 提供对象查询接口
- 管理占用地图

#### 2. **UniGoalInterface**
- LLM目标分解
- 路径规划
- 导航决策

#### 3. **DualMapUniGoalBridge**
- 连接两个系统
- 数据格式转换
- 导航任务执行

## 安装依赖

### DualMap依赖

已包含在DualMap的`environment.yml`中:
- PyTorch
- Open3D
- YOLO-World
- SAM/FastSAM
- MobileCLIP

### UniGoal相关依赖

```bash
# 基础依赖
pip install habitat-sim==0.2.3
pip install lightglue
pip install scikit-image

# LLM支持 (可选)
# 选项1: Ollama (本地LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3

# 选项2: OpenAI
pip install openai
export OPENAI_API_KEY="your-api-key"
```

## 配置说明

### 配置文件结构

配置文件位于 `config/runner_unigoal.yaml`:

```yaml
# DualMap配置
dualmap:
  use_parallel: True
  active_window_size: 10
  run_detection: True

# UniGoal配置
unigoal:
  use_llm_decomposition: True
  llm_backend: "ollama"  # "ollama" | "openai" | "none"
  llm_model: "llama3"

# 桥接配置
bridge:
  coordinate_system: "habitat"
  map_resolution: 0.05
  update_frequency: 10

# 导航配置
navigation:
  goal_mode: "llm_inquiry"
  max_steps: 500
  success_threshold: 0.3
```

### LLM后端配置

#### Ollama (推荐用于本地部署)

```yaml
unigoal:
  use_llm_decomposition: True
  llm_backend: "ollama"
  llm_model: "llama3"
  ollama:
    host: "http://localhost:11434"
    timeout: 30
```

启动Ollama:
```bash
ollama serve
ollama pull llama3
```

#### OpenAI

```yaml
unigoal:
  use_llm_decomposition: True
  llm_backend: "openai"
  openai:
    api_key: "${oc.env:OPENAI_API_KEY}"
    model: "gpt-4"
    temperature: 0.7
```

设置API密钥:
```bash
export OPENAI_API_KEY="sk-..."
```

#### 无LLM (简单规则)

```yaml
unigoal:
  use_llm_decomposition: False
```

## 使用示例

### 示例1: 基础导航任务

```python
from dualmap import Dualmap
from utils.unigoal_bridge import DualMapUniGoalBridge
from omegaconf import OmegaConf

# 加载配置
cfg = OmegaConf.load("config/runner_unigoal.yaml")

# 初始化系统
dualmap = Dualmap(cfg)
bridge = DualMapUniGoalBridge(
    dualmap,
    OmegaConf.to_container(cfg.unigoal, resolve=True)
)

# 设置导航目标
goal = "找到厨房里的咖啡机"
bridge.set_navigation_goal(goal)

# 执行导航
current_pos = np.array([0.0, 0.0, 0.0])
while True:
    action = bridge.execute_navigation(current_pos)

    if action.get('done', False):
        print("导航完成!")
        break

    # 应用动作到机器人
    # robot.move(action['linear'], action['angular'])
```

### 示例2: 复杂目标分解

```python
# 设置复杂目标
goal = "先去厨房找到微波炉,然后去客厅找到遥控器,最后去卧室"

bridge.set_navigation_goal(goal)

# 查看目标分解
status = bridge.get_status()
print(f"总共{status['total_sub_goals']}个子目标")
print(f"当前子目标: {status['current_sub_goal']}")
```

### 示例3: 对象查询

```python
# 查询对象
results = bridge.dualmap_interface.query_object_by_text(
    "红色的杯子",
    top_k=3
)

for obj, similarity in results:
    print(f"对象: {obj.class_name}")
    print(f"位置: {obj.position}")
    print(f"相似度: {similarity:.3f}")
```

### 示例4: 数据集模式

```bash
# 使用Replica数据集
python applications/runner_unigoal.py \
    dataset.name=replica \
    dataset.scene=room_0 \
    navigation.goal_mode=inquiry \
    inquiry_sentence="找到椅子"
```

### 示例5: ROS模式 (未来支持)

```bash
# ROS2模式
python applications/runner_unigoal.py \
    ros.enable=True \
    ros.version=2 \
    navigation.goal_mode=llm_inquiry
```

## API参考

### DualMapInterface

```python
class DualMapInterface:
    def get_global_objects(self) -> List[SemanticObject]:
        """获取所有全局对象"""

    def get_occupancy_map(self) -> np.ndarray:
        """获取占用地图"""

    def query_object_by_text(self, text: str, top_k: int = 1) -> List[Tuple[SemanticObject, float]]:
        """通过文本查询对象"""

    def get_navigation_graph(self) -> nx.Graph:
        """获取导航图"""
```

### UniGoalInterface

```python
class UniGoalInterface:
    def set_scene_objects(self, objects: List[SemanticObject]):
        """设置场景对象"""

    def set_occupancy_map(self, occ_map: np.ndarray):
        """设置占用地图"""

    def decompose_goal(self, goal: str) -> List[SubGoal]:
        """分解目标"""

    def plan_path_to_object(self, target_obj: SemanticObject, start_pos: np.ndarray) -> np.ndarray:
        """规划到对象的路径"""
```

### DualMapUniGoalBridge

```python
class DualMapUniGoalBridge:
    def set_navigation_goal(self, goal: str) -> bool:
        """设置导航目标"""

    def execute_navigation(self, current_pos: np.ndarray, current_heading: float = 0.0) -> Dict[str, Any]:
        """执行导航步骤"""

    def get_status(self) -> Dict[str, Any]:
        """获取导航状态"""

    def update_map_data(self):
        """更新地图数据"""
```

## 工作流程

### 典型导航流程

```
1. 初始化系统
   ├─ 加载DualMap
   ├─ 初始化UniGoal接口
   └─ 创建Bridge

2. 构建语义地图
   ├─ 接收传感器数据
   ├─ DualMap处理(检测、映射)
   └─ 更新全局地图

3. 设置导航目标
   ├─ 接收自然语言指令
   ├─ LLM分解为子目标
   └─ 匹配地图中的对象

4. 路径规划
   ├─ 查询目标对象位置
   ├─ 获取占用地图
   ├─ 使用FMM/A*规划路径
   └─ 生成航点序列

5. 执行导航
   ├─ 跟随航点
   ├─ 实时更新地图
   ├─ 避障和重规划
   └─ 到达目标

6. 任务完成
   ├─ 验证目标达成
   ├─ 记录统计数据
   └─ 保存结果
```

## 性能优化

### 1. 并行处理

```yaml
dualmap:
  use_parallel: True  # 启用并行处理
```

### 2. 降低地图更新频率

```yaml
bridge:
  update_frequency: 5  # Hz (降低从10Hz到5Hz)
```

### 3. 使用FastSAM

```yaml
dualmap:
  use_fastsam: True  # 比SAM更快
```

### 4. 调整检测阈值

```yaml
system_config:
  yolo:
    conf_threshold: 0.3  # 提高阈值减少误检
```

## 评估指标

系统会自动记录以下指标:

- **Success Rate (SR)**: 导航成功率
- **Path Length**: 路径长度
- **SPL**: Success weighted by Path Length
- **Navigation Time**: 导航时间
- **Steps**: 总步数

查看结果:
```bash
# 结果保存在
outputs/unigoal_eval/metrics.json
```

## 常见问题

### Q1: LLM分解效果不好怎么办?

**A**: 尝试以下方法:
1. 使用更强大的模型 (如GPT-4)
2. 调整prompt设计
3. 增加few-shot示例
4. 使用更简单的目标描述

### Q2: 路径规划失败?

**A**: 检查:
1. 占用地图是否正确构建
2. 目标对象是否在地图中
3. 起点和终点是否在自由空间
4. 调整规划参数 (inflation_radius等)

### Q3: 对象检测不准确?

**A**: 优化方法:
1. 调整YOLO检测阈值
2. 更新class_list添加目标类别
3. 使用更高分辨率图像
4. 收集更多视角的观测

### Q4: 系统运行很慢?

**A**: 性能优化:
1. 启用并行处理
2. 使用FastSAM代替SAM
3. 降低地图分辨率
4. 减少活动窗口大小
5. 使用GPU加速

### Q5: 如何集成真实机器人?

**A**: 步骤:
1. 确保ROS安装正确
2. 配置相机和深度传感器topic
3. 实现cmd_vel发布
4. 测试传感器数据流
5. 逐步测试建图->导航

### Q6: 支持哪些坐标系?

**A**: 当前支持:
- **habitat**: Habitat模拟器坐标系
- **ros**: ROS标准坐标系 (REP-103)
- **opengl**: OpenGL坐标系

通过`bridge.coordinate_system`配置。

### Q7: 如何可视化导航过程?

**A**: 可视化选项:
1. **Rerun** (推荐): `dualmap.use_rerun: True`
2. **Rviz** (ROS): `dualmap.use_rviz: True`
3. 保存轨迹: `evaluation.save_trajectory: True`

## 进阶使用

### 自定义LLM提示词

在`utils/unigoal_bridge.py`中修改`_llm_decompose_goal`方法:

```python
def _llm_decompose_goal(self, goal: str) -> List[SubGoal]:
    prompt = f"""
    请将以下导航任务分解为步骤:
    任务: {goal}

    输出格式:
    1. 第一步
    2. 第二步
    ...
    """
    # 调用LLM API
    response = self.call_llm(prompt)
    # 解析响应
    sub_goals = self.parse_llm_response(response)
    return sub_goals
```

### 集成自定义规划器

继承`UniGoalInterface`并重写`plan_path_to_object`:

```python
class CustomUniGoalInterface(UniGoalInterface):
    def plan_path_to_object(self, target_obj, start_pos):
        # 使用你的自定义规划器
        path = my_custom_planner(start_pos, target_obj.position)
        return path
```

### 添加新的目标模式

在配置中添加:

```yaml
navigation:
  goal_mode: "custom"
  custom_goal_handler: "path.to.your.handler"
```

## 引用

如果使用本融合系统,请引用:

```bibtex
@article{dualmap2025,
  title={DualMap: Online Open-Vocabulary Semantic Mapping},
  author={...},
  journal={IEEE Robotics and Automation Letters},
  year={2025}
}

@article{unigoal,
  title={UniGoal: Universal Goal-Oriented Navigation},
  author={...},
  year={2024}
}
```

## 贡献

欢迎贡献! 请提交Issue或Pull Request。

## 许可证

遵循DualMap和UniGoal的原始许可证。

## 联系方式

- DualMap Issues: https://github.com/Eku127/DualMap/issues
- UniGoal Issues: https://github.com/bagh2178/UniGoal/issues

---

**最后更新**: 2025-11-12
