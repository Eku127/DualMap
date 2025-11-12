# DualMap-UniGoal 融合方案

## 概述

本文档描述了如何将DualMap的语义映射系统与UniGoal的零样本具身导航系统进行融合。

## 架构设计

### 系统层次

```
┌────────────────────────────────────────────────────────┐
│                应用层 (Application Layer)               │
│  - 语义导航任务                                         │
│  - 多模态目标查询                                       │
│  - 动态场景探索                                         │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│              融合层 (Integration Layer)                 │
│  - DualMapUniGoalBridge                                │
│  - SharedSemanticMap                                   │
│  - GoalDecompositionModule                             │
└────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────┬─────────────────────────────────┐
│    DualMap Core      │      UniGoal Agent              │
│  - 语义映射          │  - 目标分解                      │
│  - 对象检测          │  - 路径规划                      │
│  - 局部/全局地图     │  - 决策推理                      │
└──────────────────────┴─────────────────────────────────┘
```

## 核心组件

### 1. DualMapUniGoalBridge (桥接器)

**职责**: 连接DualMap和UniGoal,负责数据转换和通信

**主要功能**:
- 将DualMap的对象地图转换为UniGoal的场景图
- 将占用地图传递给UniGoal的路径规划器
- 接收UniGoal的导航指令并转换为机器人控制命令

**数据流**:
```
DualMap GlobalMap → Bridge → UniGoal SceneGraph
DualMap LayoutMap → Bridge → UniGoal OccupancyMap
UniGoal Actions   → Bridge → Robot Commands
```

### 2. SharedSemanticMap (共享语义地图)

**职责**: 提供统一的语义地图表示,供两个系统共享

**数据结构**:
```python
{
    "occupancy": np.ndarray,           # 占用地图 (HxW)
    "objects": List[SemanticObject],   # 语义对象列表
    "graph": nx.Graph,                 # 对象关系图
    "features": Dict[str, np.ndarray], # CLIP特征映射
}
```

### 3. GoalDecompositionModule (目标分解模块)

**职责**: 使用UniGoal的LLM能力分解复杂导航目标

**工作流程**:
1. 接收自然语言导航指令
2. 使用LLM分解为子目标序列
3. 在DualMap语义地图中定位子目标
4. 生成导航路径

## 集成接口设计

### DualMap侧接口

```python
class DualMapInterface:
    """DualMap系统接口"""

    def get_global_objects(self) -> List[GlobalObject]:
        """获取全局对象列表"""
        pass

    def get_occupancy_map(self) -> np.ndarray:
        """获取占用地图"""
        pass

    def get_object_features(self, obj_id: str) -> np.ndarray:
        """获取对象的CLIP特征"""
        pass

    def query_object_by_text(self, text: str) -> Optional[GlobalObject]:
        """通过文本查询对象"""
        pass

    def get_navigation_graph(self) -> nx.Graph:
        """获取导航图"""
        pass
```

### UniGoal侧接口

```python
class UniGoalInterface:
    """UniGoal系统接口"""

    def set_scene_graph(self, objects: List, graph: nx.Graph):
        """设置场景图"""
        pass

    def set_occupancy_map(self, occ_map: np.ndarray):
        """设置占用地图"""
        pass

    def decompose_goal(self, goal: str) -> List[SubGoal]:
        """分解目标为子目标序列"""
        pass

    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """规划路径"""
        pass

    def get_next_action(self, obs: Dict) -> Dict:
        """获取下一步动作"""
        pass
```

## 融合策略

### 策略1: DualMap主导 + UniGoal增强

**适用场景**: 已有DualMap部署,希望增强导航能力

**实现方式**:
1. DualMap负责建图和语义理解
2. UniGoal作为决策模块,接收DualMap的地图
3. UniGoal的路径规划替换DualMap原有规划器

**优势**:
- 保持DualMap的实时映射能力
- 利用UniGoal的零样本导航决策
- 最小化对现有系统的改动

### 策略2: 深度融合

**适用场景**: 从零开始构建新系统

**实现方式**:
1. 重构数据流,创建统一的状态表示
2. DualMap的检测结果直接输入UniGoal的场景图
3. 共享占用地图和语义特征
4. 统一的LLM/VLM推理引擎

**优势**:
- 更高的效率和一致性
- 避免重复计算
- 更好的端到端性能

## 实现路线图

### 阶段1: 基础集成 (1-2周)

**任务**:
- [ ] 创建桥接器基础框架
- [ ] 实现数据格式转换
- [ ] 基本的地图共享功能
- [ ] 简单的目标导航测试

**产出**:
- `applications/runner_unigoal.py` - UniGoal集成运行器
- `utils/unigoal_bridge.py` - 桥接器实现
- `config/runner_unigoal.yaml` - 配置文件

### 阶段2: 功能增强 (2-3周)

**任务**:
- [ ] 集成UniGoal的LLM目标分解
- [ ] 优化路径规划集成
- [ ] 添加动态场景支持
- [ ] 实现多模态目标查询

**产出**:
- 完整的LLM集成
- 增强的导航能力
- 测试用例和评估脚本

### 阶段3: 优化与评估 (1-2周)

**任务**:
- [ ] 性能优化
- [ ] 导航成功率评估
- [ ] 文档和示例
- [ ] 论文实验支持

**产出**:
- 性能报告
- 用户文档
- 示例代码

## 技术挑战与解决方案

### 挑战1: 坐标系统不一致

**问题**: DualMap和UniGoal可能使用不同的坐标系统

**解决方案**:
- 创建统一的坐标转换模块
- 在桥接器中处理所有坐标变换
- 使用配置文件管理坐标系参数

### 挑战2: 实时性要求

**问题**: UniGoal的LLM推理可能较慢

**解决方案**:
- 使用异步处理
- 缓存LLM结果
- 在简单场景下跳过LLM推理

### 挑战3: 语义一致性

**问题**: 两个系统使用不同的语义表示

**解决方案**:
- 统一使用CLIP特征
- 建立共享的对象类别词汇表
- 实现特征对齐模块

## 评估指标

### 映射质量
- 对象检测准确率
- 地图覆盖率
- 语义一致性

### 导航性能
- 导航成功率 (SR)
- 路径效率 (SPL)
- 目标找到时间

### 系统性能
- 实时性 (FPS)
- 内存占用
- 计算开销

## 依赖项

### 新增依赖
```yaml
# UniGoal相关
habitat-sim: 0.2.3
lightglue: latest
ollama: latest  # 可选,用于本地LLM

# 额外工具
networkx: >=2.8
scikit-image: latest
```

### 模型依赖
- GroundingDINO (UniGoal)
- YOLO-World (DualMap)
- SAM/FastSAM (共享)
- CLIP/MobileCLIP (共享)
- LLM (GPT-4/Ollama)

## 配置示例

```yaml
# config/runner_unigoal.yaml

# DualMap配置
dualmap:
  use_parallel: True
  active_window_size: 10
  run_detection: True

# UniGoal配置
unigoal:
  use_llm_decomposition: True
  llm_backend: "ollama"  # or "openai"
  llm_model: "llama3"
  use_graph_representation: True
  planning_method: "fmm"  # Fast Marching Method

# 桥接配置
bridge:
  coordinate_system: "habitat"  # or "ros"
  map_resolution: 0.05  # 5cm per pixel
  update_frequency: 10  # Hz

# 导航配置
navigation:
  goal_mode: "llm_inquiry"  # "inquiry", "llm_inquiry", "image_goal"
  max_steps: 500
  success_threshold: 0.3  # meters
```

## 使用示例

### 示例1: 文本导航

```python
from dualmap import Dualmap
from utils.unigoal_bridge import DualMapUniGoalBridge

# 初始化系统
dualmap = Dualmap(cfg)
bridge = DualMapUniGoalBridge(dualmap, unigoal_cfg)

# 设置导航目标
goal_text = "找到厨房里的咖啡机"

# 目标分解
sub_goals = bridge.decompose_goal(goal_text)
# 输出: ["导航到厨房", "在厨房中找到咖啡机"]

# 执行导航
for sub_goal in sub_goals:
    path = bridge.plan_to_goal(sub_goal)
    bridge.execute_path(path)
```

### 示例2: 图像目标导航

```python
# 使用参考图像作为目标
reference_image = load_image("target_object.jpg")

# UniGoal匹配目标
target_obj = bridge.match_image_goal(reference_image)

# 导航到目标
path = bridge.plan_to_object(target_obj)
bridge.execute_path(path)
```

## 参考资料

- [DualMap论文](https://arxiv.org/abs/2506.01950)
- [UniGoal代码](https://github.com/bagh2178/UniGoal)
- [Habitat文档](https://aihabitat.org/docs/habitat-sim/)
- [ROS Navigation](http://wiki.ros.org/navigation)

## 贡献者

本融合方案设计和实现。

## 许可证

遵循DualMap和UniGoal的原始许可证。
