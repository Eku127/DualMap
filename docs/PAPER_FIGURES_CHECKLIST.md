# 论文图表准备清单

## 📊 必需图表 (Essential Figures)

### Figure 1: 系统架构图 ⭐⭐⭐
**类型**: 系统架构图
**优先级**: 最高（通常是论文第一张图）
**位置**: Introduction或Method开始
**内容**:
- [ ] 完整系统Pipeline流程
- [ ] 四个主要模块：Perception, Dual-Map, LLM, Explorer
- [ ] 数据流向箭头
- [ ] 输入输出标注
- [ ] 关键组件高亮

**设计要求**:
- 清晰的模块划分（用不同颜色区分）
- 简洁的图标和符号
- 专业的配色方案（建议：蓝色系主色调）
- 矢量图格式（SVG/PDF）

**工具推荐**: draw.io, Inkscape, Adobe Illustrator

**参考示例**:
```
输入: RGB-D + Pose
     ↓
┌─────────────────┐
│  Perception     │ → YOLO-World, SAM, CLIP
└────────┬────────┘
         ↓
┌─────────────────┐
│  Dual-Level Map │
│  ┌──────────┐   │
│  │Local Map │   │ ← Real-time tracking
│  └──────────┘   │
│  ┌──────────┐   │
│  │Global Map│   │ ← Semantic aggregation
│  └──────────┘   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ LLM Decomposer  │ → SubGoals
└────────┬────────┘
         ↓
┌─────────────────┐
│ Semantic Exp.   │ → Frontier Selection
└────────┬────────┘
         ↓
      Actions
```

---

### Figure 2: 双层地图可视化对比 ⭐⭐⭐
**类型**: 概念图 + 实际可视化
**优先级**: 最高
**位置**: Method - Dual-Level Mapping章节
**内容**:
- [ ] 左侧：局部地图（活动窗口内的对象）
- [ ] 右侧：全局地图（持久稳定对象）
- [ ] 中间：当前RGB-D观测
- [ ] 动态对象标注（红色边框）
- [ ] 稳定对象标注（绿色边框）
- [ ] 时间轴显示

**可视化要素**:
- 3D点云渲染（俯视图 + 侧视图）
- 边界框绘制
- 对象语义标签
- 颜色编码：新鲜度/稳定性

**生成方式**:
```python
# 使用Rerun记录数据
# 导出高质量截图
# 使用Blender后处理（可选）
```

---

### Figure 3: 语义引导的Frontier选择 ⭐⭐⭐
**类型**: 热力图 + 轨迹对比
**优先级**: 最高
**位置**: Method - Semantic Exploration章节
**内容**:
- [ ] (a) 场景俯视图
- [ ] (b) Frontier热力图（评分可视化）
- [ ] (c) 对比：Random vs. Ours的轨迹
- [ ] (d) 目标对象位置标注

**可视化技术**:
- 热力图：matplotlib imshow + colorbar
- 轨迹：不同颜色的路径曲线
- 标注：箭头指示探索方向

**Python代码**:
```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) 场景图
axes[0,0].imshow(occupancy_map, cmap='gray')
axes[0,0].set_title('Scene Layout')

# (b) Frontier评分热力图
heat = axes[0,1].imshow(frontier_scores, cmap='hot')
plt.colorbar(heat, ax=axes[0,1])
axes[0,1].set_title('Semantic Frontier Scores')

# (c) 轨迹对比 - Random
axes[1,0].imshow(occupancy_map, cmap='gray', alpha=0.5)
axes[1,0].plot(traj_random[:, 0], traj_random[:, 1], 'b-', label='Random')
axes[1,0].set_title('Baseline: Random Exploration')

# (d) 轨迹对比 - Ours
axes[1,1].imshow(occupancy_map, cmap='gray', alpha=0.5)
axes[1,1].plot(traj_ours[:, 0], traj_ours[:, 1], 'r-', label='Ours')
axes[1,1].scatter(goal_x, goal_y, c='green', s=200, marker='*')
axes[1,1].set_title('Ours: Semantic-Guided')

plt.tight_layout()
plt.savefig('frontier_selection.pdf', dpi=300)
```

---

### Figure 4: 动态场景适应 ⭐⭐
**类型**: 时间序列
**优先级**: 高
**位置**: Method - Dynamic Adaptation或Experiments
**内容**:
- [ ] t=0: 初始观测（静态场景）
- [ ] t=5: 检测到动态对象（椅子移动）
- [ ] t=10: 地图更新，路径重规划
- [ ] t=15: 成功避开动态障碍

**设计要求**:
- 时间序列展示（4-6帧）
- 动态对象用红色高亮
- 路径变化用不同颜色标注
- 添加文字说明

---

### Figure 5: 探索轨迹对比 ⭐⭐
**类型**: 轨迹可视化
**优先级**: 高
**位置**: Experiments - Exploration Efficiency
**内容**:
- [ ] 多个方法的轨迹并排对比
- [ ] Random, FBE, ESC, VLFM, Ours
- [ ] 标注步数、时间
- [ ] 目标位置用星号标记

**布局**:
```
┌────────┬────────┬────────┐
│ Random │  FBE   │  ESC   │
├────────┼────────┼────────┤
│  VLFM  │  Ours  │ Legend │
└────────┴────────┴────────┘
```

---

### Figure 6: 主要结果对比 ⭐⭐
**类型**: 柱状图 / 雷达图
**优先级**: 高
**位置**: Experiments - Main Results
**内容**:
- [ ] (a) Success Rate对比
- [ ] (b) SPL对比
- [ ] (c) Exploration Efficiency对比
- [ ] (d) FPS对比

**Python代码**:
```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['Random', 'FBE', 'ESC', 'VLFM', 'OmniNav', 'Ours']
sr = [45.2, 58.7, 72.4, 78.9, 81.3, 86.7]
spl = [28.3, 41.2, 53.6, 61.2, 64.7, 72.4]
eer = [0.42, 0.58, 0.71, 0.79, 0.82, 0.89]
fps = [15.3, 14.8, 3.2, 5.1, 8.3, 12.5]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# SR
axes[0,0].bar(methods, sr, color=['gray']*5 + ['red'])
axes[0,0].set_ylabel('Success Rate (%)')
axes[0,0].set_title('(a) Success Rate')

# SPL
axes[0,1].bar(methods, spl, color=['gray']*5 + ['red'])
axes[0,1].set_ylabel('SPL')
axes[0,1].set_title('(b) SPL')

# EER
axes[1,0].bar(methods, eer, color=['gray']*5 + ['red'])
axes[1,0].set_ylabel('Exploration Efficiency Ratio')
axes[1,0].set_title('(c) Exploration Efficiency')

# FPS
axes[1,1].bar(methods, fps, color=['gray']*5 + ['red'])
axes[1,1].set_ylabel('FPS')
axes[1,1].set_title('(d) Real-Time Performance')

plt.tight_layout()
plt.savefig('main_results.pdf', dpi=300)
```

---

### Figure 7: 消融实验结果 ⭐
**类型**: 热力图 / 折线图
**优先级**: 中
**位置**: Experiments - Ablation Study
**内容**:
- [ ] 各个组件对性能的影响
- [ ] 热力图显示不同配置的性能
- [ ] 折线图显示累积效果

---

### Figure 8: 定性结果展示 ⭐
**类型**: 案例研究
**优先级**: 中
**位置**: Experiments - Qualitative Results
**内容**:
- [ ] 成功案例3个
- [ ] 失败案例1个（诚实展示）
- [ ] 每个案例包含：RGB视图、语义地图、轨迹

---

### Figure 9: 实时性能分解 ⭐
**类型**: 堆叠柱状图
**优先级**: 中
**位置**: Experiments - Timing Analysis
**内容**:
- [ ] 每帧时间分解
- [ ] 各个组件耗时占比
- [ ] 与baseline对比

**代码**:
```python
components = ['Detection', 'Segmentation', 'Features', 'Local Map', 'Planning']
ours_time = [32, 8, 5, 12, 18]
baseline_time = [45, 15, 10, 30, 20]

x = np.arange(len(components))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, ours_time, width, label='Ours')
ax.bar(x + width/2, baseline_time, width, label='Baseline')

ax.set_ylabel('Time (ms)')
ax.set_title('Per-Frame Timing Breakdown')
ax.set_xticks(x)
ax.set_xticklabels(components, rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig('timing_breakdown.pdf')
```

---

## 📈 必需表格 (Essential Tables)

### Table 1: 主要结果对比 (Replica Dataset) ⭐⭐⭐
**内容**:
| Method | SR ↑ | SPL ↑ | EER ↑ | FPS ↑ |
|--------|------|-------|-------|-------|
| Random | 45.2 | 28.3 | 0.42 | 15.3 |
| ...    | ...  | ...   | ...   | ...   |
| Ours   | **86.7** | **72.4** | **0.89** | **12.5** |

**LaTeX模板**:
```latex
\begin{table}[t]
\centering
\caption{Performance comparison on Replica dataset}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
Method & SR $\uparrow$ & SPL $\uparrow$ & EER $\uparrow$ & FPS $\uparrow$ \\
\midrule
Random & 45.2 & 0.283 & 0.42 & 15.3 \\
FBE & 58.7 & 0.412 & 0.58 & 14.8 \\
ESC \cite{esc2023} & 72.4 & 0.536 & 0.71 & 3.2 \\
VLFM \cite{vlfm2024} & 78.9 & 0.612 & 0.79 & 5.1 \\
OmniNav \cite{omninav2024} & 81.3 & 0.647 & 0.82 & 8.3 \\
\textbf{Ours} & \textbf{86.7} & \textbf{0.724} & \textbf{0.89} & \textbf{12.5} \\
\bottomrule
\end{tabular}
\end{table}
```

---

### Table 2: 动态场景性能 (DOZE Dataset) ⭐⭐⭐
**内容**:
| Method | SR ↑ | Replans ↓ | Adapt Time (s) ↓ |
|--------|------|-----------|------------------|
| ESC | 52.3 | 8.7 | 12.4 |
| ...    | ...  | ...       | ...              |
| Ours | **85.2** | **3.4** | **2.1** |

---

### Table 3: 消融实验 ⭐⭐
**内容**:
| Variant | SR | SPL | FPS | EER |
|---------|----|----|-----|-----|
| Full Model | 86.7 | 0.724 | 12.5 | 0.89 |
| w/o Dual-Level | 78.2 | 0.641 | 15.2 | 0.85 |
| ...     | ... | ... | ... | ... |

---

### Table 4: 探索效率对比 ⭐⭐
**内容**:
| Method | Coverage (m²/min) ↑ | Steps to Goal ↓ | Efficiency ↑ |
|--------|---------------------|-----------------|-------------|
| Random | 8.2 | 445 | 0.42 |
| ...    | ... | ... | ... |
| Ours | **31.8** | **178** | **0.89** |

---

### Table 5: 时间复杂度分析 ⭐
**内容**:
| Component | Time Complexity | Actual Time (ms) |
|-----------|-----------------|------------------|
| Detection | O(HW) | 32 |
| Local Map Update | O(k log n) | 12 |
| ...       | ...   | ... |

---

## 🎬 补充材料 (Supplementary Material)

### Video 1: 系统演示视频 (3-5分钟)
**内容**:
- [ ] 0:00-0:30: 系统介绍
- [ ] 0:30-1:30: 双层地图构建过程
- [ ] 1:30-2:30: 语义引导探索
- [ ] 2:30-3:30: 动态场景适应
- [ ] 3:30-4:00: 结果总结

**制作工具**: Rerun录屏 + OBS Studio + DaVinci Resolve

---

### Appendix A: 更多实验结果
- [ ] 所有18个Replica场景的详细结果
- [ ] 不同LLM的对比（GPT-4, Llama, etc.）
- [ ] 不同目标类别的成功率分布

---

### Appendix B: 失败案例分析
- [ ] 典型失败模式分类
- [ ] 失败原因分析
- [ ] 改进方向讨论

---

### Appendix C: 实现细节
- [ ] 超参数设置完整列表
- [ ] 硬件配置
- [ ] 训练细节（如果有）

---

## ✅ 图表质量检查清单

### 通用要求
- [ ] 所有图表为矢量格式（PDF/SVG）或高分辨率（≥300 DPI）
- [ ] 字体大小适中（图中文字 ≥8pt）
- [ ] 颜色对色盲友好（使用ColorBrewer配色）
- [ ] 所有坐标轴标注清晰
- [ ] 图例位置合理不遮挡内容
- [ ] 子图标号清晰（(a), (b), (c)...）

### 会议特定要求
**IROS/ICRA**:
- [ ] 图表宽度适配双栏格式（3.5英寸或7英寸）
- [ ] 避免过小的字体（建议 ≥10pt）

**NeurIPS**:
- [ ] 严格遵循NeurIPS样式指南
- [ ] 避免彩色打印问题（关键信息不仅依赖颜色）

---

## 📐 图表生成脚本

### 创建输出目录
```bash
mkdir -p outputs/paper_figures
mkdir -p outputs/paper_tables
mkdir -p outputs/supplementary
```

### 自动生成所有图表
```bash
python scripts/visualization/generate_all_figures.py \
    --results_dir outputs/paper_experiments \
    --output_dir outputs/paper_figures
```

### 单独生成特定图表
```bash
# Figure 1: 系统架构
python scripts/visualization/draw_architecture.py

# Figure 2: 双层地图
python scripts/visualization/visualize_dual_maps.py \
    --scene replica_room_0

# Figure 3: Frontier选择
python scripts/visualization/plot_frontier_selection.py \
    --episode_id 42

# Figure 6: 主要结果
python scripts/visualization/plot_main_results.py \
    --results outputs/paper_experiments/main_results.json
```

---

## 🎨 配色方案建议

### 方法对比配色
```python
colors = {
    'Random': '#CCCCCC',      # 灰色
    'FBE': '#A0A0A0',         # 深灰
    'ESC': '#4A90E2',         # 浅蓝
    'VLFM': '#7B68EE',        # 紫色
    'OmniNav': '#50C878',     # 绿色
    'Ours': '#E74C3C'         # 红色（突出）
}
```

### 状态配色
```python
status_colors = {
    'static': '#2ECC71',      # 绿色
    'dynamic': '#E74C3C',     # 红色
    'unknown': '#95A5A6',     # 灰色
    'frontier': '#F39C12'     # 橙色
}
```

---

## 📊 数据可视化最佳实践

1. **简洁性**: 每张图只传达一个核心信息
2. **对比性**: 基线方法用灰色，我们的方法用高亮色
3. **可读性**: 字体大小适中，线条粗细合适
4. **一致性**: 同一概念在不同图中使用相同颜色/符号
5. **完整性**: 包含误差线、置信区间（如适用）

---

## 🔄 迭代流程

1. **初稿**: 使用脚本快速生成原始图表
2. **反馈**: 与合作者讨论，收集意见
3. **优化**: 调整布局、配色、标注
4. **审查**: 模拟审稿人视角检查
5. **定稿**: 导出高质量版本

---

## 📝 Caption写作建议

### Figure Caption结构
```
**Figure X: [简短标题].**
[详细描述] (a) [子图1描述]. (b) [子图2描述].
[关键观察或结论]. [对比说明].
```

### 示例
```
**Figure 2: Dual-Level Semantic Mapping.**
Visualization of our dual-level map architecture.
(a) Local map maintains objects within the active window for real-time tracking.
(b) Global map aggregates stable observations for persistent semantic representation.
(c) Current RGB-D observation with detected objects.
Dynamic objects (red) are tracked in the local map,
while stable objects (green) are promoted to the global map.
```

---

**准备时间估计**: 2-3周完成所有高质量图表
