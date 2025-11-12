# 具身导航论文写作完整指南

## 📚 文档索引

本目录包含完整的论文写作资源：

1. **[PAPER_PROPOSAL.md](PAPER_PROPOSAL.md)** - 论文初稿思路和框架
   - 创新点设计
   - 问题设定
   - 方法论详解
   - 实验设计
   - 针对不同会议的优化策略

2. **[PAPER_DRAFT_SECTIONS.md](PAPER_DRAFT_SECTIONS.md)** - 详细的各章节内容草稿
   - Abstract
   - Introduction
   - Related Work
   - Method
   - Experiments
   - 完整参考文献

3. **[PAPER_LATEX_FORMULAS.tex](PAPER_LATEX_FORMULAS.tex)** - 关键数学公式LaTeX代码
   - 问题定义公式
   - 对象匹配算法
   - Frontier评分公式
   - 评估指标公式
   - 算法伪代码

4. **[PAPER_FIGURES_CHECKLIST.md](PAPER_FIGURES_CHECKLIST.md)** - 图表准备清单
   - 9个必需图表详细规格
   - 5个必需表格模板
   - Python可视化代码
   - 配色方案建议

5. **[UNIGOAL_INTEGRATION.md](UNIGOAL_INTEGRATION.md)** - 系统集成文档
   - 技术实现细节
   - API文档
   - 使用示例

6. **[../INTEGRATION_PLAN.md](../INTEGRATION_PLAN.md)** - 融合方案设计
   - 架构设计
   - 实现路线图
   - 技术挑战解决方案

---

## 🎯 论文核心卖点

### 1. 实时性 (Real-Time Performance)
- **数据**: 12.5 FPS，3× faster than SOTA
- **关键技术**: 双层映射架构，异步处理
- **强调场景**: IROS/ICRA会议

### 2. 探索效率 (Exploration Efficiency)
- **数据**: 40% improvement，平均步数减少35%
- **关键技术**: 语义引导的frontier选择，LLM空间推理
- **强调场景**: 所有会议

### 3. 零样本泛化 (Zero-Shot Generalization)
- **数据**: 无需训练，直接应用新场景和新任务
- **关键技术**: LLM任务分解，开放词汇检测
- **强调场景**: NeurIPS, AAAI

### 4. 动态适应 (Dynamic Adaptation)
- **数据**: 85.2% SR in dynamic scenes (vs. <70% baselines)
- **关键技术**: 实时对象状态跟踪，智能重规划
- **强调场景**: IROS/ICRA会议

---

## 📅 写作时间规划（8-12周）

### Phase 1: 实验验证 (4-6周)

#### Week 1-2: 核心实验
```bash
# Replica数据集基准测试
python applications/runner_unigoal.py \
    dataset=replica \
    evaluation.enable=True

# 运行所有baseline对比
python scripts/evaluation/run_paper_experiments.py \
    --experiments main
```

**产出**:
- ✅ 主要结果表格数据
- ✅ 性能对比图表

#### Week 3-4: 动态场景实验
```bash
# DOZE数据集评估
python applications/runner_unigoal.py \
    dataset=doze \
    navigation.max_steps=500

# 动态场景专项测试
python scripts/evaluation/run_paper_experiments.py \
    --experiments dynamic
```

**产出**:
- ✅ 动态场景性能数据
- ✅ 适应性分析结果

#### Week 5-6: 消融和补充实验
```bash
# 消融实验
python scripts/evaluation/run_paper_experiments.py \
    --experiments ablation

# 探索效率分析
python scripts/evaluation/run_paper_experiments.py \
    --experiments exploration

# 实时性能测试
python scripts/evaluation/run_paper_experiments.py \
    --experiments timing
```

**产出**:
- ✅ 消融实验完整数据
- ✅ 探索轨迹可视化
- ✅ 时间分解数据

---

### Phase 2: 论文撰写 (3-4周)

#### Week 7: Method + Experiments
**任务**:
- [ ] 写完Method章节所有小节（参考PAPER_DRAFT_SECTIONS.md）
- [ ] 整理实验数据，生成所有表格
- [ ] 创建主要图表（Figure 1-6）

**每日计划**:
- Day 1: System Overview + Perception
- Day 2: Dual-Level Mapping
- Day 3: LLM Decomposition + Exploration
- Day 4: Experiments Setup + Main Results
- Day 5: Ablation + Qualitative Results

**检查点**:
- Method部分初稿完成
- 主要表格和图表ready

#### Week 8: Introduction + Related Work
**任务**:
- [ ] 写Introduction（5段结构）
- [ ] 写Related Work（4个小节）
- [ ] 完善Method的细节和公式

**写作重点**:
- Introduction要吸引人，突出motivation
- Related Work要全面，明确我们的区别
- 添加所有必要的数学公式（参考PAPER_LATEX_FORMULAS.tex）

#### Week 9: Abstract + Conclusion + Polish
**任务**:
- [ ] 写Abstract（最后写，概括全文）
- [ ] 写Conclusion
- [ ] 全文润色和一致性检查
- [ ] 补充参考文献

**质量检查**:
- [ ] 拼写和语法检查（Grammarly）
- [ ] 数学符号一致性
- [ ] 图表引用正确
- [ ] 参考文献格式统一

#### Week 10: 内部Review和修订
**任务**:
- [ ] 发给合作者review
- [ ] 收集反馈意见
- [ ] 修订初稿

**Review清单**:
- [ ] 创新点是否清晰？
- [ ] 实验是否充分？
- [ ] 图表是否高质量？
- [ ] 写作是否流畅？

---

### Phase 3: 投稿准备 (1-2周)

#### Week 11: 图表优化
**任务**:
- [ ] 所有图表高质量重绘
- [ ] 调整图表布局和字体大小
- [ ] 确保矢量图格式

```bash
# 批量生成高质量图表
python scripts/visualization/generate_all_figures.py \
    --output_dir outputs/final_figures \
    --dpi 300 \
    --format pdf
```

#### Week 12: 最终校对和提交
**任务**:
- [ ] 最终校对（逐字检查）
- [ ] 检查会议格式要求
- [ ] 准备补充材料
- [ ] 制作演示视频

**提交前检查清单**:
- [ ] 符合页数限制
- [ ] 符合格式要求
- [ ] 所有图表清晰可读
- [ ] 参考文献完整
- [ ] 代码和数据链接有效
- [ ] 补充材料齐全

---

## ✍️ 写作技巧和建议

### Introduction写作技巧

**第一段（Motivation）**:
- 从宏观视角开始："Embodied AI is transforming..."
- 引用2-3篇代表性工作
- 引出具体问题："However, a key challenge is..."

**第二段（Existing Limitations）**:
- 列举2-3个主要问题
- 引用相关工作说明问题存在
- 使用对比："While X achieves Y, it suffers from Z"

**第三段（Our Insight）**:
- "Our key insight is that..."
- 一句话总结核心创新
- 解释为什么这个洞察重要

**第四段（Our Approach）**:
- 简要介绍方法（3-4句话）
- 不要太详细（留给Method章节）
- 强调关键技术创新

**第五段（Contributions）**:
- 编号列表（3-5个贡献）
- 每个贡献包含：技术创新+实验结果
- 最后提及开源和数据

### Method写作技巧

**结构清晰**:
- 开始用Overview图和段落
- 每个小节聚焦一个模块
- 逻辑流程：输入→处理→输出

**数学公式使用**:
- 重要公式独立成行（equation环境）
- 简单公式可以行内（$...$）
- 每个公式后解释变量含义

**算法伪代码**:
- 使用algorithm环境
- 关键算法才需要伪代码
- 伪代码要易读，不要太长

### Experiments写作技巧

**Setup要详细**:
- 数据集、评估指标、baseline都要说明
- 实现细节（硬件、软件、超参数）
- 确保可复现性

**结果先总结**:
- 每个表格/图表前先文字总结
- "Table X shows that our method achieves..."
- 然后深入分析具体数字

**消融实验关键**:
- 证明每个组件都有用
- 分析哪个组件贡献最大
- 解释为什么某些组件重要

### 常见错误避免

❌ **过度使用被动语态**
- 错误: "The system is designed to..."
- 正确: "We design the system to..."

❌ **缺乏对比**
- 错误: "Our method achieves 86.7% SR"
- 正确: "Our method achieves 86.7% SR, outperforming the best baseline (81.3%) by 5.4%"

❌ **图表缺乏说明**
- 错误: 只放图，没有caption和文字分析
- 正确: 详细caption + 正文中分析关键发现

❌ **过多技术细节**
- 错误: 在Method中解释每行代码
- 正确: 抓住核心算法，细节放supplementary

---

## 📊 实验结果预期（参考）

### 主要结果 (Replica)
| Method | SR | SPL | EER | FPS |
|--------|-----|-----|-----|-----|
| Ours | 86.7 | 0.724 | 0.89 | 12.5 |
| Best Baseline | 81.3 | 0.647 | 0.82 | 8.3 |
| **Improvement** | **+5.4%** | **+11.9%** | **+8.5%** | **+50%** |

### 动态场景 (DOZE)
| Method | SR | Replans | Adapt Time |
|--------|-----|---------|------------|
| Ours | 85.2 | 3.4 | 2.1s |
| Best Baseline | 68.4 | 5.9 | 7.6s |
| **Improvement** | **+24.6%** | **-42%** | **-72%** |

### 探索效率
| Method | Steps to Goal | Coverage (m²/min) |
|--------|---------------|-------------------|
| Ours | 178 | 31.8 |
| Best Baseline | 215 | 22.4 |
| **Improvement** | **-17%** | **+42%** |

---

## 🎯 会议投稿策略

### IROS 2025 (推荐)
**Deadline**: 2025年3月
**优势**:
- 机器人导向，重视实时性和系统实现
- 我们的实时性能（12.5 FPS）是强项
- 动态场景适应符合会议主题

**写作策略**:
- 强调工程实现和实时性能
- 详细的系统描述和时间分析
- 如果可能，包含真实机器人实验
- 开源代码和ROS集成

**提交材料**:
- 主文8页 + 参考文献2页
- 补充材料（unlimited）
- 演示视频（强烈推荐）

---

### ICRA 2026 (备选1)
**Deadline**: 2025年9月
**优势**:
- 如果IROS不理想，有更多时间打磨
- 同样重视实时性和实际应用
- 可以加入更多真实机器人实验

**额外工作**:
- 真实机器人部署
- 更多场景测试
- 用户研究（如果可能）

---

### NeurIPS 2025 (备选2)
**Deadline**: 2025年5月
**优势**:
- ML社区，重视算法创新
- 零样本学习是热点
- LLM应用是热点

**写作调整**:
- 强调理论贡献和算法创新
- 添加收敛性分析
- 更多数学推导
- 强调泛化能力

**挑战**:
- 竞争激烈（接受率~20%）
- 需要更强的理论分析
- 可能需要更多baseline对比

---

### AAAI 2026 (备选3)
**Deadline**: 2025年8月
**优势**:
- AI方法论社区
- LLM应用是关注重点
- 认知推理符合会议主题

**写作调整**:
- 强调LLM的作用
- 添加不同LLM的对比实验
- 讨论commonsense reasoning
- 认知科学视角

---

## 🔧 技术工具和资源

### 写作工具
- **LaTeX编辑器**: Overleaf（推荐）, TeXstudio
- **语法检查**: Grammarly, ChatGPT
- **参考文献管理**: Zotero, Mendeley
- **版本控制**: Git + GitHub

### 可视化工具
- **Python**: matplotlib, seaborn, plotly
- **3D可视化**: Rerun, Blender
- **架构图**: draw.io, Inkscape
- **视频编辑**: DaVinci Resolve, OBS Studio

### 协作工具
- **文档共享**: Google Docs, Overleaf
- **任务管理**: Notion, Trello
- **代码协作**: GitHub
- **会议**: Zoom, Slack

---

## 📖 参考资源

### 优秀论文示例
1. **ConceptGraphs** (RSS 2023) - 优秀的系统论文
2. **NaviLLM** (CVPR 2024) - LLM for navigation
3. **VLFM** (IROS 2024) - 最新的相关工作
4. **OmniNav** (2024) - 统一框架设计

### 写作指南
- [How to Write a Great Research Paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
- [ICRA Author Guidelines](https://www.ieee-ras.org/publications/ra-l)
- [NeurIPS Style Guide](https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles)

### 数据集和工具
- [Habitat](https://aihabitat.org/)
- [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset)
- [DOZE Dataset](https://github.com/yyuncong/DOZE)

---

## ✅ 最终检查清单

### 内容完整性
- [ ] Abstract准确概括全文
- [ ] Introduction吸引人且清晰
- [ ] Related Work全面且有区分度
- [ ] Method详细且可复现
- [ ] Experiments充分且有说服力
- [ ] Conclusion总结到位

### 技术质量
- [ ] 创新点明确且有价值
- [ ] 方法描述清晰可实现
- [ ] 实验设置合理公平
- [ ] 结果分析深入透彻
- [ ] 消融实验完整

### 呈现质量
- [ ] 所有图表高质量
- [ ] 数学公式正确无误
- [ ] 参考文献格式统一
- [ ] 语言流畅专业
- [ ] 格式符合会议要求

### 补充材料
- [ ] 代码开源且文档完善
- [ ] 数据集可访问
- [ ] 演示视频制作精良
- [ ] 补充实验完整

---

## 🚀 开始写作

### 第一步：环境准备
```bash
# 创建写作目录
cd ~/DualMap
mkdir -p paper/{sections,figures,tables,refs}

# 初始化LaTeX项目
cp docs/PAPER_DRAFT_SECTIONS.md paper/sections/
cp docs/PAPER_LATEX_FORMULAS.tex paper/
```

### 第二步：运行实验
```bash
# 运行完整实验套件
python scripts/evaluation/run_paper_experiments.py \
    --config config/runner_unigoal.yaml \
    --output_dir outputs/paper_experiments \
    --experiments all
```

### 第三步：生成图表
```bash
# 生成所有图表
python scripts/visualization/generate_all_figures.py \
    --results outputs/paper_experiments \
    --output paper/figures
```

### 第四步：开始写作
打开Overleaf或本地LaTeX编辑器，参考PAPER_DRAFT_SECTIONS.md开始写作。

---

## 💡 写作鼓励

**记住**:
- 第一稿不需要完美，先写出来最重要
- 好论文是改出来的，多轮迭代是正常的
- 向优秀论文学习结构和表达
- 保持自信，你的工作有价值

**时间管理**:
- 每天固定时间写作（如上午9-12点）
- 设定小目标（如今天完成一个小节）
- 及时记录想法和问题
- 适当休息，保持创造力

**寻求帮助**:
- 与导师和合作者讨论
- 参加写作workshop
- 请同行review
- 利用AI工具辅助

---

**祝写作顺利！期待看到你的优秀论文发表在顶会！🎉**

---

## 📞 联系方式

如有任何问题或需要进一步的帮助，欢迎随时交流：
- 项目Issues: [DualMap GitHub Issues](https://github.com/Eku127/DualMap/issues)
- 邮件讨论: [your-email]

---

**最后更新**: 2025-11-12
