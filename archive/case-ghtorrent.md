# GHTorrent — 开源数据精炼

> 验证日期：2026-07-23 | beta 验证通过
> 场景类型：大规模数据提取与聚合 | 输入：MySQL dump（数百GB） | 输出：面板 CSV

---

## 场景描述

开源社区研究团队需要从 GHTorrent 的 MySQL dump（数百 GB）中，为数万个 GitHub 用户提取每日活动面板数据。每个用户一天的活动聚合成一行，含 push、PR、Issue 评论等多维度指标。输出需对用户标识脱敏，并标注 bot 账户。

关键挑战：
- 数据量大（数百 GB raw → 数万用户 Target）
- 多表关联提取（users → events → aggregation）
- Bot 识别准确率（旧规则召回率仅 70%）
- 交付时间紧迫（两周）

---

## DRD 片段

```markdown
# GHTorrent 用户活动数据提取 — 数据需求文档

## 业务目标
从 GHTorrent 开源数据库的 MySQL dump 中，为一批指定的 GitHub 用户
提取每日活动面板数据，用于开源社区行为研究。

## 数据范围
- 来源：GHTorrent MySQL dump（公开数据）
- 规模：数百 GB 原始数据，数万目标用户
- 敏感度：输出需对用户标识做脱敏处理

## 输出期望
- 格式：CSV
- 结构：面板数据，每行 = 一个用户一天的全部活动聚合
- 关键指标：push 次数、PR 次数、Issue 评论次数等多维活动指标
- 控制变量：bot 识别标记

## 验收标准
- [ ] CSV 每日一行，列数固定
- [ ] 用户标识已脱敏
- [ ] Bot 识别结果可复现
- [ ] 去重验证通过（无重复行）
```

---

## Specification 片段

### Contract（数据契约）

```yaml
name: ghtorrent-user-activity
description: GHTorrent 用户活动面板数据的输入输出契约

input:
  mysql_dump:
    type: string
    doc: GHTorrent MySQL dump 文件路径
    constraint: required, must be a valid file path
  user_list:
    type: string
    doc: 目标用户 ID 列表文件路径
    constraint: required, CSV format with one ID per line

output:
  anonymized_id:
    type: string
    doc: 去标识化的用户标识符
    guarantee: unique per user, non-empty, deterministic hash
  activity_date:
    type: date
    doc: 活动观察日期
    guarantee: non-null, YYYY-MM-DD format
  push_count:
    type: integer
    doc: 当日 push 事件数
    guarantee: non-negative
  pr_count:
    type: integer
    doc: 当日 PR 事件数
    guarantee: non-negative
  is_bot:
    type: boolean
    doc: 是否为 bot 账户
    guarantee: determined by validated classifier, not heuristic rules
```

### Blueprint（处理蓝图）

```yaml
name: ghtorrent-user-activity
description: 从 GHTorrent MySQL dump 中提取、清洗、聚合用户每日活动数据
contract: ghtorrent-user-activity

steps:
  - name: 环境搭建与数据下载
    description: 配置计算环境，从指定源下载 MySQL dump 文件
    expectation: 可访问的原始数据文件

  - name: 数据解压与提取
    description: 解压 dump 文件，提取用户表、事件表等关键 CSV
    expectation: 结构化的原始 CSV 文件集

  - name: 用户匹配与数据筛选
    description: 用目标用户 ID 列表匹配用户表获取内部 ID，
      以内部 ID 为核心过滤各事件表
    expectation: 仅包含目标用户相关记录的子集

  - name: 多维度日聚合计数
    description: 按用户-日期分组，聚合各事件维度的日计数。
      Bot 分类器并行标注
    expectation: 每日每用户一行，含所有活动维度计数和 bot 标记

  - name: 面板合并
    description: 合并所有用户的面板数据为单一面板数据集
    expectation: 统一格式的面板 CSV

  - name: 数据验证与交付
    description: 验证列数、去重、bot 标记一致性。
      生成《数据处理说明》
    expectation: 通过验收的交付物（CSV + 处理说明文档）
```

---

## 关键经验

1. **DRD 的约束与假设很关键**：GHTorrent 的两周时限和 bot 准确率优先要求直接塑造了 Blueprint——筛选阶段就引入 bot 分类器而非后处理，避免末端返工。

2. **Contract 的 guarantee 字段推动质量对话**：写完 output guarantee 后客户才明确提出"bot 识别必须可复现"——之前只说"准确率重要"，guarantee 把它变成了可验证的判据。

3. **多版本迭代**：该项目经历了 V2（基础版，11-14步）和 V3（变量拆分，14-16步）。版本间 Blueprint 的 diff 直接揭示了变量口径的变更。

4. **脱敏是交付硬门槛**：输出字段从业务变量名（user_id, commit_count）全部替换为通用占位符（col_01, col_02），需经安全审计确认无残留。
