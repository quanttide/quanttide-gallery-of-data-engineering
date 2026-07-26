# Cosmetics Inspection Report — 结构化抽取

> 验证日期：2026-07-24 | beta 验证通过
> 场景类型：非结构化报告标准化抽取 | 输入：多机构 PDF 检测报告 | 输出：长表结构数据

---

## 场景描述

化妆品功效检测机构（SGS、华测、谱尼等）出具的检测报告格式各异，每份报告含多个样品、多个时间点、多个指标的检测值（均值、标准差、变化率、P值）。需要将几十份 PDF 报告统一抽取为结构化长表数据，且不同机构对同一指标的叫法不同（如"皮肤水分含量" vs "角质层水分含量" vs "皮肤含水量"），需要指标本体映射。

关键挑战：
- 多机构报告版式不统一
- 指标名称同义异构（需要本体映射）
- 数值精度不能丢失（30.37% 就是 30.37%）
- 每个抽取值需可追溯到原始报告页面位置

---

## DRD 片段

```markdown
# 化妆品检测报告标准化 — 数据需求文档

## 业务目标
对多家检测机构出具的化妆品功效检测报告 PDF 进行标准化处理，
提取检测指标数据并统一为结构化数据，支持跨机构对比分析。

## 数据范围
- 来源：SGS、华测、谱尼等多家检测机构出具的功效检测报告
- 格式：PDF（非结构化）
- 规模：数十份报告，每份含多个样品、多时间点、多指标

## 输出期望
- 格式：长表（CSV / Parquet）
- 结构：一行 = "某报告、某样品、某指标、某时间点、某组别"的一次观测
- 指标统一：不同机构同义指标名映射到标准指标名
- 可追溯：每个抽取值记录原始报告的页码和区域

## 验收标准
- [ ] 数值精度不丢失（如 30.37% 保持 30.37%）
- [ ] 指标本体映射覆盖所有已见机构
- [ ] 每个 observation 附带 source_page 追溯字段
- [ ] 长表格式通过 schema 验证
```

---

## Specification 片段

### Contract（数据契约）

```yaml
name: cosmetics-inspection-report
description: 化妆品功效检测报告的结构化抽取契约

input:
  report_pdf:
    type: string
    doc: 检测报告 PDF 文件路径
    constraint: required, must be a readable PDF file
  report_metadata:
    type: map
    doc: 报告级元数据
    constraint: must contain report_no, testing_agency, issue_date

output:
  report_id:
    type: string
    doc: 报告唯一标识
    guarantee: unique per report, non-empty
  sample_id:
    type: string
    doc: 样品标识
    guarantee: unique per sample within a report
  indicator_code:
    type: string
    doc: 标准指标码（来自指标本体）
    guarantee: must match an entry in indicator_ontology
  timepoint:
    type: string
    doc: 观测时间点（如 D0, D7, D14, D28）
    guarantee: non-empty
  group_name:
    type: string
    doc: 组别名称（如"测试组""对照组"）
  value_mean:
    type: number
    doc: 均值
    guarantee: precision preserved from source
  value_sd:
    type: number
    doc: 标准差
  change_rate_pct:
    type: number
    doc: 变化率（%）
  p_value:
    type: number
    doc: 统计显著性 P 值
  source_page:
    type: integer
    doc: 数据来源页码
    guarantee: must be traceable back to original report page
  extraction_confidence:
    type: number
    doc: 抽取置信度（0-1）
```

### Blueprint（处理蓝图）

```yaml
name: cosmetics-inspection-report
description: 从多家检测机构 PDF 报告中结构化抽取功效检测数据
contract: cosmetics-inspection-report

steps:
  - name: PDF 解析与文本提取
    description: 解析 PDF 结构，提取每页文本、表格和位置信息。
      保留字符级坐标用于后续追溯
    expectation: 每页的结构化文本块（含坐标）

  - name: 报告元数据识别
    description: 从封面和首页提取报告编号、检测机构、委托方、日期。
      不同机构封面版式不同，使用 position-based 规则 + LLM fallback
    expectation: 标准化元数据记录

  - name: 指标本体映射
    description: 扫描全文出现的指标名称，匹配到标准 indicator_code。
      已知映射表：皮肤水分含量/角质层水分含量/皮肤含水量 → skin_moisture_content。
      未匹配到的指标名称记录到待扩充列表
    expectation: 指标名称到标准码的映射列表 + 未匹配指标列表

  - name: 观测数据抽取
    description: 定位每个检测表格，按"样品-指标-时间点-组别"四维提取数值。
      保留 source_page 和 source_bbox 用于追溯。
      数值精度完整保留（不做四舍五入）
    expectation: 结构化 observation_fact 记录集

  - name: 质量校验
    description: 校验：数值精度是否与原文一致、时间点是否完整、
      指标映射覆盖率、必填字段是否为空。
      异常标记 extraction_confidence < 阈值的记录
    expectation: 通过校验的数据集 + 质量报告 + 异常记录列表
```

---

## 关键经验

1. **指标本体是核心资产**：不同机构对同一指标的叫法不同（"皮肤水分含量"/"角质层水分含量"/"皮肤含水量"）。Contract 的 `indicator_code` 字段 + 本体映射表是这类项目的核心竞争力。

2. **长表 > 宽表**：一行一个观测的设计天然支持多机构、多指标、多时间点的异构数据，避免了"先对齐列再合并"的痛苦。

3. **source_page 追溯是质量底线**：数值看起来不对时，业务方可以直接打开原报告对应页面复核。没有这个字段，30.37% 和 30.73% 的差异永远无法定位。

4. **初次覆盖率不要追求 100%**：指标本体映射初期覆盖 80% 已见指标即可，未匹配的计入待扩充列表。追求全覆盖会阻塞项目——新机构的新指标名是不可预见的。
