# SEC Credit Agreement — 文档分类

> 验证日期：2026-07-24 | beta 验证通过
> 场景类型：法律文档智能分类 | 输入：SEC EDGAR 8-K filings | 输出：Exhibit 级分类标记

---

## 场景描述

金融研究团队需要从 SEC EDGAR 数据库批量下载的 8-K filings 中，筛选出 Credit Agreement 类型的附件。问题在于所有匹配 `loan`/`credit` 关键词的附件都被处理，大量 Indenture 和 Notes 文件被误召回。真正的需求是在 Exhibit 级别做多类别精准分类。

关键挑战：
- 误召回严重（关键词匹配无区分度）
- 目标子类型多（Original/Amendment/Extension/Related Letter）
- Indenture 与 Credit Agreement 结构相似（都有 Article I/II 章节）
- 分类准确率比召回率更重要

---

## DRD 片段

```markdown
# SEC Credit Agreement 识别 — 数据需求文档

## 业务目标
从 SEC EDGAR 8-K filings 的附件中，精准识别 Credit Agreement 类型文档
（包括 Original、Amendment、Extension、Related Letter），排除 Indenture
和 Notes 等非目标文档。

## 数据范围
- 来源：SEC EDGAR 数据库（公开数据）
- 规模：数千份 8-K filings，每个含多个 Exhibit 附件
- 分类级别：Exhibit（非 Filing）

## 输出期望
- 每个 Exhibit 输出：document_type、is_target、confidence、证据列表
- 分类准确率优先于召回率
- 支持人工复核（冲突样本）

## 验收标准
- [ ] 8-K 主文不进入合同抽取
- [ ] Indenture/Notes 被排除（负例准确率 > 95%）
- [ ] Amendment/Extension 类不因缺少 Article I/II 被误删
- [ ] 每个分类结果附带证据
```

---

## Specification 片段

### Contract（数据契约）

```yaml
name: sec-credit-agreement-classification
description: SEC 8-K Exhibit 级别的 Credit Agreement 分类契约

input:
  filing_accession_number:
    type: string
    doc: SEC filing accession number
    constraint: required, unique per filing
  exhibit_description:
    type: string
    doc: Exhibit 在 filing 中的描述文本
  text_head:
    type: string
    doc: 附件前若干 KB 文本，用于分类判断
    constraint: required, enough to capture header and opening sections

output:
  document_type:
    type: enum
    doc: 文档分类结果
    values:
      - credit_agreement_original
      - credit_agreement_amendment
      - credit_agreement_extension
      - credit_agreement_related_letter
      - indenture_or_notes
      - eight_k_summary
      - other
  is_target:
    type: boolean
    doc: 是否为 Credit Agreement 类目标文档
  confidence:
    type: number
    doc: 分类置信度（0-1）
    guarantee: must be reported alongside evidence
  positive_evidence:
    type: string
    doc: 支持分类的证据列表
  negative_evidence:
    type: string
    doc: 反对分类的证据列表
```

### Blueprint（处理蓝图）

```yaml
name: sec-credit-agreement-classification
description: 对 SEC 8-K Exhibit 做 Credit Agreement 多类别分类
contract: sec-credit-agreement-classification

steps:
  - name: Exhibit 解析
    description: 对每个 8-K filing，解析 Exhibit Index 或 Exhibit links，
      得到每个附件的基础信息（form_type, exhibit_description, file_name, text_head）
    expectation: 每个 Exhibit 的元数据和文本头部

  - name: 证据抽取
    description: 从 text_head 中抽取分类证据。
      正证据：标题关键词（Credit Agreement/Loan Agreement/Amendment）、
      合同角色（Borrower/Lender/Administrative Agent）、
      章节主题（The Loans/Commitments）。
      负证据：Indenture Trustee、Issuer、The Notes、Trust Indenture Act。
      不对全文做判断，仅抽取可解释的证据片段
    expectation: 结构化的正/负证据列表

  - name: LLM 分类
    description: LLM 基于已抽取的证据判断 document_type 和 is_target。
      输出结构化结果含 confidence 和 evidence。
      不做自由文本判断，仅基于上一步抽取的证据
    expectation: 带置信度和证据的初步分类结果

  - name: 规则验收
    description: 用规则引擎校验 LLM 分类结果：
      - 标题含 INDENTURE + 正文出现 Indenture Trustee/The Notes → 降级为 indenture_or_notes
      - 标题含 Extension Agreement + 正文引用 Credit Agreement → credit_agreement_extension
      - 8-K 主文摘要 → 标记为 eight_k_summary，不进入后续抽取
      规则与 LLM 冲突的样本记录至冲突日志
    expectation: 经规则校正的最终分类结果 + 冲突日志

  - name: 人工复核路由
    description: 仅以下情况路由人工：
      - LLM 判为正 + 规则检测强负特征
      - LLM 判为负 + 标题明确含 Credit Agreement
      - 无法归入现有 document_type 的新文档类型
    expectation: 人工复核样本集（预期 < 5% 总量）
```

---

## 关键经验

1. **特征工程决定分类质量**：通用关键词（loan/credit）→ 合同角色（Borrower/Lender/Administrative Agent）→ 章节主题（The Loans vs The Notes）三层递进。角色实体比关键词稳定得多。

2. **证据驱动 > 黑盒分类**：输出 `positive_evidence` + `negative_evidence` 而非简单 true/false。这让 PM 可以直接复核分类理由，不需要打开原始 PDF。

3. **规则 + LLM 组合的必要性**：LLM 对标题含 `INDENTURE` + 正文出现 Indenture Trustee 的负例有时会误判。规则兜底后，负例准确率可满足 >95% 标准。

4. **分类级别要明确**：客户最初说"分类 8-K 文件"，实际需求是 Exhibit 级别分类。Contract 中明确 `input` 的粒度避免了大范围返工。
