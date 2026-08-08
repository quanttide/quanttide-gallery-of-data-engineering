# ROADMAP

> 格式：Keep a Changelog + checkbox 任务清单。
> ROADMAP 面向未来计划；发布后将已完成条目迁移到 CHANGELOG。
> 当前：v0.0.2（内容积累期）；目标 v0.1.0：首批完整案例。

## [0.1.0]

> 核心目标：让用例库成型——案例结构模板稳定、覆盖多个场景类型，可作为外部客户参考的案例集。满足交付标准即发 v0.1.0，不按时间排期。

### Added

- [x] 案例结构模板：案例目录由 `index.md`（场景描述 + 关键挑战）+ `requirement.md`（需求文档）+ `specification.yaml`（数据契约与蓝图）组成（参照 `quanttide/index.md`）
- [ ] 量潮科技数字化第一期交付闭环：历史周会批量整理完成后，补全 `quanttide/requirement.md` 的执行与传输阶段记录
- [ ] 新增 1 个完整三件套案例（如将 `data/archive/gallery/` 中 beta 验证案例迁回，补全执行记录后重新收录）
- [ ] 案例覆盖至少 2 种场景类型（当前：电商价格采集、议事决议数据治理），更新 `index.md` 案例列表
- [ ] 案例目录结构模板确定后不再变动，新案例套用 `quanttide/index.md` 三件套结构
- [ ] 案例列表（`index.md`）与案例目录一一对应
- [ ] 至少 3 个 active 案例，均收录于 `index.md` 案例列表
- [ ] 至少 1 个案例具备完整生命周期（需求 → 规格 → 实现 → 执行 → 传输），如 `quanttide/requirement.md` 覆盖全流程
- [ ] 案例列表可被外部客户阅读参考，`index.md` 提供案例导航