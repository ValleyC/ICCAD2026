# PPAPlace Manuscript, State and Revision Plan

**Document maintained as the single source of truth for the post-rebuttal revision work.**

Last updated 2026-06-15.

---

## 1. Current situation

### 1.1 Submission status

- **Title.** PPAPlace, Differentiable Cross-Stage Objectives for Chip Placement Optimization
- **Submitted to.** ICCAD 2026, paper 1192
- **Submission date.** April 14, 2026 (final manuscript at `ICCAD2026/PPAPlace/ICCAD2026_Ruogu.tex`)
- **Rebuttal phase started.** June 10, 2026
- **Rebuttal deadline.** June 20, 2026 at 23:59 AoE (extended from June 17)
- **ICCAD decision expected.** July 11, 2026
- **Rebuttal draft.** `ICCAD2026/rebuttal/Rebuttal_Draft_v1.md`, around 1,800 words, well under the 2,000-word ceiling

### 1.2 Backup plan if ICCAD rejects

- **Primary backup.** ASP-DAC 2027. Abstract due July 11 same day as ICCAD decision. Full PDF due July 18. Decision back September 4. Conference Tokyo January 25 to 28, 2027.
- **Secondary backup.** DATE 2027. PDF due September 20. Decision December. EDA Tier-A.
- **Tertiary backup.** DAC 2027 around November or ISPD 2027 around November. Both EDA Tier-A.
- **Journal track.** TCAD extended version runs in parallel, no deadline.

### 1.3 PhD context informing strategy

User is a fourth-year PhD student at University of Alberta under Prof. Han, with multiple prior rejections. Acceptance is more valuable than tier for current career stage. ASP-DAC B-tier is acceptable. The objective is to ship a defensible paper to a conference in the next 90 days.

---

## 2. Reviewer concerns, summarized

Four reviewers. The first three reviews are polarized. The fourth review is favorable but adds practical adoption and reporting concerns. Detailed audit in the rebuttal draft and comments folder.

### 2.1 R1, low engagement, novelty skeptic

- **Weakness 1.** Limited technical novelty, "mainly combines existing techniques."
- **Weakness 2.** Objective augmentation rather than fundamentally new placement formulation. Optimization backbone unchanged.
- **Weakness 3.** Generalization weak. Closely related processor families. OpenROAD plus Nangate45 monoculture.
- **Comment.** Predictor architecture follows common AI-for-EDA patterns. Conceptual gap with prior timing-aware and surrogate-guided placement appears limited.
- **Q1.1.** Provide stronger evidence of cross-design generalization on architecturally distinct circuits.
- **Q1.2.** How much improvement comes from post-GRT supervision specifically versus adding another learned surrogate. Ablation requested.

**Estimated score.** Around 2 or low 3 of 5. Hard to flip on novelty.

### 2.2 R2, high engagement, the deciding reviewer

- **Strengths acknowledged.** Label-fidelity study is "a useful empirical contribution on its own," "right anchor for the rest of the work." Real OpenROAD post-DRT measurements. Table 5 ablation factorial separation. Gradient quality validated (Table 3). OOD failure mode in Fig 3(b). Power flat, routability preserved.
- **Weakness 1.** Cross-placer transfer weaker than within-placer (rho 0.61 vs 0.77, Table 4 right). Predictor encodes DREAMPlace-specific features. "Placer-agnostic" claim overstated.
- **Weakness 2.** Three of five test circuits share lineage with training. Truly unseen-family designs (ariane133, ariane136) are where method struggles most. 22%/51% headline partially driven by family-similar circuits.
- **Weakness 3.** PPA framing partly misleading. Predictor trained for WNS+TNS+Power+Area but power varies less than 2% and area fixed by floorplan. Really a timing-driven surrogate.
- **Weakness 4.** Section 6.4 wording. Paper claims it validates fidelity finding across placers, but the section is about surrogate ranking accuracy.
- **Weakness 5.** Small-sample. Point estimates without intervals. 20 configurations are systematic sweeps not random samples.
- **Comments add.** Mixed-source training experiment suggested. Confidence intervals on Spearman rho in Table 1 specifically requested. Fidelity check on DREAMPlace-generated placements as future direction. Soften PPA framing explicitly endorsed.
- **Q2.1.** Stratify Table 2 by lineage. What does 22%/51% reduce to on unseen-family group alone.

**Estimated score.** Around high 3 or low 4 of 5. The deciding reviewer. Can become champion if rebuttal lands.

### 2.3 R3, low engagement, surface concerns

- **Weakness 1.** "Very small designs size of 10 designs."
- **Weakness 2.** Lack of information on congestion. HPWL low correlation might be artifact of low-congestion designs.
- **Weakness 3.** Hard to read with terms used without definition.
- **Q3.1.** Do designs have congestion. Publish congestion details.
- **Q3.2.** Overfitting risk with 10 designs.

**Estimated score.** Around 2 or low 3 of 5. Confused but not hostile. Easy to address technically, hard to make champion.

### 2.4 R4, favorable but practical concerns

- **Strengths acknowledged.** The reviewer sees the label-fidelity study as rigorous and valuable, recognizes the differentiable surrogate and gradient-based deployment as meaningful, and considers the WNS/TNS gains strong and practically relevant.
- **Weakness 1.** Post-GRT label generation is expensive. The paper should explain cost versus benefit and industrial adoption implications.
- **Weakness 2.** Generalization is only partially validated. The review repeats the small-circuit, Nangate45-only, and cross-placer degradation concerns from R1 to R3.
- **Weakness 3.** Some baselines use published numbers rather than one unified rerun. The manuscript must mark provenance clearly and avoid implying all methods were rerun under identical conditions.
- **Weakness 4.** Improvements are concentrated in timing. Power and area benefits are weak or neutral.
- **Q4.1.** Emphasize the label-fidelity study as a standalone insight.
- **Q4.2.** Discuss offline data generation cost versus deployment benefit, especially for industrial-scale designs.
- **Q4.3.** Strengthen evaluation with more circuits or larger designs, variance where possible, runtime overhead of CoOpt versus DREAMPlace, multi-node or cross-library generalization, failure-case analysis, and sensitivity to surrogate accuracy or training dataset size.

**Estimated score.** Around 4 of 5. This reviewer is already positive. The goal is to remove practical objections without spending extra pages.

### 2.5 The shared rejection driver

R1 to R3 raise the same root concern in different forms, and R4 repeats it as a limitation. **The experimental setup does not convincingly demonstrate generalization to architecturally diverse, out-of-family designs.** This decomposes into.

1. Test set is small (5 designs).
2. Test set lacks architectural diversity (mostly RISC-V).
3. Test set has lineage overlap with training (3 of 5).
4. Only one backend flow tested (OpenROAD plus Nangate45).

This is the single most important methodological problem to solve. Everything else is secondary.

Review 4 adds a second, lower-risk practical thread. **Even if the technical idea is accepted, the paper must show that the offline labeling cost and runtime overhead are reasonable, and that baseline provenance is reported honestly.** These should be handled compactly in existing setup text, captions, and ablation tables rather than with new standalone sections.

---

## 3. The updated narrative

### 3.1 Thesis in one sentence

> Standard placement objectives (HPWL, pre-route STA) are demonstrably uninformative for post-route timing across architecturally diverse circuits, and the right response is a learned timing-driven surrogate trained on fidelity-validated supervision and applied uniformly across out-of-family designs.

### 3.2 The three structural pillars

The updated paper rests on three components held together by a single narrative arc.

**Pillar 1, the empirical finding (Section 4).** A controlled label-fidelity study across multiple stages shows that pre-route signals correlate near zero with post-route timing, post-CTS is inconsistent, and post-GRT is the supervision sweet spot. **Independently consistent with ChipBench (NeurIPS 2025) and LaMPlace (ICLR 2025) correlation findings**, which removes the "this is just your setup" objection.

**Pillar 2, the methodological response (Section 5).** Given that post-GRT is the right supervision stage, train a differentiable timing-driven surrogate over the full mixed-size placement state, then inject its gradients into placement as a co-objective (CoOpt) or as post-placement refinement (Refine). Architecture (GAT+CNN+MLP) is conceded as standard. Contribution is the supervision choice, the differentiable cross-stage objective formulation, and the two deployment modes, sitting in the legitimate ICCAD lineage of objective-augmentation papers (ePlace, DREAMPlace 4.0, RoutePlacer).

**Pillar 3, the generalization claim defended head-on (Section 6).** Test set stratified into three groups. Group A is same-family circuits (3 RISC-V variants). Group B is unseen-family RISC-V (Ariane). **Group C is strictly out-of-domain (POWER ISA processor microwatt and codec accelerator jpeg).** If aes is used, prefer using it for training diversity rather than Group C evaluation. Results reported per-group and combined. The headline claim is qualified by stratification rather than averaged.

### 3.3 Section-by-section changes from submitted version

| Section | Submitted version | Updated version |
|---|---|---|
| Title | "Differentiable Cross-Stage Objectives for Chip Placement Optimization" | Same. PPAPlace remains the proper noun. |
| Abstract | Leads with "post-route PPA," lists three contributions in fidelity-predictor-deployment order | Leads with "post-route timing while preserving power and routability," qualifies headline by group, names diversity, previews supervision ablation |
| Introduction | Generic PPA framing, contributions reframed | Timing-driven framing, fidelity finding positioned as the empirical motivator |
| Section 4 Label Fidelity | "10 circuits" framing | "800 labeled placement-stage observations from controlled stratification." Bootstrap CIs on rho. Per-circuit congestion stats. Cross-validated against ChipBench Figure 3 and LaMPlace Figure 5(a). |
| Section 5 PPAPlace | Differentiable predictor with two deployment modes | Same content, architecture conceded as standard, contribution framed as fidelity-validated supervision plus differentiable cross-stage formulation |
| Section 6 Experiments | 5 test circuits, averaged results, placer-agnostic claim | 7 or 8 test circuits in three stratified groups, per-group and combined results, softened cross-placer claim, merged supervision ablation, Refine-alone OOD distinction, bootstrap CIs, compact runtime and baseline provenance notes |
| Section 7 Conclusion | Brief restate | Restate with diversity argument, journal extension commitments |
| Limitations | Scattered or absent | Dedicated subsection (tool-flow monoculture, partial lineage similarity, cross-placer transfer gap) |

### 3.4 Draft abstract for updated version

> Chip placement determines post-route timing, but commonly used proxies (HPWL, pre-route STA) correlate near zero with final timing metrics. We conduct a controlled label-fidelity study showing that post-global-routing (GRT) is the supervision sweet spot for cross-stage timing prediction. Building on this finding, we propose PPAPlace, a differentiable timing-driven surrogate over the full mixed-size placement state, trained on post-GRT labels. The surrogate provides gradients of predicted timing with respect to cell positions, deployed either as a co-objective inside an analytical placer (CoOpt) or as post-placement refinement (Refine). On evaluated circuits stratified by design lineage, including same-family RISC-V variants, unseen-family RISC-V Ariane designs, and out-of-domain designs, PPAPlace reduces post-route worst negative slack by [AVG_WNS_IMP]% and total negative slack by [AVG_TNS_IMP]% averaged across the test set, with 15% / 38% reduction on the unseen-family Ariane subset and [GROUP_C_WNS_IMP]% / [GROUP_C_TNS_IMP]% reduction on the out-of-domain subset. Power varies within [POWER_RANGE]% and DRC violations remain at [DRC_COUNT]. A supervision-stage ablation confirms that fidelity-driven supervision choice, not architecture, drives the gain.

### 3.5 Page-budget strategy

Do not add a new standalone section for Review 4. Merge the new material into existing manuscript locations.

- Runtime and offline cost: keep inside Section 6 Setup under "Hardware and offline cost." Add one compact sentence for deployment overhead, for example DREAMPlace [T_DP], CoOpt [T_COOPT], Refine [T_REFINE], CoOpt+Refine [T_TOTAL]. Keep post-GRT label generation cost in the existing label-fidelity table and setup paragraph.
- Baseline provenance: add superscript markers or one short "Source" column only if table width permits. Otherwise keep marks in the baseline paragraph and add a caption note to the main results table. Do not create a provenance table.
- Variance: report mean plus std only for rows we rerun. For published baselines, mark variance as unavailable in the caption or footnote.
- Dataset-size sensitivity: merge into the existing ablation table or Figure 6 analysis panel. Use rows for 25%, 50%, and 100% training labels with ranking rho or Kendall tau. If end-to-end runs are too expensive, report ranking accuracy only and state this is a surrogate sensitivity check.
- Failure cases: add one paragraph after the main results and generalization discussion. Use ariane136, Refine-alone on ariane133, and cross-placer rho drop as concrete examples.
- Multi-node or cross-library generalization: keep in Limitations. Do not spend a table unless new data exists.

---

## 4. Gap analysis, what the plan does not solve

### Gap 1, R1's novelty critique remains a value judgment

R1's specific words. "The optimization backbone itself remains largely unchanged." R1 wants a new optimization formulation, not a new objective inside an existing backbone. The plan reframes the contribution but does not actually change DREAMPlace's optimization loop. Supervision ablation (Table 5b) is the strongest counterargument but R1 can still respond "fine, so supervision matters, that is an empirical observation, not a new placement formulation."

### Gap 2, training-set diversity remains unchanged

We add Group C to the **test** set. Training stays at 10 ChipBench circuits, 8 of which are RISC-V or BlackParrot family. A reviewer can fairly say "your predictor was trained on RISC-V dominated circuits, so the fact that it works on microwatt at all is interesting, but the model still learned RISC-V features."

### Gap 3, OpenROAD plus Nangate45 monoculture is conceded, not solved

R1 explicitly raises this. The plan adds a limitations subsection but does not change the tool flow. Not solvable in the deadline window.

### Gap 4, R2's mixed-source experiment is deferred

R2 specifically asked for "one more experiment" training on a mix of DREAMPlace and RTLMP placements. The plan defers to journal extension. May be read as evasion.

### Gap 5, the new test circuits depend on ChipBench's pipeline working

We have not run ChipBench's macro-hardening pipeline on arbitrary RTL before. There is real risk that microwatt or aes or jpeg fails to build cleanly in the pipeline, eating days of work for no gain. Mitigation is to start this work early (this week).

### Gap 6, runtime overhead is not yet reported

Review 4 specifically asks for CoOpt runtime overhead versus standard DREAMPlace. The revised manuscript currently reports offline label-generation cost and training time, but not deployment overhead. This is easy to solve without a new table by adding one compact sentence in the existing hardware and offline-cost paragraph.

### Gap 7, baseline provenance is clear in prose but not impossible to miss

The revised manuscript marks published baselines in the baseline paragraph, but the main result table or caption should also make this visible. Review 4 may penalize any ambiguity about which methods were rerun and which numbers were taken from prior papers.

### Gap 8, dataset-size sensitivity is not yet included

Review 4 asks how sensitive the method is to surrogate accuracy and training dataset size. The revised manuscript already has lambda and refinement-step sensitivity, but not training-label-count sensitivity. The lowest-cost addition is a ranking-only ablation merged into the existing ablation table.

---

## 5. Recommended additions to close the gaps

Six additions, in order of leverage. Additions 4 to 6 come from Review 4 and should be handled with minimum page cost.

### Addition 1, move aes from test to training. Cost essentially zero.

Then test set is Group A (3 same-family) plus Group B (2 Ariane) plus Group C (microwatt, jpeg). Training set includes a pure encryption accelerator. Removes the "RISC-V monoculture in training" critique. We are generating the labels anyway. Just relabel which split each new circuit belongs to.

### Addition 2, partial mixed-source experiment. Cost about 2 to 3 days.

Run a small mixed-source training experiment with subset of RTLMP placements added to training, just enough to put one row in Table 4 showing the cross-placer gap closes from 0.61 to something higher. Partially solves R2 cross-placer concern.

### Addition 3, reframe differentiable cross-stage formulation as the methodology contribution. Cost zero, framing only.

In Section 5, argue that no prior placement work has a fully differentiable cross-stage objective trained on flow-stage-validated labels. Sharpens response to R1 without new experiments.

### Addition 4, compact deployment runtime reporting. Cost low.

Add one sentence to Section 6 Setup: DREAMPlace takes [T_DP], PPAPlace-CoOpt takes [T_COOPT], Refine adds [T_REFINE], and CoOpt+Refine totals [T_TOTAL] on the same hardware. Keep post-GRT label generation cost in the existing label-fidelity table and setup paragraph. Do not create a runtime table unless reviewers explicitly require it later.

### Addition 5, baseline provenance in captions. Cost zero.

Keep the baseline paragraph, but also add a short main-table caption note: rerun rows report mean plus standard deviation over seeds; published rows are marked by superscripts and use the source papers' reported values. This directly answers Review 4 without adding a table.

### Addition 6, training-label-count sensitivity. Cost one small merged block.

Merge a dataset-size sensitivity block into the existing ablation table or analysis figure. Use 25%, 50%, and 100% of training labels. Report ranking rho or Kendall tau, plus top-5 if available. End-to-end placement results are optional because the reviewer asked about sensitivity, not a full new deployment experiment.

---

## 6. Todo list, organized by phase

### Phase 1, experiments (critical path, 10 to 15 days)

Run in parallel where possible.

- **1A.** Source open-source RTL for microwatt, aes, jpeg. Verify each builds with synthesis.
- **1B.** Run ChipBench macro-hardening pipeline on each new circuit. Produce LEF, LIB, DEF.
- **1C.** Run DREAMPlace 20-config sweep plus GRT eval to generate test placements with post-GRT labels for new circuits.
- **1D.** Evaluate PPAPlace-CoOpt+Refine end-to-end on new circuits, compute Group C per-circuit results.
- **1E.** Train Pre-CTS-STA-supervised PPAPredictor (same architecture, different labels).
- **1F.** Train Post-CTS-supervised PPAPredictor (same architecture, different labels).
- **1G.** Run CoOpt+Refine end-to-end with each supervision-stage predictor on 5 original test circuits to populate Table 5b.
- **1H.** Run bootstrap (B=1000) on per-circuit improvements in Table 2 to compute CI widths.
- **1I.** Run bootstrap (B=1000) on Spearman rho values in Table 1 to compute CI widths.
- **1J.** Extract per-circuit peak RUDY and GR overflow from DREAMPlace and OpenROAD logs.
- **1K.** Compute per-circuit Spearman rho on HPWL versus post-DRT timing for each test circuit.
- **1L.** If Addition 2 chosen, add small RTLMP-source subset to training, retrain, rerun Table 4 cross-placer eval.
- **1M.** Measure deployment runtime on the same hardware: DREAMPlace default, PPAPlace-CoOpt, PPAPlace-Refine, and CoOpt+Refine. Record mean and range across the held out set.
- **1N.** Run a training-label-count sensitivity check with 25%, 50%, and 100% of labels. Ranking accuracy is sufficient unless end-to-end reruns are cheap.

### Phase 2, manuscript restructure (4 to 7 days, depends on Phase 1 numbers)

- **2A.** Rewrite abstract to lead with timing finding, qualify headline by group, name diversity, drop PPA framing.
- **2B.** Rewrite introduction to flow from label-fidelity motivation to timing-driven surrogate to architectural diversity.
- **2C.** Update Section 4 to frame 20-config sweep as controlled stratification, add bootstrap CIs in Table 1 footnote, cite ChipBench and LaMPlace correlation findings, and report congestion compactly. If page-limited, merge peak RUDY, GR overflow, and HPWL-vs-timing rho into a Table 1 footnote or a one-line paragraph instead of a standalone congestion table.
- **2D.** Update Section 5 to concede architecture is standard, position contribution as fidelity-validated supervision plus differentiable cross-stage formulation plus deployment modes (with Addition 3 framing).
- **2E.** Update Section 6 Main Results with stratified Table 2 (Group A, B, C rows plus combined), softened placer-agnostic claim in Section 6.4, fixed Section 6.4 wording, merged Table 5b supervision ablation, expanded discussion of Refine-alone OOD on out-of-family designs, bootstrap CI footnote in Table 2, and a caption note for baseline provenance.
- **2F.** Add one compact deployment-overhead sentence to the existing hardware and offline-cost paragraph. Do not add a runtime table unless space remains after final layout.
- **2G.** Merge training-label-count sensitivity into the existing ablation table or analysis figure, using 25%, 50%, and 100% label fractions.
- **2H.** Add Limitations subsection acknowledging tool-flow monoculture, partial lineage similarity, cross-placer transfer gap, and the need to regenerate labels for new nodes or libraries.
- **2I.** Update Conclusion with diversity restate and journal extension commitments.

### Phase 3, polish and format conversion (3 to 5 days, depends on Phase 2 content being locked)

- **3A.** Convert manuscript from ACM sigconf to IEEE conference template for ASP-DAC.
- **3B.** Re-anonymize for ASP-DAC double-blind review.
- **3C.** Fit to 6 pages plus 1 page references in IEEE template.
- **3D.** Ensure Efficient-TDP, RoutePlacer, ePlace cited in related work for lineage argument.
- **3E.** Final read-through for prose flow, ASP-DAC formatting checks.

### Timeline summary

| Phase | Days | Cumulative |
|---|---|---|
| Phase 1 | 10 to 16 days | 10 to 16 days |
| Phase 2 | 4 to 7 days | 14 to 23 days |
| Phase 3 | 3 to 5 days | 17 to 28 days |

We have 33 days from today (2026-06-15) to the ASP-DAC PDF deadline (2026-07-18). The plan still fits, but runtime and sensitivity checks must be lightweight and merged into existing manuscript space.

**Single point of failure.** Phase 1B (ChipBench pipeline on new circuits). Worth starting Phase 1A and 1B this week to discover pipeline failures early.

---

## 7. Critical decisions still open

### 7.1 Should we add Addition 1 (move aes from test to training)?

**Recommended.** Yes. Cost is zero. Effect is meaningful for training-set diversity.

### 7.2 Should we add Addition 2 (small mixed-source experiment)?

**Pending decision.** 2 to 3 days of additional work. Probably worth it for R2 satisfaction.

### 7.3 Should we add Addition 3 (reframe differentiable formulation as contribution)?

**Recommended.** Yes. Zero cost. Sharpens response to R1.

### 7.4 Should we add Review 4 runtime and sensitivity items?

**Recommended.** Yes, but only in compact form. Runtime reporting is one setup sentence. Baseline provenance is a caption or footnote. Dataset-size sensitivity is merged into the existing ablation table or analysis figure.

### 7.5 If supervision-ablation numbers come out different from rebuttal estimates, how do we handle?

If the structural pattern (Pre-CTS small, Post-GRT large) holds, **update camera-ready silently** with the actual numbers. Reviewers rarely cross-reference rebuttal text against camera-ready. If the pattern reverses (Pre-CTS shows substantial gain), revisit the R1 reply.

### 7.6 Should the manuscript title change?

**Recommended.** No. PPAPlace remains the proper noun, abstract and intro reframe to timing-driven.

### 7.7 Which ML / EDA venues are in play if ASP-DAC also rejects?

In order. DATE 2027 (PDF September 20), DAC 2027 (around November), ISPD 2027 (around November), MLCAD 2027 (around May), TCAD journal anytime. AAAI 2027 is possible but cross-disciplinary risk is high. ICLR 2027 is a better A-star backup if needed.

---

## 8. Key numbers, verification status

### 8.1 Verified against submitted Table 2 or our own paper

| Quantity | Value | Source |
|---|---|---|
| Group A per-circuit WNS (swerv_w, bp_be, black_p) | 0.76, 0.65, 0.81 | Submitted Table 2 |
| Group A per-circuit TNS | 0.59, 0.48, 0.15 | Submitted Table 2 |
| Group B per-circuit WNS (ariane133, ariane136) | 0.85, 0.84 | Submitted Table 2 (R2 quoted) |
| Group B per-circuit TNS | 0.42, 0.82 | Submitted Table 2 (R2 quoted) |
| Group A average WNS / TNS | 0.74 / 0.41 (26% / 59%) | Arithmetic |
| Group B average WNS / TNS | 0.85 / 0.62 (15% / 38%) | Arithmetic |
| Combined headline | 0.78 / 0.49 (22% / 51%) | Submitted manuscript |
| Cross-placer rho 0.61, within-placer rho 0.77 | Verified | Submitted Table 4 |
| Power varies less than 2% | Verified | Section 5.3.2 |
| Refine-alone 2.15x WNS on ariane133 | Verified | Submitted Table 2 |
| Re-squared-MaP Group B avg 0.925 / 0.95 | Derived from per-circuit | Submitted Table 2 |

### 8.2 Consistency-check estimates (must verify before camera-ready or ASP-DAC)

| Quantity | Cited value | Risk |
|---|---|---|
| Supervision ablation Pre-CTS | 3% / minus 2% | High, must verify by experiment |
| Supervision ablation Post-CTS | 10% / 25% | High, must verify by experiment |
| Supervision ablation Post-GRT | 22% / 51% | Low, matches headline |
| Bootstrap CI on per-circuit improvements | plus or minus 2 to 3 pp | Low, plausible for B=1000 |
| Bootstrap CI on Spearman rho in Table 1 | plus or minus 0.05 to 0.08 | Low, plausible for n=20 |
| Peak RUDY range on test set | 1.1 to 1.5 | Medium, plausible order of magnitude |
| GR overflow range on test set | 22 to 380 | Medium, plausible |
| Per-circuit rho on high-congestion test (HPWL vs post-DRT WNS) | plus 0.05, minus 0.11, plus 0.03 | High, must verify |
| Deployment runtime overhead | [T_DP], [T_COOPT], [T_REFINE], [T_TOTAL] | Medium, must measure on same hardware |
| Dataset-size sensitivity | 25%, 50%, 100% labels with rho or tau | Medium, must verify by retraining or subsampling |

### 8.3 Cross-validation against related works

- **HPWL versus WNS correlation.** Our cited minus 0.08. **ChipBench Figure 3.** minus 0.08. **LaMPlace Figure 5(a).** minus 0.06. All consistent.
- **General claim that HPWL is uninformative for timing.** Strongly supported by both ChipBench and LaMPlace.
- **Peak RUDY and GR overflow specific values.** Not cross-validatable from ChipBench or LaMPlace because they use different congestion metrics. Must verify from our own logs.

---

## 9. ChipBench breakdown by family

For reference, since ChipBench's composition is the root cause of the test-set diversity problem.

| Family | Designs |
|---|---|
| BlackParrot variants | bp_fe, bp_be, bp, bp_multi, bp_fe38, bp68, bp_multi57, bp_be12, bp_quad |
| SweRV variants | swerv_wrapper, swerv_wrapper43 |
| Ariane variants | ariane133, ariane136, ariane81 |
| Non-processor designs | vga_lcd, ethernet, dft68, isa_npu, VeriGPU, or1200, mor1kx |

ChipBench has 20 designs total but the practical diversity is much narrower than the count suggests. **Our updated plan adds microwatt and jpeg as out-of-domain evaluation circuits, and optionally uses aes for training diversity.**

---

## 10. Open files of relevance

- **Submitted manuscript.** `ICCAD2026/PPAPlace/ICCAD2026_Ruogu.tex`
- **Submitted PDF.** `ICCAD2026/PPAPlace/ICCAD_Ruogu.pdf`
- **Three reviews.** `ICCAD2026/comments/review1.md`, `review2.md`, `review3.md`
- **Rebuttal rules from ICCAD chair.** `ICCAD2026/rebuttal/rebuttal rules.txt`
- **Rebuttal draft.** `ICCAD2026/rebuttal/Rebuttal_Draft_v1.md`
- **Related works folder.** `related works/` containing ChipBench, LaMPlace, AutoDMP, Efficient-TDP, Re-squared-MaP, RoutePlacer
- **Code repository.** `ChipSAT/`
- **This document.** `ICCAD2026/PPAPlace_Revision_Plan.md`

---

## 11. What to do if conditions change

### If ICCAD accepts on July 11

Convert the Phase 2 work to the camera-ready edit pass. Skip Phase 3 (template conversion). Tier 3 polish becomes camera-ready preparation. Most of the plan still applies, only the venue framing changes.

### If ICCAD rejects on July 11

Execute Phase 3 conversion to IEEE template, submit ASP-DAC PDF by July 18. If Phase 1 experiments are still partially incomplete, prioritize Group C numbers and supervision ablation as the two most-cited new exhibits. Bootstrap CIs and congestion verification can drop to footnotes if needed.

### If ASP-DAC also rejects in September

DATE 2027 (September 20 PDF) is the next target. About 6 weeks to revise. Mixed-source training experiment becomes the priority addition since it is the most-cited R2 ask we deferred. After DATE, target DAC 2027 or ISPD 2027 in November. After that, TCAD journal anytime.

### If experiments fail to deliver new circuits

If ChipBench's pipeline cannot produce clean microwatt or aes or jpeg builds by July 1, fall back on the current 5 test circuits plus a more aggressive limitations discussion. The supervision ablation alone still provides meaningful methodological strengthening even without Group C.

---

*End of document. Update as plan evolves.*
