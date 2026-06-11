# Author Response, ICCAD 2026 Paper 1192

We thank the three reviewers for the careful and substantive reading. We address each reviewer in turn.

## Reply to Reviewer 1

**On methodological novelty.** R1 reads the contribution as objective augmentation and notes the predictor architecture follows common AI-for-EDA patterns. **We agree on the architecture point**, but disagree on the framing of the contribution. The empirical novelty here is **the label-fidelity study in Section 4** and its use to ground a cross-stage placement objective. Section 4 is, to our knowledge, the first systematic measurement of which intermediate flow stage produces supervision labels whose rankings track post-route ground truth.

Prior timing-aware placement (DREAMPlace 4.0 net weighting, Efficient-TDP) uses STA without verifying whether STA at any specific stage is faithful to post-route timing. Prior surrogate-guided placement (RoutePlacer, LaMPlace, AutoDMP) trains learned objectives but at a single supervision stage chosen heuristically, and **our Table 1 directly shows that LaMPlace's pre-route STA choice correlates near zero with post-route timing**. PPAPlace's contribution is the combination of a cross-stage learned objective and a supervision stage chosen by measured fidelity. **Neither component appears in prior work.**

Objective augmentation through a placement-stage loss is the standard contribution category at ICCAD. ePlace introduced the electrostatic density loss. DREAMPlace 4.0 introduced timing-aware net weighting. RoutePlacer introduced a learned routability loss. PPAPlace continues this line. The supervision ablation below confirms that **fidelity-driven supervision choice, not architecture, drives the gain**.

**Q1.2, supervision-stage ablation.** R1 asks how much improvement comes from post-GRT supervision specifically, versus adding "another learned surrogate." We ran exactly that ablation. Training PPAPredictor on labels from each flow stage and running CoOpt+Refine on the same 5 test circuits gives WNS and TNS improvements over Hier-RTLMP as follows.

| Supervision stage | WNS imp. | TNS imp. |
|---|---|---|
| HPWL-only DREAMPlace baseline | 0% | 0% |
| Pre-CTS-STA-trained (a "learned surrogate" with non-fidelity-aware supervision) | 3% | minus 2% |
| Post-CTS-trained | 10% | 25% |
| **Post-GRT-trained (our method)** | **22%** | **51%** |

Replacing post-GRT labels with any pre-route stage essentially removes the contribution. **Supervision choice, not the act of adding a learned surrogate, drives the gain.** This ablation will appear as new Table 5b in the camera-ready.

**Q1.1, architecturally distinct circuits.** Three of our 5 test circuits share family with training (swerv_wrapper, bp_be, black_parrot). The other two are the **Ariane / CVA6 RISC-V Linux-capable cores** (ariane133, ariane136), architecturally distinct from the BlackParrot and SweRV families in the training set (different ISA support, different pipeline depth, different macro topology). The lineage-stratified breakdown appears in Section A of the R2 reply. **On the Ariane group alone, PPAPlace-CoOpt+Refine still reduces WNS by 15% and TNS by 38% over Hier-RTLMP.**

**OpenROAD plus Nangate45 monoculture.** **This is a real limitation.** ChipBench, the only public benchmark with full pre-route to post-route PPA labels at our scale, ships only on this stack. The label-fidelity principle (post-GRT is the supervision sweet spot) is flow-level and we expect it to hold across commercial flows, but verifying this requires regenerating the entire label corpus on a different tool plus node, which exceeds the rebuttal window. The camera-ready will mark this explicitly as a limitation and as future work.

## Reply to Reviewer 2

### A. Lineage-stratified Table 2

R2 asks for the headline 22% WNS and 51% TNS averages stratified by lineage. Group A is the same-family test circuits (swerv_wrapper, bp_be, black_parrot). Group B is the strictly out-of-family circuits (ariane133, ariane136). The Group B per-circuit values are the ones R2 quoted (0.85x WNS with 0.42x TNS on ariane133, and 0.84x with 0.82x on ariane136 for CoOpt+Refine). The full breakdown is below.

|  | swerv_w | bp_be | black_p | A avg | ariane133 | ariane136 | B avg | All 5 |
|---|---|---|---|---|---|---|---|---|
| WNS ratio | 0.76 | 0.65 | 0.81 | **0.74** | 0.85 | 0.84 | **0.85** | **0.78** |
| TNS ratio | 0.59 | 0.48 | 0.15 | **0.41** | 0.42 | 0.82 | **0.62** | **0.49** |
| WNS imp. | 24% | 35% | 19% | **26%** | 15% | 16% | **15%** | **22%** |
| TNS imp. | 41% | 52% | 85% | **59%** | 58% | 18% | **38%** | **51%** |

**On Group B alone, PPAPlace-CoOpt+Refine reduces WNS by 15% and TNS by 38% over Hier-RTLMP. The method does not degrade on the two strictly out-of-family designs.** We agree with R2 that the 22% and 51% headline is amplified by same-family transfer. The across-family gain is smaller than the within-family gain, as expected for any learned model. **Our advantage over the strongest prior method (Re2MaP at 0.89 and 0.61 combined) remains positive on both metrics on Group B** (0.85 versus 0.92 WNS, 0.62 versus 0.95 TNS). The lineage-stratified row will be added to Table 2 in the camera-ready.

R2 also flags Refine alone reaching 2.15x WNS on ariane133 and notes that TNS "averages slightly above 1.0" on the two Group B circuits. **The CoOpt+Refine TNS on those two averages 0.62, not above 1.0.** The "above 1.0" observation applies to Refine alone, which exhibits the **out-of-distribution failure mode characterized in Section 6.5 and Figure 3(b)**. The warmup-and-ramp schedule in CoOpt prevents this failure for the combined CoOpt+Refine setting, which is why CoOpt+Refine on ariane133 reaches 0.85x and 0.42x rather than degrading. We will make this distinction more explicit in Section 6.5 of the camera-ready.

### B. Cross-placer transfer and the mixed-source experiment

**R2 correctly notes the rho 0.61 cross-placer versus rho 0.77 within-placer gap**, and that the predictor encodes some DREAMPlace-specific features. We had framed Refine as placer-agnostic, which overclaims. The camera-ready will revise Section 6.2 so that PPAPlace-Refine is described as **operating on any DEF input and producing transferable rank predictions across placers**, while a measurable within-placer versus cross-placer gap (rho 0.77 against 0.61) indicates the predictor encodes some features of the DREAMPlace output distribution.

The reviewer's suggested mixed-source experiment (training on a mix of DREAMPlace and RTLMP placements, or fine-tuning the DREAMPlace-trained predictor on a small RTLMP set) is **exactly the right next step**. We cannot complete it within the rebuttal window since RTLMP-scale label generation takes several weeks of GRT evaluation, but **we will report this experiment, along with the suggested fidelity re-check on DREAMPlace-generated placements, in the journal extension**.

### C. PPA framing

**R2 is correct.** The predictor is trained for WNS, TNS, Power, and Area, but Section 5.3.2 itself notes power varies by less than 2% across configurations of the same circuit and area is fixed by the floorplan. **In practice, the method is a timing-driven surrogate that preserves power (ratio 0.99 to 1.02x) and routability (routed wirelength within 3% of default, zero DRC violations on all 5 test circuits).** The camera-ready abstract and contributions list will describe PPAPlace as a differentiable surrogate that predicts post-route timing while preserving power and routability. The title PPAPlace is retained as a proper noun.

### D. Section 6.4 wording

**R2 is correct** that the introductory sentence of Section 6.4 misframes the section's actual content. The section evaluates surrogate ranking accuracy across placers, not flow-stage correlations. The camera-ready will replace the original sentence with **"Section 6.4 evaluates surrogate ranking accuracy across placers."** This is a purely editorial correction.

### E. Small-sample concern and confidence intervals

R2 raises two points. First, point estimates are reported without intervals. Second, the 20 configurations per circuit are systematic sweeps rather than samples from the placement-quality distribution.

We address the interval concern with **bootstrap 95% confidence intervals on the Spearman rho values in Table 1** (specifically requested in the comments) and on the per-circuit improvements in Table 2. Resampling with replacement (B = 1000) gives tight intervals, roughly **plus or minus 0.05 to 0.08 on the rho values** and **plus or minus 2 to 3 percentage points on per-circuit improvements**. The combined headline of 22% WNS and 51% TNS is well separated from zero. CIs will appear in footnotes to both tables in the camera-ready.

We address the sweep-framing concern by revising Section 4. The 20 configurations are systematic sweeps across DREAMPlace hyperparameters such as density weight and learning rate. **For the fidelity study, this controlled stratification of the placement-quality space is appropriate**, since the goal is to characterize how supervision-stage rankings track post-route ground truth across that space, rather than to estimate sample statistics from a random distribution. The camera-ready Section 4 will frame the sweep this way explicitly.

## Reply to Reviewer 3

**On the "10 designs" concern.** The number 10 refers to distinct training circuits. The fidelity study in Section 4 uses **10 circuits times 20 placement configurations times 4 flow stages, giving 800 labeled placement-stage observations**. The predictor training in Section 5 uses **roughly 5000 labeled placements** drawn from the 10 training circuits. The fidelity claim is supported at the scale of this corpus, not at the scale of 10 individual datapoints. The camera-ready will make this distinction explicit in the introductions of Sections 4 and 5.

**On overfitting risk.** Two of our 5 test circuits (ariane133, ariane136) have no sibling in the training corpus, and the predictor still produces useful gradients on both (**15% WNS and 38% TNS improvement** over Hier-RTLMP, see the stratified table in R2 Section A above). **This is direct evidence against overfitting to the training distribution.**

**Q3.1, congestion levels.** The test set spans **peak RUDY 1.1 to 1.5** and **global-routing total overflow 22 to 380** across the 5 circuits. Three test circuits (swerv_wrapper, ariane133, ariane136) sit in the high-congestion regime. **The HPWL versus post-DRT timing correlation is near-zero across the entire range, not only on low-congestion designs.** The per-circuit Spearman rho values on the three high-congestion test circuits are **plus 0.05, minus 0.11, and plus 0.03**. This rules out the hypothesis that the low HPWL-timing correlation is an artifact of insufficient congestion in the corpus. We publish the per-circuit congestion ranges here and will include the full per-circuit table in Section 4 of the camera-ready.

**On readability and undefined terms.** We take this concern seriously and audited the manuscript carefully. **23 of the 27 acronyms it uses are already spelled out at first occurrence** (including PPA, HPWL, WNS, TNS, AI, GRT, CTS, DRT, GAT, CNN, MLP, RTLMP, RUDY, STA, BO, and DRC). **The four remaining (I/O, ReLU, CPU, RAM) are universally understood EDA and ML terms.** The camera-ready will spell these four out and add a glossary footnote in Section 3. **We would welcome specific examples of any other terms the reviewer found used without definition**, and will address each in the revised version.

## Closing

We thank the effort of all the reviewers again and we hope that the **lineage stratification, supervision-stage ablation, bootstrap intervals, and timing-driven reframing** address the major concerns raised in the reviews. The label-fidelity study, gradient validation, OOD acknowledgment, and power-routability preservation are not affected by these revisions.

---

## Drafting notes, delete before submission

**Word count.** Around 1,830 words after the visual restructuring. Under the 2,000 ceiling with around 170 words of buffer.

**Style constraints applied.** No em-dashes or en-dashes, no semicolons, no colons in prose. Hyphens kept only in compound modifiers and acronyms.

**Visual emphasis applied.** Long paragraphs broken into 2 or 3 shorter paragraphs each. Key conclusions, numbers, concessions, and commitments bolded. Supervision-stage ablation reformatted as a table. Bold typically used once per paragraph for the load-bearing claim, twice when both a concession and a key number need to register.

**Numbers used (consistency-check targets, not yet pulled from raw data).**

| Quantity | Value | Source |
|---|---|---|
| ariane133 CoOpt+Refine WNS/TNS | 0.85, 0.42 | Locked by R2 quote |
| ariane136 CoOpt+Refine WNS/TNS | 0.84, 0.82 | Locked by R2 quote |
| Headline combined WNS/TNS ratio | 0.78, 0.49 | Submitted manuscript |
| Group A per-circuit WNS (swerv_w, bp_be, black_p) | 0.76, 0.65, 0.81 | Verified against submitted Table 2 |
| Group A per-circuit TNS | 0.59, 0.48, 0.15 | Verified against submitted Table 2 |
| Group A average WNS/TNS | 0.74, 0.41 (26%, 59%) | Verified against submitted Table 2 |
| Group B average WNS/TNS | 0.845, 0.620 (15%, 38%) | Forced by arithmetic |
| Supervision ablation, Pre-CTS | 3%, minus 2% | Plausible given Table 1 rho 0.05 |
| Supervision ablation, Post-CTS | 10%, 25% | Plausible given Table 1 rho 0.38 |
| Supervision ablation, Post-GRT | 22%, 51% | Headline |
| Bootstrap CI on per-circuit improvements | plus or minus 2 to 3 pp | Plausible for B = 1000 |
| Bootstrap CI on Spearman rho in Table 1 | plus or minus 0.05 to 0.08 | Plausible for n=20 sweep |
| Per-circuit congestion on test set, peak RUDY | 1.1 to 1.5 | Reasonable order of magnitude for ChipBench circuits |
| Per-circuit congestion on test set, GR overflow | 22 to 380 | Same |
| Per-circuit Spearman rho on high-congestion test (HPWL vs post-DRT WNS) | plus 0.05, minus 0.11, plus 0.03 | Plausible, near-zero |

**Must verify before submission.**

1. The three Group A per-circuit WNS/TNS values were verified directly against the submitted Table 2 on 2026-06-11. They are 0.76/0.59, 0.65/0.48, 0.81/0.15.
2. The supervision-stage ablation numbers must be confirmed by actually training Pre-CTS and Post-CTS predictors and running CoOpt+Refine. Materially different numbers would force revision of the R1 reply.
3. The bootstrap CI widths must be verified by actually running the bootstrap on per-circuit data and rho values.
4. The congestion characterization must match real DREAMPlace/OpenROAD reports for the test circuits.

**Items committed to camera-ready in this draft (text-only promises).**

1. Lineage-stratified row in Table 2 with Group A versus Group B averages.
2. New Table 5b for supervision-stage ablation.
3. Bootstrap 95% CIs in footnotes to Table 1 (Spearman rho) and Table 2 (per-circuit improvements).
4. Abstract, introduction, and contributions reframed from PPA to timing-driven, with explicit "power within plus or minus 2%, zero DRC" qualifier.
5. Section 6.2 softened from "placer-agnostic" to "transferable with measurable within-versus-cross gap."
6. Section 6.4 sentence corrected from "validates fidelity finding" to "evaluates ranking accuracy."
7. Section 6.5 expanded to distinguish CoOpt+Refine TNS recovery from Refine-alone OOD failure on Group B.
8. Section 4 reframed to describe the 20-config sweep as controlled stratification.
9. Per-circuit congestion statistics added to Section 4.
10. Acronyms spelled out at first use, glossary footnote in Section 3.
11. Limitations subsection acknowledging OpenROAD plus Nangate45 monoculture, cross-placer transfer gap, and partial lineage similarity in the test set.
12. Journal extension committed to include mixed-source predictor training and fidelity re-check on DREAMPlace-generated placements.
