# Author Response, ICCAD 2026 Paper 1192

We thank all four reviewers for the careful reading. The central clarification is that PPAPlace is not meant to be a new neural architecture or a broad optimizer for every PPA metric. Its core claim is **timing driven placement through fidelity validated post GRT supervision**. The GAT and CNN components are standard. The contribution is choosing the supervision stage by measured downstream fidelity, then deploying the resulting differentiable timing surrogate inside placement. The attached sheets give the detailed numbers that support the concise answers below.

## R1 Novelty and supervision

R1 correctly notes that PPAPlace augments an analytical placement objective rather than replacing the full optimizer. We agree with that categorization. Objective augmentation is also a well established placement contribution path. DREAMPlace 4.0 adds pre route timing driven net weighting [3]. RoutePlacer adds a learned routability objective [4]. PPAPlace follows this line, but changes the target from a convenient proxy to a **post GRT supervised timing objective** whose rank fidelity is measured against post DRT timing.

The novelty is therefore not that GAT plus CNN is unusual. The novelty is the pairing of two pieces. First, Section 4 asks which flow stage preserves final timing rankings. It shows that HPWL and pre route STA are weak timing supervision sources, while post GRT reaches average Spearman rho 0.86 against post DRT WNS. Second, PPAPlace makes that post GRT timing signal differentiable with respect to placement coordinates, so the learned objective can guide CoOpt and Refine rather than only rank completed placements.

The submitted ablation already supports this mechanism and directly tests the supervision choice used by LaMPlace [2]. With the same GAT plus CNN model, pre route STA supervision gives Kendall tau 0.13, post CTS gives 0.16, and post GRT gives 0.31. The architecture is fixed, so the improvement comes from supervision fidelity. Attachment 2 separates this submitted predictor ablation from the deployment stage ablation that we will add in the camera ready version.

On cross design generalization, we agree that family overlap must be visible. The revised table will report same family and unseen family rows explicitly. This is detailed in Attachment 1. We also agree that OpenROAD plus Nangate45 is a scope limitation. The label fidelity principle is flow level, but broader claims across technology nodes and commercial flows require regenerating post GRT labels in those environments.

## R2 Lineage, transfer, and PPA framing

R2 asks what the 22 percent WNS and 51 percent TNS headline becomes on the two unseen family Ariane circuits. The answer is **15 percent WNS and 38 percent TNS improvement** over Hier RTLMP. Same family circuits give WNS and TNS ratios of 0.74 and 0.41. The Ariane unseen family group gives 0.85 and 0.62. The all five average is 0.78 and 0.49. Thus the headline is stronger on same family designs, but the unseen family result remains positive on both timing metrics.

The Ariane result also remains competitive with the strongest prior method. On the same two Ariane circuits, Re2MaP [5] averages 0.93 WNS and 0.95 TNS from the submitted table, while PPAPlace CoOpt plus Refine averages 0.85 and 0.62. This does not remove the family overlap concern, but it shows that the unseen family subset still supports the main timing claim.

R2 also notes that Refine alone can fail on ariane133 and that TNS may look weak on the unseen family group. This is an important distinction. The degradation is for **Refine alone**, which starts from a fixed placement topology and can move outside the training distribution. The reported main method is **CoOpt plus Refine**. On ariane133 and ariane136, CoOpt plus Refine gives TNS ratios of 0.42 and 0.82, averaging 0.62. It does not average above 1.0. The warmup and ramp in CoOpt let the global topology adapt before local refinement.

For cross placer transfer, we will soften the interface claim. Refine can operate on any legal DEF placement, but learned ranking accuracy depends on placement distribution. Submitted results show within placer WNS rho 0.77 and cross placer RTLMP WNS rho 0.61, with TNS rho 0.56. This is positive transfer, not full placer independence. Mixed source or small target placer fine tuning is the natural next experiment.

We also agree with R2 that the paper should be framed as timing driven placement with preserved power and routability. Area is fixed by the floorplan. Power varies weakly. In the submitted results, power remains 0.99 to 1.02 times baseline, post DRT reports have zero DRC violations, and routed wirelength remains within 3 percent of default. The camera ready text will use power and routability as preservation evidence, not as the main optimization claim.

R2 also asks for statistical care in the label fidelity study. We agree. The 20 configurations per circuit are controlled sweeps of placement quality, not random samples from all possible placements. We will state this explicitly. We will also add bootstrap confidence intervals for Table 1 and Table 2 after computing them. We do not quote interval widths here because they must be verified from the underlying data. Finally, the Section 6.4 sentence will be corrected to say that the section evaluates surrogate ranking accuracy across placers, not flow stage fidelity across placers.

This wording matters because the paper should not imply that cross placer ranking accuracy proves the flow stage result. The flow stage result comes from Section 4. The cross placer table tests whether the trained surrogate can rank placements from a different placement distribution.

## R3 HPWL, congestion, and overfitting

R3 asks whether the near zero HPWL timing correlation is based only on 10 designs. The broad HPWL timing mismatch is **not based only on our 10 circuits**. ChiPBench evaluates 20 circuits and six AI based placers through a full OpenROAD flow, then reports weak correlation between wirelength style placement metrics and final timing [1]. In its Figure 3, MacroHPWL versus WNS has Pearson correlation minus 0.08.

Our Section 4 is a separate controlled confirmation. It uses 10 circuits and 20 deterministic placement configurations per circuit, producing 200 placements and 800 stage level observations across HPWL, pre CTS, post CTS, and post GRT signals. In this controlled setting, HPWL again has near zero Spearman correlation with post DRT WNS, averaging minus 0.03, while post GRT averages 0.86. The point is not that 10 circuits alone prove a general rule. The point is that our flow stage study agrees with the larger ChiPBench observation and identifies post GRT as the practical supervision stage.

On congestion, the reviewer is right that the manuscript should expose the congestion regime. We will add peak RUDY and global route overflow summaries to the camera ready text after verifying them from OpenROAD logs. We do not include unverified congestion numbers in this rebuttal. The submitted method already includes RUDY congestion as an input feature and evaluates all final results after detailed routing, which reduces the risk that the timing gains come from ignoring routability.

On overfitting, the strongest submitted evidence is not only the five final placements. The predictor ranks 500 held out DREAMPlace placements per test circuit with average WNS rho 0.77, Kendall tau 0.58, and top 5 accuracy 68 percent. It also transfers to RTLMP placements with WNS rho 0.61. These tests include unseen family Ariane circuits and cross placer inputs.

On readability, we will spell out remaining acronyms at first use and add a compact notation note. This is a presentation issue, and the camera ready version can improve it without changing the method or results.

## R4 Practical cost and reporting

R4 recognizes the label fidelity study as a strong standalone contribution. We will emphasize it more clearly. The cost tradeoff is also important. Post GRT labels are expensive, but they are generated offline and amortized. In the submitted setup, post GRT costs 0.20 hours per sample on average, compared with 3.7 hours for post DRT. Generating 5000 labels takes about 63 hours wall clock with parallel OpenROAD evaluation, and model training takes about 45 minutes. Deployment then uses the trained surrogate without rerunning post GRT during optimization.

We will also make baseline provenance clearer. The submitted Table 2 already marks LaMPlace values from its paper, DREAMPlace, AutoDMP, and MaskRegulate from ChiPBench, and Re2MaP ratios computed from its published results. The camera ready caption will make this provenance impossible to miss and will report variance only for rows we reran.

For sensitivity and failure cases, the submitted paper already includes lambda and refinement step sweeps, plus gradient alignment against true post GRT perturbations. It also shows the Refine alone failure on ariane133 and the smaller gain on ariane136. We will discuss these as distribution gap examples. We will add a compact training label count sensitivity check if space allows. We will not claim that result before it is computed.

## Closing

The revised framing is precise. **Label fidelity** explains why post GRT is used. **Lineage stratification** shows where the method generalizes most strongly and where gains are smaller. **Cross placer results** show positive but distribution dependent transfer. **Timing first framing** removes the ambiguity around PPA. We will reflect these clarifications in the camera ready manuscript while keeping all numerical claims tied to submitted or verified evidence.

## References

[1] Zhihai Wang, Zijie Geng, Zhaojie Tu, Jie Wang, Yuxi Qian, Zhexuan Xu, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, Bin Li, Yongdong Zhang, and Feng Wu. “Benchmarking End To End Performance of AI Based Chip Placement Algorithms.” arXiv:2407.15026, 2024. NeurIPS Datasets and Benchmarks Track, 2025.

[2] Zijie Geng, Jie Wang, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, and Feng Wu. “LaMPlace: Learning to Optimize Cross Stage Metrics in Macro Placement.” International Conference on Learning Representations, 2025.

[3] Peiyu Liao, Siting Liu, Zhitang Chen, Wenlong Lv, Yibo Lin, and Bei Yu. “DREAMPlace 4.0: Timing Driven Global Placement with Momentum Based Net Weighting.” Design, Automation and Test in Europe Conference, 2022, pages 939 to 944.

[4] Yunbo Hou, Haoran Ye, Yingxue Zhang, Siyuan Xu, and Guojie Song. “RoutePlacer: An End to End Routability Aware Placer with Graph Neural Network.” Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pages 1085 to 1095.

[5] Yunqi Shi, Xi Lin, Zhiang Wang, Siyuan Xu, Shixiong Kai, Yao Lai, Chengrui Gao, Ke Xue, Mingxuan Yuan, Chao Qian, and Zhi-Hua Zhou. “Re2MaP: Macro Placement by Recursively Prototyping and Packing Tree based Relocating.” arXiv:2511.08054, 2025.
