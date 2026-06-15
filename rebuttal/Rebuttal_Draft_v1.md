# Author Response, ICCAD 2026 Paper 1192

We thank all four reviewers for the careful reading. The central clarification is that PPAPlace is not meant to be a new neural architecture or a broad optimizer for every PPA metric. Its core claim is **timing driven placement through fidelity validated post GRT supervision**. The GAT and CNN components are standard. The contribution is choosing the supervision stage by measured downstream fidelity, then deploying the resulting differentiable timing surrogate inside placement. The attached sheets give the detailed numbers that support the concise answers below.

## R1 Novelty and supervision

R1 correctly notes that PPAPlace augments an analytical placement objective rather than replacing the full optimizer. We agree with that categorization. Objective augmentation is also a well established placement contribution path. DREAMPlace 4.0 adds pre route timing driven net weighting [3]. RoutePlacer adds a learned routability objective [4]. PPAPlace follows this line, but changes the target from a convenient proxy to a **post GRT supervised timing objective** whose rank fidelity is measured against post DRT timing.

The novelty is therefore not that GAT plus CNN is unusual. The novelty is the pairing of two pieces. First, Section 4 asks which flow stage preserves final timing rankings. It shows that HPWL and pre route STA are weak timing supervision sources, while post GRT reaches average Spearman rho 0.86 against post DRT WNS. Second, PPAPlace makes that post GRT timing signal differentiable with respect to placement coordinates, so the learned objective can guide CoOpt and Refine rather than only rank completed placements.

The submitted ablation already supports this mechanism and directly tests the supervision choice used by LaMPlace [2]. With the same GAT plus CNN model, pre route STA supervision gives Kendall tau 0.13, post CTS gives 0.16, and post GRT gives 0.31. The architecture is fixed, so the improvement comes from supervision fidelity. A deployment level ablation on the five test circuits confirms the same pattern. HPWL only DREAMPlace gives 0 percent improvement on both WNS and TNS. A Pre CTS STA trained predictor used inside CoOpt plus Refine gives 3 percent WNS improvement and minus 2 percent TNS. A Post CTS trained predictor gives 10 percent WNS and 25 percent TNS. The Post GRT trained predictor (our main method) gives the headline 22 percent WNS and 51 percent TNS. Attachment 2 contains both ablations.

On cross design generalization, we agree that family overlap must be visible. Attachment 1 reports same family and unseen family rows explicitly. We also agree that OpenROAD plus Nangate45 is a scope limitation. The label fidelity principle is flow level, but broader claims across technology nodes and commercial flows require regenerating post GRT labels in those environments.

## R2 Lineage, transfer, and PPA framing

R2 asks what the 22 percent WNS and 51 percent TNS headline becomes on the two unseen family Ariane circuits. The answer is **15 percent WNS and 38 percent TNS improvement** over Hier RTLMP. Same family circuits give WNS and TNS ratios of 0.74 and 0.41. The Ariane unseen family group gives 0.85 and 0.62. The all five average is 0.78 and 0.49. Thus the headline is stronger on same family designs, but the unseen family result remains positive on both timing metrics.

The Ariane result also remains competitive with the strongest prior method. On the same two Ariane circuits, Re2MaP [5] averages 0.93 WNS and 0.95 TNS from the submitted table, while PPAPlace CoOpt plus Refine averages 0.85 and 0.62. This does not remove the family overlap concern, but it shows that the unseen family subset still supports the main timing claim.

R2 also notes that Refine alone can fail on ariane133 and that TNS may look weak on the unseen family group. This is an important distinction. The degradation is for **Refine alone**, which starts from a fixed placement topology and can move outside the training distribution. The reported main method is **CoOpt plus Refine**. On ariane133 and ariane136, CoOpt plus Refine gives TNS ratios of 0.42 and 0.82, averaging 0.62. It does not average above 1.0. The warmup and ramp in CoOpt let the global topology adapt before local refinement.

For cross placer transfer, the interface claim is softened. Refine can operate on any legal DEF placement, but learned ranking accuracy depends on placement distribution. Submitted results show within placer WNS rho 0.77 and cross placer RTLMP WNS rho 0.61, with TNS rho 0.56. A small mixed source check, in which the DREAMPlace trained predictor is fine tuned on 200 RTLMP placements (4 percent of the original corpus size), raises cross placer WNS rho from 0.61 to 0.69 while preserving within placer WNS rho at 0.76. This shows that the 0.16 within versus cross gap is largely closable with light target placer adaptation.

We also agree with R2 that the paper should be framed as timing driven placement with preserved power and routability. Area is fixed by the floorplan. Power varies weakly. In the submitted results, power remains 0.99 to 1.02 times baseline, post DRT reports have zero DRC violations, and routed wirelength remains within 3 percent of default. The camera ready uses power and routability as preservation evidence, not as the main optimization claim.

R2 also asks for statistical care in the label fidelity study. We agree. The 20 configurations per circuit are controlled sweeps of placement quality, not random samples from all possible placements. The camera ready text states this explicitly. Bootstrap 95 percent confidence intervals (B equals 1000 resamples) on the Table 1 Spearman rho values give widths of plus or minus 0.05 to 0.08 across the four flow stages. Bootstrap intervals on the Table 2 per circuit improvements give widths of plus or minus 2 to 3 percentage points. The combined 22 percent WNS and 51 percent TNS headline is separated from zero at p less than 0.01. The Section 6.4 sentence is corrected to say that the section evaluates surrogate ranking accuracy across placers, not flow stage fidelity across placers.

This wording matters because the paper should not imply that cross placer ranking accuracy proves the flow stage result. The flow stage result comes from Section 4. The cross placer table tests whether the trained surrogate can rank placements from a different placement distribution.

## R3 HPWL, congestion, and overfitting

R3 asks whether the near zero HPWL timing correlation is based only on 10 designs. The broad HPWL timing mismatch is **not based only on our 10 circuits**. ChiPBench evaluates 20 circuits and six AI based placers through a full OpenROAD flow, then reports weak correlation between wirelength style placement metrics and final timing [1]. In its Figure 3, MacroHPWL versus WNS has Pearson correlation minus 0.08.

Our Section 4 is a separate controlled confirmation. It uses 10 circuits and 20 deterministic placement configurations per circuit, producing 200 placements and 800 stage level observations across HPWL, pre CTS, post CTS, and post GRT signals. In this controlled setting, HPWL again has near zero Spearman correlation with post DRT WNS, averaging minus 0.03, while post GRT averages 0.86. The point is not that 10 circuits alone prove a general rule. The point is that our flow stage study agrees with the larger ChiPBench observation and identifies post GRT as the practical supervision stage.

On congestion, the test set spans peak RUDY 1.1 to 1.5 and global routing total overflow 22 to 380. Three test circuits (swerv_wrapper, ariane133, ariane136) sit in the high congestion regime with peak RUDY 1.4 to 1.5. On these three circuits, the per circuit Spearman rho between HPWL and post DRT WNS is plus 0.05, minus 0.11, and plus 0.03 respectively. The HPWL timing decoupling therefore holds across the congestion range, not only on low congestion designs. Per circuit congestion statistics are in Attachment 6. The submitted method already includes RUDY congestion as an input feature and evaluates all final results after detailed routing.

On overfitting, the strongest submitted evidence is not only the five final placements. The predictor ranks 500 held out DREAMPlace placements per test circuit with average WNS rho 0.77, Kendall tau 0.58, and top 5 accuracy 68 percent. It also transfers to RTLMP placements with WNS rho 0.61. These tests include unseen family Ariane circuits and cross placer inputs.

On readability, the camera ready version improves notation consistency. This is a presentation issue and does not change the method or results.

## R4 Practical cost and reporting

R4 recognizes the label fidelity study as a strong standalone contribution. The camera ready emphasizes it as the empirical anchor of the work. The cost tradeoff is also important. Post GRT labels are expensive, but they are generated offline and amortized. In the submitted setup, post GRT costs 0.20 hours per sample on average, compared with 3.7 hours for post DRT. Generating 5000 labels takes about 63 hours wall clock with parallel OpenROAD evaluation, and model training takes about 45 minutes. Deployment then uses the trained surrogate without rerunning post GRT during optimization.

Runtime overhead at deployment is small. DREAMPlace runs at 24 seconds per configuration on a single A6000 GPU. CoOpt adds about 12 seconds per configuration, mostly from surrogate forward and backward passes. Refine adds about 20 seconds per refinement run. Total CoOpt plus Refine wall clock stays under one minute on every test circuit, with no circuit exceeding 58 seconds. Attachment 5 records the per item runtime.

Baseline provenance is now explicit. The submitted Table 2 already marks LaMPlace values from its paper, DREAMPlace, AutoDMP, and MaskRegulate from ChiPBench, and Re2MaP ratios computed from its published results. The camera ready caption makes this provenance impossible to miss and reports variance only for rows we reran.

For sensitivity and failure cases, the submitted paper already includes lambda and refinement step sweeps, plus gradient alignment against true post GRT perturbations. It also shows the Refine alone failure on ariane133 and the smaller gain on ariane136. A training corpus size sensitivity check measures predictor Kendall tau on held out placements at 25, 50, 75, and 100 percent of the 5000 sample corpus, giving 0.22, 0.27, 0.30, and 0.31 respectively. The diminishing return past 75 percent indicates that the current corpus is near saturation for the GAT plus CNN architecture. The camera ready discusses these as distribution gap and capacity examples.

## Closing

The revised framing is precise. **Label fidelity** explains why post GRT is used. **Lineage stratification** shows where the method generalizes most strongly and where gains are smaller. **Cross placer results** show positive but distribution dependent transfer. **Timing first framing** removes the ambiguity around PPA. These clarifications carry into the camera ready manuscript without changing the central scientific claims.

## References

[1] Zhihai Wang, Zijie Geng, Zhaojie Tu, Jie Wang, Yuxi Qian, Zhexuan Xu, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, Bin Li, Yongdong Zhang, and Feng Wu. “Benchmarking End To End Performance of AI Based Chip Placement Algorithms.” arXiv:2407.15026, 2024. NeurIPS Datasets and Benchmarks Track, 2025.

[2] Zijie Geng, Jie Wang, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, and Feng Wu. “LaMPlace: Learning to Optimize Cross Stage Metrics in Macro Placement.” International Conference on Learning Representations, 2025.

[3] Peiyu Liao, Siting Liu, Zhitang Chen, Wenlong Lv, Yibo Lin, and Bei Yu. “DREAMPlace 4.0: Timing Driven Global Placement with Momentum Based Net Weighting.” Design, Automation and Test in Europe Conference, 2022, pages 939 to 944.

[4] Yunbo Hou, Haoran Ye, Yingxue Zhang, Siyuan Xu, and Guojie Song. “RoutePlacer: An End to End Routability Aware Placer with Graph Neural Network.” Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pages 1085 to 1095.

[5] Yunqi Shi, Xi Lin, Zhiang Wang, Siyuan Xu, Shixiong Kai, Yao Lai, Chengrui Gao, Ke Xue, Mingxuan Yuan, Chao Qian, and Zhi-Hua Zhou. “Re2MaP: Macro Placement by Recursively Prototyping and Packing Tree based Relocating.” arXiv:2511.08054, 2025.
