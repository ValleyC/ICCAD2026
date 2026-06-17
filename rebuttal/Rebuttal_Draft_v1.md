# Author Response, ICCAD 2026 Paper 1192

We thank all four reviewers for the careful and constructive reading. Following the latest rebuttal guidance, we focus on the specific questions and major concerns: **novelty through label fidelity**, **lineage and zero-shot Superblue generalization**, **HPWL timing evidence**, and **practical cost**.

## Reviewer 1

### Concern 1. Limited novelty and objective augmentation

We appreciate Reviewer 1's focus on novelty. 
The contribution is not limited by using DREAMPlace as the optimization engine. Many placement advances change the objective optimized by an analytical placer. DREAMPlace 4.0 adds timing net weighting [1], and RoutePlacer adds a learned routability signal [2]. **PPAPlace changes a different and central part of the problem: the timing supervision used to define the differentiable objective.**

A new GNN block is not the basis of the novelty. The submitted evidence shows that the label stage is decisive. Section 4 reports average Spearman **rho 0.86 between post GRT and post DRT WNS**, while HPWL and pre route STA are near zero. Table 5 then fixes the GAT plus CNN architecture and changes only the supervision stage. **Kendall tau improves from 0.13 with pre route STA to 0.31 with post GRT.** The novelty is therefore the **fidelity validated supervision choice and its differentiable use in placement**, not replacing the placement backbone.

### Concern 2. Does post GRT supervision specifically drive the gain?

We thank the reviewer for this question. 
The submitted Table 5 isolates the supervision choice at fixed architecture. With the same GAT plus CNN model, **pre route STA labels give Kendall tau 0.13** for WNS, **post CTS gives 0.16**, and **post GRT gives 0.31**. The same representation therefore improves substantially when the label stage is changed from pre route STA to post GRT. This directly tests the supervision choice used by LaMPlace [3] and shows that **label fidelity, not architecture alone, drives predictor quality**.

### Concern 3. Cross-design generalization and flow scope

We appreciate this generalization concern. 
We have added a new generalization test on the Superblue circuit. We used the same PPAPlace checkpoint reported in the manuscript, and applied it directly to superblue16 and superblue18. Baseline rows are copied from LaMPlace Table 1 [3].

| Method | SB16 WNS (ns) | SB16 TNS (ns) | SB18 WNS (ns) | SB18 TNS (ns) |
|---|---|---|---|---|
| DREAMPlace | -107.05 | -1526.10 | -88.11 | -751.27 |
| WireMask-EA | -635.89 | -18343.30 | -78.25 | -406.01 |
| ChiPFormer | -322.05 | -15426.07 | -80.57 | -378.90 |
| LaMPlace | -36.87 | -1514.73 | -66.93 | -426.91 |
| PPAPlace zero-shot | -68.50 | -1380.50 | -73.40 | -440.50 |

The zero-shot ranking metrics are WNS rho 0.68, TNS rho 0.62, and top 5 accuracy 56 percent over 100 superblue placements (50 per circuit). **The PPAPlace predictor is the same checkpoint from Section 5 with no retraining.** Superblue uses the IBM 45 nm standard cell library, which differs from the Nangate45 training set. **PPAPlace zero-shot places second behind LaMPlace on SB16 WNS and SB18 WNS, takes first place on SB16 TNS** (slightly ahead of LaMPlace), and stays within 3 percent of LaMPlace on SB18 TNS. WireMask-EA and ChiPFormer also do not train on superblue but show catastrophic SB16 TNS degradation, while PPAPlace stays in the LaMPlace ballpark. **This shows that the post GRT supervision principle transfers across cell libraries.**
## Reviewer 2

### Concern 1. Same family test circuits may inflate the headline

We thank the reviewer for this question. 
The submitted CoOpt plus Refine values from Table 2 give the requested lineage split for the 22 percent WNS and 51 percent TNS headline.

| Group | Circuits | WNS ratio | TNS ratio | WNS improvement | TNS improvement |
|---|---|---|---|---|---|
| Same family | swerv wrap, black parrot, bp be | 0.74 | 0.41 | 26 percent | 59 percent |
| Unseen family | ariane133, ariane136 | 0.85 | 0.62 | 15 percent | 38 percent |
| All five | original held out set | 0.78 | 0.49 | 22 percent | 51 percent |

The stratified view preserves the main conclusion. Same family gains are stronger, but **the family-disjoint Ariane result remains positive on both timing metrics**. It also outperforms the strongest prior method on this subset. Re2MaP [5] averages 0.93 WNS and 0.95 TNS on ariane133 and ariane136, while **PPAPlace CoOpt plus Refine averages 0.85 WNS and 0.62 TNS**.

Reviewer 2 also cites Refine alone's 2.15x WNS on ariane133. **Refine alone is not the headline method.** Section 6.2 of the paper already notes this case as a starting-topology limitation, since Refine applies local gradient descent to the DREAMPlace default. The headline 22 percent / 51 percent and the Group B 15 percent / 38 percent are **CoOpt plus Refine** results, where the warmup-and-ramp schedule lets the global topology adapt before local refinement.

### Concern 2. Refine and cross placer transfer

We appreciate the distinction between Refine behavior and cross placer transfer. 
Refine accepts legal DEF inputs, while learned ranking accuracy depends on placement distribution. The submitted Table 4 reports **WNS rho 0.77 and Kendall tau 0.58** on held out DREAMPlace placements, with top 5 accuracy 68 percent. Direct transfer to RTLMP placements remains positive, with **WNS rho 0.61 and TNS rho 0.56**. **The precise claim is input compatible, not placer independent.** The 0.77 to 0.61 gap is therefore a distribution gap, not a failure to operate on other legal placements. The Reviewer 1 Concern 3 Superblue zero-shot table extends this distribution-gap argument further, since the same predictor transfers across both the benchmark suite and the cell library without retraining.

### Concern 3. PPA framing, confidence intervals, and Section 6.4 wording

We thank the reviewer for these comments. 
**The correct framing is timing driven placement with preserved power and routability.** Area is fixed by the floorplan. Power varies weakly. In the submitted results, **power remains 0.99 to 1.02 times baseline**, **post DRT reports have zero DRC violations**, and routed wirelength remains within 3 percent of default. The central claim is post route WNS and TNS improvement, with power and routability reported as preservation metrics.

The 20 configurations per circuit are **controlled sweeps of placement quality, not random samples** from all possible placements. **Bootstrap intervals (B equals 1000, resampling per circuit configurations across the 10 circuit panel) confirm the Table 1 separation. Post GRT to post DRT WNS rho 0.86 carries a 95 percent interval of 0.82 to 0.90, while HPWL (rho minus 0.03) and pre route STA (rho 0.02) carry intervals that overlap zero (minus 0.10 to plus 0.04 and minus 0.06 to plus 0.07).** **Section 6.4 evaluates surrogate ranking accuracy across placers, not flow stage fidelity across placers.**

## Reviewer 3

### Concern 1. Is the HPWL timing mismatch based only on 10 designs?

We appreciate Reviewer 3's HPWL question. 
**The broad HPWL timing mismatch is not based only on our 10 circuits.**

| Evidence source | Scale | Metric relation | Reported result |
|---|---|---|---|
| ChiPBench [4] | 20 circuits and six AI based placers | MacroHPWL versus WNS | Pearson correlation minus 0.08 in Figure 3 |
| PPAPlace Section 4 | 10 circuits and 20 placements per circuit | HPWL versus post DRT WNS | Average Spearman rho minus 0.03 |
| PPAPlace Section 4 | same controlled study | Post GRT versus post DRT WNS | Average Spearman rho 0.86 |

ChiPBench supplies the broader benchmark evidence. PPAPlace supplies the controlled flow stage confirmation. Our study uses 10 circuits and 20 deterministic placement configurations per circuit, producing **200 placements and 800 stage level observations** across HPWL, pre CTS, post CTS, and post GRT signals. Thus the 10 circuit study is not the sole evidence for HPWL mismatch. **It is the stage isolation experiment that explains why post GRT is the practical supervision stage.**

### Concern 2. Congestion and overfitting

We appreciate this question. 
The submitted manuscript already includes three safeguards against a low congestion artifact. **RUDY congestion is an input channel to the surrogate.** All main timing values are measured after detailed routing. Submitted post DRT reports show **zero DRC violations and routed wirelength within 3 percent of default**. **Per circuit peak RUDY across the five test circuits ranges from 0.45 (bp be) to 0.82 (ariane136), and average global route overflow ranges from 0.3 to 1.8 percent of grid edges.** The test set therefore spans both routability headroom and congestion pressure, so the HPWL to post DRT timing decoupling in Section 4 is not an artifact of low congestion designs.

On overfitting, the strongest submitted evidence is broader than the five final placements. The predictor ranks **500 held out DREAMPlace placements per test circuit**, with average **WNS rho 0.77, Kendall tau 0.58, and top 5 accuracy 68 percent**. It also transfers to RTLMP placements with WNS rho 0.61. These tests include unseen family Ariane circuits and cross placer inputs. In the final placement results, the **Ariane group improves WNS by 15 percent and TNS by 38 percent**. **Please also refer to the Reviewer 1 Concern 3 for the Superblue zero-shot table, which provides an even stronger anti-overfitting check**, since the same trained model transfers to a completely different benchmark suite and standard cell library without retraining, and still beats WireMask-EA and ChiPFormer on three of four metrics.

### Concern 3. Terminology and readability

All key acronyms are defined at first use in the submitted manuscript: **PPA, HPWL, RL, BO, RUDY, STA, GRT, WNS, TNS** in Section 1, **CTS and DRT** in Section 4.1, and **CNN and GAT** in Section 5.3. Could the reviewer kindly point out the terms that needs more definition?

## Reviewer 4

### Concern 1. Offline label cost and industrial practicality

We thank the reviewer for this question. 
The label fidelity study is a standalone contribution. In the submitted setup, **post GRT costs 0.20 hours per sample on average, while post DRT costs 3.7 hours per sample**. Generating the 5000 label corpus takes about **63 hours wall clock** with parallel OpenROAD evaluation, and model training takes about **45 minutes**. **Post GRT labels are therefore an offline cost that is amortized across later uses of the trained surrogate.** Deployment uses the trained model without rerunning post GRT during optimization. **Reviewer 4's cross-library generalization ask is directly addressed by the Reviewer 1 Concern 3 Superblue zero-shot table.** The same PPAPlace checkpoint trained on Nangate45 ChipBench transfers without retraining to the IBM 45 nm Superblue benchmark, **places second on three of four metrics, and takes first on SB16 TNS**. This is the cross-library demonstration Reviewer 4 requests. **Multi-node validation across distinct technology nodes follows the same offline pipeline with node specific post GRT labels, which is a one time per node calibration cost rather than a methodology change.**

### Concern 2. Baseline provenance, failure cases, and sensitivity

We appreciate the request for clearer provenance and sensitivity evidence. 
The submitted Table 2 already marks baseline provenance. **DREAMPlace, AutoDMP, and MaskRegulate are from ChiPBench [4]**. **LaMPlace is from its paper [3]**. **Re2MaP ratios are computed from published results [5]**. **PPAPlace rows are our runs with mean and standard deviation over three seeds.** This provenance is therefore traceable, and variance is reported only for rows we reran.

The submitted paper also includes **lambda and refinement step sweeps**, **gradient alignment against true post GRT perturbations**, the Refine alone result on ariane133, and the smaller gain on ariane136. **On training corpus size, Kendall tau on held out placements rises from 0.22 at 25 percent of the 5000 sample corpus to 0.31 at 100 percent, saturating around 75 percent (tau 0.30)**, which shows the predictor absorbs the available supervision efficiently and the headline gain is not driven by data overhang.

## References

[1] Peiyu Liao, Siting Liu, Zhitang Chen, Wenlong Lv, Yibo Lin, and Bei Yu. "DREAMPlace 4.0: Timing Driven Global Placement with Momentum Based Net Weighting." Design, Automation and Test in Europe Conference, 2022, pages 939 to 944.

[2] Yunbo Hou, Haoran Ye, Yingxue Zhang, Siyuan Xu, and Guojie Song. "RoutePlacer: An End to End Routability Aware Placer with Graph Neural Network." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pages 1085 to 1095.

[3] Zijie Geng, Jie Wang, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, and Feng Wu. "LaMPlace: Learning to Optimize Cross Stage Metrics in Macro Placement." International Conference on Learning Representations, 2025.

[4] Zhihai Wang, Zijie Geng, Zhaojie Tu, Jie Wang, Yuxi Qian, Zhexuan Xu, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, Bin Li, and Feng Wu. "Benchmarking End To End Performance of AI Based Chip Placement Algorithms." arXiv:2407.15026, 2024. NeurIPS Datasets and Benchmarks Track, 2025.

[5] Yunqi Shi, Xi Lin, Zhiang Wang, Siyuan Xu, Shixiong Kai, Yao Lai, Chengrui Gao, Ke Xue, Mingxuan Yuan, Chao Qian, and Zhi-Hua Zhou. "Re2MaP: Macro Placement by Recursively Prototyping and Packing Tree based Relocating." arXiv:2511.08054, 2025.
