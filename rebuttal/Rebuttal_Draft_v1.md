# Author Response, ICCAD 2026 Paper 1192

We thank all four reviewers for the careful and constructive reading. Following the latest rebuttal guidance, we focus on the specific questions and major concerns: novelty through label fidelity, lineage and zero-shot Superblue generalization, HPWL timing evidence, and practical cost.

## Reviewer 1

### Concern 1. Limited novelty and objective augmentation

The contribution is not limited by using DREAMPlace as the optimization engine. Many placement advances change the objective optimized by an analytical placer. DREAMPlace 4.0 adds timing net weighting [1], and RoutePlacer adds a learned routability signal [2]. PPAPlace changes a different and central part of the problem: the timing supervision used to define the differentiable objective.

A new GNN block is not the basis of the novelty. The submitted evidence shows that the label stage is decisive. Section 4 reports average Spearman rho 0.86 between post GRT and post DRT WNS, while HPWL and pre route STA are near zero. Table 5 then fixes the GAT plus CNN architecture and changes only the supervision stage. Kendall tau improves from 0.13 with pre route STA to 0.31 with post GRT. The novelty is therefore the fidelity validated supervision choice and its differentiable use in placement, not replacing the placement backbone.

### Concern 2. Does post GRT supervision specifically drive the gain?

The submitted Table 5 isolates the supervision choice at fixed architecture. With the same GAT plus CNN model, pre route STA labels give Kendall tau 0.13 for WNS, post CTS gives 0.16, and post GRT gives 0.31. The same representation therefore improves substantially when the label stage is changed from pre route STA to post GRT. This directly tests the supervision choice used by LaMPlace [3] and shows that label fidelity, not architecture alone, drives predictor quality.

### Concern 3. Cross-design generalization and flow scope

Generalization is evaluated at two levels. Ariane tests family-disjoint transfer within ChiPBench, where CoOpt plus Refine reaches 0.85 WNS and 0.62 TNS ratios. The new Superblue check keeps the submitted checkpoint fixed, uses no Superblue labels, and applies it directly to superblue16 and superblue18. Baseline rows are copied from LaMPlace Table 1 [3]. Units follow that table. Higher timing values are better.

<table>
<thead>
<tr>
<th>Method</th>
<th>SB16 WNS</th>
<th>SB16 TNS</th>
<th>SB18 WNS</th>
<th>SB18 TNS</th>
</tr>
</thead>
<tbody>
<tr>
<td>DREAMPlace</td>
<td>-107.05</td>
<td>-1526.10</td>
<td>-88.11</td>
<td>-751.27</td>
</tr>
<tr>
<td>WireMask-EA</td>
<td>-635.89</td>
<td>-18343.30</td>
<td>-78.25</td>
<td>-406.01</td>
</tr>
<tr>
<td>ChiPFormer</td>
<td>-322.05</td>
<td>-15426.07</td>
<td>-80.57</td>
<td>-378.90</td>
</tr>
<tr>
<td>LaMPlace</td>
<td>-36.87</td>
<td>-1514.73</td>
<td>-66.93</td>
<td>-426.91</td>
</tr>
<tr>
<td>PPAPlace zero-shot</td>
<td>[SB16_WNS]</td>
<td>[SB16_TNS]</td>
<td>[SB18_WNS]</td>
<td>[SB18_TNS]</td>
</tr>
</tbody>
</table>

The zero-shot ranking metrics are WNS rho [SB_WNS_RHO], TNS rho [SB_TNS_RHO], and top 5 [SB_TOP5] over [SB_N] placements. Superblue is a different benchmark and design distribution from the ChiPBench Nangate45 training set. Full multi-node validation still requires regenerated labels.

## Reviewer 2

### Concern 1. Same family test circuits may inflate the headline

The submitted CoOpt plus Refine values from Table 2 give the requested lineage split for the 22 percent WNS and 51 percent TNS headline.

<table>
<thead>
<tr>
<th>Group</th>
<th>Circuits</th>
<th>WNS ratio</th>
<th>TNS ratio</th>
<th>WNS improvement</th>
<th>TNS improvement</th>
</tr>
</thead>
<tbody>
<tr>
<td>Same family</td>
<td>swerv wrap, black parrot, bp be</td>
<td>0.74</td>
<td>0.41</td>
<td>26 percent</td>
<td>59 percent</td>
</tr>
<tr>
<td>Unseen family</td>
<td>ariane133, ariane136</td>
<td>0.85</td>
<td>0.62</td>
<td>15 percent</td>
<td>38 percent</td>
</tr>
<tr>
<td>All five</td>
<td>original held out set</td>
<td>0.78</td>
<td>0.49</td>
<td>22 percent</td>
<td>51 percent</td>
</tr>
</tbody>
</table>

The stratified view preserves the main conclusion. Same family gains are stronger, but the family-disjoint Ariane result remains positive on both timing metrics. It also outperforms the strongest prior method on this subset. Re2MaP [5] averages 0.93 WNS and 0.95 TNS on ariane133 and ariane136, while PPAPlace CoOpt plus Refine averages 0.85 WNS and 0.62 TNS.

The Refine-alone result on ariane133 is a distribution-gap case already discussed in the paper. Refine alone starts from a fixed topology and can move outside the training distribution. The reported main method is CoOpt plus Refine. On ariane133 and ariane136, CoOpt plus Refine gives TNS ratios of 0.42 and 0.82, averaging 0.62. The TNS concern above 1.0 applies to Refine alone, not the combined method.

### Concern 2. Refine and cross placer transfer

Refine accepts legal DEF inputs, while learned ranking accuracy depends on placement distribution. The submitted Table 4 reports WNS rho 0.77 and Kendall tau 0.58 on held out DREAMPlace placements, with top 5 accuracy 68 percent. Direct transfer to RTLMP placements remains positive, with WNS rho 0.61 and TNS rho 0.56. The precise claim is input compatible, not placer independent. The 0.77 to 0.61 gap is therefore a distribution gap, not a failure to operate on other legal placements.

### Concern 3. PPA framing, confidence intervals, and Section 6.4 wording

The correct framing is timing driven placement with preserved power and routability. Area is fixed by the floorplan. Power varies weakly. In the submitted results, power remains 0.99 to 1.02 times baseline, post DRT reports have zero DRC violations, and routed wirelength remains within 3 percent of default. The central claim is post route WNS and TNS improvement, with power and routability reported as preservation metrics.

The 20 configurations per circuit are controlled sweeps of placement quality, not random samples from all possible placements. Bootstrap intervals are not quoted in the rebuttal because interval widths must be verified from the logs. Section 6.4 evaluates surrogate ranking accuracy across placers, not flow stage fidelity across placers.

## Reviewer 3

### Concern 1. Is the HPWL timing mismatch based only on 10 designs?

The broad HPWL timing mismatch is not based only on our 10 circuits.

<table>
<thead>
<tr>
<th>Evidence source</th>
<th>Scale</th>
<th>Metric relation</th>
<th>Reported result</th>
</tr>
</thead>
<tbody>
<tr>
<td>ChiPBench [4]</td>
<td>20 circuits and six AI based placers</td>
<td>MacroHPWL versus WNS</td>
<td>Pearson correlation minus 0.08 in Figure 3</td>
</tr>
<tr>
<td>PPAPlace Section 4</td>
<td>10 circuits and 20 placements per circuit</td>
<td>HPWL versus post DRT WNS</td>
<td>Average Spearman rho minus 0.03</td>
</tr>
<tr>
<td>PPAPlace Section 4</td>
<td>same controlled study</td>
<td>Post GRT versus post DRT WNS</td>
<td>Average Spearman rho 0.86</td>
</tr>
</tbody>
</table>

ChiPBench supplies the broader benchmark evidence. PPAPlace supplies the controlled flow stage confirmation. Our study uses 10 circuits and 20 deterministic placement configurations per circuit, producing 200 placements and 800 stage level observations across HPWL, pre CTS, post CTS, and post GRT signals. Thus the 10 circuit study is not the sole evidence for HPWL mismatch. It is the stage isolation experiment that explains why post GRT is the practical supervision stage.

### Concern 2. Congestion and overfitting

The submitted manuscript already includes three safeguards against a low congestion artifact. RUDY congestion is an input channel to the surrogate. All main timing values are measured after detailed routing. Submitted post DRT reports show zero DRC violations and routed wirelength within 3 percent of default. Together these rule out a routability artifact in the main result. Peak RUDY and global route overflow are the appropriate per circuit summaries after verification from OpenROAD logs.

On overfitting, the strongest submitted evidence is broader than the five final placements. The predictor ranks 500 held out DREAMPlace placements per test circuit, with average WNS rho 0.77, Kendall tau 0.58, and top 5 accuracy 68 percent. It also transfers to RTLMP placements with WNS rho 0.61. These tests include unseen family Ariane circuits and cross placer inputs. In the final placement results, the Ariane group improves WNS by 15 percent and TNS by 38 percent.

## Reviewer 4

### Concern 1. Offline label cost and industrial practicality

The label fidelity study is a standalone contribution. In the submitted setup, post GRT costs 0.20 hours per sample on average, while post DRT costs 3.7 hours per sample. Generating the 5000 label corpus takes about 63 hours wall clock with parallel OpenROAD evaluation, and model training takes about 45 minutes. Post GRT labels are therefore an offline cost that is amortized across later uses of the trained surrogate. Deployment uses the trained model without rerunning post GRT during optimization. The R1 Superblue table gives the cross-benchmark transfer check. Broader node or library generalization still requires regenerated labels in those environments.

### Concern 2. Baseline provenance, failure cases, and sensitivity

The submitted Table 2 already marks baseline provenance. DREAMPlace, AutoDMP, and MaskRegulate are from ChiPBench [4]. LaMPlace is from its paper [3]. Re2MaP ratios are computed from published results [5]. PPAPlace rows are our runs with mean and standard deviation over three seeds. This provenance is therefore traceable, and variance is reported only for rows we reran.

The submitted paper also includes lambda and refinement step sweeps, gradient alignment against true post GRT perturbations, the Refine alone result on ariane133, and the smaller gain on ariane136. These are concrete failure-case and sensitivity evidence. No training-label-count sensitivity claim is made before it is computed.

## References

[1] Peiyu Liao, Siting Liu, Zhitang Chen, Wenlong Lv, Yibo Lin, and Bei Yu. "DREAMPlace 4.0: Timing Driven Global Placement with Momentum Based Net Weighting." Design, Automation and Test in Europe Conference, 2022, pages 939 to 944.

[2] Yunbo Hou, Haoran Ye, Yingxue Zhang, Siyuan Xu, and Guojie Song. "RoutePlacer: An End to End Routability Aware Placer with Graph Neural Network." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pages 1085 to 1095.

[3] Zijie Geng, Jie Wang, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, and Feng Wu. "LaMPlace: Learning to Optimize Cross Stage Metrics in Macro Placement." International Conference on Learning Representations, 2025.

[4] Zhihai Wang, Zijie Geng, Zhaojie Tu, Jie Wang, Yuxi Qian, Zhexuan Xu, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, Bin Li, and Feng Wu. "Benchmarking End To End Performance of AI Based Chip Placement Algorithms." arXiv:2407.15026, 2024. NeurIPS Datasets and Benchmarks Track, 2025.

[5] Yunqi Shi, Xi Lin, Zhiang Wang, Siyuan Xu, Shixiong Kai, Yao Lai, Chengrui Gao, Ke Xue, Mingxuan Yuan, Chao Qian, and Zhi-Hua Zhou. "Re2MaP: Macro Placement by Recursively Prototyping and Packing Tree based Relocating." arXiv:2511.08054, 2025.
