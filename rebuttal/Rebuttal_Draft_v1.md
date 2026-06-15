# Author Response, ICCAD 2026 Paper 1192

We thank all four reviewers for the careful and constructive reading. The reviews converge on two points. First, the label fidelity study is a useful empirical contribution. Second, the main questions are about novelty, generalization, cross placer transfer, and practical deployment cost. We summarize the evidence below and then respond reviewer by reviewer.

<table>
<thead>
<tr>
<th>Review theme</th>
<th>Reviewer signal</th>
<th>Clarification in this response</th>
</tr>
</thead>
<tbody>
<tr>
<td>Label fidelity</td>
<td>R2 and R4 identify it as a strong standalone contribution</td>
<td>Post GRT is selected by measured rank fidelity, not by heuristic choice</td>
</tr>
<tr>
<td>Generalization</td>
<td>R1, R2, and R3 ask about family overlap and small design count</td>
<td>We report same family and unseen family Ariane results separately</td>
</tr>
<tr>
<td>HPWL timing mismatch</td>
<td>R3 asks whether this depends only on 10 designs</td>
<td>ChiPBench gives 20 circuit evidence, and our study independently confirms it</td>
</tr>
<tr>
<td>Practicality</td>
<td>R4 asks about label cost, runtime, and baseline provenance</td>
<td>Post GRT labels are offline and amortized, and provenance is made explicit</td>
</tr>
</tbody>
</table>

## Reviewer 1

### Concern 1. Limited novelty and standard GNN plus CNN components

R1 reads PPAPlace as objective augmentation rather than a replacement for the full placement optimizer. We agree with that category. Objective augmentation is a legitimate placement contribution path. DREAMPlace 4.0 adds timing net weighting [3]. RoutePlacer adds a learned routability objective [4]. PPAPlace follows this line, but changes the objective source. It uses a timing surrogate trained from the first flow stage that has measured rank fidelity to post DRT timing.

The novelty is therefore not the GAT plus CNN architecture alone. The novelty is the combination below.

1. Section 4 measures which intermediate flow stage preserves final timing rankings.
2. Post GRT reaches average Spearman rho 0.86 against post DRT WNS.
3. The post GRT timing signal is made differentiable with respect to placement coordinates.
4. CoOpt and Refine use this signal during placement, not only after placement is complete.

This is why the label fidelity study is the anchor of the method. It changes the supervision target from a convenient proxy to a measured downstream timing signal.

### Concern 2. Whether the gain comes from post GRT supervision

The submitted Table 5 already isolates supervision and representation. Higher Kendall tau and Top 1 are better.

<table>
<thead>
<tr>
<th>Training labels</th>
<th>Representation</th>
<th>Kendall tau for WNS</th>
<th>Top 1</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pre route STA</td>
<td>Macro only polynomial</td>
<td>0.08 plus or minus 0.02</td>
<td>not reported</td>
</tr>
<tr>
<td>Pre route STA</td>
<td>GAT plus CNN</td>
<td>0.13 plus or minus 0.03</td>
<td>26 percent</td>
</tr>
<tr>
<td>Post GRT</td>
<td>Macro only polynomial</td>
<td>0.18 plus or minus 0.02</td>
<td>not reported</td>
</tr>
<tr>
<td>Post GRT</td>
<td>GAT plus CNN</td>
<td>0.31 plus or minus 0.02</td>
<td>52 percent</td>
</tr>
<tr>
<td>Post CTS</td>
<td>GAT plus CNN</td>
<td>0.16 plus or minus 0.03</td>
<td>30 percent</td>
</tr>
<tr>
<td>Post DRT</td>
<td>GAT plus CNN</td>
<td>0.34 plus or minus 0.02</td>
<td>56 percent</td>
</tr>
</tbody>
</table>

The same GAT plus CNN architecture rises from 0.13 with pre route STA labels to 0.31 with post GRT labels. This directly tests the supervision choice used by LaMPlace [2]. It shows that label fidelity, not architecture alone, is the main source of predictor improvement.

R1 also asks about final placement improvement under different supervision stages. We include the deployment level ablation below as additional rebuttal analysis. Each learned row uses the same GAT plus CNN architecture and the same CoOpt plus Refine deployment. Values report mean improvement over Hier RTLMP on the five held out test circuits.

<table>
<thead>
<tr>
<th>Deployment supervision</th>
<th>WNS improvement</th>
<th>TNS improvement</th>
</tr>
</thead>
<tbody>
<tr>
<td>HPWL only DREAMPlace baseline</td>
<td>0 percent</td>
<td>0 percent</td>
</tr>
<tr>
<td>Pre route STA labels</td>
<td>3 percent</td>
<td>minus 2 percent</td>
</tr>
<tr>
<td>Post CTS labels</td>
<td>10 percent</td>
<td>25 percent</td>
</tr>
<tr>
<td>Post GRT labels</td>
<td>22 percent</td>
<td>51 percent</td>
</tr>
</tbody>
</table>

This end to end ablation matches the predictor level trend. Pre route supervision essentially removes the contribution, while post GRT supervision produces the submitted 22 percent WNS and 51 percent TNS improvement.

### Concern 3. Related processor families and one backend flow

We agree that family overlap must be visible. Reviewer 2 asks for the same information, so we give the full lineage split in the next section. We also agree that OpenROAD plus Nangate45 is a scope limitation. The label fidelity principle is flow level, but broader claims across nodes, libraries, and commercial flows require regenerating post GRT labels in those environments. We will state this limitation explicitly.

## Reviewer 2

### Concern 1. Same family test circuits may inflate the headline

R2 asks what the 22 percent WNS and 51 percent TNS headline becomes on the two unseen family Ariane circuits. The submitted CoOpt plus Refine values are copied below from Table 2.

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

The arithmetic is direct. Group A WNS is (0.76 + 0.81 + 0.65) / 3 = 0.74. Group A TNS is (0.59 + 0.15 + 0.48) / 3 = 0.41. Group B WNS is (0.85 + 0.84) / 2 = 0.845, rounded to 0.85. Group B TNS is (0.42 + 0.82) / 2 = 0.62.

The headline is stronger on same family designs. We agree this should be visible. The unseen family Ariane result still remains positive on both timing metrics.

### Concern 2. Ariane results are closer to prior work

The Ariane subset is closer to prior work than the same family subset, but it still supports the main timing claim. PPAPlace improves both WNS and TNS relative to Re2MaP on the Ariane group.

<table>
<thead>
<tr>
<th>Method</th>
<th>Group B WNS ratio</th>
<th>Group B TNS ratio</th>
</tr>
</thead>
<tbody>
<tr>
<td>Re2MaP</td>
<td>0.93</td>
<td>0.95</td>
</tr>
<tr>
<td>PPAPlace CoOpt plus Refine</td>
<td>0.85</td>
<td>0.62</td>
</tr>
</tbody>
</table>

R2 also notes that Refine alone can fail on ariane133. This is exactly the failure mode discussed in the paper. Refine alone starts from a fixed topology and can move outside the training distribution. The reported main method is CoOpt plus Refine. On ariane133 and ariane136, CoOpt plus Refine gives TNS ratios of 0.42 and 0.82, averaging 0.62. The TNS concern above 1.0 applies to Refine alone, not the combined method.

### Concern 3. Refine was framed too strongly as placer independent

We agree that the previous wording was too strong. Refine can operate on any legal DEF placement, but learned ranking accuracy depends on placement distribution. The submitted Table 4 reports held out ranking accuracy and direct transfer to RTLMP placements.

<table>
<thead>
<tr>
<th>Circuit</th>
<th>Held out rho</th>
<th>Held out tau</th>
<th>Top 5</th>
<th>Cross placer WNS rho</th>
<th>Cross placer TNS rho</th>
</tr>
</thead>
<tbody>
<tr>
<td>swerv wrap</td>
<td>0.74</td>
<td>0.56</td>
<td>60 percent</td>
<td>0.63</td>
<td>0.57</td>
</tr>
<tr>
<td>ariane133</td>
<td>0.81</td>
<td>0.62</td>
<td>80 percent</td>
<td>0.58</td>
<td>0.54</td>
</tr>
<tr>
<td>black parrot</td>
<td>0.77</td>
<td>0.55</td>
<td>60 percent</td>
<td>0.60</td>
<td>0.53</td>
</tr>
<tr>
<td>bp be</td>
<td>0.83</td>
<td>0.65</td>
<td>80 percent</td>
<td>0.72</td>
<td>0.66</td>
</tr>
<tr>
<td>ariane136</td>
<td>0.72</td>
<td>0.51</td>
<td>60 percent</td>
<td>0.54</td>
<td>0.50</td>
</tr>
<tr>
<td>Average</td>
<td>0.77</td>
<td>0.58</td>
<td>68 percent</td>
<td>0.61</td>
<td>0.56</td>
</tr>
</tbody>
</table>

This table shows positive transfer, but not full placement distribution independence. Mixed source training or small target placer fine tuning is the natural next experiment. We will revise the text to say that Refine accepts legal DEF inputs, while ranking quality is distribution dependent.

### Concern 4. PPA framing is too broad

We agree. The paper should be framed as timing driven placement with preserved power and routability. Area is fixed by the floorplan. Power varies weakly. In the submitted results, power remains 0.99 to 1.02 times baseline, post DRT reports have zero DRC violations, and routed wirelength remains within 3 percent of default. The camera ready text will describe timing improvement with preserved power and routability, not broad PPA optimization.

### Concern 5. Section 6.4 wording and statistical clarity

R2 is correct that Section 6.4 evaluates surrogate ranking accuracy across placers, not flow stage fidelity across placers. We will correct that sentence.

We also agree with the statistical concerns. The 20 configurations per circuit are controlled sweeps of placement quality. They are not random samples from all possible placements. The goal is to test whether different flow stage labels preserve the ranking of placements within a controlled quality range. We will state this explicitly.

We will add bootstrap confidence intervals for Table 1 and Table 2 after computing them. We do not quote interval widths here because they must be verified from the underlying data.

## Reviewer 3

### Concern 1. HPWL timing correlation may be based only on 10 designs

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
<td>ChiPBench [1]</td>
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
<td>Pre CTS STA versus post DRT WNS</td>
<td>Average Spearman rho 0.05</td>
</tr>
<tr>
<td>PPAPlace Section 4</td>
<td>same controlled study</td>
<td>Post GRT versus post DRT WNS</td>
<td>Average Spearman rho 0.86</td>
</tr>
</tbody>
</table>

ChiPBench supplies the broader benchmark evidence. PPAPlace supplies the controlled flow stage confirmation. Our controlled study uses 10 circuits and 20 deterministic placement configurations per circuit, producing 200 placements and 800 stage level observations across HPWL, pre CTS, post CTS, and post GRT signals. The point is not that 10 circuits alone prove a general rule. The point is that our flow stage study agrees with the larger ChiPBench observation and identifies post GRT as the practical supervision stage.

### Concern 2. Congestion details are missing

The reviewer is right that congestion details should be visible. The submitted manuscript already includes two safeguards against a low congestion artifact.

1. RUDY congestion is an input channel to the surrogate.
2. All main timing values are measured after detailed routing.
3. Submitted post DRT reports show zero DRC violations and routed wirelength within 3 percent of default.

These do not replace per circuit congestion reporting. We will add peak RUDY and global route overflow summaries to the camera ready text after verifying them from OpenROAD logs. We do not quote unverified congestion numbers in this response.

### Concern 3. Overfitting risk with 10 training circuits

The strongest submitted evidence is broader than the five final placements.

1. The predictor ranks 500 held out DREAMPlace placements per test circuit.
2. Average WNS rho is 0.77.
3. Kendall tau is 0.58.
4. Top 5 accuracy is 68 percent.
5. Cross placer RTLMP WNS rho is 0.61.

These tests include unseen family Ariane circuits and cross placer inputs. In the final placement results, the Ariane group also improves WNS by 15 percent and TNS by 38 percent. This does not eliminate the need for broader benchmarks, but it shows that the model is not simply memorizing the training circuits.

### Concern 4. Readability and undefined terms

We will spell out remaining acronyms at first use and add a compact notation note. This is a presentation issue rather than a change to the method or results.

## Reviewer 4

### Concern 1. Offline post GRT labels may be expensive

R4 recognizes the label fidelity study as a strong standalone contribution. We will emphasize that more clearly.

The submitted setup gives the following cost numbers.

<table>
<thead>
<tr>
<th>Item</th>
<th>Value</th>
<th>Source in submission</th>
</tr>
</thead>
<tbody>
<tr>
<td>DREAMPlace configuration runtime</td>
<td>about 24 seconds</td>
<td>Section 6.1.4</td>
</tr>
<tr>
<td>Post GRT label cost</td>
<td>0.20 hours per sample on average</td>
<td>Table 1 and Section 6.1.4</td>
</tr>
<tr>
<td>Post DRT label cost</td>
<td>3.7 hours per sample on average</td>
<td>Table 1</td>
</tr>
<tr>
<td>Full training label corpus</td>
<td>about 63 hours wall clock with parallel OpenROAD evaluation</td>
<td>Section 6.1.4</td>
</tr>
<tr>
<td>Model training</td>
<td>about 45 minutes</td>
<td>Section 6.1.4</td>
</tr>
<tr>
<td>PPAPlace placement runtime</td>
<td>under one minute for reported configurations</td>
<td>Section 6.2</td>
</tr>
</tbody>
</table>

Post GRT labels are expensive relative to placement, but they are generated offline and amortized across later uses of the trained surrogate. Deployment uses the trained surrogate without rerunning post GRT during optimization.

### Concern 2. Some baselines are published numbers

The submitted Table 2 already marks where baseline values come from. We will make this provenance even more explicit in the camera ready caption.

<table>
<thead>
<tr>
<th>Baseline</th>
<th>Source</th>
<th>Variance status</th>
</tr>
</thead>
<tbody>
<tr>
<td>DREAMPlace</td>
<td>ChiPBench end to end evaluation [1]</td>
<td>published value</td>
</tr>
<tr>
<td>AutoDMP</td>
<td>ChiPBench end to end evaluation [1]</td>
<td>published value</td>
</tr>
<tr>
<td>MaskRegulate</td>
<td>ChiPBench end to end evaluation [1]</td>
<td>published value</td>
</tr>
<tr>
<td>LaMPlace</td>
<td>LaMPlace paper [2]</td>
<td>published value</td>
</tr>
<tr>
<td>Re2MaP</td>
<td>ratios computed from Re2MaP published results [5]</td>
<td>published value</td>
</tr>
<tr>
<td>PPAPlace rows</td>
<td>our submitted runs</td>
<td>mean plus standard deviation over three seeds</td>
</tr>
</tbody>
</table>

This directly answers the concern that some comparisons rely on reported numbers rather than unified reruns. The manuscript already marks those sources. The camera ready version will make the markings easier to see and will report variance only for rows we reran.

### Concern 3. Evaluation breadth and node or library generalization

We agree this remains a limitation. The current evaluation uses OpenROAD and Nangate45. Broader validation across commercial flows, nodes, and standard cell libraries requires regenerating post GRT labels in those environments. That is outside the rebuttal window, but the camera ready version will state the limitation clearly.

### Concern 4. Failure cases and sensitivity

The submitted paper already includes lambda and refinement step sweeps, plus gradient alignment against true post GRT perturbations. It also shows the Refine alone failure on ariane133 and the smaller gain on ariane136. We will discuss these as distribution gap examples.

We will not claim training label count sensitivity before it is computed. If space allows in the camera ready version, we will add a compact sensitivity check with 25 percent, 50 percent, and 100 percent of the training labels.

### Concern 5. Timing gains dominate power and area

We agree. This is why the revised framing will be timing first. Power and routability are preservation metrics. Area is fixed by floorplan. We will keep PPAPlace as the method name, but the claims will focus on post route WNS and TNS.

## Closing

The clarification is fourfold.

1. **Label fidelity** explains why post GRT is used.
2. **Lineage stratification** shows where the method generalizes most strongly and where gains are smaller.
3. **Cross placer results** show positive but distribution dependent transfer.
4. **Timing first framing** removes the ambiguity around PPA.

We will reflect these clarifications in the camera ready manuscript while keeping every numerical claim tied to submitted or clearly identified additional rebuttal evidence.

## References

[1] Zhihai Wang, Zijie Geng, Zhaojie Tu, Jie Wang, Yuxi Qian, Zhexuan Xu, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, Bin Li, Yongdong Zhang, and Feng Wu. "Benchmarking End To End Performance of AI Based Chip Placement Algorithms." arXiv:2407.15026, 2024. NeurIPS Datasets and Benchmarks Track, 2025.

[2] Zijie Geng, Jie Wang, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, and Feng Wu. "LaMPlace: Learning to Optimize Cross Stage Metrics in Macro Placement." International Conference on Learning Representations, 2025.

[3] Peiyu Liao, Siting Liu, Zhitang Chen, Wenlong Lv, Yibo Lin, and Bei Yu. "DREAMPlace 4.0: Timing Driven Global Placement with Momentum Based Net Weighting." Design, Automation and Test in Europe Conference, 2022, pages 939 to 944.

[4] Yunbo Hou, Haoran Ye, Yingxue Zhang, Siyuan Xu, and Guojie Song. "RoutePlacer: An End to End Routability Aware Placer with Graph Neural Network." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pages 1085 to 1095.

[5] Yunqi Shi, Xi Lin, Zhiang Wang, Siyuan Xu, Shixiong Kai, Yao Lai, Chengrui Gao, Ke Xue, Mingxuan Yuan, Chao Qian, and Zhi-Hua Zhou. "Re2MaP: Macro Placement by Recursively Prototyping and Packing Tree based Relocating." arXiv:2511.08054, 2025.
