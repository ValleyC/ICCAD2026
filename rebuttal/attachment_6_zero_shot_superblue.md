# Attachment 6. Zero-shot Superblue generalization

Protocol for zero-shot cross-benchmark evaluation. The submitted PPAPlace checkpoint is trained on ChiPBench Nangate45 only and is applied to Superblue without Superblue labels, retraining, or fine tuning.

## Protocol

<table>
<thead>
<tr>
<th>Source training set</th>
<th>Target circuits</th>
<th>Superblue labels used in training</th>
<th>Fine tuning</th>
<th>Reported evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>ChiPBench Nangate45 10 circuits</td>
<td>superblue16, superblue18</td>
<td>0</td>
<td>no</td>
<td>ranking plus Refine</td>
</tr>
</tbody>
</table>

## PPAPlace zero-shot results

<table>
<thead>
<tr>
<th>Circuit</th>
<th>Placements</th>
<th>WNS rho</th>
<th>TNS rho</th>
<th>WNS tau</th>
<th>TNS tau</th>
<th>Top 5 WNS</th>
<th>Refine WNS ratio</th>
<th>Refine TNS ratio</th>
</tr>
</thead>
<tbody>
<tr>
<td>superblue16</td>
<td>[N16]</td>
<td>[SB16_WNS_RHO]</td>
<td>[SB16_TNS_RHO]</td>
<td>[SB16_WNS_TAU]</td>
<td>[SB16_TNS_TAU]</td>
<td>[SB16_TOP5]</td>
<td>[SB16_WNS_RATIO]</td>
<td>[SB16_TNS_RATIO]</td>
</tr>
<tr>
<td>superblue18</td>
<td>[N18]</td>
<td>[SB18_WNS_RHO]</td>
<td>[SB18_TNS_RHO]</td>
<td>[SB18_WNS_TAU]</td>
<td>[SB18_TNS_TAU]</td>
<td>[SB18_TOP5]</td>
<td>[SB18_WNS_RATIO]</td>
<td>[SB18_TNS_RATIO]</td>
</tr>
<tr>
<td>Average</td>
<td>[SB_N]</td>
<td>[SB_WNS_RHO]</td>
<td>[SB_TNS_RHO]</td>
<td>[SB_WNS_TAU]</td>
<td>[SB_TNS_TAU]</td>
<td>[SB_TOP5]</td>
<td>[SB_WNS_RATIO]</td>
<td>[SB_TNS_RATIO]</td>
</tr>
</tbody>
</table>

The checkpoint is the submitted PPAPlace model trained on ChiPBench Nangate45 only. No Superblue labels are used for training or fine tuning.

## Published Superblue baselines from LaMPlace Table 1

Units follow LaMPlace Table 1. TNS is reported in 10^5 ps and WNS is reported in 10^3 ps. Higher is better for both timing metrics.

<table>
<thead>
<tr>
<th>Method</th>
<th>superblue16 TNS</th>
<th>superblue16 WNS</th>
<th>superblue18 TNS</th>
<th>superblue18 WNS</th>
</tr>
</thead>
<tbody>
<tr>
<td>DREAMPlace</td>
<td>-1526.10</td>
<td>-107.05</td>
<td>-751.27</td>
<td>-88.11</td>
</tr>
<tr>
<td>WireMask-EA</td>
<td>-18343.30</td>
<td>-635.89</td>
<td>-406.01</td>
<td>-78.25</td>
</tr>
<tr>
<td>ChiPFormer</td>
<td>-15426.07</td>
<td>-322.05</td>
<td>-378.90</td>
<td>-80.57</td>
</tr>
<tr>
<td>LaMPlace</td>
<td>-1514.73</td>
<td>-36.87</td>
<td>-426.91</td>
<td>-66.93</td>
</tr>
</tbody>
</table>

Baseline note: the published LaMPlace values are context for the Superblue benchmark. A direct method comparison should only be made if PPAPlace uses the same metric definitions and evaluation flow.
