# Attachment 2. Supervision stage evidence

Predictor level ablation from Table 5. Higher Kendall tau and Top 1 are better.

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

Deployment level ablation. Each predictor is trained with the indicated labels using the same GAT plus CNN architecture and deployed inside CoOpt plus Refine on the five test circuits. Numbers are mean improvement over Hier RTLMP across the test set.

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
<td>HPWL only (DREAMPlace baseline)</td>
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
<td>Post GRT labels (our main method)</td>
<td>22 percent</td>
<td>51 percent</td>
</tr>
</tbody>
</table>

Interpretation: both ablations isolate supervision fidelity at fixed architecture. The predictor level table shows Kendall tau rising from 0.13 (pre route STA) to 0.31 (post GRT). The deployment level table shows end to end CoOpt plus Refine improvement rising from 0 percent (HPWL baseline) to 22 percent WNS and 51 percent TNS (post GRT). Pre route supervision essentially removes the contribution. Supervision stage choice, not architecture, drives the gain.
