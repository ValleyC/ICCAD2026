# Attachment 2. Supervision stage evidence

Submitted predictor ablation from Table 5. Higher Kendall tau and top 1 are better.

<table>
<thead>
<tr>
<th>Training labels</th>
<th>Representation</th>
<th>Kendall tau for WNS</th>
<th>Top 1</th>
<th>Status</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pre route STA</td>
<td>Macro only polynomial</td>
<td>0.08 plus or minus 0.02</td>
<td>not reported</td>
<td>submitted</td>
</tr>
<tr>
<td>Pre route STA</td>
<td>GAT plus CNN</td>
<td>0.13 plus or minus 0.03</td>
<td>26 percent</td>
<td>submitted</td>
</tr>
<tr>
<td>Post GRT</td>
<td>Macro only polynomial</td>
<td>0.18 plus or minus 0.02</td>
<td>not reported</td>
<td>submitted</td>
</tr>
<tr>
<td>Post GRT</td>
<td>GAT plus CNN</td>
<td>0.31 plus or minus 0.02</td>
<td>52 percent</td>
<td>submitted</td>
</tr>
<tr>
<td>Post CTS</td>
<td>GAT plus CNN</td>
<td>0.16 plus or minus 0.03</td>
<td>30 percent</td>
<td>submitted label stage check</td>
</tr>
<tr>
<td>Post DRT</td>
<td>GAT plus CNN</td>
<td>0.34 plus or minus 0.02</td>
<td>56 percent</td>
<td>submitted upper bound</td>
</tr>
</tbody>
</table>

Planned camera ready deployment check.

<table>
<thead>
<tr>
<th>Deployment supervision</th>
<th>WNS improvement</th>
<th>TNS improvement</th>
<th>Status</th>
</tr>
</thead>
<tbody>
<tr>
<td>HPWL only</td>
<td>pending</td>
<td>pending</td>
<td>planned camera ready experiment</td>
</tr>
<tr>
<td>Pre route STA labels</td>
<td>pending</td>
<td>pending</td>
<td>planned camera ready experiment</td>
</tr>
<tr>
<td>Post CTS labels</td>
<td>pending</td>
<td>pending</td>
<td>planned camera ready experiment</td>
</tr>
<tr>
<td>Post GRT labels</td>
<td>22 percent</td>
<td>51 percent</td>
<td>submitted main method</td>
</tr>
</tbody>
</table>

Interpretation: the submitted ablation already isolates supervision fidelity at the predictor level, while the deployment stage ablation should not be quoted until rerun values are verified.
