# Attachment 5. Cost, runtime, and baseline provenance

Submitted cost and runtime evidence.

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

Baseline provenance from submitted Table 2.

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

Interpretation: post GRT labeling is an offline cost that is amortized across deployments, while the submitted table already distinguishes rerun PPAPlace rows from published baselines.
