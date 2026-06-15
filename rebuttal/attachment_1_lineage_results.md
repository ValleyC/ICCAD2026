# Attachment 1. Lineage stratified timing results

Source: submitted Table 2, PPAPlace CoOpt plus Refine row. Lower ratios are better.

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
<td>swerv_wrap, black_parrot, bp_be</td>
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

Strongest prior method on Group B from submitted Table 2.

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

Arithmetic check: Group A WNS is (0.76 + 0.81 + 0.65) / 3 = 0.74. Group A TNS is (0.59 + 0.15 + 0.48) / 3 = 0.41. Group B WNS is (0.85 + 0.84) / 2 = 0.845, rounded to 0.85. Group B TNS is (0.42 + 0.82) / 2 = 0.62.

Interpretation: same family transfer is stronger, but the unseen family Ariane group remains positive on both WNS and TNS.
