# Attachment 6. Per circuit congestion summary

Source: DREAMPlace and OpenROAD GRT logs for the five test circuits. Peak RUDY measures peak local wire density (values above 1.0 indicate routing demand exceeding capacity). Global routing total overflow is the sum of overflowed grid cells across the chip.

<table>
<thead>
<tr>
<th>Test circuit</th>
<th>Peak RUDY</th>
<th>GR total overflow</th>
<th>Placement utilization</th>
<th>Congestion regime</th>
</tr>
</thead>
<tbody>
<tr>
<td>bp_be</td>
<td>1.1</td>
<td>22</td>
<td>0.68</td>
<td>Low to moderate</td>
</tr>
<tr>
<td>black_parrot</td>
<td>1.2</td>
<td>50</td>
<td>0.71</td>
<td>Moderate</td>
</tr>
<tr>
<td>swerv_wrapper</td>
<td>1.4</td>
<td>245</td>
<td>0.78</td>
<td>High</td>
</tr>
<tr>
<td>ariane133</td>
<td>1.5</td>
<td>380</td>
<td>0.82</td>
<td>High</td>
</tr>
<tr>
<td>ariane136</td>
<td>1.4</td>
<td>312</td>
<td>0.80</td>
<td>High</td>
</tr>
</tbody>
</table>

Per circuit Spearman rho between HPWL and post DRT WNS on the high congestion subset.

<table>
<thead>
<tr>
<th>Circuit</th>
<th>Spearman rho (HPWL versus post DRT WNS)</th>
</tr>
</thead>
<tbody>
<tr>
<td>swerv_wrapper</td>
<td>plus 0.05</td>
</tr>
<tr>
<td>ariane133</td>
<td>minus 0.11</td>
</tr>
<tr>
<td>ariane136</td>
<td>plus 0.03</td>
</tr>
</tbody>
</table>

Interpretation: the test set spans low to high congestion. On the three high congestion circuits, HPWL still has near zero correlation with post DRT timing. The HPWL timing decoupling in Section 4 is not an artifact of testing only low congestion designs.
