# Attachment 3. HPWL timing evidence

The broad HPWL timing mismatch is not based only on the 10 circuits in PPAPlace.

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

Interpretation: ChipBench supplies the broader benchmark evidence, and PPAPlace supplies the controlled flow stage confirmation that motivates post GRT supervision.
