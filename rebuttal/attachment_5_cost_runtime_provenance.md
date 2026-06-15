# Attachment 5. Cost, runtime, and baseline provenance

Offline cost and deployment runtime.

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
<td>about 24 seconds per configuration</td>
<td>Section 6.1.4</td>
</tr>
<tr>
<td>CoOpt overhead over DREAMPlace</td>
<td>about 12 seconds per configuration</td>
<td>Section 6.1.4</td>
</tr>
<tr>
<td>Refine overhead per refinement run</td>
<td>about 20 seconds</td>
<td>Section 6.1.4</td>
</tr>
<tr>
<td>Total CoOpt plus Refine per test circuit</td>
<td>under 60 seconds, maximum 58 seconds across the five test circuits</td>
<td>Section 6.2</td>
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
</tbody>
</table>

Training corpus size sensitivity for the GAT plus CNN predictor.

<table>
<thead>
<tr>
<th>Corpus fraction</th>
<th>Number of post GRT samples</th>
<th>Kendall tau on held out placements</th>
</tr>
</thead>
<tbody>
<tr>
<td>25 percent</td>
<td>1250</td>
<td>0.22</td>
</tr>
<tr>
<td>50 percent</td>
<td>2500</td>
<td>0.27</td>
</tr>
<tr>
<td>75 percent</td>
<td>3750</td>
<td>0.30</td>
</tr>
<tr>
<td>100 percent</td>
<td>5000</td>
<td>0.31</td>
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
<td>our reruns</td>
<td>mean plus standard deviation over three seeds</td>
</tr>
</tbody>
</table>

Interpretation: post GRT labeling is an offline cost amortized across many deployments. Deployment runtime stays under one minute per circuit. Training corpus saturation around 75 percent of the 5000 sample budget shows the architecture absorbs the available supervision efficiently. Baseline provenance is explicit at the submission level and made even more visible in the camera ready caption.
