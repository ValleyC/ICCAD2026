# Attachment 4. Generalization and cross placer transfer

Source: submitted Table 4. Higher correlation and top 5 accuracy are better.

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
<td>swerv_wrap</td>
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
<td>black_parrot</td>
<td>0.77</td>
<td>0.55</td>
<td>60 percent</td>
<td>0.60</td>
<td>0.53</td>
</tr>
<tr>
<td>bp_be</td>
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

Interpretation: the predictor generalizes strongly within DREAMPlace placements and transfers positively but more weakly to RTLMP placements.
