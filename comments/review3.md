Review #1192C
Paper summary (3-4 sentences)
The paper enhances dream place to add timing to the total objective. The paper does this by using post-global routing data (e.g. timing) as training data and creating a surrogate function that estimates post-global routing timing based on current placement state.

Strengths
For its cross-stage predictor the paper uses a collection of metrics that correspond to good placement ("Spacial" in terms of cell density, pin density, congestion, etc and "Graph" in terms of the local connectivity).This is a good set of data to train a prediction function. The addition of timing to the total objective function makes sense.

Weaknesses
The paper basis their results on a very small designs size of 10 designs. The paper claims a near-zero correlation between HPWL and post-routing timing metrics but with the lack of information on congestion it is hard to know whether their near zero correlation is due to a lack of congested designs where HWPL reduction would play a role in good timing. The paper is hard to read with a lot of terms used without definition.

Rebuttal questions for authors
Do your designs have much congestion? You claim that there is near-zero correlation between HPWL and post-routing timing metrics but you base this on only 10 designs none of which you publish details on the congestion levels of these designs. With only 10 designs what is your perception of the risk that you have over trained for the data set?