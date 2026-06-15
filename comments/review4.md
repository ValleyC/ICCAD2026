Review #1192D3h
Paper summary (3-4 sentences)
PPAPlace proposes a differentiable surrogate model that directly predicts post-route PPA from placement and feeds gradients back into placement optimization. The key insight is that common proxies like HPWL or pre-route timing poorly correlate with final PPA, which the authors validate through a label fidelity study across flow stages. The method uses a dual-stream model (GNN + CNN) trained on post-global-routing labels and integrates it into placement via co-optimization and post-placement refinement. Experiments show significant improvements in WNS and TNS over prior methods on ChiPBench.

Strengths
The paper addresses a fundamental and well-known weakness in placement optimization i.e. poor correlation between proxy objectives and final PPA and backs it with a rigorous empirical study. It proposes a novel differentiable surrogate that directly optimizes post-route metrics and integrates seamlessly into analytical placement. The method is well-validated with strong baselines, ablations, and consistent improvements in timing metrics, making it both practically relevant and technically meaningful.

Weaknesses
The approach depends heavily on expensive offline data generation (post-GRT labels), which may limit adoption in industrial settings. Generalization is only partially validated and results are restricted to Nangate45 and a small set of circuits, with noticeable degradation in cross-placer transfer. Some comparisons rely on reported numbers rather than unified experimental setups, and improvements are concentrated on timing while power and area benefits remain limited.

Comments for authors
The paper introduces several genuinely novel elements:

A systematic label fidelity study across flow stages (rare and valuable contribution)
A fully differentiable cross-stage surrogate with end-to-end gradient flow
Integration of PPA gradients directly into placement optimization The combination of learned PPA prediction + differentiable optimization is a meaningful step beyond prior proxy-based or macro-only approaches.
Results are strong and generally convincing:

Evaluated on standard ChiPBench with full OpenROAD flow
Includes strong baselines (DREAMPlace, AutoDMP, LaMPlace, ReMaP)
Demonstrates consistent improvements across circuits
Includes ablations (labels, architecture, gradients) and generalization studies
Notable strengths:

Clear demonstration of HPWL–PPA mismatch
Strong improvements in WNS/TNS
Proper reporting of variance and training setup
Limitations:

Some comparisons rely on published numbers (not fully reproduced)
Evaluation limited to 5 test circuits
Power/area improvements are weak or neutral
Heavy reliance on OpenROAD/Nangate45
Rebuttal questions for authors
The label fidelity study is a strong contribution. Consider emphasizing it more clearly as a standalone insight.

Provide more discussion on data generation cost vs. benefit tradeoff, especially for industrial-scale designs.

Strengthen evaluation by:

Expanding test circuits or including larger-scale designs
Reporting variance for all baselines where possible
Including runtime overhead of CoOpt vs standard DREAMPlace
Investigate multi-node or cross-library generalization, as this is critical for adoption.

Consider analyzing failure cases (e.g., circuits where gains are smaller).

Clarify how sensitive results are to surrogate accuracy and training dataset size.Review #1192D3h
Paper summary (3-4 sentences)
PPAPlace proposes a differentiable surrogate model that directly predicts post-route PPA from placement and feeds gradients back into placement optimization. The key insight is that common proxies like HPWL or pre-route timing poorly correlate with final PPA, which the authors validate through a label fidelity study across flow stages. The method uses a dual-stream model (GNN + CNN) trained on post-global-routing labels and integrates it into placement via co-optimization and post-placement refinement. Experiments show significant improvements in WNS and TNS over prior methods on ChiPBench.

Strengths
The paper addresses a fundamental and well-known weakness in placement optimization i.e. poor correlation between proxy objectives and final PPA and backs it with a rigorous empirical study. It proposes a novel differentiable surrogate that directly optimizes post-route metrics and integrates seamlessly into analytical placement. The method is well-validated with strong baselines, ablations, and consistent improvements in timing metrics, making it both practically relevant and technically meaningful.

Weaknesses
The approach depends heavily on expensive offline data generation (post-GRT labels), which may limit adoption in industrial settings. Generalization is only partially validated and results are restricted to Nangate45 and a small set of circuits, with noticeable degradation in cross-placer transfer. Some comparisons rely on reported numbers rather than unified experimental setups, and improvements are concentrated on timing while power and area benefits remain limited.

Comments for authors
The paper introduces several genuinely novel elements:

A systematic label fidelity study across flow stages (rare and valuable contribution)
A fully differentiable cross-stage surrogate with end-to-end gradient flow
Integration of PPA gradients directly into placement optimization The combination of learned PPA prediction + differentiable optimization is a meaningful step beyond prior proxy-based or macro-only approaches.
Results are strong and generally convincing:

Evaluated on standard ChiPBench with full OpenROAD flow
Includes strong baselines (DREAMPlace, AutoDMP, LaMPlace, ReMaP)
Demonstrates consistent improvements across circuits
Includes ablations (labels, architecture, gradients) and generalization studies
Notable strengths:

Clear demonstration of HPWL–PPA mismatch
Strong improvements in WNS/TNS
Proper reporting of variance and training setup
Limitations:

Some comparisons rely on published numbers (not fully reproduced)
Evaluation limited to 5 test circuits
Power/area improvements are weak or neutral
Heavy reliance on OpenROAD/Nangate45
Rebuttal questions for authors
The label fidelity study is a strong contribution. Consider emphasizing it more clearly as a standalone insight.

Provide more discussion on data generation cost vs. benefit tradeoff, especially for industrial-scale designs.

Strengthen evaluation by:

Expanding test circuits or including larger-scale designs
Reporting variance for all baselines where possible
Including runtime overhead of CoOpt vs standard DREAMPlace
Investigate multi-node or cross-library generalization, as this is critical for adoption.

Consider analyzing failure cases (e.g., circuits where gains are smaller).

Clarify how sensitive results are to surrogate accuracy and training dataset size.