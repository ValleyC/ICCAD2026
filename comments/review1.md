Review #1192A
Paper summary (3-4 sentences)
This paper proposes PPAPlace, a differentiable surrogate-guided placement framework for post-route PPA optimization. The method combines CNN-based spatial features and GAT-based graph representations to predict timing metrics using post-global-routing supervision. The learned surrogate is integrated into DREAMPlace as either a co-optimization objective or a gradient-based refinement module. Experimental results on ChiPBench demonstrate timing improvements over several prior placement approaches.

Strengths
The paper is generally well written and easy to follow. The motivation regarding the weak correlation between HPWL and final post-route timing is reasonable and supported by empirical analysis. The authors also evaluate the method using a complete OpenROAD flow and report post-route timing metrics.

Weaknesses
The overall technical novelty appears limited. The proposed framework mainly combines existing techniques including GNN/CNN surrogate modeling, differentiable objectives, and gradient-based refinement within the DREAMPlace framework.

The core contribution is primarily objective augmentation rather than a fundamentally new placement formulation or optimization strategy. The optimization backbone itself remains largely unchanged.

The claimed generalization capability is not sufficiently convincing because several training and testing circuits belong to closely related processor families, while all experiments are limited to the OpenROAD + Nangate45 flow.

Comments for authors
The paper is generally well written and the experimental section is reasonably comprehensive. The motivation regarding the weak correlation between HPWL and final post-route timing is also reasonable and supported by empirical analysis.

However, I remain unconvinced that the overall methodological contribution is sufficiently novel for ICCAD. The proposed framework mainly combines existing techniques including GNN/CNN surrogate modeling, differentiable placement objectives, and gradient-based refinement within the DREAMPlace framework. From a technical perspective, the work appears closer to objective augmentation through a learned surrogate rather than a fundamentally new placement formulation or optimization strategy.

In addition, the predictor architecture itself follows common AI-for-EDA design patterns, including graph-based netlist encoding and CNN-based spatial modeling. While the engineering effort is nontrivial, the conceptual gap compared with prior timing-aware or surrogate-guided placement approaches still appears relatively limited

The evaluation results are encouraging, but the claimed generalization capability would be more convincing if validated on architecturally distinct circuits and under more diverse backend flows beyond OpenROAD + Nangate45.

Rebuttal questions for authors
Several training and testing circuits appear to belong to closely related processor families. Can the authors provide stronger evidence of cross-design generalization on architecturally distinct circuits?

How much improvement comes specifically from post-GRT supervision, compared with simply adding another learned surrogate objective into DREAMPlace? An ablation using different supervision stages would help clarify this point.