Below is a manuscript-fix checklist organized by concern category. I would treat items marked **critical** as required before either ICCAD camera-ready or ASP-DAC resubmission.

**0. Version Control / Evidence Consistency**
- [ ] **Critical:** Decide which manuscript is authoritative. The folder contains a submitted-looking PDF with older “configuration selection” framing and 18%/33% gains, while the `.tex` and rebuttal use the newer CoOpt/Refine framing and 22%/51%.
- [ ] Reconcile all numbers across manuscript, rebuttal, figures, README, and captions.
- [ ] Remove or update stale claims: README says 14%/24%; current `.tex` says 22%/51%; PDF says 18%/33%.
- [ ] Verify every rebuttal number before submission, especially supervision ablation, bootstrap CIs, congestion ranges, and per-circuit HPWL-vs-timing correlations.

**1. Novelty / Contribution Framing**
Reviewer source: R1, partly R2.

- [ ] Reframe novelty away from “new architecture” and toward **fidelity-validated cross-stage objective design**.
- [ ] Explicitly concede that GAT+CNN is standard AI-for-EDA practice.
- [ ] Emphasize that the novelty is: label-fidelity study, post-GRT supervision choice, differentiable timing surrogate, and two deployment modes.
- [ ] Compare more sharply against LaMPlace, DREAMPlace 4.0, Efficient-TDP, RoutePlacer, AutoDMP, and Re2MaP.
- [ ] Avoid overclaiming “fundamentally new placement formulation.” This is objective augmentation, so position it in the ePlace / DREAMPlace 4.0 / RoutePlacer line of work.
- [ ] Add a supervision-stage ablation showing that post-GRT labels, not merely “adding a learned surrogate,” drive the gain.

My comment: R1 may not fully flip on novelty. The manuscript should make the empirical label-fidelity contribution unavoidable and defensible, rather than trying to pretend the optimizer itself is radically new.

**2. Generalization / Test-Set Lineage**
Reviewer source: R1, R2, R3.

- [ ] **Critical:** Add lineage-stratified results: same-family test circuits vs truly unseen-family circuits.
- [ ] Clearly define family relationships: `bp_be/bp_be12`, `swerv_wrapper/swerv_wrapper43`, `black_parrot/bp_fe`, Ariane group.
- [ ] Stop using “unseen” alone if it means only held-out circuit, not held-out family.
- [ ] Report Group B Ariane-only averages explicitly.
- [ ] Discuss that 22%/51% is partly amplified by same-family transfer.
- [ ] For ASP-DAC resubmission, strongly consider adding out-of-domain circuits such as `microwatt`, `aes`, `jpeg`, or similar.
- [ ] Add a limitations paragraph on OpenROAD + Nangate45 monoculture.

My comment: This is the most important scientific weakness. For ICCAD camera-ready, disclosure plus stratification may be enough. For ASP-DAC, I would add new out-of-domain experiments if at all feasible.

**3. Cross-Placer Transfer**
Reviewer source: R2.

- [ ] Soften “placer-agnostic” to “can operate on any DEF input, but transfer quality is lower across placement distributions.”
- [ ] Explicitly discuss the gap: within-placer rho 0.77 vs cross-placer rho 0.61.
- [ ] Add or promise mixed-source training: DREAMPlace + RTLMP placements.
- [ ] If ASP-DAC: run a small mixed-source or fine-tuning experiment and add one row showing whether the 0.61 gap improves.

My comment: “Placer-agnostic” currently reads too strong. The method interface is placer-agnostic; the learned distribution is not fully placer-agnostic.

**4. PPA Framing**
Reviewer source: R2.

- [ ] Reframe as **timing-driven surrogate placement that preserves power and routability**.
- [ ] Avoid implying meaningful area optimization; area is fixed by floorplan.
- [ ] Avoid implying strong power optimization; power varies less than 2%.
- [ ] Keep “PPAPlace” as the method name, but revise abstract/introduction/conclusion wording.
- [ ] Report power and routability as preservation metrics: power within ~0.99–1.02x, zero DRC, routed wirelength within 3%.

My comment: This is easy to fix and worth doing thoroughly. Reviewers are likely to reward this concession.

**5. Label-Fidelity Study / Statistical Rigor**
Reviewer source: R1, R2, R3.

- [ ] Add confidence intervals for Table 1 Spearman correlations.
- [ ] Clarify that the 20 configurations are controlled sweeps, not random samples.
- [ ] Report the number of observations clearly: 10 circuits × 20 configs × stages, not just “10 designs.”
- [ ] Add per-circuit or aggregate congestion statistics.
- [ ] Show HPWL-vs-post-route timing remains near-zero in congested designs.
- [ ] Add the requested supervision-stage ablation: pre-CTS/pre-route, post-CTS, post-GRT.

My comment: Do not include CI/congestion numbers unless actually computed. The current rebuttal draft contains plausible placeholders.

**6. Experimental Breadth / Baselines**
Reviewer source: R1, R2, AI review.

- [ ] Keep Re2MaP comparison prominent, since R2 treats it as strong prior SOTA.
- [ ] Keep DREAMPlace 4.0 timing-driven baseline.
- [ ] Clarify which baselines are rerun vs copied from published sources.
- [ ] Note any flow/version mismatch for published baseline ratios.
- [ ] If ASP-DAC: add stronger controlled reruns where possible, especially Re2MaP or mixed-source comparisons.

My comment: The manuscript is vulnerable if it mixes published numbers and controlled numbers without very clear notation.

**7. Method Robustness / OOD Behavior**
Reviewer source: R2, AI review.

- [ ] Explain Refine-alone failures on Ariane and distinguish them from CoOpt+Refine.
- [ ] Expand discussion of OOD degradation after too many refinement steps.
- [ ] Justify best-checkpoint / early-stopping strategy.
- [ ] Add sensitivity for `lambda_p`, refinement steps, and learning rate if space permits.
- [ ] Avoid presenting Refine as universally safe.

My comment: The OOD failure is not fatal if framed honestly. It actually supports the need for CoOpt warmup/ramp and checkpointing.

**8. Congestion / Routability**
Reviewer source: R3, AI review.

- [ ] Report peak RUDY, global-route overflow, DRC count, and routed wirelength.
- [ ] Answer whether designs are congested.
- [ ] Show that timing gains are not purchased by DRC/routability degradation.
- [ ] Add a small table or footnote with per-circuit congestion range.

My comment: This directly answers R3 and strengthens the “not just HPWL” argument.

**9. Readability / Definitions**
Reviewer source: R3.

- [ ] Spell out all acronyms at first use.
- [ ] Add a compact glossary footnote or notation table.
- [ ] Reduce overloaded “PPA” language where only WNS/TNS matter.
- [ ] Clean small prose issues: missing spaces, inconsistent circuit abbreviations, and ambiguous “unseen.”

My comment: This is low-effort and should be done regardless of venue.

**10. Camera-Ready vs ASP-DAC Strategy**
- [ ] If ICCAD accepts: focus on honest reframing, lineage-stratified results, limitations, CI/congestion verification, and consistency cleanup.
- [ ] If ICCAD rejects: add new experiments before ASP-DAC, especially out-of-domain circuits and mixed-source/cross-placer training.
- [ ] For ASP-DAC, restructure the paper around the stronger thesis: **post-GRT label fidelity + differentiable timing objective + stratified generalization**.

The highest-priority fixes are: version consistency, verified rebuttal numbers, lineage-stratified results, PPA-to-timing reframing, and a clearer novelty argument centered on fidelity-validated supervision.