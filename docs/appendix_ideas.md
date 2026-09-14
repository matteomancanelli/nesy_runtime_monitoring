# Possible Appendix Sections

Running list of candidate appendix material for the paper.

- **CILP encoding.** The Garcez & Zaverucha translation used throughout: one hidden unit per rule, weight/bias scheme realizing each Horn clause, the fixpoint of the evaluation phase recovered by iterating the forward pass. Fills the equivalence claims relied on in `4_rulerunner.tex` §Architecture (flat/structured encodings) and the differentiability discussion (sign→tanh is future work, see `artur_future_work/`).
- **Counterexample represented as a recurrent network.** The nested-temporal counterexample (`4_rulerunner.tex` §The Nested-Temporal Limitation) ($F(a \land Xb)$) rendered in the CILP recurrent-network form, making concrete that the shared truth value is a shared neuron and the conflation is present in the network too (cf. the output-layer impossibility argument).

Both entries stay with **Paper A**. The second is now more important than it looks: it is
the network-level witness that the conflation is present in the CILP network itself, not only
in the rule system.
