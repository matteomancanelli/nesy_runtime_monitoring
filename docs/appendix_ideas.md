# Possible Appendix Sections

Running list of candidate appendix material for the paper.

- **CILP encoding.** The Garcez & Zaverucha translation used throughout: one hidden unit per rule, weight/bias scheme realizing each Horn clause, the fixpoint of the evaluation phase recovered by iterating the forward pass. Fills the equivalence claims relied on in §3.1 (flat/structured encodings) and §3.3 (differentiability under sign→tanh).
- **Counterexample represented as a recurrent network.** The §3.2 nested-temporal counterexample ($F(a \land Xb)$) rendered in the CILP recurrent-network form, making concrete that the shared truth value is a shared neuron and the conflation is present in the network too (cf. the output-layer impossibility argument).