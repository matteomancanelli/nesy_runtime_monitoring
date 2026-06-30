#!/usr/bin/env bash
# Regenerate the counterexample DFA figures directly from ltlf2dfa/MONA.
# Each formula is a nested-temporal LTLf spec that RuleRunner's
# one-literal-per-subformula encoding cannot monitor correctly, but whose
# minimal DFA is exact (see CLAUDE.md, "nested temporal under F/G/U/R").
#
# Usage:  bash gen_dfa_figs.sh
# Requires: the nesy-monitoring conda env (ltlf2dfa + MONA) and graphviz `dot`.
set -euo pipefail
cd "$(dirname "$0")"

PY=/home/matteo/miniconda3/envs/nesy-monitoring/bin/python

# name -> formula
declare -A FORMULAS=(
  [response]="G(a -> F b)"          # BPM response pattern (headline)
  [chain_response]="G(a -> X b)"    # chain response
  [eventually_next]="F(a & X b)"    # X under propositional under F
)

for name in "${!FORMULAS[@]}"; do
  "$PY" - "$name" "${FORMULAS[$name]}" <<'PY'
import sys
from ltlf2dfa.parser.ltlf import LTLfParser
name, formula = sys.argv[1], sys.argv[2]
dot = LTLfParser()(formula).to_dfa()
open(f"fig_{name}_dfa.dot", "w").write(dot)
PY
  dot -Tpdf "fig_${name}_dfa.dot" -o "fig_${name}_dfa.pdf"
  dot -Tpng -Gdpi=150 "fig_${name}_dfa.dot" -o "fig_${name}_dfa.png"
  echo "generated fig_${name}_dfa.{dot,pdf,png}"
done
