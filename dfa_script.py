from ltlf2dfa.parser.ltlf import LTLfParser
p = LTLfParser()

for name, f in [('chain_response', 'G(a -> X b)'), ('eventually_next', 'F(a & X b)')]:
    dot = p(f).to_dfa()
    open(f'fig_{name}_dfa.dot','w').write(dot)


# dot -Tpdf <name>.dot -o <name>.pdf