import penman
g = penman.decode("""(z0 / present-01
    :ARG0 (z1 / essay)
    :ARG1 (z2 / reason
              :quant (z3 / some)
              :ARG1-of (z4 / relate-01
                           :ARG2 (z5 / affect-01
                                     :ARG0 (z6 / computer)
                                     :ARG1 (z7 / person))))
    :concession-of (z8 / and
                       :op1 (z9 / develop-02
                                :ARG1 (z10 / idea)
                                :mod (z11 / under))
                       :op2 (z12 / lack-01
                                 :ARG1 (z13 / support-01
                                            :ARG1 z10
                                            :ARG0-of (z14 / suffice-01)))))""")
for a, b, concept in g.instances():
    print(a,b,concept)

print(penman.format_triples(g.triples))

print(penman.parse("""(z0 / present-01
    :ARG0 (z1 / essay)
    :ARG1 (z2 / reason
              :quant (z3 / some)
              :ARG1-of (z4 / relate-01
                           :ARG2 (z5 / affect-01
                                     :ARG0 (z6 / computer)
                                     :ARG1 (z7 / person))))
    :concession-of (z8 / and
                       :op1 (z9 / develop-02
                                :ARG1 (z10 / idea)
                                :mod (z11 / under))
                       :op2 (z12 / lack-01
                                 :ARG1 (z13 / support-01
                                            :ARG1 z10
                                            :ARG0-of (z14 / suffice-01)))))"""))  # noqa
# import penman
# g = penman.decode('(b / bark-01 :ARG0 (d / dog))')
