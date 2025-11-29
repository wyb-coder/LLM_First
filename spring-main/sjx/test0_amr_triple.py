def parse_amr(amr):
    triples = []

    def recurse(node, parent=None, relation=None):
        if isinstance(node, tuple):
            # node is in the form (var, concept)
            var, concept = node
            if parent:
                triples.append((parent, relation, var))
            # If the concept has a parent relation, add it to triples
            if isinstance(concept, dict):
                for rel, child in concept.items():
                    recurse(child, var, rel)
        elif isinstance(node, dict):
            # node is a dict of relations and children
            for rel, child in node.items():
                recurse(child, parent, rel)

    recurse(amr)
    return triples

amr_example = ("z0", {
    "/": "present-01",
    ":ARG0": ("z1", {
        "/": "essay"
    }),
    ":ARG1": ("z2", {
        "/": "reason",
        ":quant": ("z3", {
            "/": "some"
        }),
        ":ARG1-of": ("z4", {
            "/": "relate-01",
            ":ARG2": ("z5", {
                "/": "affect-01",
                ":ARG0": ("z6", {
                    "/": "computer"
                }),
                ":ARG1": ("z7", {
                    "/": "person"
                })
            })
        })
    }),
    ":concession-of": ("z8", {
        "/": "and",
        ":op1": ("z9", {
            "/": "develop-02",
            ":ARG1": ("z10", {
                "/": "idea"
            }),
            ":mod": ("z11", {
                "/": "under"
            })
        }),
        ":op2": ("z12", {
            "/": "lack-01",
            ":ARG1": ("z13", {
                "/": "support-01",
                ":ARG1": "z10",
                ":ARG0-of": ("z14", {
                    "/": "suffice-01"
                })
            })
        })
    })
})

import penman
g = penman.decode('(b / bark-01 :ARG0 (d / dog))')
print(penman.format_triples(g.triples))

triples = parse_amr(amr_example)
for triple in triples:
    print(triple)
