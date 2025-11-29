from amr_to_instance import amr_to_instance
from merge_triplese import merge_triples
from triple_to_essay_id import tripleEssayToinput

l = [8]
for i in l:
    amr_to_instance(i) # triples
    merge_triples(i) # triples -> string
tripleEssayToinput(l)

