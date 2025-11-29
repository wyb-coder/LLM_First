def extract_triplets(relationships):
    role_priority = [f'ARG{i}' for i in range(8)]  # ARG0-ARG7
    predicates = {}

    for rel in relationships:
        rel = rel.strip(' ^')
        parts = rel.split('(')
        if len(parts) < 2:
            continue
        role = parts[0]
        variables = parts[1].strip(')').split(', ')
        if len(variables) < 2:
            continue
        predicate = variables[0]
        variable = variables[1]

        if predicate not in predicates:
            predicates[predicate] = {}
        predicates[predicate].setdefault(role, []).append(variable)

    triplets = []
    for predicate, role_map in predicates.items():
        collected = []
        for role in role_priority:
            nodes = role_map.get(role)
            if nodes:
                collected.append(nodes[0])
            if len(collected) == 2:
                break
        if len(collected) == 2:
            triplets.append([collected[0], predicate, collected[1]])
    return triplets



# # 输入关系列表
# relationships = [
#     'ARG0(z0, z1) ^',
#     'ARG1(z0, z2) ^',
#     'ARG0(z4, z1) ^',
#     'ARG1(z4, z5) ^',
#     'ARG0(z6, z1) ^',
#     'ARG1(z6, z7) ^',
#     'ARG1(z9, z2) ^'
# ]
#
# # 提取三元组
# triplets = extract_triplets(relationships)
#
# # 打印结果
# print(triplets)
