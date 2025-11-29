def extract_triplets(relationships):
    # 存储动作谓词与发出者和承受者的映射
    predicates = {}

    for rel in relationships:
        # 去除结尾的 ' ^' 并拆分成关系和变量
        rel = rel.strip(' ^')
        parts = rel.split('(')
        role = parts[0]  # ARG0, ARG1, etc.
        variables = parts[1].strip(')').split(', ')  # z0, z1

        # 获取动作谓词和变量
        predicate = variables[0]
        variable = variables[1]

        # 根据角色（ARG0或ARG1）填充谓词映射
        if predicate not in predicates:
            predicates[predicate] = {'ARG0': None, 'ARG1': None}

        if role == 'ARG0':
            predicates[predicate]['ARG0'] = variable
        elif role == 'ARG1':
            predicates[predicate]['ARG1'] = variable

    # 提取三元组
    triplets = []
    for predicate, args in predicates.items():
        if args['ARG0'] and args['ARG1']:
            triplet = (args['ARG0'], predicate, args['ARG1'])
            triplets.append(triplet)

    return triplets


# 输入关系列表
relationships = [
    'ARG0(z0, z1) ^',
    'ARG1(z0, z2) ^',
    'ARG0(z4, z1) ^',
    'ARG1(z4, z5) ^',
    'ARG0(z6, z1) ^',
    'ARG1(z6, z7) ^',
    'ARG1(z9, z2) ^'
]

# 提取三元组
triplets = extract_triplets(relationships)

# 打印结果
print(triplets)
