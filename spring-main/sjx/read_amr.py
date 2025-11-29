def read_amr(prompt):
    # 读取每个句子的amr图，并将amr图存入列表中
    # num = 0
    # if prompt == 1:
    #     amr = ["""""" for i in range(10065)]
    # if prompt == 2:
    #     amr = ["""""" for i in range(19070)]
    # if prompt == 3:
    #     amr = ["""""" for i in range(11270)]
    # if prompt == 4:
    #     amr = ["""""" for i in range(6975)]
    # if prompt == 5:
    #     amr = ["""""" for i in range(7899)]
    # if prompt == 6:
    #     amr = ["""""" for i in range(9775)]
    # if prompt == 7:
    #     amr = ["""""" for i in range(6374)]
    # if prompt == 8:
    #     amr = ["""""" for i in range(3754)]
    id = []
    num = -1
    with open("./data/amr_ouput/pred.amr{}.new.txt".format(prompt), 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 读取所有行
        nn = 0 # 该prompt的句子数量，即amr图的数量
        for line in lines:
            if "::id" in line:
                nn += 1
                index = line.find(".")
                id.append(int(line[index + 1:]))

        amr = ["""""" for i in range(nn)]
        for line in lines:
            if line[0] == "(":
                num += 1
                amr[num] += line
            if line[0] == " ":
                amr[num] += line


    print(len(id)) # 4501
    return amr, id

# read_amr()

