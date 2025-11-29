import pandas as pd
from Detach_Explain import detach_explain
def detach_score_explain(prompt):
    data = pd.read_csv("./data/score_explain_prompt{}_new.csv".format(prompt), encoding='utf-8')
    score_explain = data["score_explain"]
    # print(score_explain)

    num = 0
    overall_score = []
    score = []
    explain = []
    idd = []
    dic = {}
    label=[]
    for i in range(len(score_explain)):

        se = data["score_explain"][i]
        print(se, data["id"][i])
        # 获取总分
        s = se[8]
        if se[9] <= "9" and se[9] >= "0":
            s += se[9]
        overall_score.append(s)


        # 去除总分，只取traits分数
        index = se.find("Ideas")
        new_explain = se[index:]

        e = detach_explain(new_explain)
        explain.append(e)
        traits = []
        traits_explain = []
        num = 0
        for j in range(len(new_explain)):
            if new_explain[j] == ":" and new_explain[j + 2] >= "0" and new_explain[j + 2] <= "6" and j <= len(
                    new_explain) - 3:
                if not (new_explain[j + 3] >= "0" and new_explain[j + 3] <= "9"):
                    num += 1

                    feature_score = new_explain[j + 2]
                    traits.append(feature_score)

                    # 提取解释
                    # index1 = new_explain[j + 3:].find("\n")
                    # if num < 6:
                    #     t = new_explain[j+3:index1]
                    # else:
                    #     t = new_explain[j+3:]
                    #
                    # traits_explain.append(t)

        label.append(data["label"][i])
        idd.append(data["id"][i])
        if len(traits) == 6:
            score.append(traits)
        elif len(traits) == 7:
            score.append(traits[:-1])
        else:
            score.append([])
            # explain.append(traits_explain)

    print(len(idd), len(label), len(overall_score), len(score), len(explain))

    dic["id"] = idd
    dic["label"] = label
    dic["overall_score"] = overall_score
    dic["score"] = score
    dic["explain"] = explain

    data = pd.DataFrame(dic)
    data.to_csv("./data/detach/detach_score_explain_prompt{}_new.csv".format(prompt))
    # id 1635 425 结尾不是 "]"

for i in range(8, 9):
    # 将分数与解释分离，其中提取分数用一个函数，提取解释用另一个函数。提取完成之后，解释中包含大量的“乱码”需要手动处理
    detach_score_explain(i)