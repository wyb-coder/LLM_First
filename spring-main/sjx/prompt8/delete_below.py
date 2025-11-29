import pandas as pd

# 将字符串形式的分数转换为int形式，并按照asap提供的公式计算总分
data = pd.read_csv("data/detach/detach_score_explain_prompt8_newnew.csv", encoding='utf-8')
score_feature = data["score"]
score_list = []
for i in range(len(score_feature)):
    score_feature_i = score_feature[i]
    feature_score_i_list = []
    if len(score_feature_i) != 2:
        for j in range(len(score_feature_i)):
            if score_feature_i[j] >= "0" and score_feature_i[j] <= "9":
                feature_score_i_list.append(int(score_feature_i[j]))
        score = 2 * (feature_score_i_list[0]+feature_score_i_list[1]+feature_score_i_list[4]) + 4 * feature_score_i_list[5]
    else:
        score = -1

    score_list.append(score)

dic = {}

dic["score_new"] = score_list

data = pd.DataFrame(dic)
data.to_csv("./data/detach/score_new.csv")