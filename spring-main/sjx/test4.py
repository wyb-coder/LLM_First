import pandas as pd
# data_id = pd.read_csv("./data/triples.csv")
# print(len(data_id))
import pandas as pd
import re

file_path = "./data/prompt1_8/detach_score_explain_prompt{}.csv".format(4)

# 读取数据
data = pd.read_csv(file_path)

# 逐行检查，寻找非字符串数据
for index, value in enumerate(data["explain"]):
    if not isinstance(value, str):
        print(f"Non-string data found at index {index}: {value} (type: {type(value)})")



