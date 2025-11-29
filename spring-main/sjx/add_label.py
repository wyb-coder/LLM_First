import pandas as pd
import xlrd
import xlwt
from xlutils.copy import copy
from openpyxl import load_workbook
all_essay = pd.read_excel("./data/prompt1_8_sent/training_set_rel3.xls")
def find_label(id):

    label = 0
    for i in range(len(all_essay)):
        if all_essay["essay_id"][i] == id:
            label = all_essay["domain1_score"][i]
            break

    return label # 行数(+抬头占一行)

def generate_prompt_space_label(prompt):

    essay_prompt = pd.read_csv("./data/prompt1_8_sent/prompt{}_space_new.csv".format(prompt))

    Id = []
    Score = []
    Label = []
    New_explain = []

    for i in range(len(essay_prompt)):
        iid = int(essay_prompt["id"][i])
        label = find_label(iid)
        # 修改指定单元格的内容 (行, 列, 新值)
        Id.append(essay_prompt["id"][i])
        Score.append(essay_prompt["score"][i])
        Label.append(label)
        New_explain.append(essay_prompt["new_explain"][i])

    data = {"id": Id, "score": Score, "label": Label, "new_explain": New_explain}
    df = pd.DataFrame(data)
        # 保存到新的文件
    df.to_csv('./data/prompt1_8_sent/prompt{}_space_label_new.csv'.format(prompt))

for i in range(6, 7):
    generate_prompt_space_label(i)