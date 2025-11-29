import pandas as pd
import xlrd
import xlwt
from xlutils.copy import copy
from openpyxl import load_workbook
# 该代码的作用为删除最终数据中triple_essay为空的行，即被删除的这些数据的LLM预测分数与标签相差过大

data = pd.read_excel("./data/complete_data/modified_example_6_new.xls")


df_cleaned = data.dropna(subset=['triple_essay'])


# print(df_cleaned)

df_cleaned.to_excel('./data/complete_data/cleaned_data.xls', index=False, engine='xlwt')

# 最后需要手动删除cleaned_data.xls中的"'", "[", "]"等列表中的符号