# import pandas as pd
# l = [["a"], ["b"], ["c"]]
# d = {"1":l}
# data = pd.DataFrame(d)
# data.to_csv("./test.csv")
import xlrd
import xlwt
from xlutils.copy import copy

# 打开现有的xls文件
workbook = xlrd.open_workbook('./data/training_set_rel3.xls', formatting_info=True)


# 复制工作簿
workbook_copy = copy(workbook)

# 选择要修改的工作表（例如第一个工作表）
worksheet_copy = workbook_copy.get_sheet(0)

# 修改指定单元格的内容 (行, 列, 新值)
worksheet_copy.write(1, 1, 'New Value')

# 保存到新的文件
workbook_copy.save('./data/modified_example.xls')

print("文件已成功修改并保存为 modified_example.xls")

