import nltk
import pandas as pd

template = """# ::id sdl_0002.{} ::date 2013-07-02T03:00:13 ::annotator SDL-AMR-09 ::preferred
        # ::snt {}
        # ::save-date Thu Jul 11, 2013 ::file sdl_0002_1.txt
        (s / say-01
              :ARG0 (p / person :wiki "Johan_Rockström"
                    :name (n / name :op1 "Rockström"))
              :ARG1 (a / accelerate-01
                    :ARG0 (c / country
                          :mod (r / rich))
                    :ARG1 (e / exploit-01
                          :ARG0 c
                          :ARG1 (r2 / resource
                                :mod (n2 / nation)
                                :poss (w / world)))
                    :mod (a2 / also)
                    :time (i / increase-01
                          :ARG0 c
                          :ARG1 (c2 / consume-01
                                :ARG0 c))
                    :ARG0-of (c3 / cause-01
                          :ARG1 (e2 / emit-01
                                :ARG0 c
                                :ARG1 (g / gas
                                      :mod (g2 / greenhouse)
                                      :mod (m / more))))))
        """

def para_to_sent(prompt):
    data = pd.read_csv("./data/prompt1_8_sent/sent{}.csv".format(prompt))
    input_data = []
    # 将每个解释划分为句子，并将每条句子加工为模型的输入格式
    if prompt == 1:
        for i in range(len(data)):
            if abs(data["label"][i] - 2 * data["score"][i]) <= 2:
                text = data["new_explain"][i]
                sentences = nltk.sent_tokenize(text) # 按空格分、
                # 打印结果
                print(sentences)
                for sentence in sentences:
                    formatting = template.format(data["id"][i], sentence)
                    input_data.append(formatting)

    if prompt >= 2 and prompt <= 6:
        for i in range(len(data)):
            if abs(data["label"][i] - data["score"][i]) <= 1:
                text = data["new_explain"][i]
                sentences = nltk.sent_tokenize(text) # 按空格分、
                # 打印结果
                # print(sentences)
                for sentence in sentences:
                    formatting = template.format(data["id"][i], sentence)
                    input_data.append(formatting)

    if prompt == 7:
        for i in range(len(data)):
            if abs(data["label"][i] - 2 * data["score"][i]) <= 4:
                text = data["new_explain"][i]
                sentences = nltk.sent_tokenize(text) # 按空格分、
                # 打印结果
                # print(sentences)
                for sentence in sentences:
                    formatting = template.format(data["id"][i], sentence)
                    input_data.append(formatting)

    if prompt == 8:
        for i in range(len(data)):
            # if abs(data["label"][i] - data["score"][i]) <= 8:
            text = data["new_explain"][i]
            sentences = nltk.sent_tokenize(text) # 按空格分、
            # 打印结果
            # print(sentences)
            for sentence in sentences:
                formatting = template.format(data["id"][i], sentence)
                input_data.append(formatting)

    print("prompt: {}, len: {}".format(prompt, len(input_data)))
    with open("./data/input_data/strings_prompt{}_test_new.txt".format(prompt), "w", encoding="utf-8") as file:
        # 遍历字符串列表，并写入文件
        for string in input_data:
            file.write(string + "\n")  # 在每个字符串后添加换行符


for i in range(8, 9):
    para_to_sent(i)