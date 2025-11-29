import nltk
import pandas as pd
import re

def para_to_sent(prompt):
    data = pd.read_csv("./data/prompt1_8/detach_score_explain_prompt{}_newnew.csv".format(prompt))
    new_explain = []
    id = []
    score = []
    explain = []
    label = []
    for i in range(len(data)):

        id.append(data["id"][i])
        label.append(data["label"][i])
        score.append(data["score"][i])
        explain.append(data["explain"][i])

        text = text_processing(data["explain"][i], data["id"][i])

        old = re.findall(r"\.[A-Z]", text)
        old = set(old) # 最多有四个不同类型的".[A-Z]"
        old = list(old)
        for j in range(len(old)):
            text = text.replace(old[j], old[j][0] + " " + old[j][1])
        new_explain.append(text)

    data1 = {"id": id, "label": label, "explain": explain, "new_explain": new_explain}
    df = pd.DataFrame(data1)
    df.to_csv("./data/prompt1_8_sent/sent{}.csv".format(prompt))

def text_processing(text, id):
    print(id)
    print(text)

    cleaned_text = re.sub(r"[^a-zA-Z0-9\s,\.!?'\":]", "", text)
    cleaned_text = cleaned_text.replace(":", ".")
    return cleaned_text


for i in range(8, 9):
    para_to_sent(i)
# input_text = """我我The author chose to conclude the paragraph with the sentence "when they come back, Saeng vowed silently to herself, in the spring , when the snows melt and the geese return and this hibiscus is budding, then I will take that test again." because it shows how the strenghth of the hibiscus is related to Saeng's strenghth. In the story, after failing her drivers test, Saeng comfort in a hibiscus native to her country. when she gets home her mother says “I’ve seen this kind blooming along the lake. Its flowers aren't as pretty, but its strong enough to make it through the cold months here." This shows that saeng is faltiering in her new foriegn country. The end quote sums up that saeng strenghth in the strenghth of the flowers. almost thinking if flower can survive in a land away from its home, why can't I?" This shows that Saeng is determined to overcome her struggles and take the test again, just like the hibiscus which is able to survive through the cold months."""
# cleaned_text = text_processing(input_text)
# print(cleaned_text)
