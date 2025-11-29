def detach_explain(explain):
    # explain = """Ideas and Content Score: 4
    # The student shows a clear understanding of the prompt and effectively conveys a story about laughter between friends. However, the essay lacks depth and the supporting details are somewhat general. The main idea is not fully developed, and the content is not always relevant to the prompt.
    #
    # Organization Score: 4
    # The essay has a clear structure, but the transitions between sentences and ideas could be smoother. The conclusion effectively ties the story back to the prompt, but it is somewhat abrupt and could be more satisfying.
    #
    # Voice Score: 4
    # The student has an engaging and conversational tone, but the voice could be more consistent. There are moments where the essay sounds overly casual or formal, which detracts from the overall effect.
    #
    # Word Choice Score: 4
    # The student uses a variety of words, but some of the choices are not very precise or appropriate for the context. The vocabulary could be richer and more varied.
    #
    # Sentence Fluency Score: 4
    # The sentences have a natural flow, but the transitions between them could be smoother. The sentences lack variety and could be more engaging.
    #
    # Convention Score: 3
    # The student demonstrates adequate control of standard writing conventions, but there are some errors in punctuation and capitalization that detract from readability.
    #
    # Overall, this essay demonstrates some strengths, but also some weaknesses in terms of content, organization, voice, and conventions. With some revisions to address these areas, the essay could be improved to achieve a higher score."""
    #
    # s = """Ideas and Content Score: 4
    # The stude"""

    index1 = explain.find("Ideas")
    index2 = explain.find("Organization")
    index3 = explain.find("Voice")
    index4 = explain.find("Word")
    index5 = explain.find("Sentence")
    index6 = explain.find("Convention")
    # 利用对冒号和每个特征分数名称的识别来分割句子，去除特征名称和分数
    if index1 != -1 and index2 != -1 and index3 != -1 and index4 != -1 and index5 != -1 and index6 != -1:
        if explain.find(":") == -1:
            index = "\n"
        else:
            index = ":"
        e1 = explain[explain.find(index):index2]
        e2 = explain[index2+explain[index2:].find(index):index3]
        # print(explain[index2:].find("\n"))
        e3 = explain[index3+explain[index3:].find(index):index4]
        e4 = explain[index4+explain[index4:].find(index):index5]
        e5 = explain[index5+explain[index5:].find(index):index6]
        e6 = explain[index6+explain[index6:].find(index):]

        e = e1 + e2 + e3 + e4 +e5 +e6

        return e
    return "Null"






