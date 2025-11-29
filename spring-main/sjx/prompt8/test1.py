s  ="""The student's essay has a clear purpose and presents a true story about their experiences with making people laugh. However, the content is somewhat limited and lacks depth or complexity. The essay could benefit from more specific details and insights.

Organization Score: 5"""

for i in range(len(s)):
    if s[i] == "\n":
        print(s[i-1])

