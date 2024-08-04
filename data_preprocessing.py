import pandas as pd

df1 = pd.read_csv('data\emh dataset_sw.csv', encoding='cp1252')
df2 = pd.read_csv(r'data\bt_self-prepared_dataset_QA_1.csv', encoding='cp1252')
df3 = pd.read_csv(r'data\bt_self-prepared_dataset_paragraph.csv', encoding='cp1252')

df1 = df1[['Question', 'Answer', 'Difficulty']]
print(df1.columns)

df2 = df2[['Paragraph ID', 'Question', 'Answer', "Bloom's Taxonomy Level"]]

print(df3.columns)

def change_difficulty_label(level):
    if level=="Knowledge" or level=="Comprehension":
        label = "Easy"
    if level=="Application" or level=="Analysis":
        label = "Medium"
    if level=="Synthesis" or level=="Evaluation":
        label = "Hard"
    return label

df2["Bloom's Taxonomy Level"] = df2["Bloom's Taxonomy Level"].apply(change_difficulty_label)
df2 = df2.rename(columns={"Bloom's Taxonomy Level": "Difficulty"})

df2 = df2.merge(df3[['Paragraph ID', 'Paragraph']], left_on='Paragraph ID', right_on='Paragraph ID', how='left')

df2 = df2[['Paragraph', 'Question', 'Answer', "Difficulty"]]

print(df2.columns)
print(df2.head())

print(df1.shape)
print(df2.shape)

df2.to_csv('data\dataset - ONE.csv', index=False)

"""
df = pd.concat([df1, df2])

print(df.head())
print(df.shape)

df.to_csv('data\dataset-FINAL.csv', index=False)
"""



