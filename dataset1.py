import pandas as pd


df1 = pd.read_csv('data/qna_kaggle/S08_question_answer_pairs.txt', sep='\t')
df2 = pd.read_csv('data/qna_kaggle/S09_question_answer_pairs.txt', sep='\t')
df3 = pd.read_csv('data/qna_kaggle/S10_question_answer_pairs.txt', sep='\t', encoding = 'ISO-8859-1')

print(df1.head())
print(df1.shape)
print(df1.columns)

all_data = pd.concat([df1, df2, df3])
all_data.info()

#all_data['Question'] = all_data['ArticleTitle'].str.replace('_', ' ') + ' ' + all_data['Question']
all_data = all_data[['Question', 'Answer', 'DifficultyFromQuestioner']]
print(all_data.shape)

all_data = all_data.drop_duplicates(subset='Question')
print(all_data.shape)

all_data = all_data.dropna()
print(all_data.shape)

all_data.to_csv('data/emh_dataset_kaggle.csv', index=False)
