import pandas as pd


x=8
df_a = pd.read_csv('data\dataset - ONE.csv')


df1 = pd.read_csv('data\manual dataset\manual 1.csv', encoding='cp1252')
df2 = pd.read_csv('data\manual dataset\manual 2.csv', encoding='cp1252')
df3 = pd.read_csv('data\manual dataset\manual 3.csv', encoding='cp1252')
df4 = pd.read_csv('data\manual dataset\manual 4.csv')
df5 = pd.read_csv('data\manual dataset\manual 5.csv', encoding='cp1252')
df6 = pd.read_csv('data\manual dataset\manual 6.csv', encoding='cp1252')
df7 = pd.read_csv('data\manual dataset\manual 7.csv', encoding='cp1252')
df8 = pd.read_csv('data\manual dataset\manual 8.csv', encoding='cp1252')
df9 =pd.read_csv('data\manual dataset\manual 9.csv', encoding='cp1252')
df10 =pd.read_csv('data\manual dataset\manual 10.csv', encoding='cp1252')
df11 =pd.read_csv('data\manual dataset\manual 11.csv', encoding='cp1252')
df12 =pd.read_csv('data\manual dataset\manual 12.csv', encoding='cp1252')
df13 =pd.read_csv('data\manual dataset\manual 13.csv', encoding='cp1252')
df14 =pd.read_csv('data\manual dataset\manual 14.csv')
df15 =pd.read_csv('data\manual dataset\manual 15.csv', )
df16 =pd.read_csv('data\manual dataset\manual 16.csv')
df17 =pd.read_csv('data\manual dataset\manual 17.csv',)
df18 =pd.read_csv('data\manual dataset\manual 18.csv',)


df = pd.concat([ df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18])

print(df.shape)
print(df.columns)

df.to_csv('data/dataset -  MANUAL.csv', index=False)
