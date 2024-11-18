# EMH-question-classification

<h3>Dataset Creation</h3>
The dataset has 3 features <ul>
<li>Document: it is the context from which question and answer is derived</li>
<li>Question</li>
<li>Answer</li>
  </ul>
And the target variable Difficulty with values Easy/Medium/Hard based on how easily the answer can be formed from the given Document.

Paragraphs from textbooks such as for DBMS or DSA were chosen to form questions and respective answers and label them as E/M/H.

<h3>Model</h3>
1. Semantic similarity<br>
2. Transformer - BERT model
