import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer, BertModel
from transformers import Trainer, TrainingArguments

from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

from transformers import BertConfig


# Load the dataset
df = pd.read_csv('data\dataset - MANUAL FINAL.csv')

# Encode the labels
label_encoder = LabelEncoder()
df['Difficulty'] = label_encoder.fit_transform(df['Difficulty'])

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(df[['Document', 'Question', 'Answer']], df['Difficulty'], test_size=0.2)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input texts
train_paragraphs = tokenizer(list(train_texts['Document']), truncation=True, padding=True, max_length=256, return_tensors="pt")
train_questions = tokenizer(list(train_texts['Question']), truncation=True, padding=True, max_length=256, return_tensors="pt")
train_answers = tokenizer(list(train_texts['Answer']), truncation=True, padding=True, max_length=256, return_tensors="pt")

val_paragraphs = tokenizer(list(val_texts['Document']), truncation=True, padding=True, max_length=256, return_tensors="pt")
val_questions = tokenizer(list(val_texts['Question']), truncation=True, padding=True, max_length=256, return_tensors="pt")
val_answers = tokenizer(list(val_texts['Answer']), truncation=True, padding=True, max_length=256, return_tensors="pt")


class BloomDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, questions, answers, labels):
        self.paragraphs = paragraphs
        self.questions = questions
        self.answers = answers
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids_paragraph': self.paragraphs['input_ids'][idx],
            'attention_mask_paragraph': self.paragraphs['attention_mask'][idx],
            'input_ids_question': self.questions['input_ids'][idx],
            'attention_mask_question': self.questions['attention_mask'][idx],
            'input_ids_answer': self.answers['input_ids'][idx],
            'attention_mask_answer': self.answers['attention_mask'][idx],
            'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)  # Convert labels to LongTensor
        }
        return item

    def __len__(self):
        return len(self.labels)



train_dataset = BloomDataset(train_paragraphs, train_questions, train_answers, train_labels)
val_dataset = BloomDataset(val_paragraphs, val_questions, val_answers, val_labels)



class BertForMultiInputClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)  # Combine three BERT outputs

    def forward(self, input_ids_paragraph=None, attention_mask_paragraph=None, input_ids_question=None, attention_mask_question=None, input_ids_answer=None, attention_mask_answer=None, labels=None):
        outputs_paragraph = self.bert(input_ids=input_ids_paragraph, attention_mask=attention_mask_paragraph).last_hidden_state[:, 0, :]
        outputs_question = self.bert(input_ids=input_ids_question, attention_mask=attention_mask_question).last_hidden_state[:, 0, :]
        outputs_answer = self.bert(input_ids=input_ids_answer, attention_mask=attention_mask_answer).last_hidden_state[:, 0, :]

        # Concatenate the outputs
        concatenated_output = torch.cat((outputs_paragraph, outputs_question, outputs_answer), dim=1)

        logits = self.classifier(concatenated_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))

        return (loss, logits)



config = BertConfig.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model = BertForMultiInputClassification.from_pretrained('bert-base-uncased', config=config)

# Define training arguments
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()




# Evaluate the model
trainer.evaluate()

# Predict on new data
def classify_question(paragraph, question, answer):
    encoding_paragraph = tokenizer(paragraph, return_tensors='pt', truncation=True, padding=True, max_length=256)
    encoding_question = tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=256)
    encoding_answer = tokenizer(answer, return_tensors='pt', truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        output = model(
            input_ids_paragraph=encoding_paragraph['input_ids'],
            attention_mask_paragraph=encoding_paragraph['attention_mask'],
            input_ids_question=encoding_question['input_ids'],
            attention_mask_question=encoding_question['attention_mask'],
            input_ids_answer=encoding_answer['input_ids'],
            attention_mask_answer=encoding_answer['attention_mask']
        )
    
    prediction = torch.argmax(output[1], dim=1)
    return label_encoder.inverse_transform(prediction.cpu().numpy())[0]

# Example usage
paragraph = r"""We must have a way to specify how tuples within a given relation are distinguished. This is expressed in terms of their attributes. That is, the values of the attribute values of a tuple must be such that they can uniquely identify the tuple. In other words, no two tuples in a relation are allowed to have exactly the same value for all attributes.

A superkey is a set of one or more attributes that, taken collectively, allow us to identify uniquely a tuple in the relation. For example, the ID attribute of the relation instructor is sufficient to distinguish one instructor tuple from another. Thus, ID is a superkey. The name attribute of instructor, on the other hand, is not a superkey, because several instructors might have the same name.

Formally, let R denote the set of attributes in the schema of relation r. If we say that a subset K of R is a superkey for r, we are restricting consideration to instances of relations r in which no two distinct tuples have the same values on all attributes in K. That is, if t1 and t2 are in r and t1 â‰  t2, then t1.K â‰  t2.K. A superkey may contain extraneous attributes. For example, the combination of ID and name is a superkey for the relation instructor. If K is a superkey, then so is any superset of K. We are often interested in superkeys for which no proper subset is a superkey. Such minimal superkeys are called candidate keys.

It is possible that several distinct sets of attributes could serve as a candidate key. Suppose that a combination of name and dept_name is sufficient to distinguish among members of the instructor relation. Then, both {ID} and {name, dept_name} are candidate keys. Although the attributes ID and name together can distinguish instructor tuples, their combination, {ID, name}, does not form a candidate key, since the attribute ID alone is a candidate key.

We shall use the term primary key to denote a candidate key that is chosen by the database designer as the principal means of identifying tuples within a relation. A key (whether primary, candidate, or super) is a property of the entire relation, rather than of the individual tuples. Any two individual tuples in the relation are prohibited from having the same value on the key attributes at the same time. The designation of a key represents a constraint in the real-world enterprise being modeled. """
question = "What term is used to describe a set of one or more attributes that allow us to uniquely identify a tuple in a relation?"
answer = "Superkey" # knowledge or easy
print(classify_question(paragraph, question, answer))
