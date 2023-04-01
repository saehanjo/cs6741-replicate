# Target

Model | Val Accuracy | Test Accuracy
----------|---------|----
Table-BERT-Horizontal-F+T-Template (Paper) |  66.0 |  65.1
Mine | 70.01 | 70.05


# Experimental Setup (and what was different)
- I downloaded their datasets (in the data folder) from their website: https://tabfact.github.io
- Then, I used the huggingface library to load a pretrained bert model: "bert-base-multilingual-cased". I used this model since they state that they use an implementation of "BERT using the pre-trained model with 12-layer, 768-hidden, 12-heads, and 110M parameters trained in 104 languages".
- Following their description, the statement and the table text is joined with a [SEP] token. The [CLS] token is attached to the front and another [SEP] token at the end. For each statement, the table text is generated from a table by extracting relevant columns and horizontally concatenating the column values based on a natural lanauge template of the form: `row 1 is : {col_name1} is {val11} ; {col_name2} involved is {val12} ; ... . row 2 is : {col_name1} is {val21} ; ... `. We truncate the table text if the input size exceeds the input limit of Bert.
- The model is loaded with the `AutoModelForSequenceClassification.from_pretrained` and trained with the huggingface Trainer library.
- Trial 1 (log/train-1.log and log/test-1.log): I used mostly the default parameter values of the TrainerArgument class to finetune the Bert model. It took ~2 hours to train for 3 epochs with a batch size of 8 on RTX 2080-Ti. The paper states that they achieved the best performance with ~10K steps with a batch size of 6 (~3 hours of training on a single TITAN X GPU). However, the accuracy numbers on both validation and test sets were really low: 50.64% for validation and 50.28% for test. The paper states they acheived 66.0% and 65.1% on validation and test sets, respectively.
- Trial 2 (log/train-2.log): Then, I figured I would check their source code to make sure that I am using the same hyperparameter values for training. I found a couple of configurations that seemed interesting to try out: weight_decay=0.01 and warmup_ratio=0.1. I ran the model to get the first evaluation result on the validation set after the first epoch. I still got only 50.64% accuracy.
- Trial 3 (log/train-3.log): I was converting Pandas dataframes into huggingface Datasets for automated batching during training. I realized that, maybe, I should have converted the 'label' column to a ClassLabel type. After this fix, I was able to achieve 70.01% accuracy on the validation set and 70.05% accuracy on the test set, which are higher than the reported numbers.

