from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np

"""Functions for Training and Evaluation process"""

metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    # compute accuracy for dev_set 

    # shape predictions: size_dev x padding_size x len(labels) 
    # shape labels: size_dev x padding_size
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2) # size_dev x padding_size

    predictions = predictions.reshape(-1, 1)
    labels = labels.reshape(-1, 1)

    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]

    accuracy =  metric.compute(predictions=predictions, references=labels)

    return accuracy


def evaluate_on_all_languages(trainer, test_sets):
  # evaluate fine-tuned model on all chosen languages
  languages = ["fr", "es", "de", "tr", "cs"]
  scores = {}

  for c, test_set in enumerate(test_sets):
    scores[languages[c]] = trainer.evaluate(test_set)["eval_accuracy"]
    print(f"{languages[c]}: {scores[languages[c]]}")

  return scores


def fine_tune_model(language, model, batch_size, train, dev, learning_rate, nb_epochs, tests):

    training_args = TrainingArguments(
    output_dir=f"./training_{language}",
    evaluation_strategy = "epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=nb_epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train, # the train set
    eval_dataset= dev, # the dev set
    compute_metrics= compute_metrics
    )

    trainer.train()

    # evaluate on all languages
    return evaluate_on_all_languages(trainer, tests)



