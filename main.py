import pandas as pd
import seaborn as sns
from transformers import AutoModelForTokenClassification
from lab_2.mbert_git.fine_tuning import fine_tune_model
from pre_treatment import model_args

"""Load Data"""

model_checkpoint = "bert-base-multilingual-cased"

# load datasets
# load French corpus
filename_fr = "fr_sequoia-ud-test.conllu"
train_fr, test_fr, dev_fr, len_labels_fr = model_args(filename_fr)

# load Spanish corpus
filename_es = "es_ancora-ud-test.conllu"
train_es, test_es, dev_es, len_labels_es = model_args(filename=filename_es)

# load German corpus
filename_de = "de_gsd-ud-test.conllu"
train_de, test_de, dev_de, len_labels_de = model_args(filename=filename_de)

#load Turkish corpus
filename_tr = "tr_penn-ud-test.conllu"
train_tr, test_tr, dev_tr, len_labels_tr = model_args(filename=filename_tr)

# load Czech corpus
filename_cs = "cs_cac-ud-test.conllu"
train_cs, test_cs, dev_cs, len_labels_cs = model_args(filename=filename_cs)

test_sets = [test_fr, test_es, test_de, test_tr, test_cs]


"""Train the Models"""

# determine the max unique label length so all languages can be tested on the trained model
max_labels = max(len_labels_cs, len_labels_de, len_labels_es, len_labels_fr, len_labels_tr)
print("Max number of labels:", max_labels)

# train French model 
model_fr = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=max_labels)
scores_fr = fine_tune_model("fr", model=model_fr, batch_size=16, train=train_fr, dev=dev_fr, learning_rate=2e-5, nb_epochs=7, tests=test_sets)


# train Spanish model
model_es = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=max_labels)
scores_es = fine_tune_model("es", model=model_es, batch_size=16, train=train_es, dev=dev_es, learning_rate=2e-5, nb_epochs=3,tests=test_sets)


# train German model
model_de = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=max_labels)
scores_de = fine_tune_model("de", model=model_de, batch_size=16, train=train_de, dev=dev_de, learning_rate=2e-5, nb_epochs=5, tests=test_sets)

# train Turkish model
model_tr = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=max_labels)
scores_tr = fine_tune_model("tr", model=model_tr, batch_size=16, train=train_tr, dev=dev_tr, learning_rate=2e-5, nb_epochs=7, tests=test_sets)

# train Czech model
model_cs = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=max_labels)
scores_cs = fine_tune_model("cs", model=model_cs, batch_size=16, train=train_cs, dev=dev_cs, learning_rate=2e-5, nb_epochs=5, tests=test_sets)


"""Visualize results"""

df = pd.DataFrame({"fr":scores_fr, "es":scores_es, "de":scores_de, "tr":scores_tr, "cs":scores_cs})
languages = ["fr", "es", "de", "tr", "cs"]

ax = sns.heatmap(df, cmap="Blues", annot=True, xticklabels=languages, yticklabels=languages, cbar=False)
ax.set(title="Accuracy for mBERT", xlabel="trained models", ylabel="test on models")
