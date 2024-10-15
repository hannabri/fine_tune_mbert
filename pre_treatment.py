import conllu
from transformers import AutoTokenizer
from datasets import Dataset

"""Functions for Universal Dependencies corpus pre-treatment"""

def multiword(token, pos, id):
    # function to "normalize PoS to compound PoS"
    elements_to_delete = []

    compound = True

    # the "_" does not have a proper index in the UD, therefore
    # we have to count and add the number of "_" present in the list
    nb_comp = 0

    while compound: 

        compound = False        

        if "_" in set(pos): 

            compound = True
            # get index of the multiword token
            index = pos.index("_")

            # define starting and ending point for multiword token
            start,_, end = id[index]

            # change PoS to compound PoS
            new_pos = [pos[i] for i in range(start+nb_comp, end+nb_comp+1)]
            pos[index] = "+".join(new_pos)

            # delete the elements describing the token
            elements_to_delete.extend([i for i in range(start+nb_comp, end+nb_comp+1)])
            nb_comp+=1

    pos_list = [j for i, j in enumerate(pos) if i not in elements_to_delete]
    token_list = [j for i, j in enumerate(token) if i not in elements_to_delete]
        
    return (token_list, pos_list)

def add_token(token): 
    # function to delte whitespaces in token
    if " " in token:
        return "".join(token.split(" "))
    else: 
        return token

def load_conllu(filename):
    # function to load the corpus

    all_sentences = [] # list of list of str
    all_pos = [] # list of list of str

    for sentence in conllu.parse(open(filename, "rt", encoding="utf-8").read()):
            # lists to give into the function 
            ids = [token["id"] for token in sentence]
            pos = [token["upos"] for token in sentence]
            tokens = [add_token(token["form"]) for token in sentence]

            # output of the function 
            
            new_tokens, new_pos = (multiword(tokens, pos, ids))
            
            all_sentences.append(new_tokens)
            all_pos.append(new_pos)
    
    return (all_sentences, all_pos)


def add_pad(pos, offset):
    # add <pad> token to the PoS
    # input: list of PoS and offset returned by mBERT tokenizer

    padding_length = len(offset[0])
    padded_pos = []

    for s in range(len(offset)): 
        for t in range(len(offset[s])):            

            # if the tuple starts not with 0 or ends with 0
            if (offset[s][t][0] != 0) or (offset[s][t][1] == 0):
                pos[s].insert(t, "<pad>")

        # adding only the first padding_length elements --> padding
        padded_pos.append(pos[s][:padding_length]) 

    return padded_pos


def encode_pos(pos):
    # function to encode list of PoS tags into integers

    encoder = {}
    unique_pos = set()

    # get a unique set of PoS
    for sentence in pos: 
        unique_pos.update(set(sentence))

    # create dict with key = PoS and value = encoding
    for c, up in enumerate(list(unique_pos)):
        if up == "<pad>": 
            encoder["<pad>"] = -100
        else: 
            encoder[up] = c

    # encode given PoS list
    encoded_pos = []
    
    for sentence in pos: 
        encoded_pos.append([encoder[t] for t in sentence])

    return (encoded_pos, len(unique_pos))


def create_datasets(input_ids, attention_mask, labels):
    # function to create a HuggingFace Dataset with the data

    # 70% of the data for train and 15% of the data for each test and dev
    splits = [int(len(labels)*0.7), int(len(labels)*0.15)]
    
    train_dict = []
    test_dict = []
    dev_dict = []

    for i in range(len(labels)):

        # train set --> 70% of the data
        if i < splits[0]: 
            set_to_append = train_dict

        # test set --> 15% of the data
        elif i < splits[0]+splits[1]: 
            set_to_append = test_dict

        # dev set --> 15% of the data
        else: set_to_append = dev_dict

        set_to_append.append({
            "input_ids": input_ids[i], 
            "attention_mask": attention_mask[i],
            "labels": labels[i]
        })

    train = Dataset.from_list(train_dict)
    test = Dataset.from_list(test_dict)
    dev = Dataset.from_list(dev_dict)

    return (train, test, dev)


def model_args(filename):
    # returns everything the model needs to be fine-tuned

    # load the UD sentences, already "normalized"
    print("Normalize sentences")
    sentences, pos = load_conllu(filename)

    # tokenize the sentences
    print("Tokenize sentences")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    mbert_tokenizer = tokenizer(sentences, return_tensors="pt", is_split_into_words=True,
                return_offsets_mapping=True, padding=True,
                truncation=True)

    # add <pad> in the lables and encode them into int
    print("add <pad> to PoS")
    labels, len_unique_labels = encode_pos(add_pad(pos, mbert_tokenizer["offset_mapping"]))

    # compute train, test and dev set
    print("Split dataset into test, dev and train\n")
    train, test, dev = create_datasets(
        mbert_tokenizer["input_ids"], 
        mbert_tokenizer["attention_mask"], 
        labels)

    return (train, test, dev, len_unique_labels)
