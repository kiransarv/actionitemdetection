import spacy;
import pandas as pd;
from nltk import RegexpParser, sent_tokenize;
from projroot import get_slang_words_path;
import nltk;
import re;

fororg_email_pattern = re.compile("-{5,}");

QUES_WORDS = ["who", "what", "when", "where", "why", "how", "is", "can", "does", "do", "aren't", "isn't"]

slang_words_action = set();
with open(get_slang_words_path()) as f:
    for line in f.readlines():
        slang_words_action.add(line.rstrip("\n"));
    print("Loaded slang words :: ", len(slang_words_action), slang_words_action)

vp_chunks_reg_exp = r"""
                    MODAL-Phrase: {<,><MD><RB>*<PRP><.>*}
                    MODAL-Phrase: {<MD|NN|NNP><PRP|DT><VB|VBP>*}
                    """
vp_broad_chunks_reg_exp = r"""
                    INTJ-PHRASE: {<INTJ|DET>+<PROPN|NOUN>*<VERB>+<.*>*<PROPN|NOUN>+}
                    VERB-PRO-PHRASE: {<VERB>+<PRON>+<.*>*<VERB>+}
                    PRON-PHRASE : {<PRON>+<VERB>+<.*>*<NOUN|PROPN>+}
                    """
#PRONOUN-PHRASE : {<PRON>+<VERB>+<.*>*<NOUN|PROPN>+}
#SVO : {<NOUN>+<.*>*<VERB>+<.*>*<NOUN>+}

chunker = RegexpParser(vp_chunks_reg_exp);
broad_chunker = RegexpParser(vp_broad_chunks_reg_exp);

def get_tokens(doc):
    return [token.text for token in doc]

def get_chunks(tagged_sen):
    chunks = chunker.parse(tagged_sen);
    return chunks;

def get_broad_chunks(tagged_sen):
    broad_chunks = broad_chunker.parse(tagged_sen);
    return broad_chunks;

def tag_sentences_verbose(doc):
    return [(token.text, token.pos_, token.tag_, token.dep_, token.is_punct, token.is_digit, token.is_upper) for token
            in doc]

def pos_tag_granular(doc):
    return [(token.text, token.tag_) for token in doc]


def pos_tag_broader(doc):
    return [(token.text, token.pos_) for token in doc]

def get_entities(doc):
    entityDict = dict();
    for ent in doc.ents:
        splits = ent.text.split(" ");
        for split in splits:
            entityDict[split] = ent.label_;
    return entityDict;

def parse_tree(doc):
    return [
        (token.text, token.pos_, token.head.text, token.head.pos_, token.head.tag_, [child for child in token.children])
        for token in doc]


def classify_action_item(doc):
    if "?" in doc.text[-3:]:
        return 1.0;

    tagged_granular_sen = pos_tag_granular(doc);
    tagged_broad_sen = pos_tag_broader(doc);

    #print(tagged_granular_sen);
    #print(tagged_broad_sen);

    granular_chunks = get_chunks(tagged_granular_sen);
    #print(granular_chunks);
    for chunk in granular_chunks:
        if type(chunk) is nltk.tree.Tree:
            label = chunk.label();
            #print(doc.text, label);
            if "MODAL-" in label:
                return 1.0;

    broad_chunks = get_broad_chunks(tagged_broad_sen);
    #print(broad_chunks)
    for chunk in broad_chunks:
        if type(chunk) is nltk.tree.Tree:
            label = chunk.label();
            #print(doc.text, label)
            if "INTJ-PHRASE" in label or "VERB-PRO-PHRASE" in label:
                return 1.0;
            elif "PRON-PHRASE" in label:
                return 1.0;

    tokens = get_tokens(doc);
    for token in tokens:
        if token.lower() in slang_words_action:
            return 1.0;

    if tokens[0].lower() in QUES_WORDS:
        return 1.0

    ent_dict = get_entities(doc);
    dep_tree = parse_tree(doc);

    #print(ent_dict)
    #print(dep_tree)

    for tree in dep_tree:
        text = tree[0];
        if text in ent_dict:
            ent_label = ent_dict[text];
            if (ent_label == "DATE" or ent_label == "TIME") and (tree[3] == "VERB" or tree[3] == "NOUN"):
                #print(text, ent_label)
                return 1.0;

    return 0.0;


nlp = spacy.load("en_core_web_sm");

dataset = pd.read_csv("/home/kiran/Datasets/Huddl/emails.csv");
#dataset = dataset.head(100);
dataset = dataset.drop(columns="file");
msgs = dataset["message"];

for msg in msgs:
    index = -1;
    if "X-FileName" in msg:
        index = msg.index("X-FileName");

    if index != -1:
        msgsubstr = msg[index:];
        lines = msgsubstr.split("\n");
        if len(lines) > 1:
            text = " ".join(lines[1:]);
            sens = sent_tokenize(text);
            for sen in sens:
                match = re.search(fororg_email_pattern, sen)
                if match:
                    continue;

                doc = nlp(sen);
                prob = classify_action_item(doc);
                print(sen, prob);

    print("*******************");

'''sentences = ["Can you complete assignment by EOD", "Are you sure about the meeting", "Lets fix our time as 1:30 PM",
             "Please pass on the material and minutes as soon as possible",
             "Who all are coming to meeting tomorrow ?",
             "Also can you look at EOG as a play on rising oil and gas prices",
             "Greg,   How about either next Tuesday or Thursday?",
             "Who can do this work in two days",
             "Finish it by EOD",
             " Colleen,   Please add Mike Grigsby to the distribution",
             "How is progress on creating the  spreadsheets.",
             "It would also be good to have each person's phone number, in the event we need to reach them.",
             "Open the \"utility\" spreadsheet and try to complete the analysis of whether it  is better to be a small commercial or a medium commercial (LP-1). "]

for sen in sentences:
    doc = nlp(sen);
    prob = classify_action_item(doc);
    print(sen, prob);'''