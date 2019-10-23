import spacy;
from preprocessing.text_preprocessing import get_sentences;
from nltk import RegexpParser;

vp_chunks_reg_exp = r"""
                    MODAL-Phrase: {<,><MD><RB>*<PRP><.>*}
                    MODAL-Phrase: {<MD|NN|NNP><PRP|DT><VB|VBP>*}
                    WH-PHRASE:{<WP|WRB|WDT|WP$><DT|PRP|RB><VB|VBP|VBZ>}
                    PRONOUN_PHRASE : {<PRON>+<DET|ADV>*<VERB>+} 
                    VERB-PHRASE: {<VERB>?<ADV>*<VERB>+}
                    VB-Phrase: {<NN.?>+<,>*<VB>}
                    VB-Phrase: {<DT><,>*<VB>}
                    VB-Phrase: {<RB><VB>}
                    VB-Phrase: {<UH><,>*<VB>}
                    VB-Phrase: {<UH><,><VBP>}
                    VB-Phrase: {<PRP><VB>}
                    """
chunker = RegexpParser(vp_chunks_reg_exp);

def get_chunks(tagged_sen):
    chunks = chunker.parse(tagged_sen);
    return chunks;

def tag_sentences_verbose(doc):
    return [(token.text, token.pos_, token.tag_, token.dep_, token.is_punct, token.is_digit, token.is_upper) for token in doc]

def pos_tag_granular(doc):
    return [(token.text, token.tag_) for token in doc]

def pos_tag_broader(doc):
    return [(token.text, token.pos_) for token in doc]

def parse_tree(doc):
    return [(token.text, token.pos_, token.head.text, token.head.pos_, token.head.tag_, [child for child in token.children]) for token in doc]

def classify_action_items(sens=None):
    return False;

nlp = spacy.load("en_core_web_sm");
sens = get_sentences("/home/kiran/Datasets/Huddl/emails.csv");
docs = list(nlp.pipe(sens));

for doc in docs:
    tagged_granular_sen = pos_tag_granular(doc);
    print(tagged_granular_sen);
    print(get_chunks(tagged_granular_sen));

    tagged_broad_sen = pos_tag_broader(doc);
    print(tagged_broad_sen);
    print(get_chunks(tagged_broad_sen));

    print(parse_tree(doc))
