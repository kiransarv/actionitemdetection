import pandas as pd;
from nltk.tokenize import sent_tokenize;

import re;

fororg_email_pattern = re.compile("-{5,}");

def get_sentences(file=None):
    sentences = ["Can you complete assignment by EOD", "Are you sure about the meeting", "Lets fix our time as 1:30 PM",
                 "Please pass on the material and minutes as soon as possible",
                 "Who all are coming to meeting tomorrow ?",
                 "Also can you look at EOG as a play on rising oil and gas prices",
                 "Greg,   How about either next Tuesday or Thursday?"]


    '''dataset = pd.read_csv(file);
    dataset = dataset.head(100);
    dataset = dataset.drop(columns="file");

    msgs = dataset["message"];
    count = 0;

    sentences = [];

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

                    print(sen);
                    sentences.append(sen);
        print("*******************");'''

    return sentences;

if __name__=="__main__":
    get_sentences("/home/kiran/Datasets/Huddl/emails.csv");