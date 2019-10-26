import os;

working_dir = os.path.dirname("__file__");
print(working_dir);

def get_slang_words_path():
    return "/home/kiran/Workspace/ActionItem/resources/slang_words";
    #return os.path.join(working_dir, "resources/slang_words");
