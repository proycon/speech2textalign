from deepmultilingualpunctuation import PunctuationModel as pm
import string

"""
Add punctuation to wav2vec2 model output based on named entity recognition model

article:
http://ceur-ws.org/Vol-2957/sepp_paper4.pdf
huggingface:
https://huggingface.co/oliverguhr/fullstop-dutch-punctuation-prediction

Adds the following punctuation:
. , ? - :

Optionally capitalize letters at start of sentence
Optionally add end of sentence markers
"""
model = None

def set_model(m):
    global model
    model = m

def load_model(model_name= "oliverguhr/fullstop-dutch-punctuation-prediction"):
    """Loads a punctutation model from hugging face
    https://huggingface.co/oliverguhr/fullstop-dutch-punctuation-prediction
    It is not possible to set cache dir
    """
    global model
    model = pm(model= model_name)

def _restore_punctuation(text): 
    """restores punctuation of a text (removes any present)
    does not add capital letters
    cannot set gpu device (always device 0)
    """
    if model is None: load_model()
    return model.restore_punctuation(text)

def _capitalize_start_of_sentence(text, sep = ". "):
    return string.capwords(text,sep = sep)

def _add_eol(text):
    return text.replace(".",".\n")


def restore_punctuation(text, capitalize = True, add_eol = True):
    output_text = _restore_punctuation(text)
    if capitalize: output_text = _capitalize_start_of_sentence(output_text)
    if add_eol: output_text = _add_eol(output_text)
    return output_text

def restore_punctuation_text_file(filename, output_filename = None, 
    capitalize = True,add_eol= True):
    text = open(filename).read()
    text = restore_punctuation(text,capitalize,add_eol)
    if not output_filename:
        stem = ".".join(filename.split(".")[:-1])
        output_filename = stem + "_punct." + filename.split(".")[-1]
    with open(output_filename, "w") as fout:
        fout.write(text)



