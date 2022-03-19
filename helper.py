# helper functions

import string
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download nltk dependencies
dependencies = ["vader_lexicon", "omw-1.4", "stopwords", "brown", "names", 
"wordnet", "averaged_perceptron_tagger", "universal_tagset"]
for dependency in dependencies:
    nltk.download(dependency)

# return the wordnet object value corresponding to the POS tag
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# return the cleaned text
def clean_text(text, digits=False, stop_words=False, lemmatize=False, only_noun=False):
    # lower text
    text = str(text).lower()
    
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    
    # remove words that contain numbers
    if digits:
        text = [word for word in text if not any(c.isdigit() for c in word)]
        
    # remove stop words
    if stop_words:
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
    
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    
    # pos tag text
    if lemmatize:
        pos_tags = pos_tag(text)    
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        
    if only_noun:
        # select only nouns
        is_noun = lambda pos: pos[:2] == 'NN'
        text = [word for (word, pos) in pos_tag(text) if is_noun(pos)]
    
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    
    # join all
    text = " ".join(text)
    
    return(text)