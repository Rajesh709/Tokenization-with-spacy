import spacy
nlp = spacy.load("en_core_web_sm")

doc=nlp("spacy is workin on MAC!")

for token in doc:
    print(token.text,token.pos_,token.dep_)