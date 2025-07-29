import spacy

# Tokenization

nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en_core_web_sm")

doc=nlp("spacy is workin on MAC!")

for token in doc:
    print(token.text,token.pos_,token.dep_)

#2. Lemmatization#
for token in doc:
        print(f"{token.text} → {token.lemma_}")
