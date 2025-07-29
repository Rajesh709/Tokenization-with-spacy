import spacy

# ✅ 1 Tokenization

nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en_core_web_sm")

doc=nlp("spacy is workin on MAC!")

for token in doc:
    print(token.text,token.pos_,token.dep_)

#✅ 2. Lemmatization#
for token in doc:
        print(f"{token.text} → {token.lemma_}")


#✅ 3. Part-of-Speech (POS) Tagging
for token in doc:
    print(f"{token.text}: {token.pos_}")


#✅ 4. Named Entity Recognition (NER)
doc = nlp("Elon Musk founded SpaceX in California.")
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")


#✅ 5. Stop Words Removal
tokens = [token.text for token in doc if not token.is_stop]
print(tokens)


#✅ 6. Dependency Parsing
for token in doc:
    print(f"{token.text} → Head: {token.head.text}, Dep: {token.dep_}")

#✅ 7. Sentence Segmentation
doc = nlp("This is the first sentence. And here is another one.")
for sent in doc.sents:
    print(sent.text)

#✅ 8. Similarity Check
doc1 = nlp("Apple")
doc2 = nlp("Orange")
print("Similarity:", doc1.similarity(doc2))

#✅ 9. Custom Stop Word Addition
from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS.add("amazing")


#✅ 10. Text Classification (custom or pretrained)
from spacytextblob.spacytextblob import SpacyTextBlob

nlp.add_pipe("spacytextblob")
doc = nlp("I love this product!")
print(doc._.polarity)
