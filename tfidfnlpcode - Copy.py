import pandas as pd
import random
import nltk
import string
from nltk.corpus import stopwords
import gensim
from gensim import corpora, models, similarities
import warnings
warnings.simplefilter('ignore')
def searchnlp(test_set_sentence):
    #test_set_sentence = input("Enter the sentence for Searching :")
    #Load Data Source
    with open ('upload/completedRequestDetails.csv', 'r+', encoding='utf-8') as csvfile:
        content = csvfile.read()

    with open ('upload/completedRequestDetails.csv', 'w+', encoding='utf-8') as csvfile:
        content = csvfile.write(content.replace('"', ''))
    csvfile.close()

    data = pd.read_csv('upload/completedRequestDetails.csv', delimiter='|', encoding='utf-8', dtype=str, engine='python')
    # data = pd.read_csv('upload/chatbot.csv', delimiter='|', encoding='utf-8')
    print(len(data))
    if(len(data) != 0):
        #Create Stop Word
        newstopwords = set(stopwords.words('english'))
        #define Wordnet Lemmatizer
        WNlemma = nltk.WordNetLemmatizer()

        #Create Preprocessing Function
        def pre_process(text):
            tokens = nltk.word_tokenize(text)
            tokens=[WNlemma.lemmatize(t) for t in tokens]
            tokens= [ t for t in tokens if t not in string.punctuation ]
            tokens=[word for word in tokens if word.lower() not in newstopwords]
            # bigr = nltk.bigrams(tokens[:10])
            # trigr = nltk.trigrams(tokens[:10])

            return(tokens)


        # greeting function
        GREETING_INPUTS = ("hello", "hi", "greetings", "hello i need help", "good day", "hey", "i need help", "greetings")
        GREETING_RESPONSES = ["Good day, How may i of help?", "Hello, How can i help?", "hello",
                              "I am glad! You are talking to me."]

        def greeting(sentence):
            for word in sentence.split():
                if word.lower() in GREETING_INPUTS:
                    return random.choice(GREETING_RESPONSES)


        # Preprocess Question Column
        #print(data['MESSAGE'].replace('"', ""))
        data['MESSAGE'] = data['MESSAGE'].apply(pre_process)

        # Define Questions
        question = data['MESSAGE']

        dictionary = corpora.Dictionary(question)
        corpus = [dictionary.doc2bow(a) for a in question]
        tfidf = models.TfidfModel(corpus)

        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=650)  # Threshold A
        corpus_lsi = lsi[corpus_tfidf]
        index = similarities.MatrixSimilarity(corpus_lsi)
        #print(sentence)
        sentence =  str(test_set_sentence)
        print(sentence)
        if (str(sentence).lower() == 'bye'):
            reply = "Bye! Take Care"
            #print(reply)
            return reply
        elif (str(sentence).lower() == 'thank you'):
            reply = "Its my pleasure to help you"
            return reply
        else:
            # ---------------Tokenisation of user input -----------------------------#
            tokens = pre_process(test_set_sentence)
            texts = " ".join(tokens)
            # -----------------------------------------------------------------------#

            # ---------------Find and Sort Similarity -------------------------------#
            vec_bow = dictionary.doc2bow(texts.lower().split())
            vec_tfidf = tfidf[vec_bow]
            vec_lsi = lsi[vec_tfidf]

            # If not in the topic trained.
            if not (vec_lsi):
                not_understood = "Apology, I do not understand. Can you rephrase?"
                print(not_understood)
            else:
                # sort similarity
                sims = index[vec_lsi]
                sims = sorted(enumerate(sims), key=lambda item: -item[1])

                index_s = []
                score_s = []

                for i in range(len(sims)):
                    x = sims[i][1]
                    # If similarity is less than 0.5 ask user to rephrase.
                    flag = False
                    if(x >= 0.5):
                        index_s.append(str(sims[i][0]))
                        score_s.append(str(sims[i][1]))
                        reply_indexes = pd.DataFrame({'index': index_s, 'score': score_s})

                # Find Top Questions and Score
                r_index = int(reply_indexes['index'].loc[0])
                r_score = float(reply_indexes['score'].loc[0])
                reply = str(data.iloc[:, 1][r_index])
                r_score = r_score * 100
                print(reply, r_score)
                return reply, r_score

    else:
        r_score = -1
        reply = None
        return reply, r_score


# test_set_sentence = 'Dear Secretary\nI need to know how much money has'
# searchnlp(test_set_sentence)