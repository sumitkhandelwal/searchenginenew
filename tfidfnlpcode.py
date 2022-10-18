import spacy
import pandas as pd
def searchnlp(test_set_sentence):
    # Load Data Source
    with open('upload/completedRequestDetails.csv', 'r+', encoding='utf-8') as csvfile:
        content = csvfile.read()

    with open('upload/completedRequestDetails.csv', 'w+', encoding='utf-8') as csvfile:
        content = csvfile.write(content.replace('"', ''))
    csvfile.close()

    data = pd.read_csv('upload/completedRequestDetails.csv', delimiter='|', encoding='utf-8', dtype=str,
                       engine='python')
    data.insert(2, "Score", 0, True)
    nlp = spacy.load("en_core_web_lg")
    for index, row in data.iterrows():
        doc1 = nlp(row['MESSAGE'])
        doc2 = nlp(test_set_sentence)
        reply = round(float(doc1.similarity(doc2))*100, 2)
        data.loc[index, ['Score']] = reply
    finalResult = data[data['Score'] == data['Score'].max()]
    scoreObt = float(finalResult['Score'])
    if(scoreObt >= 80):
        return finalResult['RESPONSE'].iloc[0], finalResult['Score'].iloc[0]
    else:
        response = -1
        score = None
        return response, score