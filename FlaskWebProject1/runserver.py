"""
This script runs the FlaskWebProject1 application using a development server.
"""

from os import environ
from FlaskWebProject1 import app
from flask import render_template
from datetime import datetime
import requests
import json
import webbrowser

@app.route('/')
def index():
  return render_template('template.html')

@app.route('/handle_data/', methods=['POST'])
def handle_data():
	from flask import request
	userName = request.form['userName']
	print userName
	query = request.form['query1']+ " " + request.form['query2']
	print query
	token = request.form['token'] 
	print token
	#from Algorithm import evaluate_model_line, NamedEntityChunker

	message = evaluate_model_line(query, userName, token)
	

	
	return render_template(
        'index.html',
        title='Answer',
        year=datetime.now().year,
		message = message)
        #message="Attendee: "+ parsed["attendees"][ + "\nStartTime: " + parsed["start"]["dateTime"])


import string
import os
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag, word_tokenize
import pickle
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
from nltk.chunk.util import conlltags2tree
from datetime import datetime
import parsedatetime as pdt
import json

import requests
from datetime import timedelta


def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """

    # init the stemmer
    stemmer = SnowballStemmer('english')

    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])

    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase

    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase

    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,

        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,

        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,

        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,

        'prev-iob': previob,

        'contains-dash': contains_dash,
        'contains-dot': contains_dot,

        'all-caps': allcaps,
        'capitalized': capitalized,

        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,

        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }


def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]

                        standard_form_tokens = []

                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]

                            if ner != 'O':
                                ner = ner.split('-')[0]

                            if tag in ('LQU', 'RQU'):   # Make it NLTK compatible
                                tag = "``"

                            standard_form_tokens.append((word, tag, ner))

                        conll_tokens = to_conll_iob(standard_form_tokens)

                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]


class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)

        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        #iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        #return conlltags2tree(iob_triplets)
        return chunks


def train_model(model_file='model.dat'):
    corpus_root = 'E:/Data/NameEntityExtraction/gmb-2.2.0'
    reader = read_gmb(corpus_root)
    data = list(reader)
    training_samples = data[:int(len(data) * 0.9)]
    test_samples = data[int(len(data) * 0.9):]

    print "#training samples = %s" % len(training_samples)    # training samples = 55809
    print "#test samples = %s" % len(test_samples)                # test samples = 6201

    chunker = NamedEntityChunker(data)
    f = open(model_file, 'wb')
    p = pickle.Pickler(f)
    p.dump(chunker)
    f.close()

    #Test the model
    print chunker.parse(pos_tag(word_tokenize("I'm going to Germany this Monday.")))


def evaluate_model(model_file='model.dat'):
    authToken = "eyJ0eXAiOiJKV1QiLCJub25jZSI6IkFRQUJBQUFBQUFCbmZpRy1tQTZOVGFlN0NkV1c3UWZkUl9fVXJiY2UzLTZCOXg1Q0JEYmppaldBRUl5MHp0d0dVOGhPdUJxLWZxUFIwSm1ZZzQ5LXJ6ckFRM1M5RHVRdVRlUnUwTVNvbmR6ekNhTzk4RkJfanlBQSIsImFsZyI6IlJTMjU2IiwieDV0IjoiVldWSWMxV0QxVGtzYmIzMDFzYXNNNWtPcTVRIiwia2lkIjoiVldWSWMxV0QxVGtzYmIzMDFzYXNNNWtPcTVRIn0.eyJhdWQiOiJodHRwczovL2dyYXBoLm1pY3Jvc29mdC5jb20iLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC83MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMWRiNDcvIiwiaWF0IjoxNTAxMDY5ODQxLCJuYmYiOjE1MDEwNjk4NDEsImV4cCI6MTUwMTA3Mzc0MSwiYWNyIjoiMSIsImFpbyI6IkFUUUF5LzhEQUFBQUVleWh5ajZIUHFDL0VUbkVnWlhDVHE2LzF2RTV0MXE0ckswMUNwbmpISWVJVUdWVFBsanNSVnhkdVBSTWhFYUkiLCJhbXIiOlsid2lhIiwibWZhIl0sImFwcF9kaXNwbGF5bmFtZSI6IkhhY2thdGhvbjIwMTciLCJhcHBpZCI6ImU4MWRhZjhmLTJiMzAtNDNkMC1iNmE5LTczN2RjMzM1MzEyNiIsImFwcGlkYWNyIjoiMCIsImRldmljZWlkIjoiZjQ4ZTg1N2EtNGIzNS00OGE2LTkyMGUtMTUxYzIyMTY5NTg2IiwiZV9leHAiOjI2MjgwMCwiZmFtaWx5X25hbWUiOiJZYXJvbSIsImdpdmVuX25hbWUiOiJNaWNoYWwiLCJpbl9jb3JwIjoidHJ1ZSIsImlwYWRkciI6IjE2Ny4yMjAuMTk2LjI1NCIsIm5hbWUiOiJNaWNoYWwgWWFyb20iLCJvaWQiOiIzNWM3OWQxYy1mNDEzLTQ3N2UtODUyMC1lYjFhOWMxYjE0YjkiLCJvbnByZW1fc2lkIjoiUy0xLTUtMjEtNzIwNTE2MDctMTc0NTc2MDAzNi0xMDkxODc5NTYtMjU0ODA0IiwicGxhdGYiOiIzIiwicHVpZCI6IjEwMDMzRkZGOUM2RTdCQzkiLCJzY3AiOiJDYWxlbmRhcnMuUmVhZFdyaXRlIFVzZXIuUmVhZCIsInNpZ25pbl9zdGF0ZSI6WyJkdmNfbW5nZCIsImR2Y19jbXAiLCJkdmNfZG1qZCIsImttc2kiXSwic3ViIjoiSUFZTVEzc3V1QnNYTGhBVTF5djhJcFhHQnRWTVVjV3RFenA5ZHNDX0dYMCIsInRpZCI6IjcyZjk4OGJmLTg2ZjEtNDFhZi05MWFiLTJkN2NkMDExZGI0NyIsInVuaXF1ZV9uYW1lIjoibWl5YXJvbUBtaWNyb3NvZnQuY29tIiwidXBuIjoibWl5YXJvbUBtaWNyb3NvZnQuY29tIiwidXRpIjoiTUtjTGI1eVVla2VncWJxNW0xZ0RBQSIsInZlciI6IjEuMCJ9.E-DVDHfF-BTTOwcrOLH8r7w3tgtQn1AK4cSZYxupzKG0UcqjyAuZPx6PPJS1x8YuYmrkREFSTUP_XaR4mudLF2cNjK7WzU0yaooireRGLqxIbvwj2kcrCuEEVOOZd8w-Bz_FoykUIW02t87tt-lxR-BxgBjoUafO4hCZDTwCF_z7YA1vZGHAFx5JxWDsOoKfLMLF5r98_BfGx377crkYhcb8d7ddeu-KLRGAga3ORwyyLwLvwiwDIXMDnxFckX3855YF74peSVZ7plClfh5toKKPSqvF8yZQr66g2K_rT_teyKFb6F7RuoPMxZseNDzGNDbZHGyI0fC40krDDjuVBA"
    # Load the model
    f = open(model_file, "rb")
    p = pickle.Unpickler(f)
    chunker = p.load()
    f.close()

    senList = ["Hi Ran, do you want to eat lunch tomorrow at 12:00 in Zozobra?Yes, Oren will also be here tomorrow, let's invite him too.",
               "Do you want to see a movie next week?",
               "the mail is not working, can you check it?"]

    detected_meetings = 0
    for sen in senList:
        if containsMeetingWords(sen):
            detected_meetings += 1
            timeWords, personWords, geoWords = parseMeetingDetails(sen, chunker)
            meeting_dt, status = getMeetingDateTime(sen, timeWords)
            personDetails = getPersonDetails(personWords)
            if status > 0:
                print("Found a meeting with the following details:")
                print("Time: " + meeting_dt.isoformat())
                people_dt = ""
                for person in personDetails:
                    people_dt += person[0]+ " " + person[1] + ","
                print("Attendees: " + people_dt)
                geo_st = ','.join(geoWords)
                print("Location: " + geo_st)

                sendEvent(meeting_dt, personDetails, geo_st, authToken)
    print("Detected " + str(detected_meetings) + "/" + str(len(senList)) + " meetings")


def evaluate_model_line(sen, sender_name, authToken, model_file='model.dat'):
	# Load the model
	f = open(model_file, "rb")
	p = pickle.Unpickler(f)
	chunker = p.load()
	f.close()

	# Check if the line contains meeting suggestion and return the meeting details
	if containsMeetingWords(sen):
		timeWords, personWords, geoWords = parseMeetingDetails(sen, chunker)
		meeting_dt, status = getMeetingDateTime(sen, timeWords)
		personWords.append(sender_name)
		personDetails = getPersonDetails(personWords)
		geo_st = ','.join(geoWords)
		return sendEvent(meeting_dt, personDetails, geo_st, authToken)


def parseMeetingDetails(sentence, model):
    res = model.parse(pos_tag(word_tokenize(sentence)))
    timeWords = [w for ((w, t), c) in res if "tim" in c]
    personWords = [w for ((w, t), c) in res if "per" in c]
    geoWords = [w for ((w, t), c) in res if "B-geo" in c]
    return list(set(timeWords)), personWords, geoWords


def getMeetingDateTime(sentence, timeWords):
    cal = pdt.Calendar()
    now = datetime.now()
    if len(timeWords) > 0:
        time_string = ' '.join(timeWords)
        suggestTime,status = cal.parseDT(time_string, now)
    else:
        suggestTime,status = cal.parseDT(sentence, now)

    if suggestTime < now:
        status = 0

    return suggestTime, status


def getPersonDetails(personWords):
    personDetails = []
    for nameToLookup in personWords:
        people_file = open('people.json', 'r')
        people_data = json.load(people_file)
        name_and_addresses = [(contact_data['displayName'], contact_data['mail']) for contact_data in people_data['value']]
        possible_contacts = [x for x in name_and_addresses if x[0].split(' ')[0] == nameToLookup]
        if len(possible_contacts) == 1:
            print("Resolved: "+ possible_contacts[0][0] + possible_contacts[0][1])
            personDetails.append(possible_contacts[0])
    return personDetails


def containsMeetingWords(sentence):
    meetingWords = ["Meet",
                    "Meeting",
                    "Come",
                    "Join",
                    "See you",
                    "Get together",
                    "Encounter",
                    "Junction",
                    "Coming together",
                    "Session",
                    "Discussion",
                    "Talk",
                    "Visit",
                    "Go to",
                    "Hearing",
                    "Gathering",
                    "Reunion",
                    "One on one",
                    "Schedule",
                    "Do you want to"]
    sentenceLow = sentence.lower()
    if len([word for word in meetingWords if word.lower() in sentenceLow])>0:
        return True
    else:
        return False


# Creating new meeting
def sendEvent(startTime, atendees, location, authToken):
    startTime = (startTime + timedelta(hours=-3)) #fix timezone offset
    startTimeString = startTime.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    endTimeString = (startTime + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    url = "https://graph.microsoft.com/v1.0/me/events"

    data = {
        "subject": "Meeting",
        "attendees": [],
        "location": {
            "displayName": location
        },
        "start": {
            "dateTime": startTimeString,
            "timeZone": "UTC"
        },
        "end": {
            "dateTime": endTimeString,
            "timeZone": "UTC"
        },
    }

    for atd in atendees:
        atdObj = {"status": {
            "response": "None",
            "time": startTimeString
        },
            "type": "Required",
            "emailAddress": {
                "address": atd[1],
                "name": atd[0]
            }
        }
        data["attendees"].append(atdObj)

    data_json = json.dumps(data)
    headers = {'Content-type': 'application/json',
               'Authorization':'Bearer ' + authToken}

    response = requests.post(url, data=data_json, headers=headers)
    return response.text

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
