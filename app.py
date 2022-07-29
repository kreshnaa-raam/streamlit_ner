import csv
import datarobot as dr
import pandas as pd
import streamlit as st
import sys
import json
import requests
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import pandas as pd



def main():
    st.header("Named Entity Recognition App via DataRobot")
    st.markdown(
        """ This app can be used to run NER on your text files based on SPACY NER Model. 
        We use Spacy's - https://spacy.io/api/entityrecognizer - NER Model deployed in DataRobot MLOps as the backend"""
    )
    st.header("Enter Credential Details")
    api_url = st.text_input("DataRobot API URL", value="https://app.datarobot.com/api/v2", max_chars=None,
                            key="api_url", type="default", help=None, autocomplete=None,
                            on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    api_key = st.text_input("DataRobot API Key", value="****", max_chars=None, key="api_key",
                            type="default", help=None, autocomplete=None,
                            on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    dr_key = st.text_input("DataRobot Key From Deployment", value="****", max_chars=None, key="dr_key",
                           type="default", help=None, autocomplete=None,
                           on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)

    did = st.text_input("DataRobot Deployment ID", value="", max_chars=None, key="DID",
                        type="default", help=None, autocomplete=None,
                        on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)

    ner_text = st.text_input("Input Text for NER", value="", max_chars=None, key="NER",
                             type="default", help=None, autocomplete=None,
                             on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    if st.button('Recognize Entities'):
        with open('NER_Data.txt', 'w') as f:
            f.write(ner_text)
        filename = 'NER_Data.txt'
        mimetype = 'text/plain'
        charset = 'UTF-8'
        API_URL = api_url  # noqa
        API_KEY = api_key
        DATAROBOT_KEY = dr_key
        DEPLOYMENT_ID = did
        data = open(filename, 'rb').read()
        data_size = sys.getsizeof(data)
        MAX_PREDICTION_FILE_SIZE_BYTES = 52428800
        predictions = make_datarobot_deployment_unstructured_predictions(data=data,
                                                                         deployment_id=DEPLOYMENT_ID,
                                                                         mimetype=mimetype,
                                                                         charset=charset,
                                                                         API_URL=API_URL,
                                                                         API_KEY=API_KEY,
                                                                         DATAROBOT_KEY=DATAROBOT_KEY)
        st.write(predictions)


class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""


def make_datarobot_deployment_unstructured_predictions(data, deployment_id, mimetype, charset, API_URL, API_KEY,
                                                       DATAROBOT_KEY):
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {
        'Content-Type': '{};charset={}'.format(mimetype, charset),
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }

    url = API_URL.format(deployment_id=deployment_id)

    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=data,
        headers=headers,
    )
    _raise_dataroboterror_for_status(predictions_response)
    # Return raw response content
    return predictions_response.content


def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = '{code} Error: {msg}'.format(
            code=response.status_code, msg=response.text)
        raise DataRobotPredictionError(err_msg)


if __name__ == "__main__":
    main()
