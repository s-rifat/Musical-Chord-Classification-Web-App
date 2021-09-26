import pickle
import soundfile as sf
import librosa
import numpy as np
import streamlit as st

st.title('Musical Chord Classification: Major and Minor')

def parse_audio(x):
    return x.flatten('F')[:x.shape[0]] 

def mean_mfccs(x):
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]

loaded_model = pickle.load(open('music_model.sav', 'rb'))

st.write('Major chords are considered to sound happy but minor chords are considered to sound sad.')
input_path = ''
with st.form(key='my_form'):
    input_path = st.text_input(label='Enter your music path')
    submit_button = st.form_submit_button(label='Submit')

if input_path:
    try:
        with open(input_path):
            x, sr = sf.read(input_path, always_2d=True)
            x = parse_audio(x)
            z = mean_mfccs(x)
            z = np.array(z)
            result = loaded_model.predict(z.reshape(1,-1)).item()
            if result == 1:
                st.write('It\'s a *Major* Chord! :sunglasses: :guitar:')
            else:
                st.write('It\'s a *Minor* Chord! :pensive: :guitar:')     
    except FileNotFoundError:
        st.error('No file in such directory.')