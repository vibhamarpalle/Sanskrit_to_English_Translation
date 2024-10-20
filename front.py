import streamlit as st
import requests

input_word = st.text_input("Sanskrit word : ") #निर्वर्णम्
response = requests.get("http://localhost:8080/translate/" + input_word)
st.write(response.json()['translation'])
