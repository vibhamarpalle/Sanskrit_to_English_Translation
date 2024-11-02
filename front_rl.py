# import streamlit as st
# import requests

# input_word = st.text_input("Sanskrit word : ") #निर्वर्णम्
# response = requests.get("http://localhost:8080/translate/" + input_word)
# st.write(response.json()['translation'])

import streamlit as st
import requests

if "chat" not in st.session_state:
    st.session_state.chat = []

if "feedback" not in st.session_state:
    st.session_state.feedback = False

def translate(input_word):
    # input_word = "निर्वर्णम्"
    response = requests.get("http://localhost:8080/translate/" + input_word)
    st.session_state.backend_response = response
    # response = "lmao"
    print(response.json()["translation"])
    print(input_word)
    st.session_state.chat.append([input_word, response.json()["translation"]])
    st.session_state.feedback = True


def display_chat():
    for i in range(len(st.session_state.chat)):
        st.markdown(str(i + 1) + ". " + st.session_state.chat[i][0] + " : " + st.session_state.chat[i][1])


def display_feedback():
    st.markdown("What do you think of the above translation?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🙂"):
            requests.post("http://localhost:8080/rl/" + st.session_state.backend_response.json()["translation"] + "/" + "good")
            st.session_state.feedback = False
    with col2:
        if st.button("😐"):
            requests.post("http://localhost:8080/rl/" + st.session_state.backend_response.json()["translation"] + "/" + "normal")
            st.session_state.feedback = False
    with col3:
        if st.button("🙁"):
            requests.post("http://localhost:8080/rl/" + st.session_state.backend_response.json()["translation"] + "/" +  "bad")
            st.session_state.feedback = False


input_word = st.chat_input("Enter a sanskrit word")
if input_word:
    translate(input_word)

display_chat()

if st.session_state.feedback == True:
    display_feedback()
