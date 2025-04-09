from dotenv import load_dotenv
load_dotenv() # loading all the env variables 
import requests
import json 
import streamlit as st
import os 


st.set_page_config(page_title="Chat App ")
st.title("Chwt with Jung")
user_input = st.text_input("ask Jung whats on your mind")


if st.button("Generate Answer"):
    with st.spinner("Thinking..."):
        data = {
            "model": "jung",
            "prompt": user_input,
            }
        url = os.getenv("generate_url")
        st.write("DEBUG: URL = ", url)
        response = requests.post(url=url,json=data, stream=True)
        output = ""
        for line in response.iter_lines():
            #st.write("DEBBUG: RAW line = ",  line)
            if line:
                decoded = json.loads(line.decode("utf-8"))
                #st.write("DEBUG: Decoded Json =", decoded)
                output +=  decoded.get("response", "")

        st.success("Response:")
        st.write(output)




