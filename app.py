import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

class LLM:
    def __init__(self):
        self.llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", config={"max_new_tokens": 256, "temperature": 0.01})
        self.template = """
            Write your current comments on the topic of {sample_topic} including {input_text}
            within {number_of_words} words.
        """

    def get_response(self, input_text, number_of_words, sample_topic):
        prompt = PromptTemplate(input_variables=["sample_topic", "input_text", "number_of_words"], template=self.template)
        response = self.llm(prompt.format(sample_topic=sample_topic, input_text=input_text, number_of_words=number_of_words))
        return response

class StreamlitApp:
    def __init__(self):
        st.set_page_config(page_title="LLM", page_icon="🤖", layout="centered", initial_sidebar_state="collapsed")
        self.generator = LLM()
        self.run()

    def run(self):
        st.header("LLM 🤖")
        input_text = st.text_input("Enter The Topic")
        
        col1, col2 = st.columns([5, 5])
        
        with col1:
            number_of_words = st.text_input("Number of Words")
        with col2:
            sample_topic = st.selectbox("Sample topics to write for", ("Data Science", "Large Language Models", "Global Warming"), index=0)
        
        submit = st.button("Generate")
        
        if submit:
            response = self.generator.get_response(input_text, number_of_words, sample_topic)
            st.write(response)

if __name__ == "__main__":
    StreamlitApp()
