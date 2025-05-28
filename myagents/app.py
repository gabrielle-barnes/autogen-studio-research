import streamlit as st
import asyncio
from agents import teamConfig, orchestrate

st.title('Literature Review Assistant')
desc = st.text_input('Enter a description to conduct a literature review:')

chat_container = st.container()

clicked = st.button('Search', type='primary')
if clicked:
    async def main(desc):
        team = teamConfig()
        async for message in orchestrate(team, desc):
            with chat_container:
                if message.startswith('arxiv_agent:'):
                    with st.chat_message('human'):
                        st.markdown(message[12:])
                elif message.startswith('researcher:'):
                    with st.chat_message('assistant'):
                        st.markdown(message[11:])
                elif message.startswith('user'):
                    with st.chat_message('user'):
                        st.markdown(message)
                else:
                    with st.expander('Tool Call'):
                        st.markdown(message)
    
    with st.spinner('Searching for papers...'):
        asyncio.run(main(desc))
    st.success('Search completed!')
    st.balloons()
