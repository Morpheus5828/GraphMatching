"""This module is a StreamlitApp to compare graph pickle file generated
..moduleauthor:: Marius Thorre

"""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

st.title("File Selector Example")

if 'html_content1' not in st.session_state:
    st.session_state.html_content1 = ""
    st.session_state.file_name1 = ""

if 'html_content2' not in st.session_state:
    st.session_state.html_content2 = ""
    st.session_state.file_name2 = ""

select_col1, select_col2 = st.columns(2)

with select_col1:
    uploaded_file1 = st.file_uploader("Choose a HTML file", type="html", key="file_uploader1")
    if uploaded_file1 is not None:
        st.session_state.file_name1 = uploaded_file1.name
        st.session_state.html_content1 = uploaded_file1.getvalue().decode('utf-8')

    if st.session_state.file_name1:
        st.write(f"Selected file: {st.session_state.file_name1}")
    if st.session_state.html_content1:
        components.html(st.session_state.html_content1, width=1000, height=1500)

with select_col2:
    uploaded_file2 = st.file_uploader("Choose a HTML file", type="html", key="file_uploader2")
    if uploaded_file2 is not None:
        st.session_state.file_name2 = uploaded_file2.name
        st.session_state.html_content2 = uploaded_file2.getvalue().decode('utf-8')

    if st.session_state.file_name2:
        st.write(f"Selected file: {st.session_state.file_name2}")
    if st.session_state.html_content2:
        components.html(st.session_state.html_content2, width=1000, height=1500)
