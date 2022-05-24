from os import path, listdir
import streamlit as st
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data_dir = 'data/'

st.set_page_config(page_title='Job Recommender System', layout='wide')

tfdif = TfidfVectorizer(stop_words='english')


def recommend_jobs(search: str, item_count: int = 30) -> pd.DataFrame:
    jobs_list = pd.concat(
        [pd.Series([search]), data['Full Description']],
        ignore_index=True
    )
    description_matrix = tfdif.fit_transform(jobs_list)
    similarity_matrix = linear_kernel(description_matrix)

    job_index = 0

    similarity_score = list(enumerate(similarity_matrix[job_index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:item_count + 1]

    job_indices = [i[0] for i in similarity_score]
    return data.iloc[job_indices]


@st.cache(allow_output_mutation=True)
def load_data() -> pd.DataFrame:
    csv_files = [path.join(data_dir, csv) for csv in listdir(data_dir)]
    df = pd.concat(
        map(lambda csv: pd.read_csv(csv, index_col=0), csv_files),
        ignore_index=True
    )

    df['Description'] = df['Description'].fillna('')

    return df


data = load_data()

with st.container():
    col1, col2, col3 = st.columns((2, 0.5, 2))

    with col1:
        search_input = st.text_input('Search jobs', '')
        st.write(f'Search results for: {search_input}')

    with col3:
        result_count = st.number_input('Results count', 1, 100, 30)
        st.write('')

if search_input != '':
    results = recommend_jobs(search_input, result_count)

    with st.container():
        for index, result in results.iterrows():
            with st.expander(result['Title']):
                st.markdown('**Location**')
                st.write(result['Location'])

                st.markdown('**Estado**')
                st.write(result['Estado'])

                st.markdown('**Company**')
                st.write(result['Company'])

                st.markdown('**Salary**')
                st.write(result['Salary'])
