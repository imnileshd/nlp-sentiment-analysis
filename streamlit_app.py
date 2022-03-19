import streamlit as st
import pandas as pd
import helper
from load_css import local_css
import plotly.express as px
import plotly.graph_objects as go

from nltk.sentiment.vader import SentimentIntensityAnalyzer

APP_NAME = "Sentiment Analysis!"

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# load local css
local_css("css/style.css")

st.sidebar.markdown("Made with love using [streamlit](https://streamlit.io/)")
st.sidebar.image(
    "images/sentiment-analysis.png"
)

st.sidebar.title(APP_NAME)

st.header("Sentiment Analysis for Earnings Call Transcript!")

with open('data/transcript.txt', 'r', encoding="utf8") as file:
    data = file.read()

trascript_dict = dict()
for value in data.split('\n\n'):
    speakers_data = value.split('\n')
    trascript_dict[speakers_data[0]] = speakers_data[1]

key = st.selectbox("Select Speaker", trascript_dict.keys())

transcript = trascript_dict[key]
# print(transcript)

show_text = st.checkbox("Show Transcript")
if show_text:
    st.subheader('Transcript')
    st.markdown(transcript)


sentences = [' '.join(sent.split()).strip()
             for sent in transcript.replace('\n', '').split('. ')]

df = pd.DataFrame(sentences, columns=['content'])

# clean text data
df['content_clean'] = df['content'].apply(lambda x: helper.clean_text(
    x, digits=True, stop_words=True, lemmatize=True))

# add sentiment anaylsis columns
sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['content_clean'].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['sentiment'], axis=1),
               df['sentiment'].apply(pd.Series)], axis=1)

df = df.rename(columns={'neu': 'neutral',
               'neg': 'negative', 'pos': 'positive'})

df['confidence'] = df[["negative", "neutral", "positive"]].max(axis=1)
df['sentiment'] = df[["negative", "neutral", "positive"]].idxmax(axis=1)

# visualization

grouped = pd.DataFrame(df['sentiment'].value_counts()).reset_index()
grouped.columns = ['sentiment', 'count']

st.subheader(f'Display percentage of positive, negative and neutral sentiments')
# Display percentage of positive, negative and neutral sentiments
fig = px.pie(grouped, values='count', names='sentiment', title='Sentiments')
st.plotly_chart(fig)

sentiment_ratio = df['sentiment'].value_counts(normalize=True).to_dict()

for key in ['negative', 'neutral', 'positive']:
    if key not in sentiment_ratio:
        sentiment_ratio[key] = 0.0

# Display sentiment score
st.subheader(f'Display sentiment score')
sentiment_score = (
    sentiment_ratio['neutral'] + sentiment_ratio['positive']) - sentiment_ratio['negative']

fig = go.Figure(go.Indicator(
    mode="number+delta",
    value=sentiment_score,
    delta={"reference": 0.5},
    title={"text": "Sentiment Score"},))

st.plotly_chart(fig)

# Display negative sentence locations
fig = px.scatter(df, y='sentiment', color='sentiment', size='confidence', hover_data=[
                 'content'], color_discrete_map={"negative": "firebrick", "neutral": "navajowhite", "positive": "darkgreen"})
st.subheader(f'Display negative sentence locations')

fig.update_layout(
    width=800,
    height=300,
)

st.plotly_chart(fig)

# Disply annotated trasncript
st.subheader(f'Disply annotated trasncript')

def annotate (record):
    line = f"""<span class="highlight {record['sentiment']}">{record['content']} </span>"""
    return line

text = ''
for record in df[['content', 'sentiment']].to_dict('records'):
    text += annotate(record)

st.markdown(text, unsafe_allow_html=True)
