import streamlit as st
import preprocess
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import numpy as np
import pandas as pd 
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# model import
import joblib
pipe_lr = joblib.load(open("model/emotion_classifier_pipe.pkl","rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results




st.sidebar.subheader("Text Classifier")
st.sidebar.text("WhatsApp Text Analysis Using Python")
st.sidebar.info("Frontend using Streamlit Team")

activities = ["Sentiment Analysics","Sentiment Analysics (Our Model)","WhatsApp Chat Analysis","WhatsApp Sentiment Analysics"]
choice = st.sidebar.selectbox("Choice",activities)
	

st.sidebar.subheader("Group Memebrs")
st.sidebar.text("191080032 Rishab Jain")
st.sidebar.text("191080043 Ayush Malaye")
st.sidebar.text("191070071 Abhishek Somwanshi ")


if choice == 'WhatsApp Chat Analysis':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode('utf-8')
        df = preprocess.preprocess(data)
        st.dataframe(df)

        ## fetch unique users
        users_list = df['user'].unique().tolist()
        users_list.remove("group_notification")
        users_list.sort()
        users_list.insert(0,"Overall")

        # header Start Area 

        selected_user = st.selectbox("show analysis with respective to list",users_list)

        if st.button("Show Analysis"):
            num_message,len_of_words,num_media_message,num_link = helper.fetch_status(selected_user,df)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.title("Total Messages")
                st.header(num_message)
            with col1:
                st.title("Total Words")
                st.header(len_of_words)
            with col3:
                st.title("Media Messages")
                st.header(num_media_message)
            with col4:
                st.title("Links Shared")
                st.header(num_link)
            
            # monthly timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user,df)
            fig,ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'],color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # activity map
            st.title('Activity Map')
            col1,col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user,df)
                fig,ax = plt.subplots()
                ax.bar(busy_day.index,busy_day.values,color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values,color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user,df)
            fig,ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)


            # Most Busy Users in group chat (only for Overall)
            if selected_user == "Overall":
                st.header("Most Busy Users")
                x,new_df = helper.most_busy_users(df)
                col1, col2 = st.columns(2,gap="large")
                fig,ax = plt.subplots()

                with col1:
                    ax.bar(x.index,x.values)
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)

            # #WorldCloud
            # st.title("Wordcloud")
            # df_wc = helper.create_wordcloud(selected_user,df)
            # fig,ax = plt.subplots()
            # ax.imshow(df_wc)
            # st.pyplot(fig)

            # Most Common word used
            most_common_df = helper.most_common_words(selected_user,df)

            fig,ax = plt.subplots()

            ax.barh(most_common_df[0],most_common_df[1])
            plt.xticks(rotation='vertical')
            st.title('Most commmon words')
            st.pyplot(fig)


            # emoji analysis
            emoji_df = helper.emoji_helper(selected_user,df)
            st.title("Emoji Analysis")

            col1,col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig,ax = plt.subplots()
                ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
                st.pyplot(fig)


if choice == 'Sentiment Analysics':
		st.subheader("Sentiment Analysis")
		st.write("Sentiment Analysis Using Python Library")
		raw_text = st.text_area("Enter Your Text","Type Here")
		if st.button("Analyze"):
			blob = TextBlob(raw_text)
			result = blob.sentiment.polarity
			if result > 0.0:
				custom_emoji = ':smile:'
				st.write("Positive : {}" .format(emoji.emojize(custom_emoji)))
			elif result < 0.0:
				custom_emoji = ':disappointed:'
				st.write("Negative : {}" .format(emoji.emojize(custom_emoji)))
			else:
				st.write("Netural : {}" .format(emoji.emojize(':expressionless:')))
			st.info("Polarity Score is : {}".format(result))

if choice == 'WhatsApp Sentiment Analysics': 
    sentiments = SentimentIntensityAnalyzer()
    st.subheader("Sentiment Analysis Of Whatsapp Group Chat")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode('utf-8')
        df = preprocess.preprocess(data)
        st.dataframe(df)

        df.drop(["date", "only_date", "year","month_num","month","day","day_name","hour","minute","period"], axis = 1, inplace=True)
        data = df.dropna()
        data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]]
        data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]]
        data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]]

        x = sum(data["Positive"])
        y = sum(data["Negative"])
        z = sum(data["Neutral"])

        result = helper.sentiment_score(x, y, z)

        st.dataframe(data)
        st.info("Polarity Score is : {}".format(result))

        st.subheader("Sentiment Analysis Of Whatsapp Group Chat Using Our Model ")

        
        data2 = df.dropna()
        data2["Emotion"] = [predict_emotions(i) for i in data2["message"]]
        data2["Probability"] = [max(get_prediction_proba(i)) for i in data2["message"]]
        
        # data2["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data2["message"]]
        st.dataframe(data2)





emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐",
                       "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

if choice == "Sentiment Analysics (Our Model)":
		st.subheader("Sentiment Analysics")
		st.write(emoji.emojize('Sentiment Analysics Using Our Train Model'))
		raw_text = st.text_area("Enter Your Text","Type Here")
		if st.button("Analyze"):
                    prediction = predict_emotions(raw_text)
                    emoji_icon = emotions_emoji_dict[prediction]
                    pred = get_prediction_proba(raw_text)
                    st.write(emotions_emoji_dict)
                    st.write(pred)
                    st.success("{}:{}".format(prediction, emoji_icon))


        




