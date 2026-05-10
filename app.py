import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np

nltk.download('vader_lexicon', quiet=True)

st.set_page_config(
    page_title="Indian Economic Policy Sentiment Analyzer",
    page_icon="🇮🇳",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        padding: 25px; border-radius: 15px; margin-bottom: 25px; text-align: center;
    }
    .main-header h1 { color: white; margin: 0; }
    .main-header p { color: #bbdefb; font-size: 1.1em; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('policy_sentiment_complete.csv', parse_dates=['Date'])
    daily = pd.read_csv('daily_sentiment_index.csv', parse_dates=['date'])
    return df, daily


@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)


@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


try:
    df_policy, daily_sentiment = load_data()
    finbert = load_finbert()
    vader = load_vader()
except Exception as e:
    st.error("Error loading data. Make sure CSV files are in same folder as app.py")
    st.error(str(e))
    st.stop()

# Sidebar
st.sidebar.markdown("<h2 style='text-align:center;'>🇮🇳 Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Select Page", [
    "🏠 Home",
    "🔍 Test New Headline",
    "📈 Sentiment Index",
    "📊 Topic Analysis",
    "📅 Event Study",
    "🔗 Model Comparison",
    "📋 About Project"
])

st.sidebar.markdown("---")
st.sidebar.write(f"Headlines: {len(df_policy):,}")
date_min = df_policy['Date'].min().strftime('%b %Y')
date_max = df_policy['Date'].max().strftime('%b %Y')
st.sidebar.write(f"Range: {date_min} - {date_max}")


# =============================================
# HOME PAGE
# =============================================
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1>🇮🇳 Indian Economic Policy Sentiment Analyzer</h1>
        <p>Analyzing public sentiment towards economic policies using NLP</p>
    </div>
    """, unsafe_allow_html=True)

    avg_fb = df_policy['finbert_compound'].mean()
    pos_pct = (df_policy['finbert_compound'] > 0).mean() * 100
    neg_pct = (df_policy['finbert_compound'] < 0).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📰 Headlines", f"{len(df_policy):,}")
    c2.metric("📊 Avg Sentiment", f"{avg_fb:+.4f}")
    c3.metric("🟢 Positive", f"{pos_pct:.1f}%")
    c4.metric("🔴 Negative", f"{neg_pct:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'], y=daily_sentiment['finbert_index_7d'],
        name='FinBERT', line=dict(color='steelblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'], y=daily_sentiment['flair_index_7d'],
        name='Flair', line=dict(color='coral', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'], y=daily_sentiment['vader_index_7d'],
        name='VADER', line=dict(color='seagreen', width=2)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(height=450, title='7-Day Rolling Sentiment Index',
                      xaxis_title='Date', yaxis_title='Sentiment Score')
    st.plotly_chart(fig, use_container_width=True)

    topic_counts = df_policy['primary_topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig2 = px.bar(topic_counts, x='Count', y='Topic', orientation='h',
                  color='Count', color_continuous_scale='Blues',
                  title='Headlines by Policy Topic')
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)


# =============================================
# TEST NEW HEADLINE
# =============================================
elif page == "🔍 Test New Headline":
    st.markdown("""
    <div class='main-header'>
        <h1>🔍 Test New Headline</h1>
        <p>Enter any economic news headline to analyze its sentiment</p>
    </div>
    """, unsafe_allow_html=True)

    headline = st.text_area("Enter Headline",
        placeholder="e.g., RBI holds repo rate steady amid global uncertainties",
        height=80)

    examples = [
        "RBI cuts repo rate by 25 basis points to boost growth",
        "GST collection falls below target amid slowdown",
        "India GDP growth surges to 8.4 percent",
        "Unemployment rate hits record high after lockdown",
        "Government announces massive stimulus for MSMEs",
        "Rupee falls to all time low against US dollar"
    ]
    selected = st.selectbox("Or select an example", [""] + examples)
    if selected:
        headline = selected

    if headline and len(headline.strip()) > 10:
        with st.spinner("Analyzing..."):
            fb_result = finbert(headline[:512])[0]
            label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            fb_score = label_map[fb_result['label'].lower()] * fb_result['score']
            vd_scores = vader.polarity_scores(headline)
            vd_compound = vd_scores['compound']

        col1, col2 = st.columns(2)

        with col1:
            if fb_score > 0.1:
                st.success("🟢 Overall: POSITIVE")
            elif fb_score < -0.1:
                st.error("🔴 Overall: NEGATIVE")
            else:
                st.warning("🟡 Overall: NEUTRAL")

            st.metric("FinBERT Score", f"{fb_score:+.4f}")
            st.write(f"Label: {fb_result['label']} | Confidence: {fb_result['score']:.4f}")
            st.metric("VADER Score", f"{vd_compound:+.4f}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fb_score,
                title={'text': "FinBERT Score"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "steelblue"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "#ffcdd2"},
                        {'range': [-0.3, 0.3], 'color': "#fff9c4"},
                        {'range': [0.3, 1], 'color': "#c8e6c9"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=['FinBERT', 'VADER'], y=[fb_score, vd_compound],
                marker_color=['steelblue', 'coral'],
                text=[f"{fb_score:+.3f}", f"{vd_compound:+.3f}"],
                textposition='outside'
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="black")
            fig2.update_layout(yaxis_range=[-1, 1], height=300, title="Comparison")
            st.plotly_chart(fig2, use_container_width=True)

        if fb_score > 0 and vd_compound > 0:
            st.info("Both models agree: POSITIVE tone.")
        elif fb_score < 0 and vd_compound < 0:
            st.info("Both models agree: NEGATIVE tone.")
        else:
            st.info("Models DISAGREE. FinBERT (financial expert) may be more reliable for economic news.")
# =============================================
# SENTIMENT INDEX
# =============================================
elif page == "📈 Sentiment Index":
    st.markdown("""
    <div class='main-header'>
        <h1>📈 Historical Sentiment Index</h1>
        <p>Track economic policy sentiment over time</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        start_year = st.slider("Start Year", 2017, 2021, 2017)
    with c2:
        end_year = st.slider("End Year", 2017, 2021, 2021)
    with c3:
        models_sel = st.multiselect("Models", ['FinBERT', 'Flair', 'VADER'],
                                     default=['FinBERT', 'Flair', 'VADER'])

    mask = ((daily_sentiment['date'].dt.year >= start_year) &
            (daily_sentiment['date'].dt.year <= end_year))
    filtered = daily_sentiment[mask]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)

    colors = {'FinBERT': 'steelblue', 'Flair': 'coral', 'VADER': 'seagreen'}
    col_map = {'FinBERT': 'finbert_index_7d', 'Flair': 'flair_index_7d', 'VADER': 'vader_index_7d'}

    for model in models_sel:
        if col_map[model] in filtered.columns:
            fig.add_trace(go.Scatter(
                x=filtered['date'], y=filtered[col_map[model]],
                name=model, line=dict(color=colors[model], width=2)
            ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=filtered['date'], y=filtered['article_count'],
        name='Articles', marker_color='gray', opacity=0.5
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.update_layout(height=600, title=f'Sentiment Index ({start_year}-{end_year})')
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment", row=1, col=1)
    fig.update_yaxes(title_text="Articles", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Yearly summary
    st.markdown("### 📊 Yearly Summary")
    df_policy['year'] = df_policy['Date'].dt.year
    yearly = df_policy.groupby('year').agg(
        avg_sentiment=('finbert_compound', 'mean'),
        total_articles=('finbert_compound', 'size'),
        positive_pct=('finbert_label', lambda x: (x == 'positive').mean() * 100)
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig_y = px.bar(yearly, x='year', y='avg_sentiment',
                       color='avg_sentiment', color_continuous_scale='RdYlGn',
                       title='Average Sentiment by Year')
        fig_y.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_y, use_container_width=True)
    with c2:
        fig_y2 = px.bar(yearly, x='year', y='total_articles',
                        title='Article Volume by Year', color_discrete_sequence=['steelblue'])
        st.plotly_chart(fig_y2, use_container_width=True)


# =============================================
# TOPIC ANALYSIS
# =============================================
elif page == "📊 Topic Analysis":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Policy Topic Analysis</h1>
        <p>Explore sentiment across different economic policy areas</p>
    </div>
    """, unsafe_allow_html=True)

    topics = sorted(df_policy['primary_topic'].unique().tolist())
    selected_topic = st.selectbox("Select Policy Topic", topics)

    topic_data = df_policy[df_policy['primary_topic'] == selected_topic]

    avg_sent = topic_data['finbert_compound'].mean()
    pos_pct = (topic_data['finbert_compound'] > 0).mean() * 100
    neg_pct = (topic_data['finbert_compound'] < 0).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Headlines", f"{len(topic_data):,}")
    c2.metric("Avg Sentiment", f"{avg_sent:+.4f}")
    c3.metric("Positive %", f"{pos_pct:.1f}%")
    c4.metric("Negative %", f"{neg_pct:.1f}%")

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(topic_data, x='finbert_compound', nbins=50,
                            title=f'{selected_topic}: Sentiment Distribution',
                            color_discrete_sequence=['steelblue'])
        fig1.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        monthly = topic_data.groupby(
            topic_data['Date'].dt.to_period('M')
        )['finbert_compound'].mean().reset_index()
        monthly['Date'] = monthly['Date'].dt.to_timestamp()
        fig2 = px.line(monthly, x='Date', y='finbert_compound',
                       title='Monthly Trend')
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

    # All topics comparison
    st.markdown("### All Topics Comparison")
    topic_avg = df_policy.groupby('primary_topic')['finbert_compound'].mean().sort_values()
    topic_avg = topic_avg.reset_index()
    topic_avg.columns = ['Topic', 'Avg Sentiment']

    fig3 = px.bar(topic_avg, x='Avg Sentiment', y='Topic', orientation='h',
                  color='Avg Sentiment', color_continuous_scale='RdYlGn',
                  title='Average Sentiment by Topic')
    fig3.add_vline(x=0, line_dash="dash", line_color="black")
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

# =============================================
# EVENT STUDY
# =============================================
elif page == "📅 Event Study":
    st.markdown("""
    <div class='main-header'>
        <h1>📅 Event Study Analysis</h1>
        <p>How did sentiment react to key policy announcements?</p>
    </div>
    """, unsafe_allow_html=True)

    # Events dictionary
    events = {
        'GST Launch (Jul 2017)': '2017-07-01',
        'Budget 2018 (Feb 2018)': '2018-02-01',
        'Budget 2019 (Jul 2019)': '2019-07-05',
        'Corporate Tax Cut (Sep 2019)': '2019-09-20',
        'COVID Lockdown (Mar 2020)': '2020-03-25',
        'Atmanirbhar Package (May 2020)': '2020-05-12',
        'Budget 2021 (Feb 2021)': '2021-02-01'
    }

    # UI controls
    c1, c2 = st.columns(2)
    with c1:
        selected_event = st.selectbox("Select Event", list(events.keys()))
    with c2:
        window_days = st.slider("Window (days before/after)", 7, 30, 14)

    # Convert event date
    event_date = pd.to_datetime(events[selected_event])

    # Filter window safely
    window = daily_sentiment[
        (daily_sentiment['date'] >= event_date - pd.Timedelta(days=window_days)) &
        (daily_sentiment['date'] <= event_date + pd.Timedelta(days=window_days))
    ].copy()

    # Choose correct column (IMPORTANT)
    sentiment_col = 'finbert_index_7d' if 'finbert_index_7d' in window.columns else 'finbert_index'

    # =======================
    # PLOT
    # =======================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=window['date'],
        y=window[sentiment_col],
        mode='lines+markers',
        name='FinBERT Sentiment',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6)
    ))

    # Vertical line (SAFE METHOD)
    fig.add_shape(
        type="line",
        x0=event_date,
        x1=event_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="red", dash="dash")
    )

    # Annotation
    fig.add_annotation(
        x=event_date,
        y=1,
        yref="paper",
        text="Event Day",
        showarrow=False,
        font=dict(color="red")
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    fig.update_layout(
        height=450,
        title=f'Sentiment Around: {selected_event}',
        xaxis_title='Date',
        yaxis_title='Sentiment Score'
    )

    st.plotly_chart(fig, use_container_width=True)

    # =======================
    # METRICS (SAFE)
    # =======================
    before = window[window['date'] < event_date][sentiment_col].mean()
    after = window[window['date'] >= event_date][sentiment_col].mean()

    if pd.isna(before) or pd.isna(after):
        change = 0
    else:
        change = after - before

    c1, c2, c3 = st.columns(3)
    c1.metric("Before Event", f"{before:+.4f}" if not pd.isna(before) else "N/A")
    c2.metric("After Event", f"{after:+.4f}" if not pd.isna(after) else "N/A")
    c3.metric("Change", f"{change:+.4f}")

    if change > 0:
        st.success("📈 Sentiment IMPROVED after this event")
    elif change < 0:
        st.error("📉 Sentiment DECLINED after this event")
    else:
        st.info("➡️ Sentiment remained UNCHANGED")

    # =======================
    # ALL EVENTS COMPARISON
    # =======================
    st.markdown("### All Events Comparison")

    event_results = []

    for name, date_str in events.items():
        edate = pd.to_datetime(date_str)

        w = daily_sentiment[
            (daily_sentiment['date'] >= edate - pd.Timedelta(days=14)) &
            (daily_sentiment['date'] <= edate + pd.Timedelta(days=14))
        ]

        b = w[w['date'] < edate][sentiment_col].mean()
        a = w[w['date'] >= edate][sentiment_col].mean()

        if pd.isna(b) or pd.isna(a):
            change_val = 0
        else:
            change_val = a - b

        event_results.append({
            'Event': name,
            'Before': round(b, 4) if not pd.isna(b) else None,
            'After': round(a, 4) if not pd.isna(a) else None,
            'Change': round(change_val, 4)
        })

    event_df = pd.DataFrame(event_results)

    st.dataframe(event_df, use_container_width=True)

    fig2 = px.bar(
        event_df,
        x='Change',
        y='Event',
        orientation='h',
        color='Change',
        color_continuous_scale='RdYlGn',
        title='Sentiment Change After Each Event'
    )

    fig2.add_vline(x=0, line_dash="dash", line_color="black")

    st.plotly_chart(fig2, use_container_width=True)

# =============================================
# MODEL COMPARISON
# =============================================
elif page == "🔗 Model Comparison":
    st.markdown("""
    <div class='main-header'>
        <h1>🔗 Model Comparison</h1>
        <p>Compare FinBERT, VADER and Flair sentiment models</p>
    </div>
    """, unsafe_allow_html=True)

    avg_fb = df_policy['finbert_compound'].mean()
    avg_vd = df_policy['vader_compound'].mean()
    avg_fl = df_policy['flair_compound'].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("FinBERT Avg", f"{avg_fb:+.4f}")
    c2.metric("VADER Avg", f"{avg_vd:+.4f}")
    c3.metric("Flair Avg", f"{avg_fl:+.4f}")

    # Distribution
    st.markdown("### Score Distributions")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_policy['finbert_compound'],
                               name='FinBERT', opacity=0.5,
                               marker_color='steelblue', nbinsx=50))
    fig.add_trace(go.Histogram(x=df_policy['vader_compound'],
                               name='VADER', opacity=0.5,
                               marker_color='coral', nbinsx=50))
    fig.add_trace(go.Histogram(x=df_policy['flair_compound'],
                               name='Flair', opacity=0.5,
                               marker_color='seagreen', nbinsx=50))
    fig.update_layout(barmode='overlay', height=400,
                      title='Sentiment Distribution by Model')
    st.plotly_chart(fig, use_container_width=True)

    # Agreement
    st.markdown("### Polarity Agreement")
    fb_pos = (df_policy['finbert_compound'] >= 0)
    vd_pos = (df_policy['vader_compound'] >= 0)
    fl_pos = (df_policy['flair_compound'] >= 0)

    agree_fb_vd = (fb_pos == vd_pos).mean() * 100
    agree_fb_fl = (fb_pos == fl_pos).mean() * 100
    agree_vd_fl = (vd_pos == fl_pos).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("FinBERT vs VADER", f"{agree_fb_vd:.1f}%")
    c2.metric("FinBERT vs Flair", f"{agree_fb_fl:.1f}%")
    c3.metric("VADER vs Flair", f"{agree_vd_fl:.1f}%")

    # Scatter plots
    st.markdown("### Score Comparisons")
    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.scatter(df_policy, x='finbert_compound', y='vader_compound',
                          opacity=0.1, title='FinBERT vs VADER')
        fig1.add_shape(type='line', x0=-1, y0=-1, x1=1, y1=1,
                       line=dict(color='red', dash='dash'))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.scatter(df_policy, x='finbert_compound', y='flair_compound',
                          opacity=0.1, title='FinBERT vs Flair')
        fig2.add_shape(type='line', x0=-1, y0=-1, x1=1, y1=1,
                       line=dict(color='red', dash='dash'))
        st.plotly_chart(fig2, use_container_width=True)


# =============================================
# ABOUT PROJECT
# =============================================
elif page == "📋 About Project":
    st.markdown("""
    <div class='main-header'>
        <h1>📋 About This Project</h1>
        <p>Project details and methodology</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
### Objective
- Develop a real-time sentiment analysis framework that captures public reaction
  to Indian economic policy announcements by analyzing news headlines.

### Dataset
- Source: Indian financial news headlines (2017-2021)
- Size: 40,000+ policy-related headlines
- Topics: Monetary policy, fiscal budget, GST, trade, banking, inflation, employment

### Methodology
1. Data Cleaning: Parse dates, remove duplicates, clean text
2. Policy Filtering: Keyword taxonomy to select relevant headlines
3. Entity Extraction: spaCy NER for policy actors
4. Topic Modeling: BERTopic for latent theme discovery
5. Sentiment Analysis: FinBERT, VADER, and Flair models
6. Daily Index: Aggregated daily sentiment with 7-day rolling average
7. Event Study: Sentiment analysis around key policy dates
8. Model Comparison: Cross-validation between three models

### Limitations
- Dataset ends April 2021
- Headline-only (no full article text)
- No human-annotated ground truth
- Moderate model agreement (~60-73%)

### Future Scope
- URL scraping for full article text
- Fine-tuning on Indian financial corpus
- Real-time news monitoring dashboard
- Correlation with economic indicators
""")