import os
import re
import time
import logging
import numpy as np
import pandas as pd
import streamlit as st
import urllib.parse
from datetime import datetime, timedelta
from google.cloud import bigquery
from google_play_scraper import reviews, Sort, app
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import urllib.parse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Config ===
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('BQ_KEY_PATH', 'Enter your path here') # Replace with your actual path(JSON key)
project_id = "Project_ID"  # Replace with your actual project ID
dataset_id = "Dataset_ID"  # Replace with your actual dataset ID

# === Initialization ===
client = bigquery.Client()
nltk.download('vader_lexicon')
nltk.download('stopwords')

# === Get App Info ===
def get_app_name(app_id):
    try:
        app_info = app(app_id, lang='en', country='us')
        app_name = app_info['title']
        safe_name = re.sub(r'[^\w\s-]', '', app_name).strip().replace(' ', '_')
        return safe_name
    except Exception as e:
        logging.error(f"Failed to get app name: {e}")
        return "app"

def get_app_metadata(app_id):
    try:
        metadata = app(app_id, lang='en', country='us')
        return {
            'AppID': app_id,
            'Title': metadata.get('title'),
            'Developer': metadata.get('developer'),
            'Installs': metadata.get('installs'),
            'Score': metadata.get('score'),
            'Ratings': metadata.get('ratings'),
            'Reviews': metadata.get('reviews'),
            'Version': metadata.get('version'),
            'Updated': metadata.get('updated'),
            'Genre': metadata.get('genre'),
            'Min_Android_Ver': metadata.get('androidVersion'),
            'App_URL': metadata.get('url'),
            'Retrieved_At': datetime.utcnow()
        }
    except Exception as e:
        logging.warning(f"Failed to fetch app metadata: {e}")
        return None

def get_last_review_date(table_id):
    try:
        query = f"SELECT MAX(Date) as last_date FROM `{table_id}`"
        result = client.query(query).result()
        return [row.last_date for row in result][0]
    except Exception as e:
        logging.warning(f"Error retrieving last review date: {e}")
        return None

# === Fetch Reviews ===
def fetch_all_reviews(app_id, cutoff_date, custom_end_date=None):
    logging.info(f"Fetching reviews for: {app_id}")
    all_reviews = []
    token = None
    while True:
        try:
            batch_reviews, token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=100,
                continuation_token=token
            )
        except Exception as e:
            logging.error(f"Review fetch error: {e}")
            time.sleep(2)
            continue
        if not batch_reviews:
            break
        for review in batch_reviews:
            review_date = review['at']
            if review_date < cutoff_date or (custom_end_date and review_date > custom_end_date):
                return all_reviews
            all_reviews.append({
                'ReviewID': review['reviewId'],
                'Username': review['userName'],
                'Rating': review['score'],
                'Date': review_date.strftime('%Y-%m-%d'),
                'Review': review['content'],
                'Reply': review.get('replyContent', ''),
                'Reply Date': review['repliedAt'].strftime('%Y-%m-%d') if review.get('repliedAt') else '',
                'Usefulness': review['thumbsUpCount'],
            })
        logging.info(f"Fetched {len(all_reviews)} reviews so far...")
        time.sleep(0.5)
        if not token:
            break
    return all_reviews

# === Main Pipeline Function ===
def run_pipeline(app_id, mode="append", start=None, end=None):
    app_name = get_app_name(app_id)
    unified_table_id = f"{project_id}.{dataset_id}.{app_name}_unified"

    # Decide cutoff date
    cutoff_date = datetime.now() - timedelta(days=30)
    last_date = get_last_review_date(unified_table_id)
    if last_date:
        logging.info(f"📅 Last review date in BigQuery: {last_date}")
    else:
        logging.info("ℹ️ No existing data found for this app.")

    custom_end_date = None
    if mode == "append" and last_date:
        cutoff_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    elif mode == "range" and start and end:
        cutoff_date = pd.to_datetime(start)
        custom_end_date = pd.to_datetime(end)
    elif mode == "overwrite":
        cutoff_date = datetime(2000, 1, 1)

    metadata = get_app_metadata(app_id)
    app_metadata_df = pd.DataFrame([metadata]) if metadata else pd.DataFrame()
    reviews_data = fetch_all_reviews(app_id, cutoff_date, custom_end_date)
    df = pd.DataFrame(reviews_data)
    if df.empty:
        logging.warning("No new reviews to upload.")
        return

    # === Processing ===
    df['Username'] = df['Username'].astype(str).str.strip().str.lower()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Reply Date'] = pd.to_datetime(df['Reply Date'], errors='coerce')
    df['Review'] = df['Review'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df['Reply_Time_Days'] = (df['Reply Date'] - df['Date']).dt.days
    df['Reply'] = df['Reply'].fillna('No Reply')
    df['Usefulness'] = pd.to_numeric(df['Usefulness'], errors='coerce').fillna(0).astype('Int64')

    sia = SentimentIntensityAnalyzer()
    df['Sentiment_Score'] = df['Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['Sentiment'] = pd.cut(df['Sentiment_Score'], bins=[-1, -0.05, 0.05, 1], labels=['Negative', 'Neutral', 'Positive'])

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Year'] = df['Date'].dt.strftime('%Y-%m')
    df['Weekday'] = df['Date'].dt.day_name()
    df['Response_Tier'] = pd.cut(df['Reply_Time_Days'],
                                bins=[-1, 0, 1, 3, 7, 14, np.inf],
                                labels=['Same day', '<1 day', '1-3 days', '4-7 days', '1-2 weeks', '>2 weeks'],
                                include_lowest=True)

    df['Date'] = df['Date'].dt.date
    df['Reply Date'] = df['Reply Date'].dt.date

    device_patterns = {
        'iPhone': r'iphone|ios',
        'Samsung': r'samsung|galaxy',
        'Pixel': r'pixel|google phone',
        'OnePlus': r'oneplus|one plus',
        'Xiaomi': r'xiaomi|redmi|poco',
        'Android': r'\bandroid\b(?!.*(ios|iphone))',
        'iOS': r'\bios\b|apple',
        'Tablet': r'tablet|ipad|galaxy tab'
    }
    df['Devices_Mentioned'] = df['Review'].apply(lambda text: ', '.join(
        [device for device, pattern in device_patterns.items() if re.search(pattern, str(text).lower())]) or 'Unknown')

    ui_keywords = [
        'slow', 'lag', 'bug', 'glitch', 'crash', 'freeze', 'complicated', 'hard', 'navigation', 'unresponsive',
        'delay', 'latency', 'stutter', 'load time', 'resource intensive', 'memory leak', 'instability', 'error',
        'failure', 'hang', 'confusing', 'difficult', 'intricate', 'unintuitive', 'cumbersome', 'tedious',
        'user-friendly', 'accessibility', 'workflow', 'steps', 'process', 'layout', 'design', 'interface',
        'discoverability', 'pixelated', 'distorted', 'alignment', 'animation', 'responsiveness', 'touch', 'click',
        'scroll', 'visual', 'rendering', 'display', 'font', 'color', 'data loss', 'sync', 'save', 'input', 'output',
        'search', 'filter', 'functionality', 'feature', 'compatibility', 'frustrating', 'annoying', 'irritating',
        'problem', 'issue', 'bad', 'poor', 'broken', 'useless', 'disappointing'
    ]
    df['UI_Issue'] = df['Review'].str.contains('|'.join(ui_keywords), case=False, na=False)

    performance_keywords = [
        'crash', 'freeze', 'lag', 'slow', 'bug', 'glitch', 'not responding', 'stuck', 'hangs', 'loading',
        'performance', 'unstable', 'error', 'delay', 'latency', 'stutter', 'load time', 'resource intensive',
        'memory leak', 'instability', 'failure', 'unresponsive', 'rendering', 'optimization'
    ]
    df['Performance_Issue'] = df['Review'].str.contains('|'.join(performance_keywords), case=False, na=False)

    support_categories = {
        'No Response': ['no reply', 'no answer', 'ignored', 'no help', 'no response', 'never responded', 'no feedback'],
        'Slow Response': ['slow response', 'took long', 'days to reply', 'delayed response', 'long wait', 'prolonged delay', 'late reply'],
        'Unhelpful': ['not helpful', 'useless', 'did not solve', 'waste of time', 'ineffective', 'unhelpful', 'did not assist', 'no solution', 'failed to resolve'],
        'Rude Staff': ['rude', 'arrogant', 'unprofessional', 'angry', 'impolite', 'disrespectful', 'hostile', 'offensive', 'dismissive']
    }

    df['Support_Complaint'] = df['Review'].str.contains('|'.join(
        [kw for sublist in support_categories.values() for kw in sublist]), case=False, na=False)

    def categorize_complaint(text):
        text = str(text).lower()
        return [category for category, keywords in support_categories.items() if any(k in text for k in keywords)] or ['Other']

    df['Support_Complaint_Type'] = df['Review'].apply(categorize_complaint)
    df['Support_Complaint_Type'] = df['Support_Complaint_Type'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    pricing_keywords = [
        'expensive', 'overpriced', 'pricey', 'too much', 'not worth', 'high price', 'cost too much',
        'unfair', 'cheaper', 'lower price', 'reduce price', 'price hike', 'cost', 'value', 'affordable'
    ]
    df['Pricing_Complaint'] = df['Review'].str.contains('|'.join(pricing_keywords), case=False, na=False)

    subscription_issues = {
        'Cancellation & Auto-Renewal': [
            'auto-renew', 'automatic renewal', 'unsubscribe', 'cancel', 'hard to cancel', 'difficult to cancel',
            'cancellation problems', 'cannot unsubscribe', 'subscription cancellation', 'subscription expired',
            'subscription paused', 'manage subscription'
        ],
        'Billing & Charges': [
            'billing', 'charge', 'payment', 'unexpected charge', 'hidden fee', 'surprise charge', 'unauthorized charge',
            'extra fees', 'unknown charge', 'incorrect billing', 'subscription fees', 'subscription cost', 'iap',
            'in-app purchase', 'recurring payment'
        ],
        'Refund & Disputes': [
            'refund', 'money back', 'not refund', 'no refund', 'refund denied', 'refund issues', 'refund process',
            'refund policy', 'scam'
        ],
        'Value & Experience': [
            'not worth', 'waste of money', 'better free', 'overpriced subscription', 'poor value', 'not worth the cost',
            'expensive for what it offers', 'trial', 'membership', 'subscription service'
        ],
        'Support & Account Issues': [
            'subscription', 'subscription plan', 'subscription model', 'subscription options', 'subscription terms',
            'subscription problems', 'subscription error', 'subscription active', 'subscription status',
            'subscription details', 'subscription access', 'subscription account', 'subscription support',
            'subscription help', 'subscription confirmation', 'subscription history', 'subscription information'
        ]
    }

    def categorize_sub_issue(text):
        text = str(text).lower()
        return [category for category, keywords in subscription_issues.items() if any(k in text for k in keywords)] or ['Other']

    df['Subscription_Complaint'] = df['Review'].str.contains('|'.join(
        [kw for sublist in subscription_issues.values() for kw in sublist]), case=False, na=False)
    df['Subscription_Issue_Type'] = df['Review'].apply(categorize_sub_issue)
    df['Subscription_Issue_Type'] = df['Subscription_Issue_Type'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    df['delivery_issues'] = df['Review'].str.contains('|'.join([
        'late delivery', 'delayed', 'not delivered', 'delivery time', 'driver late', 'took too long',
        'ETA', 'wrong address', 'missed delivery', 'delivery failed', 'reschedule', 'never arrived',
        'package delay', 'delivery issue', 'still waiting', 'came late', 'got it late', 'delay in delivery',
        'order late', 'order not here', 'waiting for my order', 'where is my order', 'running late',
        'delivered to wrong address', 'delivered somewhere else', "did not show up", 'arrived late'
    ]), case=False, na=False)

    df['Payment_Problems'] = df['Review'].str.contains('|'.join([
        'payment failed', 'transaction error', 'card declined', 'not processed', 'double charged',
        'overcharged', 'refund pending', 'refund delay', 'payment issue', "can't pay", 'not refunded',
        'incorrect amount', 'failed to pay', 'billing error', 'charge issue', 'charged twice',
        'money deducted', 'amount not refunded', 'payment stuck', 'did not get refund',
        'transaction declined', 'unable to pay', 'app charged me', 'no confirmation after payment',
        'payment not successful'
    ]), case=False, na=False)

    df['Food_Quality'] = df['Review'].str.contains('|'.join([
        'stale food', 'not fresh', 'cold food', 'bad taste', 'spoiled', 'poor quality', 'packaging issue',
        'leaked', 'damaged package', 'soggy', 'missing items', 'wrong item', 'undercooked', 'overcooked',
        'smells bad', 'rotten', 'food poisoning', 'not edible', 'food was cold', 'food was awful', 'not good',
        'unhygienic', 'dirty packaging', 'weird smell', 'wrong dish', 'order messed up', 'hair in food',
        'low quality', 'bad smell'
    ]), case=False, na=False)

    df['Promotions_Issues'] = df['Review'].str.contains('|'.join([
        'coupon not working', 'promo code invalid', 'offer not applied', 'discount not working',
        'code expired', "can't use promo", 'offer issue', 'not eligible', "didn't get discount",
        'cashback not received', 'free delivery not applied', 'reward not credited', 'code not accepted',
        'voucher did not work', 'promo did not apply', 'invalid promo', 'did not get offer', 'promotion failed',
        'no discount received', 'promo not working', 'discount missing', 'free item not added',
        'applying coupon failed', 'reward missing'
    ]), case=False, na=False)

    request_phrases = [
        'should have', 'need', 'want', 'please add', 'where is', 'why no', 'missing', 'would love',
        'wish there was', 'suggest', 'recommend', 'hope to see', 'require', 'desire', 'looking for',
        'could use', 'it would be great if', 'it would be helpful if', 'is there a way to',
        'is it possible to', 'consider adding', "I'd like to see", "I'm trying to find",
        'can you implement', 'how do I', 'is it possible to get', 'is there', "I'm looking for"
    ]
    df['Feature_Request'] = df['Review'].str.contains('|'.join(request_phrases), case=False, na=False)

    # Convert boolean to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # === Ensure string fields for list values ===
    for col in ['Support_Complaint_Type', 'Subscription_Issue_Type']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        else:
            df[col] = 'No Complaint'
    
    # === Merge metadata into each row of df
    if not app_metadata_df.empty:
        for col in app_metadata_df.columns:
            df[col] = app_metadata_df.iloc[0][col]

    # === Upload ===
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND, autodetect=True)

    try:
        logging.info(f"Uploading unified data to {unified_table_id}")
        client.load_table_from_dataframe(df, unified_table_id, job_config=job_config).result()
        logging.info("✅ Unified data uploaded.")
    except Exception as e:
        logging.error(f"Upload failed: {e}")

    # ✅ Always run this — not just on failure
    def get_existing_app_tables():
        query = f"""
            SELECT table_name
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
            WHERE table_name LIKE '%_unified'
                AND table_name != 'All_Apps_Unified'
        """
        result = client.query(query).result()
        tables = [row.table_name for row in result]
        apps = [(t, t.replace('_unified', '').replace('__', '_')) for t in tables]
        return apps

    apps = get_existing_app_tables()

    union_queries = []
    for table, app_name in apps:
        union_queries.append(f"""
            SELECT *, '{app_name}' AS App_Name
            FROM `{project_id}.{dataset_id}.{table}`
        """)

    union_sql = "\nUNION ALL\n".join(union_queries)

    final_query = f"""
        CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.All_Apps_Unified` AS
        {union_sql}
    """

    try:
        client.query(final_query).result()
        logging.info("✅ All_Apps_Unified table created/refreshed.")
    except Exception as e:
        logging.error(f"Failed to create unified table: {e}")

def generate_lookerstudio_url(app_id: str) -> str:
    report_id = "e0b54129-afc5-4867-bac8-a3f0ed7be6c7"
    page_id = "fhpIF"
    field_id = "df13"

    # Step 1: Construct the filter string using U+2000
    raw_filter = f"include\u20000\u2000IN\u2000{app_id}"

    # Step 2: URL encode the filter string once
    once_encoded = urllib.parse.quote(raw_filter)

    # Step 3: Build the final URL
    full_params = f'%7B%22{field_id}%22:%22{once_encoded}%22%7D'
    final_url = f"https://lookerstudio.google.com/u/0/reporting/{report_id}/page/{page_id}?params={full_params}"

    return final_url


# === Streamlit UI ===
def main():
    st.set_page_config(page_title="Google Play Review Dashboard", layout="wide")
    st.title("📱 Google Play Review Dashboard")

    LOOKER_BASE_URL = "Enter Looker Studio URL"  # Replace with your actual dashboard URL

    app_id = st.text_input("Enter Google Play App ID", "")

    if app_id:
        # Build Looker URL using AppID filter
        looker_url = generate_lookerstudio_url(app_id)

        st.markdown(f"🔍 Filtering dashboard by **AppID: {app_id}**")
        st.markdown(f"[📊 Open Dashboard for **{app_id}**]({looker_url})", unsafe_allow_html=True)

        # Mode selection
        st.subheader("⚙️ Run Review Analysis Pipeline")
        mode_option = st.selectbox("Select Mode", ["Append only new data", "Specify date range", "Overwrite"])

        if mode_option == "Append only new data":
            mode = "append"
            start_date = None
            end_date = None
        elif mode_option == "Specify date range":
            mode = "range"
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            end_date = st.date_input("End Date", value=datetime.now())
        else:
            mode = "overwrite"
            start_date = None
            end_date = None

        if st.button("Run Analysis and Open Dashboard"):
            progress = st.progress(0)
            with st.spinner("🔄 Running pipeline..."):
                try:
                    progress.progress(20)
                    run_pipeline(app_id, mode=mode, start=start_date, end=end_date)
                    progress.progress(100)
                    st.success("✅ Data processed and uploaded successfully.")
                    st.markdown(f"[📊 Open Dashboard for **{app_id}**]({looker_url})", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Pipeline execution failed: {e}")
                    progress.empty()

if __name__ == "__main__":
    main()