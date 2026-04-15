import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

import streamlit as st

st.title("Test App")
st.write("Now it should work")
# Page configuration
st.set_page_config(
    page_title="Bank Term Deposit Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Load main dataset - adjusted for data folder
    df = pd.read_csv('data/Bank-term-deposit.csv')
    
    # Load contact groups - adjusted for data folder
    group1 = pd.read_csv('data/group1_contacts.csv')
    group2 = pd.read_csv('data/group2_contacts.csv')
    group3 = pd.read_csv('data/group3_contacts.csv')
    
    # Add group labels
    group1['group'] = 'Group 1'
    group2['group'] = 'Group 2'
    group3['group'] = 'Group 3'
    
    groups_df = pd.concat([group1, group2, group3], ignore_index=True)
    
    return df, groups_df

try:
    df, groups_df = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Error loading data files: {e}")
    st.info("Make sure the CSV files are in the `data/` folder")
    st.stop()

# Sidebar
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio("Select Page", [
    "Dashboard Overview",
    "Data Exploration",
    "Customer Segmentation",
    "Campaign Analysis",
    "Contact Management"
])

# ==================== DASHBOARD OVERVIEW ====================
if page == "Dashboard Overview":
    st.title("📊 Bank Term Deposit Campaign Dashboard")
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        conversion_rate = (df['y'] == 'yes').sum() / len(df) * 100
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    
    with col3:
        st.metric("Term Deposits", f"{(df['y'] == 'yes').sum():,}")
    
    with col4:
        avg_age = df['Age'].mean()
        st.metric("Avg Customer Age", f"{avg_age:.1f}")
    
    st.markdown("---")
    
    # Conversion by Key Factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Conversion by Job Category")
        job_conversion = df.groupby('Job').apply(
            lambda x: (x['y'] == 'yes').sum() / len(x) * 100
        ).sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        job_conversion.plot(kind='barh', color='steelblue', ax=ax)
        ax.set_xlabel('Conversion Rate (%)')
        ax.set_title('Top 10 Job Categories by Conversion Rate')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Conversion by Education Level")
        edu_conversion = df.groupby('Education').apply(
            lambda x: (x['y'] == 'yes').sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        edu_conversion.plot(kind='bar', color='coral', ax=ax)
        ax.set_xlabel('Education Level')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('Conversion by Education Level')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Age vs Conversion
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Conversion by Age Group")
        df['age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 65, 100])
        age_conversion = df.groupby('age_group').apply(
            lambda x: (x['y'] == 'yes').sum() / len(x) * 100
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        age_conversion.plot(kind='bar', color='green', ax=ax)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('Conversion by Age Group')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Campaign Duration Impact")
        df['duration_group'] = pd.cut(df['duration'], bins=[0, 180, 360, 600, 5000])
        duration_conversion = df.groupby('duration_group').apply(
            lambda x: (x['y'] == 'yes').sum() / len(x) * 100
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        duration_conversion.plot(kind='bar', color='purple', ax=ax)
        ax.set_xlabel('Call Duration (seconds)')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('Conversion by Call Duration')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ==================== DATA EXPLORATION ====================
elif page == "Data Exploration":
    st.title("🔍 Data Exploration")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Total Features:** {len(df.columns)}")
        st.write(f"**Date Range:** Campaign data")
    
    with col2:
        st.subheader("Target Distribution")
        target_counts = df['y'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#ff6b6b', '#51cf66']
        ax.pie(target_counts, labels=['No', 'Yes'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Term Deposit Subscription')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Statistics
    st.subheader("Numerical Features Statistics")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    stats_df = df[numerical_cols].describe().T
    st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("---")
    
    # Categorical Features
    st.subheader("Categorical Features Distribution")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    selected_cat = st.selectbox("Select Categorical Feature", categorical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{selected_cat} Distribution**")
        cat_counts = df[selected_cat].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        cat_counts.plot(kind='barh', ax=ax, color='steelblue')
        st.pyplot(fig)
    
    with col2:
        st.write(f"**{selected_cat} vs Conversion**")
        conversion_by_cat = df.groupby(selected_cat).apply(
            lambda x: (x['y'] == 'yes').sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        conversion_by_cat.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Conversion Rate (%)')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Raw Data
    st.subheader("Raw Data View")
    st.dataframe(df, use_container_width=True, height=400)

# ==================== CUSTOMER SEGMENTATION ====================
elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown("---")
    
    # Create segments based on various criteria
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segmentation Overview")
        
        # High-Value Customers (high conversion potential)
        high_value = df[(df['duration'] > 300) & (df['y'] == 'yes')]
        st.write(f"**High-Value Customers:** {len(high_value):,}")
        
        # Age-based segments
        young_professionals = df[(df['Age'] < 35) & (df['Job'] != 'retired')]
        st.write(f"**Young Professionals:** {len(young_professionals):,}")
        
        mid_career = df[(df['Age'] >= 35) & (df['Age'] < 50)]
        st.write(f"**Mid-Career Professionals:** {len(mid_career):,}")
        
        senior = df[df['Age'] >= 50]
        st.write(f"**Senior Customers:** {len(senior):,}")
    
    with col2:
        st.subheader("Segment Performance")
        
        segments = {
            'High-Value': len(high_value),
            'Young Prof.': len(young_professionals),
            'Mid-Career': len(mid_career),
            'Senior': len(senior)
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(segments.keys(), segments.values(), color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customer Segments by Size')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Contact groups comparison
    st.subheader("Contact Groups Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Group 1 Size", len(groups_df[groups_df['group'] == 'Group 1']))
    
    with col2:
        st.metric("Group 2 Size", len(groups_df[groups_df['group'] == 'Group 2']))
    
    with col3:
        st.metric("Group 3 Size", len(groups_df[groups_df['group'] == 'Group 3']))
    
    st.markdown("---")
    
    # Job Distribution across segments
    st.subheader("Top Jobs in Each Segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Young Professionals - Top Jobs**")
        young_jobs = young_professionals['Job'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(10, 6))
        young_jobs.plot(kind='barh', ax=ax, color='steelblue')
        st.pyplot(fig)
    
    with col2:
        st.write("**Senior Customers - Top Jobs**")
        senior_jobs = senior['Job'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(10, 6))
        senior_jobs.plot(kind='barh', ax=ax, color='coral')
        st.pyplot(fig)

# ==================== CAMPAIGN ANALYSIS ====================
elif page == "Campaign Analysis":
    st.title("📈 Campaign Analysis")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_contacts = len(df)
        successful = (df['y'] == 'yes').sum()
        st.metric("Success Rate", f"{successful/total_contacts*100:.1f}%")
    
    with col2:
        avg_duration = df['duration'].mean()
        st.metric("Avg Call Duration", f"{avg_duration:.0f}s")
    
    with col3:
        avg_campaign = df['campaign'].mean()
        st.metric("Avg Contacts/Customer", f"{avg_campaign:.1f}")
    
    st.markdown("---")
    
    # Contact Frequency Impact
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campaign Contacts Impact")
        campaign_conversion = df.groupby(pd.cut(df['campaign'], bins=5)).apply(
            lambda x: (x['y'] == 'yes').sum() / len(x) * 100
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        campaign_conversion.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel('Number of Contacts')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('Conversion by Number of Campaign Contacts')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Previous Contacts Impact")
        previous_conversion = df[df['previous'] > 0].groupby(pd.cut(df[df['previous'] > 0]['previous'], bins=5)).apply(
            lambda x: (x['y'] == 'yes').sum() / len(x) * 100
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        previous_conversion.plot(kind='bar', ax=ax, color='coral')
        ax.set_xlabel('Previous Contacts')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('Conversion by Previous Contacts')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Monthly Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Conversion by Month")
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month_conversion = df.groupby('Month').apply(
            lambda x: (x['y'] == 'yes').sum() / len(x) * 100
        )
        month_conversion = month_conversion.reindex([m for m in month_order if m in month_conversion.index])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        month_conversion.plot(kind='bar', ax=ax, color='green')
        ax.set_xlabel('Month')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('Campaign Performance by Month')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Contacts by Day of Week")
        day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        day_counts = df['day_of_week'].value_counts()
        day_counts = day_counts.reindex([d for d in day_order if d in day_counts.index])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        day_counts.plot(kind='bar', ax=ax, color='purple')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Number of Contacts')
        ax.set_title('Campaign Contacts by Day of Week')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ==================== CONTACT MANAGEMENT ====================
elif page == "Contact Management":
    st.title("📞 Contact Management")
    st.markdown("---")
    
    # Group Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        g1_contacts = groups_df[groups_df['group'] == 'Group 1']
        st.metric("Group 1 Contacts", len(g1_contacts))
    
    with col2:
        g2_contacts = groups_df[groups_df['group'] == 'Group 2']
        st.metric("Group 2 Contacts", len(g2_contacts))
    
    with col3:
        g3_contacts = groups_df[groups_df['group'] == 'Group 3']
        st.metric("Group 3 Contacts", len(g3_contacts))
    
    st.markdown("---")
    
    # Select group to view
    selected_group = st.selectbox("Select Contact Group", ["Group 1", "Group 2", "Group 3"])
    
    group_data = groups_df[groups_df['group'] == selected_group].copy()
    
    st.subheader(f"{selected_group} - Contact Information")
    st.write(f"Total Contacts: {len(group_data)}")
    
    # Display contacts with search
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_id = st.text_input(f"Search customer ID in {selected_group}")
        if search_id:
            filtered = group_data[group_data['customer_id'].str.contains(search_id, case=False)]
            st.dataframe(filtered, use_container_width=True)
        else:
            st.dataframe(group_data.head(50), use_container_width=True)
    
    with col2:
        st.write("")
        if st.download_button(
            label=f"Download {selected_group}",
            data=group_data.to_csv(index=False),
            file_name=f"{selected_group.lower().replace(' ', '_')}_contacts.csv",
            mime="text/csv"
        ):
            st.success("Downloaded!")
    
    st.markdown("---")
    
    # Contact Groups Comparison
    st.subheader("Contact Distribution")
    
    group_sizes = {
        'Group 1': len(g1_contacts),
        'Group 2': len(g2_contacts),
        'Group 3': len(g3_contacts)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    wedges, texts, autotexts = ax.pie(
        group_sizes.values(),
        labels=group_sizes.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax.set_title('Contact Distribution Across Groups')
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Export options
    st.subheader("Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export All Groups Combined"):
            combined = groups_df.to_csv(index=False)
            st.download_button(
                label="Download Combined Groups",
                data=combined,
                file_name="all_contacts_combined.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Summary Statistics"):
            summary = {
                'Group': ['Group 1', 'Group 2', 'Group 3'],
                'Total Contacts': [len(g1_contacts), len(g2_contacts), len(g3_contacts)]
            }
            summary_df = pd.DataFrame(summary)
            st.download_button(
                label="Download Summary",
                data=summary_df.to_csv(index=False),
                file_name="contacts_summary.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>Bank Term Deposit Campaign Dashboard | © 2024</p>
    <p style='font-size: 12px; color: #666;'>Data-driven insights for successful campaign management</p>
    </div>
    """, unsafe_allow_html=True)