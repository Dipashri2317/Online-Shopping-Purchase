
"""Custom Streamlit styling and page layout components."""

import streamlit as st


def apply_custom_styles():
    """Apply the teal and emerald gradient theme."""

    st.markdown(
        """
        <style>

        /* Main application background */
        .stApp {
            background: linear-gradient(
                135deg,
                #F0F8F5 0%,
                #F7FBF9 50%,
                #E8F4EF 100%
            ) !important;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Header */
        .hero {
            background: linear-gradient(
                135deg,
                #102C35 0%,
                #126B68 55%,
                #20A486 100%
            ) !important;
            padding: 28px !important;
            border-radius: 18px !important;
            color: #ffffff !important;
            margin-bottom: 25px !important;
            box-shadow: 0 5px 15px rgba(16, 44, 53, 0.15) !important;
        }

        .hero h1 {
            color: #ffffff !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
        }

        .hero p {
            color: #E4F3EF !important;
            font-size: 1rem !important;
            margin-bottom: 5px !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #102C35 0%,
                #155E59 55%,
                #168574 100%
            ) !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] span {
            color: #ffffff !important;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(
                135deg,
                #FFFFFF 0%,
                #E4F3EF 100%
            ) !important;
            border: 1px solid #CFE5DC !important;
            padding: 10px !important;
            border-radius: 12px !important;
            min-height: 115px;
            box-shadow: 0 3px 10px rgba(16, 44, 53, 0.06);
        }

        div[data-testid="stMetricLabel"] {
            color: #37645C !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #126B68 !important;
            font-weight: 700 !important;
        }

        /* Sidebar file uploader */
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: rgba(255, 255, 255, 0.12) !important;
            border: 1px dashed #CFE5DC !important;
            border-radius: 10px !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section span,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section p {
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background: #ffffff !important;
            color: #153D3A !important;
            border: 1px solid #CFE5DC !important;
            border-radius: 8px !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button span,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button p {
            color: #153D3A !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            background: #ffffff !important;
            border-radius: 8px !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
            color: #153D3A !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] {
            background: #ffffff !important;
            color: #153D3A !important;
        }

        /* Buttons */
        .stButton button,
        .stDownloadButton button {
            background: linear-gradient(
                135deg,
                #126B68,
                #168574
            ) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            background: linear-gradient(
                135deg,
                #102C35,
                #126B68
            ) !important;
            color: #ffffff !important;
            border: none !important;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid #CFE5DC !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }

        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stTable"] {
            color: #153D3A !important;
        }

        /* Select boxes */
        div[data-baseweb="select"] > div {
            border-radius: 8px !important;
            border-color: #BFDCD1 !important;
        }

        /* Main headings */
        .main h1,
        .main h2,
        .main h3 {
            color: #153D3A;
        }

        .hero h1,
        .hero p {
            color: #ffffff !important;
        }

        /* Footer */
        .footer {
            background: linear-gradient(
                135deg,
                #102C35 0%,
                #126B68 100%
            ) !important;
            color: #ffffff !important;
            text-align: center !important;
            padding: 18px !important;
            border-radius: 12px !important;
            margin-top: 35px !important;
        }

        .footer p {
            color: #ffffff !important;
            margin: 0 !important;
            font-size: 0.9rem !important;
        }

        /* Responsive styling */
        @media (max-width: 768px) {
            .hero {
                padding: 20px !important;
            }

            .hero h1 {
                font-size: 1.7rem !important;
            }

            div[data-testid="stMetric"] {
                height: auto;
            }
        }

        .stApp .hero {
            background: linear-gradient(
                135deg,
                #102C35 0%,
                #126B68 55%,
                #20A486 100%
            ) !important;
            padding: 28px !important;
            border-radius: 18px !important;
            margin-bottom: 25px !important;
        }

        .stApp .hero h1 {
            color: #ffffff !important;
        }

        .stApp .hero p {
            color: #E4F3EF !important;
        }

        .stApp .footer {
            background: linear-gradient(
                135deg,
                #102C35 0%,
                #126B68 100%
            ) !important;
            color: #ffffff !important;
            text-align: center !important;
            padding: 18px !important;
            border-radius: 12px !important;
            margin-top: 35px !important;
        }

        .stApp .footer p {
            color: #ffffff !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
    """Display the application header."""

    st.html(
        """
        <div class="hero">
            <h1>🛒 Online Shoppers Purchase Prediction</h1>
            <p>Developed using Python, Scikit-learn and Streamlit</p>
            <p>Dataset: UCI Online Shoppers Purchasing Intention Dataset</p>
            <p>Machine Learning Assignment 2</p>
        </div>
        """
    )


def render_footer():
    """Display the application footer."""

    st.html(
        """
        <div class="footer">
            <p>
                Online Shoppers Purchase Prediction |
                Machine Learning Assignment 2
            </p>
        </div>
        """
    )
