import streamlit as st


def apply_custom_styles():

    st.markdown(
        """
        <style>

        .stApp {
            background: linear-gradient(
                135deg,
                #FFF4F1 0%,
                #FFF8F4 48%,
                #F3EAFE 100%
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
                #3D214F 0%,
                #75406F 55%,
                #C35D78 100%
            ) !important;

            padding: 28px !important;
            border-radius: 18px !important;
            color: white !important;
            margin-bottom: 25px !important;

            box-shadow: 0 5px 15px rgba(61, 33, 79, 0.16);
        }

        .hero h1 {
            color: white !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
        }

        .hero p {
            color: #FFEAF0 !important;
            font-size: 1rem !important;
            margin-bottom: 5px !important;
        }

        /* Sidebar */

        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #3D214F 0%,
                #65365F 55%,
                #A84F70 100%
            ) !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: white !important;
        }

        /* Metric Cards */

        div[data-testid="stMetric"] {
            background: linear-gradient(
                135deg,
                #FFFFFF 0%,
                #FBE9EE 100%
            ) !important;

            border: 1px solid #EAD2DF !important;
            padding: 10px !important;
            border-radius: 12px !important;
            min-height: 115px;

            box-shadow: 0 3px 10px rgba(61, 33, 79, 0.06);
        }

        div[data-testid="stMetricLabel"] {
            color: #70415F !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #3D214F !important;
            font-weight: 700 !important;
        }

        /* Buttons */

        .stButton button,
        .stDownloadButton button {

            background: linear-gradient(
                135deg,
                #70406F 0%,
                #C35D78 100%
            ) !important;

            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {

            background: linear-gradient(
                135deg,
                #4B2A5E 0%,
                #A84F70 100%
            ) !important;

            color: white !important;
        }

        /* Dataframes */

        div[data-testid="stDataFrame"] {
            border: 1px solid #EAD2DF !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }

        /* Headings */

        .main h1,
        .main h2,
        .main h3 {
            color: #49324F;
        }

        /* Footer */

        .footer {
            background: linear-gradient(
                135deg,
                #3D214F 0%,
                #75406F 100%
            ) !important;

            color: white !important;
            text-align: center !important;
            padding: 18px !important;
            border-radius: 12px !important;
            margin-top: 35px !important;
        }

        .footer p {
            color: white !important;
            margin: 0 !important;
            font-size: 0.9rem !important;
        }

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

       /* =========================================
   CSV FILE UPLOADER - FINAL TEXT FIX
========================================= */

div[data-testid="stFileUploader"] {
    background: linear-gradient(
        135deg,
        #3D214F 0%,
        #75406F 55%,
        #C35D78 100%
    ) !important;
    padding: 20px !important;
    border-radius: 14px !important;
    border: 1px solid #C35D78 !important;
}

/* Upload heading */
div[data-testid="stFileUploader"] label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Dropzone background */
div[data-testid="stFileUploader"] section {
    background: #3D214F !important;
    border: 2px dashed #FFFFFF !important;
    border-radius: 12px !important;
}

/* Dropzone instruction text */
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] section p,
div[data-testid="stFileUploader"] section span,
div[data-testid="stFileUploader"] section small,
div[data-testid="stFileUploader"] section div {
    color: #FFFFFF !important;
}

/* Browse files button */
div[data-testid="stFileUploader"] section button {
    background: #FFFFFF !important;
    color: #3D214F !important;
    border: 2px solid #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

/* Force ALL button text to dark purple */
div[data-testid="stFileUploader"] section button,
div[data-testid="stFileUploader"] section button *,
div[data-testid="stFileUploader"] section button p,
div[data-testid="stFileUploader"] section button span,
div[data-testid="stFileUploader"] section button div {
    color: #3D214F !important;
    fill: #3D214F !important;
    -webkit-text-fill-color: #3D214F !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Button hover */
div[data-testid="stFileUploader"] section button:hover {
    background: #FFE4EC !important;
}

/* Upload icon */
div[data-testid="stFileUploader"] section svg {
    color: #FFFFFF !important;
    stroke: #FFFFFF !important;
}

/* Uploaded file container */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 8px !important;
}

/* Uploaded filename */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
    color: #FFFFFF !important;
}

/* Remove icon */
div[data-testid="stFileUploader"] button[kind="icon"] svg {
    color: #FFFFFF !important;
    stroke: #FFFFFF !important;
}

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():

    st.html(
        """
        <div class="hero">

            <h1>
                🛒 Online Shoppers Purchase Prediction
            </h1>

            <p>
                Developed using Python, Scikit-learn and Streamlit
            </p>

            <p>
                Dataset: UCI Online Shoppers Purchasing Intention Dataset
            </p>

            <p>
                Machine Learning Assignment 2
            </p>

        </div>
        """
    )


def render_footer():

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