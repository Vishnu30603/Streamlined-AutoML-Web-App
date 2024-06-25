import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import (
    setup as setup_classification,
    compare_models as compare_models_classification,
    pull as pull_classification,
    save_model as save_model_classification,
    predict_model as predict_model_classification,
    plot_model as plot_model_classification,
    get_config as get_config_classification
)
from pycaret.regression import (
    setup as setup_regression,
    compare_models as compare_models_regression,
    pull as pull_regression,
    save_model as save_model_regression,
    predict_model as predict_model_regression,
    plot_model as plot_model_regression,
    get_config as get_config_regression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering

# Helper function to display distribution plots
def plot_distribution(df, numerical_columns):
    """Display distribution plots for numerical columns."""
    for col in numerical_columns:
        st.write(f"Distribution for {col}:")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

# Helper function to display bar plots for categorical columns
def plot_categorical(df, categorical_columns):
    """Display bar plots for categorical columns."""
    for col in categorical_columns:
        st.write(f"Distribution for {col}:")
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

# Helper function to display box plots for numerical columns
def plot_boxplot(df, numerical_columns):
    """Display box plots for numerical columns."""
    st.write("Boxplots for Numerical Columns:")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[numerical_columns], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Helper function to display correlation matrix
def plot_correlation_matrix(df):
    """Display the correlation matrix."""
    st.header("Correlation Matrix")
    corr = df.select_dtypes(include=['number']).corr()  # Only numerical columns
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size to make the heatmap larger
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Helper function to save and display plot
def save_and_display_plot(plot_func, plot_name):
    """Save and display plot."""
    plot_func(save=True)
    if os.path.exists(f"{plot_name}.png"):
        st.image(f"{plot_name}.png")
    else:
        st.warning(f"Failed to save and display {plot_name} plot.")

# Function to plot clusters
def plot_clusters(df, labels, title):
    """Plot clusters for clustering analysis."""
    df['Cluster'] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='Cluster', palette='viridis')
    plt.title(title)
    st.pyplot(plt.gcf())

# Streamlit app
with st.sidebar:
    st.image("https://r2.erweima.ai/imgcompressed/compressed_c39588fa888ad6a222a90b6ce6c2df62.webp")
    st.title("AutoML")
    choice = st.radio("Navigation Menu", ["Upload Data", "Explore Data", "Build Model", "Download Model"])
    st.info("Welcome to AutoML!")

if os.path.exists('sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)

if choice == "Upload Data":
    st.title("Upload Your Dataset for Modelling!")
    file = st.file_uploader("Upload your .csv file here")
    if file:
        try:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading file: {e}")

if choice == "Explore Data":
    st.title("Automated Exploratory Data Analysis")

    if 'df' in locals() and not df.empty:
        st.header("Dataset")
        st.dataframe(df)

        st.header("Data Overview")
        st.write("Shape of the dataset:", df.shape)
        st.write("Data Types:", df.dtypes)
        st.write("Summary Statistics:", df.describe())
        st.write("Missing Values:", df.isnull().sum())
        st.write("Unique Values:", df.nunique())

        st.header("Distribution of Numerical Columns")
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        plot_distribution(df, numerical_columns)
        plot_boxplot(df, numerical_columns)

        st.header("Distribution of Categorical Columns")
        categorical_columns = df.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            plot_categorical(df, categorical_columns)
        else:
            st.write("No categorical columns found.")

        plot_correlation_matrix(df)
    else:
        st.warning("Please upload a dataset to proceed.")

if choice == "Build Model":
    st.title("Model Building")

    if 'df' in locals() and not df.empty:
        problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression", "Clustering"])

        if problem_type != "Clustering":
            target = st.selectbox("Select Your Target", df.columns)

        feature_selection = st.checkbox('Perform Feature Selection', value=False)
        features = st.multiselect('Select Features to Exclude from Model', options=df.columns.tolist(), default=[])
        preprocess = st.checkbox('Preprocess Data', value=True)
        fix_imbalance = st.checkbox('Fix Imbalance', value=False)
        remove_outliers = st.checkbox('Remove Outliers', value=False)
        outliers_threshold = st.slider('Outliers Threshold', min_value=0.01, max_value=0.1, value=0.05)
        fold_count = st.slider('Number of Cross-Validation Folds', min_value=3, max_value=10, value=5)
        n_select_models = st.slider('Number of Models to Compare', min_value=1, max_value=5, value=3)

        if problem_type == "Clustering":
            cluster_method = st.selectbox("Select Clustering Method", ["KMeans", "Agglomerative Clustering"])
            n_clusters = st.slider('Select Number of Clusters', min_value=2, max_value=10, value=3)

        if st.button('Train Model'):
            if problem_type != "Clustering":
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target] if problem_type == "Classification" else None)
                test_df.to_csv("testdata.csv", index=None)
                train_df = train_df.dropna(subset=[target])
                selected_features = [col for col in df.columns if col not in features and col != target]
            else:
                train_df = df.dropna()
                selected_features = [col for col in df.columns if col not in features]

            with st.spinner("Training model, please wait..."):
                try:
                    if problem_type == "Classification":
                        clf1 = setup_classification(
                            data=train_df[selected_features + [target]], target=target, fix_imbalance=fix_imbalance,
                            remove_outliers=remove_outliers, outliers_threshold=outliers_threshold, preprocess=preprocess,
                            feature_selection=feature_selection, fold=fold_count, verbose=False
                        )
                        setup_df = pull_classification()
                        top_models = compare_models_classification(n_select=n_select_models)
                        compare_df = pull_classification()
                        save_model_classification(top_models[0], "best_model")

                    elif problem_type == "Regression":
                        clf1 = setup_regression(
                            data=train_df[selected_features + [target]], target=target, remove_outliers=remove_outliers,
                            outliers_threshold=outliers_threshold, preprocess=preprocess, feature_selection=feature_selection,
                            fold=fold_count, verbose=False
                        )
                        setup_df = pull_regression()
                        top_models = compare_models_regression(n_select=n_select_models)
                        compare_df = pull_regression()
                        save_model_regression(top_models[0], "best_model")

                    else:
                        # Encoding categorical features
                        for col in train_df.select_dtypes(include=['object']).columns:
                            le = LabelEncoder()
                            train_df[col] = le.fit_transform(train_df[col])

                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(train_df[selected_features])

                        if cluster_method == "KMeans":
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            labels = kmeans.fit_predict(scaled_data)
                            plot_clusters(train_df[selected_features], labels, 'KMeans Clustering')
                            # Save the clustering model
                            pd.to_pickle(kmeans, "best_cluster_model.pkl")
                        elif cluster_method == "Agglomerative Clustering":
                            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
                            labels = agg_clustering.fit_predict(scaled_data)
                            plot_clusters(train_df[selected_features], labels, 'Agglomerative Clustering')
                            # Save the clustering model
                            pd.to_pickle(agg_clustering, "best_cluster_model.pkl")

                        st.write("Cluster Model Summary")
                        st.dataframe(train_df[selected_features].assign(Cluster=labels))

                    if problem_type != "Clustering":
                        st.info("This is the ML Experiment Settings")
                        st.dataframe(setup_df)

                        st.info("This is the ML Model Comparison")
                        st.dataframe(compare_df)

                    if problem_type == "Classification":
                        st.write("Test Set Evaluation Metrics")
                        test_df = pd.read_csv("testdata.csv")

                        metrics_list = []
                        for model in top_models:
                            predictions = predict_model_classification(model, data=test_df)
                            metrics = pull_classification()
                            metrics_list.append(metrics)

                        metrics_df = pd.concat(metrics_list, keys=[f"Model {i+1}" for i in range(len(top_models))])
                        st.write(metrics_df)

                        st.header("Confusion Matrix")
                        save_and_display_plot(lambda save: plot_model_classification(top_models[0], plot='confusion_matrix', save=save), "Confusion Matrix")

                        st.header("ROC Curve")
                        save_and_display_plot(lambda save: plot_model_classification(top_models[0], plot='auc', save=save), "AUC")

                    elif problem_type == "Regression":
                        st.write("Test Set Evaluation Metrics")
                        test_df = pd.read_csv("testdata.csv")

                        metrics_list = []
                        for model in top_models:
                            predictions = predict_model_regression(model, data=test_df)
                            metrics = pull_regression()
                            metrics_list.append(metrics)

                        metrics_df = pd.concat(metrics_list, keys=[f"Model {i+1}" for i in range(len(top_models))])
                        st.write(metrics_df)

                        st.header("Residuals Plot")
                        save_and_display_plot(lambda save: plot_model_regression(top_models[0], plot='residuals', save=save), "Residuals")

                    if problem_type != "Clustering":
                        st.header("Pipeline")
                        try:
                            if problem_type == "Classification":
                                pipeline = get_config_classification('pipeline')
                            elif problem_type == "Regression":
                                pipeline = get_config_regression('pipeline')

                            st.text(pipeline)
                        except Exception as e:
                            st.warning("Pipeline plot is not available for this model.")
                            st.error(e)

                except Exception as e:
                    st.error(f"Error during model training: {e}")
    else:
        st.warning("Please upload a dataset to proceed.")

if choice == "Download Model":
    st.title("Download Your Model")
    download_clustering_model = st.checkbox('Download Clustering Model')
    if download_clustering_model:
        model_file = "best_cluster_model.pkl"
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                st.download_button("Download Clustering Model", f, model_file)
        else:
            st.warning("Clustering model file not found. Please train a model first.")
    else:
        model_file = "best_model.pkl"
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                st.download_button("Download Model", f, model_file)
        else:
            st.warning("Model file not found. Please train a model first.")

    if os.path.exists("cleaned_data.csv"):
        with open("cleaned_data.csv", 'rb') as f:
            st.download_button("Download Cleaned Dataset", f, "cleaned_data.csv")
    else:
        st.warning("Cleaned dataset file not found. Please train a model first.")































