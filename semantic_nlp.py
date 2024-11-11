import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Data Loading & Preprocessing
def load_and_preprocess_data(file_path):
    """
    Purpose: Load the dataset and perform initial data cleaning.
    
    Input:
    - file_path (str): Path to the CSV file containing the dataset.
    
    Output:
    - df (DataFrame): A cleaned Pandas DataFrame with rows containing abstracts.
    
    Steps:
    1. Load the CSV file into a DataFrame.
    2. Remove rows where the 'Abstract' field is missing.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    # Data Cleaning: Drop rows where Abstract is missing
    df = df.dropna(subset=['Abstract'])
    return df

# Step 2: Improved Filtering Using TF-IDF
def filter_relevant_papers(df, target_topics, threshold=0.2):
    """
    Purpose: Filter papers that are relevant using TF-IDF-based similarity to given topics.
    
    Input:
    - df (DataFrame): DataFrame containing paper details including abstracts.
    - target_topics (list): List of target topics to compare against.
    - threshold (float): Similarity threshold for filtering.

    Output:
    - filtered_df (DataFrame): DataFrame containing only the papers that are relevant.
    """
    # Combine all abstracts and topics
    documents = df['Abstract'].tolist() + target_topics

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity of each abstract with target topics
    similarities = cosine_similarity(tfidf_matrix[:-len(target_topics)], tfidf_matrix[-len(target_topics):])
    max_similarities = np.max(similarities, axis=1)

    # Filter based on threshold
    df['similarity'] = max_similarities
    filtered_df = df[df['similarity'] >= threshold]
    return filtered_df

# Step 3: Classifying the Papers
def classify_papers(df):
    """
    Purpose: Classify the papers based on the type of deep learning method used.
    
    Input:
    - df (DataFrame): DataFrame containing filtered papers.
    
    Output:
    - df (DataFrame): DataFrame with an additional 'method_type' column indicating the classification.
    
    Steps:
    1. Define classification categories and associated keywords.
    2. Classify each abstract into one of the predefined categories.
    """
    # Define classification categories
    categories = {
        'text mining': ['text mining', 'natural language processing', 'nlp', 'information retrieval', 'sentiment analysis', 'topic modeling', 'text classification', 'entity recognition', 'named entity recognition', 'tokenization', 'stemming', 'lemmatization', 'keyword extraction'],
        'computer vision': ['image', 'segmentation', 'cnn', 'convolutional', 'object detection', 'visual recognition', 'image classification', 'image analysis', 'semantic segmentation', 'feature extraction']
    }

    # Classification Function
    def classify_paper(abstract):
        found_categories = []
        for category, keywords in categories.items():
            if any(keyword.lower() in abstract.lower() for keyword in keywords):
                found_categories.append(category)

        if len(found_categories) == 2:
            return 'both'
        elif len(found_categories) == 1:
            return found_categories[0]
        else:
            return 'other'

    # Apply classification
    df['method_type'] = df['Abstract'].apply(classify_paper)
    return df

# Step 4: Extract and Report the Method Name
def extract_methods(df):
    """
    Purpose: Extract the names of the deep learning methods used in each paper.
    
    Input:
    - df (DataFrame): DataFrame containing classified papers.
    
    Output:
    - df (DataFrame): DataFrame with an additional 'methods_used' column listing the extracted methods.
    
    Steps:
    1. Define regex patterns for common machine learning methods.
    2. Extract the methods mentioned in each abstract using regex.
    """
    # Define regex patterns for common machine learning methods
    method_patterns = [
        # Deep Learning Methods
        r'\bCNN\b', r'\bRNN\b', r'\btransformer\b', r'\bLSTM\b',
        r'\bGAN\b', r'\bautoencoder\b', r'\bBERT\b', r'\bDeep Q-Network\b', r'\bResNet\b',
        r'\bVGG\b', r'\bInception\b', r'\bGRU\b', r'\bCapsule Network\b', r'\bDenseNet\b',
        # General Machine Learning Methods
        r'\bSVM\b', r'\bSupport Vector Machine\b', r'\bRandom Forest\b', r'\bDecision Tree\b',
        r'\bK-Nearest Neighbors\b', r'\bKNN\b', r'\bNaive Bayes\b', r'\bLogistic Regression\b',
        r'\bLinear Regression\b', r'\bPCA\b', r'\bPrincipal Component Analysis\b',
        r'\bK-Means\b', r'\bHierarchical Clustering\b',
        # General Text Mining Methods
        r'\bTF-IDF\b', r'\bBag of Words\b', r'\bBoW\b', r'\bWord2Vec\b', r'\bDoc2Vec\b',
        r'\bLatent Dirichlet Allocation\b', r'\bLDA\b', r'\bTopic Modeling\b'
    ]

    # Extraction Function
    def extract_method(abstract):
        methods_found = []
        for pattern in method_patterns:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            methods_found.extend(matches)
        return ', '.join(set(methods_found)) if methods_found else 'Unknown'

    # Apply method extraction
    df['methods_used'] = df['Abstract'].apply(extract_method)
    return df

# Step 5: Generate Results
def generate_results(df, threshold):
    """
    Purpose: Generate plots for data visualization and provide statistical overview.
    
    Input:
    - df (DataFrame): DataFrame containing filtered and classified papers.
    - threshold (int): Threshold for grouping less common methods into 'Other'.
    
    Output:
    - Saves bar and pie charts as PNG files to visualize the distribution of relevant papers and methods used.
    - Prints statistical summaries of the dataset.
    
    Steps:
    1. Plot the count of papers classified by method type.
    2. Plot a pie chart showing the proportion of each classification.
    3. Plot the count of methods used.
    4. Print dataset statistics.
    """
    # Bar plot for method types
    method_counts = df['method_type'].value_counts()
    plt.figure(figsize=(10, 6))
    method_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Papers by Method Type')
    plt.xlabel('Method Type')
    plt.ylabel('Number of Papers')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout(pad=2)
    plt.savefig('img/distribution_of_papers_by_method_type.png')
    plt.close()

    # Pie chart for method types
    plt.figure(figsize=(8, 8))
    method_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue', 'lightgreen', 'gold'])
    plt.title('Proportion of Papers by Method Type')
    plt.ylabel('')
    plt.tight_layout(pad=2)
    plt.savefig('img/proportion_of_papers_by_method_type.png')
    plt.close()

    # Bar plot for methods used (grouping less common methods as 'Other')
    method_used_counts = df['methods_used'].value_counts().sort_values(ascending=False)

    # This code sets a threshold to filter the methods used, ensuring only methods with at least 'threshold' occurrences are individually represented.
    # Less frequent methods are grouped into an 'Other' category to simplify the visualization and make the plot easier to read.
    filtered_method_counts = method_used_counts[method_used_counts >= threshold]
    filtered_method_counts['Other'] = method_used_counts[method_used_counts < threshold].sum()
    
    plt.figure(figsize=(12, 6))
    filtered_method_counts.plot(kind='bar', color='mediumpurple')
    plt.title('Distribution of Papers by Methods Used (Grouped)')
    plt.xlabel('Method Used')
    plt.ylabel('Number of Papers')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout(pad=2)
    plt.savefig('img/distribution_of_papers_by_methods_used_grouped.png')
    plt.close()

    # Pie chart for methods used (grouping less common methods as 'Other')
    plt.figure(figsize=(10, 10))
    filtered_method_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue', 'lightgreen', 'gold', 'mediumpurple', 'orange', 'cyan'])
    plt.title('Proportion of Papers by Methods Used (Grouped)')
    plt.ylabel('')
    plt.tight_layout(pad=2)
    plt.savefig('img/proportion_of_papers_by_methods_used_grouped.png')
    plt.close()

    # Statistical Overview
    total_papers = len(df)
    num_methods_used = df['methods_used'].nunique()
    most_common_method = df['methods_used'].value_counts().idxmax()
    most_common_method_count = df['methods_used'].value_counts().max()
    num_method_types = df['method_type'].nunique()
    most_common_method_type = df['method_type'].value_counts().idxmax()
    most_common_method_type_count = df['method_type'].value_counts().max()

    # List top methods and types by frequency
    top_methods = df['methods_used'].value_counts().head(10).reset_index()
    top_methods.columns = ['Method', 'Frequency']
    top_methods['Percentage'] = (top_methods['Frequency'] / total_papers) * 100
    
    top_method_types = df['method_type'].value_counts().head(10).reset_index()
    top_method_types.columns = ['Method Type', 'Frequency']
    top_method_types['Percentage'] = (top_method_types['Frequency'] / total_papers) * 100

    print("\nStatistical Overview:")
    print(f"Total number of papers: {total_papers}")
    print(f"\nNumber of unique method types: {num_method_types}")
    print(f"Most common method type: {most_common_method_type} ({most_common_method_type_count} occurrences)")
    print("\nTop Method Types by Frequency:")
    print(top_method_types)
    print(f"\n\nNumber of unique methods used: {num_methods_used}")
    print(f"Most common method used: {most_common_method} ({most_common_method_count} occurrences)")
    print("\nTop Methods by Frequency:")
    print(top_methods)

# Main function to orchestrate the workflow
def main():
    """
    Purpose: Orchestrate the entire workflow from data loading to saving the final results.
    
    Steps:
    1. Load and preprocess the data.
    2. Filter relevant papers using TF-IDF-based similarity.
    3. Classify the filtered papers.
    4. Extract the methods used in each paper.
    5. Save the results to a CSV file.
    6. Generate and save results including plots and statistical overview.
    
    Input:
    - Input file path and output file path provided as command-line arguments.
    - Optional threshold value for grouping less frequent methods.
    
    Output:
    - CSV file saved containing filtered, classified papers with extracted method information.
    - PNG files visualizing the distribution of paper classifications.
    - Prints statistical summary.
    """
    parser = argparse.ArgumentParser(description="Process and classify virology papers.")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing paper data')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file to save filtered results')
    parser.add_argument('--threshold', type=int, default=7, help='Threshold for grouping less frequent methods (default: 7)')
    parser.add_argument('--similarity_threshold', type=float, default=0.2, help='Threshold for TF-IDF similarity filtering (default: 0.2)')
    args = parser.parse_args()

    # Step 1: Load and preprocess data
    df = load_and_preprocess_data(args.input_file)

    # Step 2: Filter relevant papers using TF-IDF-based similarity
    target_topics = ["deep learning", "neural network", "virology", "epidemiology"]
    filtered_df = filter_relevant_papers(df, target_topics, threshold=args.similarity_threshold)

    # Step 3: Classify the papers
    classified_df = classify_papers(filtered_df)

    # Step 4: Extract methods used
    final_df = extract_methods(classified_df)

    # Save the final DataFrame to CSV
    final_df.to_csv(args.output_file, index=False)
    print("Filtering and classification completed. Results saved to {}".format(args.output_file))

    # Step 5: Generate results (including plots and statistical overview)
    generate_results(final_df, args.threshold)

# Run the main function
if __name__ == "__main__":
    main()

