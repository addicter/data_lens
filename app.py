# pip install streamlit pandas numpy scipy matplotlib plotly pyarrow pillow kaleido openai tiktoken
# streamlit run app.py --server.port 8080

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
from typing import Dict, List, Set, Any, Optional
import plotly.express as px
import matplotlib.pyplot as plt
import io
import base64
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os
import tempfile
from PIL import Image
from scipy import stats
import openai
from openai import OpenAI
import contextlib
import tiktoken


def num_tokens_from_string(string, model="gpt-4"):
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except:
        # If tiktoken fails, make a rough estimate
        return len(string.split()) * 1.5


def prepare_ai_dataset_summary(df, analysis, profiler, max_tokens=3500):
    """Prepare a dataset summary for the AI with token limit control"""
    # Basic information
    dataset_summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_data": df.head(3).to_dict(orient="records"),
        "schema": profiler.get_schema(),
    }
    
    # Add summary statistics for numerical columns
    if "column_stats" in analysis and analysis["column_stats"]:
        dataset_summary["numerical_stats"] = analysis["column_stats"]
    
    # Add null value information
    if "null_values" in analysis and analysis["null_values"]:
        dataset_summary["null_values"] = [{
            "column": item["Column"],
            "null_count": item["Null Count"],
            "percentage": item["Percentage"]
        } for item in analysis["null_values"]]
    
    # Add correlation information (limited)
    if ("correlation" in analysis and analysis["correlation"] and 
        "strong_correlations" in analysis["correlation"] and 
        analysis["correlation"]["strong_correlations"]):
        dataset_summary["strong_correlations"] = analysis["correlation"]["strong_correlations"]
    
    # Add duplicate information
    if "duplicates" in analysis and "exact_duplicates" in analysis["duplicates"]:
        dataset_summary["duplicate_info"] = {
            "exact_duplicates": {
                "count": analysis["duplicates"]["exact_duplicates"]["count"],
                "percentage": analysis["duplicates"]["exact_duplicates"]["percentage"]
            }
        }
    
    # Check token count and trim if necessary
    try:
        dataset_json = json.dumps(dataset_summary)
        token_count = num_tokens_from_string(dataset_json)
        
        # If token count is too high, trim progressively
        if token_count > max_tokens:
            # First, reduce sample data
            dataset_summary["sample_data"] = df.head(1).to_dict(orient="records")
            
            # Then limit schema information
            if "schema" in dataset_summary and len(dataset_summary["schema"]) > 10:
                dataset_summary["schema"] = dataset_summary["schema"][:10]
                dataset_summary["schema_note"] = "Schema truncated due to size constraints"
            
            # Finally, limit column types if still needed
            dataset_json = json.dumps(dataset_summary)
            token_count = num_tokens_from_string(dataset_json)
            
            if token_count > max_tokens and len(dataset_summary["column_types"]) > 10:
                column_keys = list(dataset_summary["column_types"].keys())
                dataset_summary["column_types"] = {k: dataset_summary["column_types"][k] for k in column_keys[:10]}
                dataset_summary["column_types_note"] = f"Only showing 10 out of {len(column_keys)} columns due to size limitations"
            
            # If still too large, remove less critical information
            dataset_json = json.dumps(dataset_summary)
            token_count = num_tokens_from_string(dataset_json)
            
            if token_count > max_tokens:
                if "numerical_stats" in dataset_summary:
                    del dataset_summary["numerical_stats"]
                    dataset_summary["removed_info"] = "Detailed numerical statistics removed due to size constraints"
    except:
        # If there's any error in token counting, just return the basic summary
        pass
        
    return dataset_summary


def query_openai(prompt, dataset_info, api_key, file_name):
    """Send query to OpenAI and get response"""
    client = OpenAI(api_key=api_key)
    
    try:
        # Prepare the messages
        messages = [
            {"role": "system", "content": f"""You are an expert data analyst assistant that helps users understand and analyze their datasets. 
             You have been provided with information about the user's dataset '{file_name}' including schema, sample data, and initial analysis.
             Always base your responses on the actual data provided. When suggesting visualizations or analyses, make them specific to the columns in this dataset.
             You can provide code examples in Python using libraries like pandas, matplotlib, seaborn, or plotly.
             If you generate code, make sure it's correct, properly indented, and handles potential errors (like missing data).
             Structure your responses with clear headings and bullet points when appropriate."""},
            {"role": "user", "content": f"Here is information about my dataset: {json.dumps(dataset_info)}"},
            {"role": "user", "content": prompt}
        ]
        
        # Call the API with gpt-4o-mini model
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Changed to gpt-4o-mini
            messages=messages,
            temperature=0.4,  # Lower temperature for more focused responses
            max_tokens=2000    # Increased token limit for more comprehensive responses
        )
        
        return response.choices[0].message.content, None
        
    except Exception as e:
        return None, str(e)

def ai_assistant_tab(tab, df, analysis, profiler, uploaded_file):
    with tab:
        st.markdown('<h2 class="sub-header">🤖 AI Dataset Assistant</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="converter-card">
            <h3>Chat with your data</h3>
            <p>Ask questions about your dataset and get AI-powered insights and visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # API key input with better validation
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input("Enter your OpenAI API Key", type="password", 
                                help="Your API key is securely used only for this session and not stored persistently.")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            test_connection = st.button("Test Connection")
        
        if test_connection and api_key:
            try:
                # Simple test to validate API key
                client = OpenAI(api_key=api_key)
                models = client.models.list()
                st.success("✅ API Key is valid! Connection successful.")
                # Store API key in session state
                st.session_state.api_key = api_key
                st.session_state.api_key_valid = True
            except Exception as e:
                st.error(f"❌ Invalid API Key or connection error: {str(e)}")
                st.session_state.api_key_valid = False
        
        # Store API key in session state when entered
        if api_key and not test_connection:
            if 'api_key' not in st.session_state or st.session_state.api_key != api_key:
                st.session_state.api_key = api_key
                st.session_state.api_key_valid = True  # Assume valid until proven otherwise
        
        # Only show the chat interface if a dataset is loaded and API key is provided
        api_key_valid = st.session_state.get('api_key_valid', False) if 'api_key' in st.session_state else False
        
        if uploaded_file is not None and df is not None and api_key and api_key_valid:
            
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []
            
            # Dataset summary for the AI to reference
            file_name = uploaded_file.name
            
            # Preparing dataset summary with progress indication
            with st.spinner("Preparing dataset for AI analysis..."):
                dataset_summary = prepare_ai_dataset_summary(df, analysis, profiler)
                st.success("Dataset prepared for AI analysis!")
            
            # User prompt section with improved UI
            st.markdown("### Ask about your data")
            
            # Add some example prompts in an expander
            with st.expander("Example prompts", expanded=False):
                st.markdown("""
                Try asking questions like:
                - Summarize this dataset and identify its key characteristics
                - What are the main data quality issues and how should I fix them?
                - Which columns have the strongest correlations? Visualize them.
                - Create a Python function to clean the missing values in this dataset
                - Generate a visualization showing the distribution of [column name]
                - What insights can I derive from this data that would be valuable for business?
                """)
            
            # User prompt input
            user_prompt = st.text_area("What would you like to know about your dataset?", 
                                    height=100,
                                    placeholder="Enter your question about the dataset here...")
            
            # Example quick-select buttons in columns
            st.markdown("#### Quick Prompts")
            example_col1, example_col2, example_col3 = st.columns(3)
            
            with example_col1:
                if st.button("Summarize dataset", key="summarize_btn"):
                    user_prompt = "Provide a comprehensive summary of this dataset including its structure, key statistics, and potential use cases."
                    
            with example_col2:
                if st.button("Identify quality issues", key="quality_btn"):
                    user_prompt = "What are the most critical data quality issues in this dataset and how should I address them?"
                    
            with example_col3:
                if st.button("Suggest analyses", key="suggest_btn"):
                    user_prompt = "Based on this dataset, what are the most insightful analyses I could perform? Suggest specific approaches."
            
            # Second row of example buttons
            example_col4, example_col5, example_col6 = st.columns(3)
            
            with example_col4:
                if st.button("Visualize key columns", key="viz_btn"):
                    user_prompt = "Create visualizations for the most important columns in this dataset and explain what they show."
                    
            with example_col5:
                if st.button("Data cleaning code", key="clean_btn"):
                    user_prompt = "Generate Python code to clean and preprocess this dataset addressing all quality issues you identified."
                    
            with example_col6:
                if st.button("Find outliers", key="outlier_btn"):
                    user_prompt = "Identify outliers in this dataset and suggest how to handle them."
            
            # Submit button with clearer UI
            if st.button("🔍 Get AI Analysis", type="primary") and user_prompt:
                with st.spinner("The AI is analyzing your data... This may take a moment."):
                    try:
                        # Call OpenAI API with the user's prompt and dataset information
                        response, error = query_openai(user_prompt, dataset_summary, st.session_state.api_key, file_name)
                        
                        if error:
                            st.error(f"Error querying OpenAI API: {error}")
                            st.markdown("Please check your API key and try again. If the problem persists, try with a simpler question.")
                        elif response:
                            # Display the response in a clean format
                            st.markdown("### 🤖 AI Analysis Results")
                            st.markdown("---")
                            st.markdown(response)
                            st.markdown("---")
                            
                            # Check if the response has code and offer to execute it
                            if "```python" in response:
                                code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
                                if code_blocks:
                                    with st.expander("⚙️ Run Generated Code", expanded=True):
                                        st.markdown("The AI generated code that you can execute to analyze your data:")
                                        st.warning("⚠️ Always review code before running it. While the AI tries to generate safe code, you should verify its correctness.")
                                        
                                        for i, code in enumerate(code_blocks):
                                            with st.container():
                                                st.markdown(f"#### Code Block {i+1}")
                                                st.code(code.strip(), language="python")
                                                if st.button(f"▶️ Execute Code Block {i+1}", key=f"exec_btn_{i}"):
                                                    try:
                                                        # Create a local namespace with access to the dataframe
                                                        local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "px": px, 
                                                                    "stats": stats}
                                                        
                                                        # Execute the code
                                                        with st.spinner("Executing code..."):
                                                            # Capture the output
                                                            output_buffer = io.StringIO()
                                                            with contextlib.redirect_stdout(output_buffer):
                                                                exec(code.strip(), globals(), local_vars)
                                                            
                                                            # Display output
                                                            output = output_buffer.getvalue()
                                                            if output:
                                                                st.text("Output:")
                                                                st.text(output)
                                                            
                                                            # If matplotlib figures were created, display them
                                                            fig_nums = plt.get_fignums()
                                                            if fig_nums:
                                                                for fig_num in fig_nums:
                                                                    fig = plt.figure(fig_num)
                                                                    st.pyplot(fig)
                                                                plt.close('all')
                                                            
                                                            # Check for Plotly figures in the locals
                                                            for var_name, var_val in local_vars.items():
                                                                if str(type(var_val)) == "<class 'plotly.graph_objs._figure.Figure'>":
                                                                    st.plotly_chart(var_val)
                                                            
                                                            st.success("✅ Code executed successfully!")
                                                    except Exception as e:
                                                        st.error(f"❌ Error executing code: {str(e)}")
                                                        st.info("Try modifying the code to fix the error or ask the AI for a different approach.")
                            
                            # Store conversation in session state for history
                            st.session_state.conversation_history.append({
                                "prompt": user_prompt,
                                "response": response
                            })
                    except Exception as general_error:
                        st.error(f"An unexpected error occurred: {str(general_error)}")
                        st.info("Please try again with a different question or check your connection.")
            
            # Show conversation history if it exists
            if st.session_state.conversation_history:
                with st.expander("💬 Conversation History", expanded=False):
                    for i, exchange in enumerate(reversed(st.session_state.conversation_history)):
                        idx = len(st.session_state.conversation_history) - i
                        st.markdown(f"### Q{idx}: {exchange['prompt']}")
                        st.markdown(f"### A{idx}:")
                        st.markdown(exchange["response"])
                        if i < len(st.session_state.conversation_history) - 1:
                            st.divider()
                    
                    if st.button("🗑️ Clear Conversation History"):
                        st.session_state.conversation_history = []
                        st.experimental_rerun()
        
        elif uploaded_file is not None and df is not None:
            # Show a message if the dataset is loaded but no API key or invalid API key
            st.warning("Please enter a valid OpenAI API key to enable the AI assistant and click 'Test Connection'.")
        else:
            # Show a message if no dataset is loaded
            st.info("Please upload a dataset first to use the AI assistant.")
            
            
class EnhancedDataProfiler:
    def __init__(self, df, date_format="%Y-%m-%d %H:%M:%S"):
        """Initialize the profiler with a DataFrame."""
        self.df = df
        self.date_format = date_format
        warnings.filterwarnings("ignore")

    def get_schema(self):
        """Get the schema of the dataframe."""
        schema_data = []

        for col in self.df.columns:
            nullable = "true" if self.df[col].isnull().any() else "false"

            if pd.api.types.is_integer_dtype(self.df[col]):
                dtype = "integer"
            elif pd.api.types.is_float_dtype(self.df[col]):
                dtype = "float"
            elif pd.api.types.is_bool_dtype(self.df[col]):
                dtype = "boolean"
            elif pd.api.types.is_datetime64_dtype(self.df[col]):
                dtype = "timestamp"
            else:
                dtype = "string"

            schema_data.append({"Column": col, "Type": dtype, "Nullable": nullable})

        return schema_data

    def get_null_values(self, columns=None):
        """Get null values in each column, optionally focusing on specific columns."""
        null_data = []

        # Use specified columns or all columns
        cols_to_check = columns if columns is not None else self.df.columns

        null_counts = self.df[cols_to_check].isnull().sum()

        # Only return columns with null values
        for col, count in null_counts.items():
            if count > 0:
                total_count = len(self.df)
                percentage = round((count / total_count) * 100, 2)

                # Get sample rows with null values (up to 5)
                sample_indices = self.df[self.df[col].isnull()].index[:5].tolist()
                sample_rows = self.df.loc[sample_indices].to_dict('records') if sample_indices else []

                null_data.append({
                    "Column": col,
                    "Null Count": count,
                    "Total Count": total_count,
                    "Percentage": percentage,
                    "Sample Rows": sample_rows
                })

        return null_data

    def get_blank_values(self, columns=None):
        """Get blank values in string columns, optionally focusing on specific columns."""
        blank_data = []

        # Determine which columns to check
        if columns is not None:
            cols_to_check = [col for col in columns if col in self.df.select_dtypes(include=['object']).columns]
        else:
            cols_to_check = self.df.select_dtypes(include=['object']).columns

        for col in cols_to_check:
            # Count empty strings
            blank_count = (self.df[col] == '').sum()

            if blank_count > 0:
                percentage = round((blank_count / len(self.df)) * 100, 2)

                # Get sample rows with blank values (up to 5)
                sample_indices = self.df[self.df[col] == ''].index[:5].tolist()
                sample_rows = self.df.loc[sample_indices].to_dict('records') if sample_indices else []

                blank_data.append({
                    "Column": col,
                    "Blank Count": blank_count,
                    "Percentage": percentage,
                    "Sample Rows": sample_rows
                })

        return blank_data

    def get_unique_values(self):
        """Get unique values in each column."""
        unique_data = {}

        for col in self.df.columns:
            unique_values = self.df[col].drop_duplicates().dropna()
            unique_count = len(unique_values)

            # Show all if less than 10, otherwise show top 10
            if unique_count > 0:
                sample_values = unique_values[:10] if unique_count > 10 else unique_values
                sample_values = [str(val) for val in sample_values]

                if unique_count > 10:
                    sample_values.append(f"... and {unique_count - 10} more values")
            else:
                sample_values = ["null"]

            unique_data[col] = {
                "count": unique_count,
                "values": sample_values
            }

        return unique_data

    def get_date_validation(self):
        """Validate date/time formats in string columns."""
        date_validation = []

        for col in self.df.select_dtypes(include=['object']).columns:
            # Skip columns with all nulls
            if self.df[col].isna().all():
                continue

            # Check if column might contain dates
            date_pattern = re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}')
            sample = self.df[col].dropna().sample(min(100, len(self.df[col].dropna())))

            if any(date_pattern.search(str(x)) for x in sample):
                # Try to parse as dates
                try:
                    invalid_count = 0
                    non_empty = self.df[col].fillna('').astype(str)
                    non_empty = non_empty[non_empty != '']  # Exclude empty strings

                    for idx, val in non_empty.items():
                        try:
                            if val:
                                datetime.strptime(val, self.date_format)
                        except ValueError:
                            invalid_count += 1

                    if invalid_count > 0:
                        date_validation.append({
                            "Column": col,
                            "Invalid Count": invalid_count,
                            "Format": self.date_format
                        })
                except:
                    pass

        return date_validation

    def get_status_analysis(self):
        """Perform analysis on status columns."""
        status_data = {}

        # Detect potential status columns
        status_columns = [col for col in self.df.columns if 'status' in col.lower()]

        if status_columns:
            for status_col in status_columns:
                status_counts = self.df[status_col].value_counts().to_dict()
                status_data[status_col] = status_counts

        return status_data

    def get_route_analysis(self):
        """Analyze route information."""
        route_data = {}

        # Check for route information
        route_columns = [col for col in self.df.columns if 'route' in col.lower()]

        if route_columns:
            for col in route_columns:
                if self.df[col].nunique() > 1:
                    route_counts = self.df[col].value_counts().to_dict()
                    route_data[col] = route_counts

        return route_data

    def get_relationships(self):
        """Analyze relationships between key columns."""
        relationship_data = {}

        # Look for reference or ID columns
        id_columns = [col for col in self.df.columns if
                      'id' in col.lower() or 'number' in col.lower() or 'reference' in col.lower()]

        if len(id_columns) >= 2:
            # Choose primary key and reference columns
            primary_key = id_columns[0]
            reference_col = id_columns[-1]

            # Get relationship data (limit to 20 rows)
            result = self.df[[primary_key, reference_col]].head(20)
            relationship_data = {
                "primary_key": primary_key,
                "reference_col": reference_col,
                "data": result.to_dict('records')
            }

        return relationship_data

    def get_invalid_rows(self):
        """Identify and report rows with null or blank values for key columns."""
        invalid_data = {}

        # Find potential primary key (usually has 'id' or 'number')
        primary_key = None
        for col in self.df.columns:
            if 'id' in col.lower() or 'number' in col.lower():
                if self.df[col].nunique() > 1:
                    primary_key = col
                    break

        if not primary_key:
            # If no primary key found, use the first column
            primary_key = self.df.columns[0]

        # Check key columns for nulls and blanks
        problem_cols = []

        # Identify columns with null or blank values
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            blank_count = 0

            if self.df[col].dtype == 'object':
                blank_count = (self.df[col] == '').sum()

            invalid_count = null_count + blank_count

            if invalid_count > 0:
                problem_cols.append((col, invalid_count))

        # Collect data for problem columns with examples
        for col, count in problem_cols:
            if count > 0:
                # Show rows with nulls or blanks in this column
                if self.df[col].dtype == 'object':
                    invalid_rows = self.df[(self.df[col].isna()) | (self.df[col] == '')][[primary_key, col]]
                else:
                    invalid_rows = self.df[self.df[col].isna()][[primary_key, col]]

                invalid_rows = invalid_rows.head(20)

                invalid_data[col] = {
                    "count": count,
                    "primary_key": primary_key,
                    "examples": invalid_rows.to_dict('records')
                }

        return invalid_data

    def get_column_stats(self):
        """Get basic statistics for numerical columns."""
        stats_data = {}

        for col in self.df.select_dtypes(include=['number']).columns:
            stats = self.df[col].describe().to_dict()
            stats_data[col] = stats

        return stats_data

    def get_correlation_analysis(self):
        """Analyze correlations between numerical columns."""
        correlation_data = {}

        # Get numerical columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()

        if len(num_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = self.df[num_cols].corr().round(2)

            # Find strong correlations (positive or negative)
            strong_correlations = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    col1 = num_cols[i]
                    col2 = num_cols[j]
                    corr_value = corr_matrix.iloc[i, j]

                    if abs(corr_value) >= 0.5:  # Consider correlations >= 0.5 as noteworthy
                        strong_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": corr_value
                        })

            correlation_data = {
                "matrix": corr_matrix.to_dict(),
                "strong_correlations": strong_correlations
            }

        return correlation_data

    def get_outlier_analysis(self):
        """Detect outliers in numerical columns using IQR method."""
        outlier_data = {}

        for col in self.df.select_dtypes(include=['number']).columns:
            # Skip columns with all nulls or all same value
            if self.df[col].nunique() <= 1:
                continue

            # Calculate IQR (Interquartile Range)
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier boundaries
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)

            # Find outliers
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]

            if not outliers.empty:
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(self.df)) * 100

                # Get some example outlier rows (limit to 10)
                example_indices = outliers.index[:10].tolist()
                example_rows = self.df.loc[example_indices].to_dict('records')

                outlier_data[col] = {
                    "count": outlier_count,
                    "percentage": round(outlier_percentage, 2),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "min_outlier": float(outliers.min()),
                    "max_outlier": float(outliers.max()),
                    "examples": example_rows
                }

        return outlier_data

    def get_distribution_analysis(self):
        """Analyze distribution characteristics of numerical columns."""
        distribution_data = {}

        for col in self.df.select_dtypes(include=['number']).columns:
            # Skip columns with too many nulls or all same value
            if self.df[col].nunique() <= 1 or self.df[col].isna().sum() > len(self.df) * 0.5:
                continue

            # Calculate basic statistics
            mean = self.df[col].mean()
            median = self.df[col].median()
            skewness = self.df[col].skew()
            kurtosis = self.df[col].kurtosis()

            # Determine distribution type based on skewness
            if abs(skewness) < 0.5:
                distribution_type = "Normal distribution (symmetric)"
            elif skewness >= 0.5:
                distribution_type = "Right-skewed (positively skewed)"
            else:
                distribution_type = "Left-skewed (negatively skewed)"

            # Check for mean-median difference as another indicator
            mean_median_diff = abs(mean - median) / median if median != 0 else 0

            distribution_data[col] = {
                "mean": mean,
                "median": median,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "distribution_type": distribution_type,
                "mean_median_difference": mean_median_diff
            }

        return distribution_data

    def get_duplicate_analysis(self, columns=None):
        """Analyze potential duplicate records in the dataset, optionally focusing on specific columns."""
        duplicate_info = {}

        # First check if any columns contain unhashable types
        has_unhashable = False
        cols_to_check = columns if columns is not None else self.df.columns

        for col in cols_to_check:
            if col in self.df.columns and self.df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                has_unhashable = True
                break

        # Check for exact duplicates (all columns or specified columns)
        if not has_unhashable:
            try:
                if columns is not None and len(columns) > 0:
                    # Check duplicates in specified columns
                    exact_duplicates = self.df.duplicated(subset=columns).sum()
                    duplicate_rows = self.df[self.df.duplicated(subset=columns, keep=False)]
                else:
                    # Check duplicates across all columns
                    exact_duplicates = self.df.duplicated().sum()
                    duplicate_rows = self.df[self.df.duplicated(keep=False)]

                duplicate_percentage = (exact_duplicates / len(self.df)) * 100 if len(self.df) > 0 else 0

                # Get sample duplicate rows (limit to 5)
                sample_duplicates = duplicate_rows.head(5).to_dict('records') if not duplicate_rows.empty else []

                duplicate_info["exact_duplicates"] = {
                    "count": int(exact_duplicates),
                    "percentage": round(duplicate_percentage, 2),
                    "sample_rows": sample_duplicates
                }
            except TypeError:
                # Fallback if TypeError occurs
                duplicate_info["exact_duplicates"] = {
                    "count": 0,
                    "percentage": 0.0,
                    "note": "Skipped due to unhashable values in dataframe",
                    "sample_rows": []
                }
        else:
            # If unhashable types detected, skip duplicate check
            duplicate_info["exact_duplicates"] = {
                "count": 0,
                "percentage": 0.0,
                "note": "Skipped due to unhashable values in dataframe",
                "sample_rows": []
            }

        # Look for potential ID/key columns to check for duplicates
        if columns is not None:
            id_columns = [col for col in columns if col in self.df.columns and (
                    'id' in col.lower() or 'key' in col.lower() or 'number' in col.lower())]
        else:
            id_columns = [col for col in self.df.columns if
                          'id' in col.lower() or 'key' in col.lower() or 'number' in col.lower()]

        if id_columns:
            key_duplicates = {}
            for col in id_columns:
                # Skip columns with all nulls or unhashable types
                if self.df[col].isna().all() or self.df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    continue

                try:
                    # Count duplicates
                    dup_count = self.df[col].duplicated().sum()
                    if dup_count > 0:
                        dup_percentage = (dup_count / len(self.df)) * 100

                        # Get example duplicated values
                        duplicated_values = self.df[self.df.duplicated(subset=[col], keep=False)][
                            col].drop_duplicates().head(5).tolist()

                        # Get sample rows with duplicate key values (up to 5)
                        sample_indices = self.df[self.df.duplicated(subset=[col], keep=False)].index[:5].tolist()
                        sample_rows = self.df.loc[sample_indices].to_dict('records') if sample_indices else []

                        key_duplicates[col] = {
                            "count": int(dup_count),
                            "percentage": round(dup_percentage, 2),
                            "examples": duplicated_values,
                            "sample_rows": sample_rows
                        }
                except TypeError:
                    # Skip this column if unhashable types are encountered
                    continue

            duplicate_info["key_duplicates"] = key_duplicates

        return duplicate_info

    def get_categorical_analysis(self):
        """Analyze categorical columns for balance/imbalance and other characteristics."""
        categorical_data = {}

        # Find potential categorical columns (object type or few unique values)
        for col in self.df.columns:
            if pd.api.types.is_object_dtype(self.df[col]) or (
                    self.df[col].nunique() < 20 and self.df[col].nunique() > 1):
                # Skip columns with too many nulls
                if self.df[col].isna().sum() > len(self.df) * 0.5:
                    continue

                # Get value counts
                value_counts = self.df[col].value_counts().reset_index()
                value_counts.columns = ["Value", "Count"]

                # Calculate percentages
                value_counts["Percentage"] = (value_counts["Count"] / len(self.df) * 100).round(2)

                # Calculate entropy (measure of balance)
                probabilities = value_counts["Count"] / value_counts["Count"].sum()
                entropy = -(probabilities * np.log2(probabilities)).sum()
                max_entropy = np.log2(len(value_counts))
                balance_score = entropy / max_entropy if max_entropy > 0 else 1

                # Determine if column is balanced or imbalanced
                if balance_score > 0.75:
                    balance_status = "Well balanced"
                elif balance_score > 0.5:
                    balance_status = "Moderately balanced"
                else:
                    balance_status = "Imbalanced"

                # Calculate dominance of top category
                top_category_dominance = value_counts.iloc[0]["Percentage"] if not value_counts.empty else 0

                categorical_data[col] = {
                    "unique_values": len(value_counts),
                    "top_categories": value_counts.head(5).to_dict('records'),
                    "balance_score": round(balance_score, 2),
                    "balance_status": balance_status,
                    "top_category_dominance": top_category_dominance
                }

        return categorical_data

    def get_string_pattern_analysis(self):
        """Analyze patterns in string columns."""
        pattern_data = {}

        for col in self.df.select_dtypes(include=['object']).columns:
            # Skip columns with too many nulls
            if self.df[col].isna().sum() > len(self.df) * 0.5:
                continue

            # Get sample of non-null values
            sample = self.df[col].dropna().sample(min(100, len(self.df[col].dropna())))

            if sample.empty:
                continue

            # Calculate length statistics
            lengths = sample.str.len()

            # Check for common patterns
            has_numeric = any(bool(re.search(r'\d', str(x))) for x in sample)
            has_special = any(bool(re.search(r'[^\w\s]', str(x))) for x in sample)

            # Check for common prefixes/suffixes
            prefixes = {}
            suffixes = {}

            for value in sample:
                if pd.notna(value) and isinstance(value, str) and len(value) >= 3:
                    prefix = value[:3].lower()
                    suffix = value[-3:].lower()

                    prefixes[prefix] = prefixes.get(prefix, 0) + 1
                    suffixes[suffix] = suffixes.get(suffix, 0) + 1

            # Find common prefixes/suffixes
            common_prefixes = sorted([(k, v) for k, v in prefixes.items() if v > 1], key=lambda x: x[1], reverse=True)[
                              :3]
            common_suffixes = sorted([(k, v) for k, v in suffixes.items() if v > 1], key=lambda x: x[1], reverse=True)[
                              :3]

            pattern_data[col] = {
                "min_length": int(lengths.min()) if not lengths.empty else 0,
                "max_length": int(lengths.max()) if not lengths.empty else 0,
                "avg_length": round(float(lengths.mean()), 2) if not lengths.empty else 0,
                "contains_numeric": has_numeric,
                "contains_special": has_special,
                "common_prefixes": common_prefixes,
                "common_suffixes": common_suffixes
            }

        return pattern_data

    def get_time_series_analysis(self):
        """Analyze time-based patterns if date columns are present."""
        time_series_data = {}

        # Find date columns
        date_columns = self.df.select_dtypes(include=['datetime64']).columns.tolist()

        # Also check string columns that might contain dates
        for col in self.df.select_dtypes(include=['object']).columns:
            # Skip columns with too many nulls
            if self.df[col].isna().sum() > len(self.df) * 0.5:
                continue

            # Check if column might contain dates
            date_pattern = re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}')
            sample = self.df[col].dropna().sample(min(10, len(self.df[col].dropna())))

            if any(date_pattern.search(str(x)) for x in sample):
                # Try to parse as dates
                try:
                    parsed_dates = pd.to_datetime(self.df[col], errors='coerce')
                    if parsed_dates.notna().sum() > len(self.df) * 0.5:  # If more than 50% can be parsed
                        date_columns.append(col)
                except:
                    pass

        if date_columns:
            for date_col in date_columns:
                # Convert to datetime if not already
                if self.df[date_col].dtype != 'datetime64[ns]':
                    try:
                        date_series = pd.to_datetime(self.df[date_col], errors='coerce')
                    except:
                        continue
                else:
                    date_series = self.df[date_col]

                # Skip if too many nulls after conversion
                if date_series.isna().sum() > len(self.df) * 0.5:
                    continue

                # Get basic time statistics
                min_date = date_series.min()
                max_date = date_series.max()

                # Calculate date range and frequency
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = (max_date - min_date).days

                    # Extract time components for analysis - ADD THIS CHECK
                    has_time = False
                    try:
                        has_time = any(d.time() != pd.Timestamp('00:00:00').time() for d in date_series if pd.notna(d))

                        # Only process these if we have valid datetime
                        if date_series.dtype == 'datetime64[ns]':
                            year_counts = date_series.dt.year.value_counts().sort_index()
                            month_counts = date_series.dt.month.value_counts().sort_index()
                            day_of_week_counts = date_series.dt.dayofweek.value_counts().sort_index()
                        else:
                            # Skip datetime operations if not properly converted
                            continue
                    except (AttributeError, TypeError):
                        # If any error occurs, skip this date column
                        continue

                    # Identify trends if data spans multiple months
                    trend = "Insufficient data"
                    if date_range > 60 and len(month_counts) > 2:
                        try:
                            # Simple trend analysis by monthly frequency
                            month_year_counts = date_series.dt.to_period('M').value_counts().sort_index()
                            if len(month_year_counts) > 2:
                                first_half = month_year_counts.iloc[:len(month_year_counts) // 2].mean()
                                second_half = month_year_counts.iloc[len(month_year_counts) // 2:].mean()

                                percent_change = (
                                        (second_half - first_half) / first_half * 100) if first_half > 0 else 0

                                if percent_change > 10:
                                    trend = "Increasing trend"
                                elif percent_change < -10:
                                    trend = "Decreasing trend"
                                else:
                                    trend = "Stable trend"
                        except:
                            trend = "Error calculating trend"

                    time_series_data[date_col] = {
                        "min_date": min_date.strftime('%Y-%m-%d'),
                        "max_date": max_date.strftime('%Y-%m-%d'),
                        "date_range_days": date_range,
                        "has_time_component": has_time,
                        "year_distribution": year_counts.to_dict(),
                        "month_distribution": month_counts.to_dict(),
                        "day_of_week_distribution": day_of_week_counts.to_dict(),
                        "trend": trend
                    }

        return time_series_data

    def get_consistency_analysis(self):
        """Check for logical consistency between related columns."""
        consistency_data = {}

        # Look for common related column pairs
        # Date related checks
        date_pattern = re.compile(
            r'(start|begin|from|creation|create)_(date|time|at)|date_(start|begin|from)|created_(date|at|on)')
        end_date_pattern = re.compile(
            r'(end|finish|to|until|close|completion|complete)_(date|time|at)|date_(end|finish|to)|closed_(date|at|on)|completed_(date|at|on)')

        # Numeric related checks
        min_pattern = re.compile(r'(min|minimum|lower|floor)')
        max_pattern = re.compile(r'(max|maximum|upper|ceiling)')

        # Find potential date pairs
        date_start_cols = [col for col in self.df.columns if date_pattern.search(str(col).lower())]
        date_end_cols = [col for col in self.df.columns if end_date_pattern.search(str(col).lower())]

        # Check date consistency
        if date_start_cols and date_end_cols:
            for start_col in date_start_cols:
                for end_col in date_end_cols:
                    # Try to parse both as dates if they're not already
                    try:
                        if self.df[start_col].dtype != 'datetime64[ns]':
                            start_dates = pd.to_datetime(self.df[start_col], errors='coerce')
                        else:
                            start_dates = self.df[start_col]

                        if self.df[end_col].dtype != 'datetime64[ns]':
                            end_dates = pd.to_datetime(self.df[end_col], errors='coerce')
                        else:
                            end_dates = self.df[end_col]

                        # Find inconsistencies where end date is before start date
                        inconsistent = self.df[(end_dates < start_dates) & start_dates.notna() & end_dates.notna()]

                        if not inconsistent.empty:
                            inconsistent_count = len(inconsistent)
                            percentage = (inconsistent_count / len(self.df)) * 100

                            consistency_data[f"{start_col} vs {end_col}"] = {
                                "type": "date_order",
                                "issue": "End date before start date",
                                "count": inconsistent_count,
                                "percentage": round(percentage, 2),
                                "examples": inconsistent.head(5).to_dict('records')
                            }
                    except:
                        pass

        # Find potential numeric min/max pairs
        num_cols = self.df.select_dtypes(include=['number']).columns
        min_cols = [col for col in num_cols if min_pattern.search(str(col).lower())]
        max_cols = [col for col in num_cols if max_pattern.search(str(col).lower())]

        # Check numeric consistency
        if min_cols and max_cols:
            for min_col in min_cols:
                for max_col in max_cols:
                    # Check if min is greater than max
                    inconsistent = self.df[
                        (self.df[min_col] > self.df[max_col]) & self.df[min_col].notna() & self.df[max_col].notna()]

                    if not inconsistent.empty:
                        inconsistent_count = len(inconsistent)
                        percentage = (inconsistent_count / len(self.df)) * 100

                        consistency_data[f"{min_col} vs {max_col}"] = {
                            "type": "numeric_order",
                            "issue": "Minimum value greater than maximum value",
                            "count": inconsistent_count,
                            "percentage": round(percentage, 2),
                            "examples": inconsistent.head(5).to_dict('records')
                        }

        return consistency_data

    def get_data_completeness(self):
        """Calculate overall data completeness and identify potential required fields."""
        completeness_data = {}

        # Calculate overall completeness
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isna().sum().sum()
        blank_cells = 0

        # Count blank strings in object columns
        for col in self.df.select_dtypes(include=['object']).columns:
            blank_cells += (self.df[col] == '').sum()

        filled_cells = total_cells - missing_cells - blank_cells
        completeness_percentage = (filled_cells / total_cells * 100) if total_cells > 0 else 0

        completeness_data["overall"] = {
            "total_cells": total_cells,
            "filled_cells": filled_cells,
            "missing_cells": int(missing_cells),
            "blank_cells": int(blank_cells),
            "completeness_percentage": round(completeness_percentage, 2)
        }

        # Identify potential required fields
        # (Fields that are almost always filled have high likelihood of being required)
        column_completeness = {}
        potential_required = []

        for col in self.df.columns:
            missing = self.df[col].isna().sum()
            blank = 0
            if pd.api.types.is_object_dtype(self.df[col]):
                blank = (self.df[col] == '').sum()

            filled = len(self.df) - missing - blank
            percentage = (filled / len(self.df) * 100) if len(self.df) > 0 else 0

            column_completeness[col] = round(percentage, 2)

            if percentage >= 95:  # Assume 95%+ filled columns might be required
                potential_required.append(col)

        completeness_data["columns"] = column_completeness
        completeness_data["potential_required_fields"] = potential_required

        return completeness_data

    def get_full_analysis(self):
        """Get all analysis results in a dictionary."""
        return {
            "schema": self.get_schema(),
            "null_values": self.get_null_values(),
            "blank_values": self.get_blank_values(),
            "unique_values": self.get_unique_values(),
            "date_validation": self.get_date_validation(),
            "status_analysis": self.get_status_analysis(),
            "route_analysis": self.get_route_analysis(),
            "relationships": self.get_relationships(),
            "invalid_rows": self.get_invalid_rows(),
            "column_stats": self.get_column_stats(),
            # New generic analyses
            "correlation": self.get_correlation_analysis(),
            "outliers": self.get_outlier_analysis(),
            "distribution": self.get_distribution_analysis(),
            "duplicates": self.get_duplicate_analysis(),
            "categorical": self.get_categorical_analysis(),
            "string_patterns": self.get_string_pattern_analysis(),
            "time_series": self.get_time_series_analysis(),
            "consistency": self.get_consistency_analysis(),
            "completeness": self.get_data_completeness()
        }


def flatten_json(nested_json, prefix=''):
    """
    Flatten a nested JSON object into a flat dictionary with concatenated keys.
    """
    flat_dict = {}

    for key, value in nested_json.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flat_dict.update(flatten_json(value, new_key))
        elif isinstance(value, list):
            # Handle lists by converting each item and using index in the key
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flat_dict.update(flatten_json(item, f"{new_key}[{i}]"))
                else:
                    flat_dict[f"{new_key}[{i}]"] = item
        else:
            # Base case: primitive value
            flat_dict[new_key] = value

    return flat_dict


def load_data(file):
    """Load data from various file formats."""
    file_extension = file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            return pd.read_csv(file)
        elif file_extension in ['xls', 'xlsx']:
            return pd.read_excel(file)
        elif file_extension == 'json':
            # For JSON files, first try the standard method
            try:
                # Try loading as a regular JSON with records
                df = pd.read_json(file)
                # Check if any columns contain dictionaries or lists
                has_complex_types = False
                for col in df.columns:
                    if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                        has_complex_types = True
                        break

                if has_complex_types:
                    # If complex types are found, fall back to flattening
                    file.seek(0)  # Reset file pointer
                    raise ValueError("Complex nested structure detected, using flattening")
                return df
            except:
                # If that fails, it might be a nested JSON
                import json
                file.seek(0)  # Reset file pointer
                try:
                    # Try to load the JSON
                    json_obj = json.load(file)

                    if isinstance(json_obj, dict):
                        # Single object - flatten it
                        flattened = flatten_json(json_obj)
                        return pd.DataFrame([flattened])
                    elif isinstance(json_obj, list):
                        # List of objects - flatten each one
                        flattened_list = []
                        for item in json_obj:
                            if isinstance(item, dict):
                                flattened_list.append(flatten_json(item))
                            else:
                                flattened_list.append({"value": item})
                        return pd.DataFrame(flattened_list)
                    else:
                        # Simple value
                        return pd.DataFrame([{"value": json_obj}])
                except Exception as json_error:
                    st.error(f"Error processing JSON: {str(json_error)}")
                    return None
        elif file_extension == 'parquet':
            return pd.read_parquet(file)
        else:
            # Try csv as default
            return pd.read_csv(file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def get_download_link(data, format_type="csv", filename="data"):
    """Generate a download link for data in various formats."""
    if format_type == "csv":
        # Convert DataFrame to CSV
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="download-button">Download CSV</a>'
        return href

    elif format_type == "parquet":
        # Convert DataFrame to Parquet
        buffer = io.BytesIO()
        data.to_parquet(buffer, index=False)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.parquet" class="download-button">Download Parquet</a>'
        return href

    elif format_type == "excel":
        # Convert DataFrame to Excel
        buffer = io.BytesIO()
        data.to_excel(buffer, index=False)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx" class="download-button">Download Excel</a>'
        return href

    elif format_type == "json":
        # Convert DataFrame to JSON
        json_str = data.to_json(orient="records")
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json" class="download-button">Download JSON</a>'
        return href

    elif format_type == "html":
        # Convert to HTML
        html_str = data
        b64 = base64.b64encode(html_str.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html" class="download-button">Download HTML Report</a>'
        return href

    return ""


def filter_dataframe(df):
    """Create interactive filter controls for a DataFrame."""
    # Create a container for filters
    filter_container = st.container()

    with filter_container:
        st.markdown("### Filter Data")

        # Select columns to filter on
        filter_columns = st.multiselect(
            "Select columns to filter on:",
            options=df.columns,
            default=[]
        )

        filtered_df = df.copy()

        # Create filters for selected columns
        for column in filter_columns:
            # Different filter types based on column data type
            if pd.api.types.is_numeric_dtype(df[column]):
                # Numeric filter with min/max slider
                min_val, max_val = float(df[column].min()), float(df[column].max())
                step = (max_val - min_val) / 100

                filter_values = st.slider(
                    f"Values for {column}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step
                )

                filtered_df = filtered_df[(filtered_df[column] >= filter_values[0]) &
                                          (filtered_df[column] <= filter_values[1])]

            elif pd.api.types.is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                # Categorical filter with multiselect
                options = df[column].unique()
                selected_options = st.multiselect(
                    f"Values for {column}",
                    options=options,
                    default=list(options)
                )

                filtered_df = filtered_df[filtered_df[column].isin(selected_options)]

            else:
                # Text filter with text input
                text_input = st.text_input(
                    f"Substring or regex in {column}"
                )

                if text_input:
                    filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(text_input, case=False)]

        # Show the filtered DataFrame
        if st.checkbox("Show filtered data", value=True):
            st.write(f"Filtered data ({len(filtered_df)} rows):")
            st.dataframe(filtered_df, use_container_width=True)

        return filtered_df


def generate_html_report(df, analysis, file_name):
    """Generate a lightweight HTML report"""
    try:
        # Sample the dataframe if it's large (prevents memory issues)
        sample_size = min(5000, len(df))
        if len(df) > sample_size:
            df_sample = df.sample(sample_size, random_state=42)
            sample_note = f"<p><em>Note: Report based on a sample of {sample_size} rows from {len(df)} total rows.</em></p>"
        else:
            df_sample = df
            sample_note = ""

        # Calculate basic metrics
        null_count = sum(item["Null Count"] for item in analysis["null_values"]) if analysis["null_values"] else 0
        blank_count = sum(item["Blank Count"] for item in analysis["blank_values"]) if analysis["blank_values"] else 0
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = null_count + blank_count
        quality_score = 100 - ((missing_cells / total_cells) * 100) if total_cells > 0 else 100

        # Create HTML report - text only, no visualizations
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report: {file_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h3 {{ color: #2980b9; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .bad {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report: {file_name}</h1>
            {sample_note}

            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Dataset Size:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
                <p><strong>Data Quality Score:</strong> 
                    <span class="{'good' if quality_score >= 90 else 'warning' if quality_score >= 70 else 'bad'}">{quality_score:.1f}%</span>
                </p>
                <p><strong>Missing Values:</strong> {missing_cells:,} ({(missing_cells / total_cells * 100):.1f}% of total cells)</p>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

            <h2>1. Schema Information</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Nullable</th>
                </tr>
        """

        # Add schema rows
        for item in analysis["schema"]:
            html += f"""
                <tr>
                    <td>{item['Column']}</td>
                    <td>{item['Type']}</td>
                    <td>{item['Nullable']}</td>
                </tr>
            """

        html += """
            </table>

            <h2>2. Data Quality Issues</h2>
        """

        # Add null values section
        if analysis["null_values"]:
            html += """
            <h3>2.1 Null Values</h3>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Null Count</th>
                    <th>Percentage</th>
                </tr>
            """

            for item in analysis["null_values"]:
                html += f"""
                <tr>
                    <td>{item['Column']}</td>
                    <td>{item['Null Count']:,}</td>
                    <td>{item['Percentage']}%</td>
                </tr>
                """

            html += """
            </table>
            """
        else:
            html += """
            <p>No null values found in the dataset.</p>
            """

        # Add blank values section
        if analysis["blank_values"]:
            html += """
            <h3>2.2 Blank Values</h3>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Blank Count</th>
                    <th>Percentage</th>
                </tr>
            """

            for item in analysis["blank_values"]:
                html += f"""
                <tr>
                    <td>{item['Column']}</td>
                    <td>{item['Blank Count']:,}</td>
                    <td>{item['Percentage']}%</td>
                </tr>
                """

            html += """
            </table>
            """

        # Add duplicate analysis
        if analysis["duplicates"]:
            html += """
            <h3>2.3 Duplicate Analysis</h3>
            """

            exact_dups = analysis["duplicates"]["exact_duplicates"]
            html += f"""
            <p><strong>Exact Duplicate Rows:</strong> {exact_dups['count']:,} ({exact_dups['percentage']}% of rows)</p>
            """

            if "key_duplicates" in analysis["duplicates"] and analysis["duplicates"]["key_duplicates"]:
                html += """
                <p><strong>Duplicate Keys:</strong></p>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Duplicate Count</th>
                        <th>Percentage</th>
                    </tr>
                """

                for col, data in analysis["duplicates"]["key_duplicates"].items():
                    html += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{data['count']:,}</td>
                        <td>{data['percentage']}%</td>
                    </tr>
                    """

                html += """
                </table>
                """

        # Add data completeness
        if analysis["completeness"]:
            completeness = analysis["completeness"]["overall"]

            html += """
            <h2>3. Data Completeness</h2>
            """

            html += f"""
            <p><strong>Overall Completeness:</strong> {completeness['completeness_percentage']}%</p>
            <p><strong>Filled Cells:</strong> {completeness['filled_cells']:,} of {completeness['total_cells']:,}</p>
            <p><strong>Missing Cells:</strong> {completeness['missing_cells']:,}</p>
            <p><strong>Blank Cells:</strong> {completeness['blank_cells']:,}</p>
            """

            # Potential required fields
            if "potential_required_fields" in analysis["completeness"] and analysis["completeness"][
                "potential_required_fields"]:
                html += """
                <h3>3.1 Potential Required Fields</h3>
                <p>These columns are almost always filled (95%+ complete) and might be required fields:</p>
                <ul>
                """

                for field in analysis["completeness"]["potential_required_fields"]:
                    html += f"""
                    <li>{field}</li>
                    """

                html += """
                </ul>
                """

        # Add recommendations section
        html += """
        <h2>4. Recommendations</h2>
        <ul>
        """

        if analysis["null_values"]:
            html += "<li>Address missing values in key columns.</li>"

        if analysis["blank_values"]:
            html += "<li>Review and clean blank values in text fields.</li>"

        if analysis["date_validation"]:
            html += "<li>Standardize date formats in problematic columns.</li>"

        if analysis["duplicates"]["exact_duplicates"]["count"] > 0:
            html += "<li>Remove or investigate duplicate records.</li>"

        if "strong_correlations" in analysis["correlation"] and analysis["correlation"]["strong_correlations"]:
            html += "<li>Review highly correlated fields to identify potential redundancies.</li>"

        html += """
        </ul>

        <div style="text-align: center; margin-top: 50px; color: #777; font-size: 12px;">
            <p>Generated by Enhanced Data Profiler - Optimized for environments</p>
        </div>
        </body>
        </html>
        """

        return html

    except Exception as e:
        # Return basic error HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Report Generation Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .error {{ color: #e74c3c; background-color: #fce4e4; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Report Generation Error</h1>
            <div class="error">
                <p>An error occurred while generating the report: {str(e)}</p>
            </div>
            <h2>Dataset Summary</h2>
            <p><strong>File:</strong> {file_name}</p>
            <p><strong>Rows:</strong> {df.shape[0]}</p>
            <p><strong>Columns:</strong> {df.shape[1]}</p>
        </body>
        </html>
        """


def main():
    st.set_page_config(page_title="Enhanced Data Profiler", page_icon="📊", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4b7bec;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3867d6;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #26de81;
        color: white;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fed330;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fc5c65;
        color: white;
    }
    .info-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f5f6fa;
        border: 1px solid #dcdde1;
        margin-bottom: 1rem;
    }
    .download-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #4b7bec;
        color: white;
        text-decoration: none;
        border-radius: 0.3rem;
        margin: 0.5rem;
        text-align: center;
    }
    .download-button:hover {
        background-color: #3867d6;
        color: white;
    }
    .format-option {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0.5rem 0;
    }
    .converter-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        border: 1px solid #dcdde1;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">Data Lens</h1>', unsafe_allow_html=True)

    # Sidebar with controls
    with st.sidebar:
        st.header("Controls")
        st.markdown("#### 1. Upload your data file")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json", "parquet"])

        if uploaded_file is not None:
            st.markdown("#### 2. Analysis Settings")
            date_format = st.text_input("Date format for validation", "%Y-%m-%d %H:%M:%S")

    # Main content
    if uploaded_file is not None:
        with st.spinner('Loading data...'):
            df = load_data(uploaded_file)
        if df is not None:
            file_name = uploaded_file.name.split('.')[0]  # Get filename without extension

            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            st.markdown(f"<div class='info-card'>File contains {df.shape[0]} rows and {df.shape[1]} columns</div>",
                        unsafe_allow_html=True)

            # Create a container for the dashboard
            dashboard = st.container()

            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
                "🧮 Overview",
                "🔍 Schema & Nulls",
                "📊 Unique Values",
                "📝 Content Analysis",
                "❌ Data Issues",
                "📈 Correlations",
                "📉 Distribution",
                "🔄 Consistency",
                "🔄 File Conversion",
                "📑 Report",
                "❓ Guide",
                "🤖 AI Assistant"  # New tab
            ])

            with dashboard:
                # Data preview section
                with st.expander("📋 Preview Data", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.dataframe(df.head(10), use_container_width=True)
                    with col2:
                        st.markdown(f"### Dataset Info")
                        st.markdown(f"**Rows:** {df.shape[0]}")
                        st.markdown(f"**Columns:** {df.shape[1]}")
                        st.markdown(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

                        # Quick download options
                        st.markdown("### Quick Download")
                        st.markdown(get_download_link(df, "csv", file_name), unsafe_allow_html=True)

                # Run the analysis
                with st.spinner('Analyzing data quality...'):
                    profiler = EnhancedDataProfiler(df, date_format)
                    analysis = profiler.get_full_analysis()

                # Tab 1: Overview
                with tab1:
                    st.markdown('<h2 class="sub-header">📊 Data Quality Dashboard</h2>', unsafe_allow_html=True)

                    # Summary metrics in columns
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        null_count = sum(item["Null Count"] for item in analysis["null_values"]) if analysis[
                            "null_values"] else 0
                        null_percentage = (null_count / (df.shape[0] * df.shape[1])) * 100

                        st.metric(
                            label="Missing Values",
                            value=f"{null_count:,}",
                            delta=f"{null_percentage:.2f}% of cells",
                            delta_color="inverse"
                        )

                    with col2:
                        blank_count = sum(item["Blank Count"] for item in analysis["blank_values"]) if analysis[
                            "blank_values"] else 0
                        st.metric(
                            label="Blank Values",
                            value=f"{blank_count:,}",
                            delta=f"In {len(analysis['blank_values'])} columns" if analysis[
                                "blank_values"] else "None found",
                            delta_color="inverse"
                        )

                    with col3:
                        invalid_date_count = sum(item["Invalid Count"] for item in analysis["date_validation"]) if \
                            analysis["date_validation"] else 0
                        st.metric(
                            label="Invalid Dates",
                            value=f"{invalid_date_count:,}",
                            delta=f"In {len(analysis['date_validation'])} columns" if analysis[
                                "date_validation"] else "None found",
                            delta_color="inverse"
                        )

                    with col4:
                        problem_cols = len(analysis["invalid_rows"]) if analysis["invalid_rows"] else 0
                        st.metric(
                            label="Problem Columns",
                            value=f"{problem_cols}",
                            delta=f"{problem_cols / df.shape[1]:.1%} of all columns" if problem_cols > 0 else "No issues found",
                            delta_color="inverse"
                        )

                    # Data quality score
                    st.markdown('### Data Quality Score')

                    # Calculate a simple data quality score
                    total_cells = df.shape[0] * df.shape[1]
                    missing_cells = null_count + blank_count
                    score = 100 - ((missing_cells / total_cells) * 100) if total_cells > 0 else 100

                    # Display score with color based on value
                    if score >= 90:
                        color = "green"
                        quality = "Excellent"
                    elif score >= 80:
                        color = "blue"
                        quality = "Good"
                    elif score >= 70:
                        color = "orange"
                        quality = "Fair"
                    else:
                        color = "red"
                        quality = "Poor"

                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 20px;">
                        <div style="width: 150px; height: 150px; border-radius: 50%; background: conic-gradient({color} {score}%, #f1f1f1 0); display: flex; align-items: center; justify-content: center; margin-right: 20px;">
                            <div style="width: 130px; height: 130px; border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                <div style="font-size: 2rem; font-weight: bold; color: {color};">{score:.1f}%</div>
                                <div style="font-size: 1rem; color: {color};">{quality}</div>
                            </div>
                        </div>
                        <div>
                            <h3>Quality Assessment</h3>
                            <p>This score is calculated based on missing values, blank cells, and other data quality issues.</p>
                            <p>A higher score indicates better data quality.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Distribution of numerical columns
                    if df.select_dtypes(include=['number']).columns.any():
                        st.markdown('### Numerical Column Distribution')

                        num_cols = df.select_dtypes(include=['number']).columns.tolist()
                        if num_cols:
                            selected_col = st.selectbox("Select column to visualize", num_cols)

                            col1, col2 = st.columns(2)
                            with col1:
                                # Histogram
                                fig_hist = px.histogram(df, x=selected_col, marginal="box",
                                                        title=f"Distribution of {selected_col}",
                                                        template="plotly_white")
                                st.plotly_chart(fig_hist, use_container_width=True, key=f"overview_hist_{selected_col}")

                            with col2:
                                # Box plot
                                fig_box = px.box(df, y=selected_col,
                                                 title=f"Box Plot of {selected_col}",
                                                 template="plotly_white")
                                st.plotly_chart(fig_box, use_container_width=True, key=f"overview_box_{selected_col}")

                    # Status column analysis if available
                    if analysis["status_analysis"]:
                        st.markdown('### Status Analysis')

                        status_cols = list(analysis["status_analysis"].keys())
                        if status_cols:
                            selected_status = st.selectbox("Select status column", status_cols)

                            status_data = analysis["status_analysis"][selected_status]
                            status_df = pd.DataFrame(list(status_data.items()), columns=["Status", "Count"])

                            fig_pie = px.pie(status_df, names="Status", values="Count",
                                             title=f"Distribution of {selected_status}",
                                             template="plotly_white", hole=0.4)
                            st.plotly_chart(fig_pie, use_container_width=True, key=f"overview_status_{selected_status}")

                # Tab 2: Schema & Nulls
                # Complete modified code for Tab 2: Schema & Nulls
                with tab2:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown('<h2 class="sub-header">Schema Information</h2>', unsafe_allow_html=True)
                        if analysis["schema"]:
                            schema_df = pd.DataFrame(analysis["schema"])
                            st.dataframe(schema_df, use_container_width=True)

                            # Visualize data types
                            type_counts = schema_df["Type"].value_counts().reset_index()
                            type_counts.columns = ["Type", "Count"]

                            fig = px.pie(type_counts, names="Type", values="Count",
                                         title="Column Data Types",
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                            st.plotly_chart(fig, use_container_width=True, key="schema_type_pie")

                    with col2:
                        st.markdown('<h2 class="sub-header">Null Values Analysis</h2>', unsafe_allow_html=True)

                        # Add column selection for null analysis
                        all_columns_null = st.checkbox("Check all columns for nulls", value=True,
                                                       key="tab2_all_columns_null")

                        if not all_columns_null:
                            selected_columns_null = st.multiselect(
                                "Select columns to check for null values:",
                                options=df.columns,
                                default=[],
                                key="tab2_selected_columns_null"
                            )
                            # Run null analysis with selected columns
                            null_analysis = profiler.get_null_values(columns=selected_columns_null)
                        else:
                            # Use pre-calculated null analysis
                            null_analysis = analysis["null_values"]

                        if null_analysis:
                            null_df = pd.DataFrame(
                                [(item["Column"], item["Null Count"], item["Total Count"], item["Percentage"])
                                 for item in null_analysis],
                                columns=["Column", "Null Count", "Total Count", "Percentage"])
                            st.dataframe(null_df, use_container_width=True)

                            # Plot the nulls
                            fig = px.bar(null_df, x="Column", y="Null Count",
                                         title="Null Values by Column",
                                         text="Percentage",
                                         color="Percentage",
                                         color_continuous_scale="Reds")
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True, key="null_values_bar")

                            # Show sample rows with null values
                            if len(null_analysis) > 0:
                                with st.expander("View Sample Rows with Null Values", expanded=False):
                                    selected_null_col = st.selectbox(
                                        "Select column to view samples with null values:",
                                        options=[item["Column"] for item in null_analysis],
                                        key="tab2_selected_null_col"
                                    )

                                    # Find the sample rows for the selected column
                                    for item in null_analysis:
                                        if item["Column"] == selected_null_col:
                                            if "Sample Rows" in item and item["Sample Rows"]:
                                                st.dataframe(pd.DataFrame(item["Sample Rows"]),
                                                             use_container_width=True)
                                            else:
                                                st.info(f"No sample rows available for {selected_null_col}")
                                            break
                        else:
                            st.markdown('<div class="success-box">No null values found in the dataset!</div>',
                                        unsafe_allow_html=True)

                        st.markdown('<h2 class="sub-header">Blank Values Analysis</h2>', unsafe_allow_html=True)

                        # Add column selection for blank analysis
                        all_columns_blank = st.checkbox("Check all columns for blanks", value=True,
                                                        key="tab2_all_columns_blank")

                        if not all_columns_blank:
                            # Filter to only show string columns
                            string_columns = df.select_dtypes(include=['object']).columns.tolist()
                            selected_columns_blank = st.multiselect(
                                "Select string columns to check for blank values:",
                                options=string_columns,
                                default=[],
                                key="tab2_selected_columns_blank"
                            )
                            # Run blank analysis with selected columns
                            blank_analysis = profiler.get_blank_values(columns=selected_columns_blank)
                        else:
                            # Use pre-calculated blank analysis
                            blank_analysis = analysis["blank_values"]

                        if blank_analysis:
                            blank_df = pd.DataFrame([(item["Column"], item["Blank Count"], item["Percentage"])
                                                     for item in blank_analysis],
                                                    columns=["Column", "Blank Count", "Percentage"])
                            st.dataframe(blank_df, use_container_width=True)

                            # Plot the blanks
                            fig = px.bar(blank_df, x="Column", y="Blank Count",
                                         title="Blank Values by Column",
                                         text="Percentage",
                                         color="Percentage",
                                         color_continuous_scale="Blues")
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True, key="blank_values_bar")

                            # Show sample rows with blank values
                            if len(blank_analysis) > 0:
                                with st.expander("View Sample Rows with Blank Values", expanded=False):
                                    selected_blank_col = st.selectbox(
                                        "Select column to view samples with blank values:",
                                        options=[item["Column"] for item in blank_analysis],
                                        key="tab2_selected_blank_col"
                                    )

                                    # Find the sample rows for the selected column
                                    for item in blank_analysis:
                                        if item["Column"] == selected_blank_col:
                                            if "Sample Rows" in item and item["Sample Rows"]:
                                                st.dataframe(pd.DataFrame(item["Sample Rows"]),
                                                             use_container_width=True)
                                            else:
                                                st.info(f"No sample rows available for {selected_blank_col}")
                                            break
                        else:
                            st.markdown('<div class="success-box">No blank values found in the dataset!</div>',
                                        unsafe_allow_html=True)

                # Tab 3: Unique Values
                with tab3:
                    st.markdown('<h2 class="sub-header">Unique Values Analysis</h2>', unsafe_allow_html=True)

                    # Show chart with column cardinality
                    st.markdown("### Column Cardinality")

                    if analysis["unique_values"]:
                        unique_counts = {col: data["count"] for col, data in analysis["unique_values"].items()}
                        if unique_counts:
                            cardinality_df = pd.DataFrame(list(unique_counts.items()),
                                                          columns=["Column", "Unique Count"])
                            cardinality_df["Ratio"] = cardinality_df["Unique Count"] / df.shape[0]
                            cardinality_df = cardinality_df.sort_values("Unique Count", ascending=False)

                            fig = px.bar(cardinality_df.head(15), x="Column", y="Unique Count",
                                         text="Unique Count",
                                         color="Ratio",
                                         title="Top 15 Columns by Unique Value Count",
                                         color_continuous_scale="Viridis")
                            fig.update_traces(textposition='outside')
                            st.plotly_chart(fig, use_container_width=True, key="cardinality_bar")

                            # Potential primary key columns
                            pk_candidates = cardinality_df[cardinality_df["Ratio"] > 0.9].sort_values("Ratio",
                                                                                                      ascending=False)
                            if not pk_candidates.empty:
                                st.markdown("### Potential Primary Key Columns")
                                st.markdown(
                                    "These columns have high cardinality (over 90% unique values) and could be primary keys:")
                                st.dataframe(pk_candidates[["Column", "Unique Count", "Ratio"]])

                    # Display unique values for each column
                    st.markdown("### Unique Values By Column")

                    if analysis["unique_values"]:
                        # Let user select a column to explore
                        columns = list(analysis["unique_values"].keys())
                        selected_column = st.selectbox("Select a column to view its unique values:", columns)

                        data = analysis["unique_values"][selected_column]
                        st.markdown(f"**{selected_column}**: {data['count']} distinct values")

                        # Display unique values
                        values = data["values"]
                        if "more values" in values[-1] if values else False:
                            st.write("Sample of unique values:")
                            for val in values[:-1]:  # Exclude the "... and more" message
                                st.markdown(f"- `{val}`")
                            st.write(values[-1])  # Display the "... and more" message
                        else:
                            st.write("All unique values:")
                            for val in values:
                                st.markdown(f"- `{val}`")

                # Tab 4: Content Analysis
                with tab4:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown('<h2 class="sub-header">Date Format Validation</h2>', unsafe_allow_html=True)

                        if analysis["date_validation"]:
                            date_df = pd.DataFrame(analysis["date_validation"])
                            st.dataframe(date_df, use_container_width=True)

                            fig = px.bar(date_df, x="Column", y="Invalid Count",
                                         title=f"Invalid Date Formats (Expected: {date_format})",
                                         text="Invalid Count",
                                         color="Invalid Count",
                                         color_continuous_scale="Reds")
                            fig.update_traces(textposition='outside')
                            st.plotly_chart(fig, use_container_width=True, key="date_validation_bar")
                        else:
                            st.markdown('<div class="success-box">No date format issues detected!</div>',
                                        unsafe_allow_html=True)

                    with col2:
                        # Column Relationships
                        st.markdown('<h2 class="sub-header">Column Relationships</h2>', unsafe_allow_html=True)

                        if analysis["relationships"]:
                            relationship = analysis["relationships"]
                            st.markdown(
                                f"Relationship between **{relationship['primary_key']}** and **{relationship['reference_col']}**:")

                            relationship_df = pd.DataFrame(relationship["data"])
                            st.dataframe(relationship_df, use_container_width=True)
                        else:
                            st.info("No relationships detected between columns.")

                    # String Pattern Analysis
                    st.markdown('<h2 class="sub-header">String Pattern Analysis</h2>', unsafe_allow_html=True)

                    if analysis["string_patterns"]:
                        pattern_cols = list(analysis["string_patterns"].keys())

                        # Summary table
                        pattern_summary = []
                        for col in pattern_cols:
                            data = analysis["string_patterns"][col]
                            pattern_summary.append({
                                "Column": col,
                                "Min Length": data["min_length"],
                                "Max Length": data["max_length"],
                                "Avg Length": data["avg_length"],
                                "Contains Numbers": "Yes" if data["contains_numeric"] else "No",
                                "Contains Special Chars": "Yes" if data["contains_special"] else "No"
                            })

                        pattern_df = pd.DataFrame(pattern_summary)
                        st.dataframe(pattern_df, use_container_width=True)

                        # Detail view for selected column
                        selected_pattern = st.selectbox("Select string column for pattern details:", pattern_cols)

                        if selected_pattern:
                            pattern_data = analysis["string_patterns"][selected_pattern]

                            st.markdown(f"### Pattern Details for '{selected_pattern}'")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Length Statistics:**")
                                st.markdown(f"- Minimum Length: {pattern_data['min_length']}")
                                st.markdown(f"- Maximum Length: {pattern_data['max_length']}")
                                st.markdown(f"- Average Length: {pattern_data['avg_length']}")

                            with col2:
                                st.markdown("**Content Characteristics:**")
                                st.markdown(f"- Contains Numeric Characters: {pattern_data['contains_numeric']}")
                                st.markdown(f"- Contains Special Characters: {pattern_data['contains_special']}")

                            if pattern_data["common_prefixes"]:
                                st.markdown("**Common Prefixes:**")
                                for prefix, count in pattern_data["common_prefixes"]:
                                    st.markdown(f"- '{prefix}' appears {count} times")

                            if pattern_data["common_suffixes"]:
                                st.markdown("**Common Suffixes:**")
                                for suffix, count in pattern_data["common_suffixes"]:
                                    st.markdown(f"- '{suffix}' appears {count} times")
                    else:
                        st.info("No text columns available for pattern analysis.")

                    # Time Series Analysis (if available)
                    if analysis["time_series"] and len(analysis["time_series"]) > 0:
                        st.markdown('<h2 class="sub-header">Time Series Analysis</h2>', unsafe_allow_html=True)

                        time_cols = list(analysis["time_series"].keys())
                        selected_time = st.selectbox("Select date/time column to analyze:", time_cols)

                        if selected_time:
                            time_data = analysis["time_series"][selected_time]

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Date Range", f"{time_data['date_range_days']} days")

                            with col2:
                                st.metric("Date From", time_data["min_date"])

                            with col3:
                                st.metric("Date To", time_data["max_date"])

                            st.markdown(f"**Trend**: {time_data['trend']}")

                            # Visualize monthly distribution
                            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                                           "Dec"]
                            month_data = []

                            for month_num, count in time_data["month_distribution"].items():
                                month_data.append({
                                    "Month": month_names[int(month_num) - 1],
                                    "Count": count
                                })

                            # Sort by month
                            month_order = {month: i for i, month in enumerate(month_names)}
                            month_data.sort(key=lambda x: month_order[x["Month"]])

                            month_df = pd.DataFrame(month_data)

                            fig = px.bar(
                                month_df, x="Month", y="Count",
                                title=f"Monthly Distribution for {selected_time}",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"monthly_dist_{selected_time}")

                            # Day of week distribution
                            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                            day_data = []

                            for day_num, count in time_data["day_of_week_distribution"].items():
                                day_data.append({
                                    "Day": day_names[int(day_num)],
                                    "Count": count
                                })

                            # Sort by day of week
                            day_order = {day: i for i, day in enumerate(day_names)}
                            day_data.sort(key=lambda x: day_order[x["Day"]])

                            day_df = pd.DataFrame(day_data)

                            fig = px.bar(
                                day_df, x="Day", y="Count",
                                title=f"Day of Week Distribution for {selected_time}",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"day_dist_{selected_time}")

                # Tab 5: Data Issues
                with tab5:
                    st.markdown('<h2 class="sub-header">Invalid Data Analysis</h2>', unsafe_allow_html=True)

                    if analysis["invalid_rows"]:
                        # Summary of issues
                        issues_data = [(col, data["count"]) for col, data in analysis["invalid_rows"].items()]
                        issues_df = pd.DataFrame(issues_data, columns=["Column", "Invalid Rows"])
                        issues_df["Percentage"] = (issues_df["Invalid Rows"] / df.shape[0] * 100).round(2)
                        issues_df = issues_df.sort_values("Invalid Rows", ascending=False)

                        # Visualize invalid rows
                        fig = px.bar(issues_df, x="Column", y="Invalid Rows",
                                     title="Invalid Rows by Column",
                                     text="Percentage",
                                     color="Percentage",
                                     color_continuous_scale="Reds")
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True, key="invalid_rows_bar")

                        # Detailed view of each problem column
                        st.markdown("### Detailed Invalid Data")
                        st.info("Expand each section to see details of problem rows")

                        for col, data in analysis["invalid_rows"].items():
                            with st.expander(
                                    f"{col}: {data['count']} invalid rows ({data['count'] / df.shape[0]:.1%} of total)"):
                                st.markdown(f"Primary key: **{data['primary_key']}**")

                                if data["examples"]:
                                    examples_df = pd.DataFrame(data["examples"])
                                    st.dataframe(examples_df, use_container_width=True)

                                    if data["count"] > 20:
                                        st.info(f"Showing top 20 out of {data['count']} invalid rows")
                    else:
                        st.markdown('<div class="success-box">No invalid data detected in key columns!</div>',
                                    unsafe_allow_html=True)

                # Tab 6: Correlations
                with tab6:
                    st.markdown('<h2 class="sub-header">📈 Data Correlation Analysis</h2>', unsafe_allow_html=True)

                    if analysis["correlation"] and analysis["correlation"].get("matrix"):
                        st.markdown("### Correlation Matrix")

                        # Convert dict back to DataFrame for display
                        corr_matrix = pd.DataFrame(analysis["correlation"]["matrix"])

                        # Display as a heatmap
                        fig = px.imshow(corr_matrix,
                                        x=corr_matrix.columns.tolist(),
                                        y=corr_matrix.columns.tolist(),
                                        color_continuous_scale="RdBu_r",
                                        zmin=-1, zmax=1,
                                        title="Correlation Matrix")
                        st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")

                        # Display strong correlations
                        if analysis["correlation"]["strong_correlations"]:
                            st.markdown("### Strong Correlations")
                            st.markdown(
                                "These pairs of columns have strong relationships (correlation ≥ 0.5 or ≤ -0.5):")

                            strong_corr_df = pd.DataFrame(analysis["correlation"]["strong_correlations"])
                            strong_corr_df = strong_corr_df.sort_values("correlation", ascending=False)

                            st.dataframe(strong_corr_df, use_container_width=True)

                            # Select a correlation pair to visualize
                            if len(strong_corr_df) > 0:
                                correlation_pairs = [(row["column1"], row["column2"], row["correlation"]) for i, row in
                                                     strong_corr_df.iterrows()]
                                selected_pair = st.selectbox(
                                    "Select a pair to visualize:",
                                    options=correlation_pairs,
                                    format_func=lambda x: f"{x[0]} vs {x[1]} (correlation: {x[2]})"
                                )

                                if selected_pair:
                                    col1, col2, corr = selected_pair

                                    fig = px.scatter(
                                        df, x=col1, y=col2,
                                        trendline="ols",
                                        title=f"Correlation between {col1} and {col2} (r = {corr})",
                                        template="plotly_white"
                                    )
                                    st.plotly_chart(fig, use_container_width=True, key=f"corr_scatter_{col1}_{col2}")
                    else:
                        st.info("Not enough numerical columns found for correlation analysis.")

                    # Outlier analysis
                    st.markdown('<h2 class="sub-header">Outlier Detection</h2>', unsafe_allow_html=True)

                    if analysis["outliers"]:
                        outlier_cols = list(analysis["outliers"].keys())

                        st.markdown("### Columns with Outliers")
                        outlier_summary = []

                        for col in outlier_cols:
                            data = analysis["outliers"][col]
                            outlier_summary.append({
                                "Column": col,
                                "Outlier Count": data["count"],
                                "Percentage": f"{data['percentage']}%",
                                "Min Outlier": data["min_outlier"],
                                "Max Outlier": data["max_outlier"]
                            })

                        outlier_df = pd.DataFrame(outlier_summary)
                        st.dataframe(outlier_df, use_container_width=True)

                        # Visualization of outliers
                        if outlier_cols:
                            selected_col = st.selectbox("Select column to visualize outliers:", outlier_cols)

                            if selected_col:
                                outlier_data = analysis["outliers"][selected_col]

                                # Create box plot
                                fig = px.box(
                                    df, y=selected_col,
                                    title=f"Outlier Analysis for {selected_col}",
                                    template="plotly_white"
                                )

                                # Add horizontal lines for bounds
                                fig.add_hline(y=outlier_data["lower_bound"], line_dash="dash", line_color="red",
                                              annotation_text="Lower Bound", annotation_position="left")
                                fig.add_hline(y=outlier_data["upper_bound"], line_dash="dash", line_color="red",
                                              annotation_text="Upper Bound", annotation_position="right")

                                st.plotly_chart(fig, use_container_width=True, key=f"outlier_box_{selected_col}")

                                # Show example outlier rows
                                st.markdown("### Example Outlier Rows")
                                if outlier_data["examples"]:
                                    example_df = pd.DataFrame(outlier_data["examples"])
                                    st.dataframe(example_df, use_container_width=True)
                    else:
                        st.info("No significant outliers detected in numerical columns.")

                # Tab 7: Distribution Analysis
                with tab7:
                    st.markdown('<h2 class="sub-header">📉 Distribution Analysis</h2>', unsafe_allow_html=True)

                    # Distribution analysis

                    if analysis["distribution"]:
                        dist_cols = list(analysis["distribution"].keys())

                        st.markdown("### Distribution Characteristics")
                        dist_summary = []

                        for col in dist_cols:
                            data = analysis["distribution"][col]
                            dist_summary.append({
                                "Column": col,
                                "Mean": round(data["mean"], 2),
                                "Median": round(data["median"], 2),
                                "Skewness": round(data["skewness"], 2),
                                "Distribution Type": data["distribution_type"]
                            })

                        dist_df = pd.DataFrame(dist_summary)
                        st.dataframe(dist_df, use_container_width=True)
                        # Visualization of distributions
                        if dist_cols:
                            selected_col = st.selectbox("Select column to analyze distribution:", dist_cols)

                            if selected_col:
                                dist_data = analysis["distribution"][selected_col]

                                col1, col2 = st.columns(2)

                                with col1:
                                    # Create histogram with distribution curve
                                    fig = px.histogram(
                                        df, x=selected_col,
                                        marginal="box",
                                        title=f"Distribution of {selected_col}",
                                        template="plotly_white"
                                    )

                                    # Add vertical lines for mean and median
                                    fig.add_vline(x=dist_data["mean"], line_dash="solid", line_color="red",
                                                  annotation_text="Mean", annotation_position="top")
                                    fig.add_vline(x=dist_data["median"], line_dash="dash", line_color="green",
                                                  annotation_text="Median", annotation_position="bottom")

                                    st.plotly_chart(fig, use_container_width=True, key=f"dist_hist_{selected_col}")

                                with col2:
                                    # Create a QQ plot to check for normality
                                    from scipy import stats

                                    # Get the data and drop nulls
                                    data = df[selected_col].dropna()

                                    # Calculate QQ data
                                    qqplot_data = stats.probplot(data, dist="norm")

                                    # Extract the data points
                                    x_data = [point[0] for point in qqplot_data[0]]
                                    y_data = [point[1] for point in qqplot_data[0]]

                                    # Create the QQ plot
                                    qq_fig = px.scatter(
                                        x=x_data, y=y_data,
                                        labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
                                        title=f"Q-Q Plot for {selected_col} (check for normality)"
                                    )

                                    # Add the reference line
                                    qq_fig.add_trace(px.line(x=x_data, y=x_data).data[0])

                                    st.plotly_chart(qq_fig, use_container_width=True, key=f"qq_plot_{selected_col}")

                                # Distribution insights
                                st.markdown("### Distribution Insights")
                                st.markdown(f"**Distribution Type**: {dist_data['distribution_type']}")

                                skewness = dist_data["skewness"]
                                if abs(skewness) < 0.5:
                                    skew_insight = "The data is approximately symmetric, suggesting a normal distribution."
                                elif skewness > 0.5:
                                    skew_insight = "The data is right-skewed (positive skew), with a longer tail to the right. This suggests the presence of high outliers."
                                else:
                                    skew_insight = "The data is left-skewed (negative skew), with a longer tail to the left. This suggests the presence of low outliers."

                                st.markdown(f"**Skewness Insight**: {skew_insight}")

                                mean_median_diff = dist_data["mean_median_difference"]
                                if mean_median_diff > 0.1:
                                    mm_insight = "There is a notable difference between mean and median, confirming the presence of skew or outliers."
                                else:
                                    mm_insight = "The mean and median are close, suggesting symmetry in the central part of the distribution."

                                st.markdown(f"**Mean vs Median**: {mm_insight}")

                                # Recommendations based on distribution
                                st.markdown("### Recommendations")

                                if abs(skewness) > 0.5:
                                    st.markdown(
                                        "- For statistical tests, consider non-parametric methods or transform the data")
                                    st.markdown(
                                        f"- When summarizing {selected_col}, use median rather than mean for central tendency")
                                    if skewness > 0.5:
                                        st.markdown("- Consider log transformation to normalize the distribution")
                                    else:
                                        st.markdown("- Consider power transformation to normalize the distribution")
                                else:
                                    st.markdown(
                                        "- Data appears normally distributed, parametric statistical methods are appropriate")
                                    st.markdown("- Mean is a reliable measure of central tendency for this column")
                    else:
                        st.info("Not enough numerical columns found for distribution analysis.")

                    # Categorical data analysis
                    st.markdown('<h2 class="sub-header">Categorical Data Analysis</h2>', unsafe_allow_html=True)

                    if analysis["categorical"]:
                        cat_cols = list(analysis["categorical"].keys())

                        st.markdown("### Categorical Columns")
                        cat_summary = []

                        for col in cat_cols:
                            data = analysis["categorical"][col]
                            cat_summary.append({
                                "Column": col,
                                "Unique Values": data["unique_values"],
                                "Balance Score": data["balance_score"],
                                "Balance Status": data["balance_status"],
                                "Top Category (%)": f"{data['top_category_dominance']}%"
                            })

                        cat_df = pd.DataFrame(cat_summary)
                        st.dataframe(cat_df, use_container_width=True)

                        # Visualization of category distribution
                        if cat_cols:
                            selected_cat = st.selectbox("Select categorical column to analyze:", cat_cols)

                            if selected_cat:
                                cat_data = analysis["categorical"][selected_cat]

                                # Create bar chart of top categories
                                top_cats = pd.DataFrame(cat_data["top_categories"])

                                fig = px.bar(
                                    top_cats, x="Value", y="Percentage",
                                    title=f"Top Categories for {selected_cat}",
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"cat_dist_{selected_cat}")

                                # Category balance insights
                                st.markdown("### Balance Analysis")
                                st.markdown(
                                    f"**Balance Score**: {cat_data['balance_score']} ({cat_data['balance_status']})")

                                if cat_data["balance_status"] == "Imbalanced":
                                    st.warning(
                                        f"This column is highly imbalanced. The top category represents {cat_data['top_category_dominance']}% of all values.")
                                    st.markdown("**Implications**:")
                                    st.markdown(
                                        "- For machine learning, consider techniques for imbalanced data like resampling")
                                    st.markdown(
                                        "- Be careful with aggregate statistics that may be dominated by the majority category")
                                elif cat_data["balance_status"] == "Moderately balanced":
                                    st.info("This column has moderate imbalance but is not severely skewed.")
                                else:
                                    st.success("This column has a good balance across categories.")
                    else:
                        st.info("No categorical columns identified for analysis.")

                # Tab 8: Consistency Analysis
                with tab8:
                    st.markdown('<h2 class="sub-header">🔄 Data Consistency & Quality</h2>', unsafe_allow_html=True)

                    # Duplicate analysis
                    st.markdown("### Duplicate Analysis")

                    # Add column selection for duplicate analysis
                    all_columns_dup = st.checkbox("Check all columns for duplicates", value=True,
                                                  key="tab8_all_columns_dup")

                    if not all_columns_dup:
                        selected_columns_dup = st.multiselect(
                            "Select columns to check for duplicates:",
                            options=df.columns,
                            default=[],
                            key="tab8_selected_columns_dup"
                        )
                        # Run duplicate analysis with selected columns
                        duplicate_analysis = profiler.get_duplicate_analysis(columns=selected_columns_dup)
                    else:
                        # Use pre-calculated duplicate analysis
                        duplicate_analysis = analysis["duplicates"]

                    if duplicate_analysis:
                        exact_dups = duplicate_analysis["exact_duplicates"]

                        st.metric(
                            label="Exact Duplicate Rows",
                            value=f"{exact_dups['count']}",
                            delta=f"{exact_dups['percentage']}% of rows",
                            delta_color="inverse"
                        )

                        if exact_dups["count"] > 0:
                            st.warning(
                                "Dataset contains exact duplicate rows, which may affect analysis and statistics.")

                            # Show sample duplicate rows
                            with st.expander("View Sample Duplicate Rows", expanded=False):
                                if "sample_rows" in exact_dups and exact_dups["sample_rows"]:
                                    st.dataframe(pd.DataFrame(exact_dups["sample_rows"]), use_container_width=True)
                                else:
                                    st.info("No sample rows available for duplicate analysis")
                        else:
                            st.success("No exact duplicate rows found in the dataset.")

                        # Check for note in exact duplicates (added by our fix)
                        if "note" in exact_dups:
                            st.warning(f"Note: {exact_dups['note']}")

                        # Key duplicates
                        if "key_duplicates" in duplicate_analysis and duplicate_analysis["key_duplicates"]:
                            st.markdown("### Duplicate Keys")
                            st.markdown("The following potential ID/key columns have duplicate values:")

                            key_dups = []
                            for col, data in duplicate_analysis["key_duplicates"].items():
                                key_dups.append({
                                    "Column": col,
                                    "Duplicate Count": data["count"],
                                    "Percentage": f"{data['percentage']}%",
                                    "Example Values": ", ".join([str(x) for x in data["examples"]])
                                })

                            key_dups_df = pd.DataFrame(key_dups)
                            st.dataframe(key_dups_df, use_container_width=True)

                            # Show sample rows with duplicate keys
                            with st.expander("View Sample Rows with Duplicate Keys", expanded=False):
                                if key_dups:
                                    selected_key_col = st.selectbox(
                                        "Select key column to view samples with duplicate values:",
                                        options=[item["Column"] for item in key_dups],
                                        key="tab8_selected_key_col"
                                    )

                                    # Find the sample rows for the selected column
                                    for col, data in duplicate_analysis["key_duplicates"].items():
                                        if col == selected_key_col:
                                            if "sample_rows" in data and data["sample_rows"]:
                                                st.dataframe(pd.DataFrame(data["sample_rows"]),
                                                             use_container_width=True)
                                            else:
                                                st.info(f"No sample rows available for {selected_key_col}")
                                            break

                            st.warning(
                                "Duplicate keys may indicate data integrity issues if these columns are intended to be unique identifiers.")

                    if analysis["duplicates"]:
                        exact_dups = analysis["duplicates"]["exact_duplicates"]

                        st.metric(
                            label="Exact Duplicate Rows",
                            value=f"{exact_dups['count']}",
                            delta=f"{exact_dups['percentage']}% of rows",
                            delta_color="inverse"
                        )

                        if exact_dups["count"] > 0:
                            st.warning(
                                "Dataset contains exact duplicate rows, which may affect analysis and statistics.")
                        else:
                            st.success("No exact duplicate rows found in the dataset.")

                        # Key duplicates
                        if "key_duplicates" in analysis["duplicates"] and analysis["duplicates"]["key_duplicates"]:
                            st.markdown("### Duplicate Keys")
                            st.markdown("The following potential ID/key columns have duplicate values:")

                            key_dups = []
                            for col, data in analysis["duplicates"]["key_duplicates"].items():
                                key_dups.append({
                                    "Column": col,
                                    "Duplicate Count": data["count"],
                                    "Percentage": f"{data['percentage']}%",
                                    "Example Values": ", ".join([str(x) for x in data["examples"]])
                                })

                            key_dups_df = pd.DataFrame(key_dups)
                            st.dataframe(key_dups_df, use_container_width=True)

                            st.warning(
                                "Duplicate keys may indicate data integrity issues if these columns are intended to be unique identifiers.")

                    # Consistency analysis
                    st.markdown("### Logical Consistency")

                    if analysis["consistency"] and len(analysis["consistency"]) > 0:
                        st.markdown("The following logical inconsistencies were detected between related columns:")

                        consistency_data = []
                        for check, data in analysis["consistency"].items():
                            consistency_data.append({
                                "Columns": check,
                                "Issue": data["issue"],
                                "Count": data["count"],
                                "Percentage": f"{data['percentage']}%"
                            })

                        consistency_df = pd.DataFrame(consistency_data)
                        st.dataframe(consistency_df, use_container_width=True)

                        # Show examples of inconsistencies
                        if consistency_data:
                            selected_check = st.selectbox(
                                "Select inconsistency to see examples:",
                                options=list(analysis["consistency"].keys())
                            )

                            if selected_check:
                                examples = analysis["consistency"][selected_check]["examples"]
                                if examples:
                                    examples_df = pd.DataFrame(examples)
                                    st.dataframe(examples_df, use_container_width=True)

                        st.warning("These inconsistencies may indicate data quality issues or errors in data entry.")
                    else:
                        st.success("No logical inconsistencies detected between related columns.")

                    # Time series analysis
                    if analysis["time_series"] and len(analysis["time_series"]) > 0:
                        st.markdown("### Time Series Analysis")

                        time_cols = list(analysis["time_series"].keys())
                        selected_time = st.selectbox("Select date/time column to analyze:", time_cols,
                                                     key="time_series_consistency")

                        if selected_time:
                            time_data = analysis["time_series"][selected_time]

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Date Range", f"{time_data['date_range_days']} days")

                            with col2:
                                st.metric("Date From", time_data["min_date"])

                            with col3:
                                st.metric("Date To", time_data["max_date"])

                            st.markdown(f"**Trend**: {time_data['trend']}")

                            # Visualize monthly distribution
                            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                                           "Dec"]
                            month_data = []

                            for month_num, count in time_data["month_distribution"].items():
                                month_data.append({
                                    "Month": month_names[int(month_num) - 1],
                                    "Count": count
                                })

                            # Sort by month
                            month_order = {month: i for i, month in enumerate(month_names)}
                            month_data.sort(key=lambda x: month_order[x["Month"]])

                            month_df = pd.DataFrame(month_data)

                            fig = px.bar(
                                month_df, x="Month", y="Count",
                                title=f"Monthly Distribution for {selected_time}",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"consistency_monthly_{selected_time}")

                    # Data completeness
                    st.markdown("### Data Completeness")

                    if analysis["completeness"]:
                        completeness = analysis["completeness"]["overall"]

                        st.metric(
                            label="Overall Completeness",
                            value=f"{completeness['completeness_percentage']}%",
                            delta=f"{completeness['filled_cells']} of {completeness['total_cells']} cells filled",
                            delta_color="normal"
                        )

                        # Create a pie chart of data completeness
                        fig = px.pie(
                            names=["Filled Cells", "Missing Values", "Blank Cells"],
                            values=[completeness["filled_cells"], completeness["missing_cells"],
                                    completeness["blank_cells"]],
                            title="Data Completeness Breakdown",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="completeness_pie")

                        # Potential required fields
                        if "potential_required_fields" in analysis["completeness"] and analysis["completeness"][
                            "potential_required_fields"]:
                            st.markdown("### Potential Required Fields")
                            st.markdown(
                                "These columns are almost always filled (95%+ complete) and might be required fields:")

                            for field in analysis["completeness"]["potential_required_fields"]:
                                st.markdown(f"- {field}")

                        # Column completeness visualization
                        st.markdown("### Column Completeness")

                        column_completeness = analysis["completeness"]["columns"]
                        completeness_df = pd.DataFrame([
                            {"Column": col, "Completeness": value}
                            for col, value in column_completeness.items()
                        ])

                        # Sort by completeness descending
                        completeness_df = completeness_df.sort_values("Completeness", ascending=False)

                        fig = px.bar(
                            completeness_df.head(20), x="Column", y="Completeness",
                            title="Top 20 Columns by Completeness (%)",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="col_completeness_bar")
                        # Add column selection for duplicate analysis
                        all_columns_dup = st.checkbox("Check all columns for duplicates", value=True,
                                                      key="all_columns_dup")

                        if not all_columns_dup:
                            selected_columns_dup = st.multiselect(
                                "Select columns to check for duplicates:",
                                options=df.columns,
                                default=[]
                            )
                            # Run duplicate analysis with selected columns
                            duplicate_analysis = profiler.get_duplicate_analysis(columns=selected_columns_dup)
                        else:
                            # Use pre-calculated duplicate analysis
                            duplicate_analysis = analysis["duplicates"]

                        if duplicate_analysis:
                            exact_dups = duplicate_analysis["exact_duplicates"]

                            st.metric(
                                label="Exact Duplicate Rows",
                                value=f"{exact_dups['count']}",
                                delta=f"{exact_dups['percentage']}% of rows",
                                delta_color="inverse"
                            )

                            if exact_dups["count"] > 0:
                                st.warning(
                                    "Dataset contains exact duplicate rows, which may affect analysis and statistics.")

                                # Show sample duplicate rows
                                st.markdown("### Sample Duplicate Rows")
                                if "sample_rows" in exact_dups and exact_dups["sample_rows"]:
                                    st.dataframe(pd.DataFrame(exact_dups["sample_rows"]), use_container_width=True)
                                else:
                                    st.info("No sample rows available for duplicate analysis")
                            else:
                                st.success("No exact duplicate rows found in the dataset.")

                            # Key duplicates
                            if "key_duplicates" in duplicate_analysis and duplicate_analysis["key_duplicates"]:
                                st.markdown("### Duplicate Keys")
                                st.markdown("The following potential ID/key columns have duplicate values:")

                                key_dups = []
                                for col, data in duplicate_analysis["key_duplicates"].items():
                                    key_dups.append({
                                        "Column": col,
                                        "Duplicate Count": data["count"],
                                        "Percentage": f"{data['percentage']}%",
                                        "Example Values": ", ".join([str(x) for x in data["examples"]])
                                    })

                                key_dups_df = pd.DataFrame(key_dups)
                                st.dataframe(key_dups_df, use_container_width=True)

                                # Show sample rows with duplicate keys
                                st.markdown("### Sample Rows with Duplicate Keys")
                                if key_dups:
                                    selected_key_col = st.selectbox(
                                        "Select key column to view samples with duplicate values:",
                                        options=[item["Column"] for item in key_dups]
                                    )

                                    # Find the sample rows for the selected column
                                    for col, data in duplicate_analysis["key_duplicates"].items():
                                        if col == selected_key_col and "sample_rows" in data:
                                            if data["sample_rows"]:
                                                st.dataframe(pd.DataFrame(data["sample_rows"]),
                                                             use_container_width=True)
                                            else:
                                                st.info(f"No sample rows available for {selected_key_col}")
                                            break

                                st.warning(
                                    "Duplicate keys may indicate data integrity issues if these columns are intended to be unique identifiers.")

                # Tab 9: File Conversion
                with tab9:
                    st.markdown('<h2 class="sub-header">🔄 File Conversion Tool</h2>', unsafe_allow_html=True)

                    st.markdown("""
                    <div class="converter-card">
                        <h3>Convert your data to different formats</h3>
                        <p>Use this tool to convert your data file to various formats for use in different applications.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Filter data before conversion
                    st.markdown("### 1. Filter Data (Optional)")

                    with st.expander("Apply filters before conversion", expanded=False):
                        filtered_df = filter_dataframe(df)

                    # Column selection
                    st.markdown("### 2. Select Columns (Optional)")
                    all_columns = st.checkbox("Include all columns", value=True)

                    if not all_columns:
                        selected_columns = st.multiselect(
                            "Choose columns to include in the output file:",
                            options=df.columns,
                            default=list(df.columns)
                        )
                        output_df = filtered_df[selected_columns]
                    else:
                        output_df = filtered_df

                    # File format selection
                    st.markdown("### 3. Choose Output Format")

                    col1, col2 = st.columns(2)

                    with col1:
                        output_format = st.selectbox(
                            "Select output file format:",
                            options=["CSV", "Parquet", "Excel", "JSON"],
                            index=0
                        )

                    with col2:
                        output_filename = st.text_input("Output filename (without extension):", file_name)

                    # Show preview of data to be converted
                    st.markdown("### 4. Preview Data to be Converted")
                    st.dataframe(output_df.head(5), use_container_width=True)
                    st.info(f"Converting {output_df.shape[0]} rows and {output_df.shape[1]} columns")

                    # Format-specific options
                    st.markdown("### 5. Format-Specific Options")

                    format_options = {}

                    if output_format == "CSV":
                        col1, col2 = st.columns(2)
                        with col1:
                            format_options["separator"] = st.selectbox(
                                "Select delimiter:",
                                options=[",", ";", "\\t", "|"],
                                index=0,
                                format_func=lambda x: {"\\t": "Tab"}.get(x, x)
                            )
                        with col2:
                            format_options["decimal"] = st.selectbox(
                                "Decimal separator:",
                                options=[".", ","],
                                index=0
                            )

                    elif output_format == "Excel":
                        format_options["sheet_name"] = st.text_input("Sheet name:", "Sheet1")

                    elif output_format == "JSON":
                        format_options["orient"] = st.selectbox(
                            "JSON orientation:",
                            options=["records", "columns", "index", "split", "table"],
                            index=0
                        )

                    elif output_format == "Parquet":
                        format_options["compression"] = st.selectbox(
                            "Compression:",
                            options=["snappy", "gzip", "brotli", "lz4", "zstd", None],
                            index=0
                        )

                    # Convert and download button
                    st.markdown("### 6. Convert and Download")

                    if st.button("Generate Download Link", type="primary"):
                        with st.spinner(f"Converting to {output_format}..."):
                            try:
                                if output_format == "CSV":
                                    sep = format_options.get("separator", ",")
                                    if sep == "\\t":
                                        sep = "\t"

                                    decimal = format_options.get("decimal", ".")

                                    # Convert to CSV with options
                                    csv_buffer = io.StringIO()
                                    output_df.to_csv(csv_buffer, index=False, sep=sep, decimal=decimal)
                                    csv_buffer.seek(0)

                                    # Generate download link
                                    b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="{output_filename}.csv" class="download-button">Download {output_filename}.csv</a>'

                                elif output_format == "Parquet":
                                    compression = format_options.get("compression", "snappy")

                                    # Convert to Parquet with options
                                    parquet_buffer = io.BytesIO()
                                    output_df.to_parquet(parquet_buffer, index=False, compression=compression)
                                    parquet_buffer.seek(0)

                                    # Generate download link
                                    b64 = base64.b64encode(parquet_buffer.getvalue()).decode()
                                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{output_filename}.parquet" class="download-button">Download {output_filename}.parquet</a>'

                                elif output_format == "Excel":
                                    sheet_name = format_options.get("sheet_name", "Sheet1")

                                    # Convert to Excel with options
                                    excel_buffer = io.BytesIO()
                                    output_df.to_excel(excel_buffer, index=False, sheet_name=sheet_name)
                                    excel_buffer.seek(0)

                                    # Generate download link
                                    b64 = base64.b64encode(excel_buffer.getvalue()).decode()
                                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{output_filename}.xlsx" class="download-button">Download {output_filename}.xlsx</a>'

                                elif output_format == "JSON":
                                    orient = format_options.get("orient", "records")

                                    # Convert to JSON with options
                                    json_str = output_df.to_json(orient=orient)

                                    # Generate download link
                                    b64 = base64.b64encode(json_str.encode()).decode()
                                    href = f'<a href="data:application/json;base64,{b64}" download="{output_filename}.json" class="download-button">Download {output_filename}.json</a>'

                                # Show success message with download link
                                st.success(f"✅ Successfully converted to {output_format}!")
                                st.markdown(href, unsafe_allow_html=True)

                                # Show file info
                                file_size = len(b64) * 3 / 4 / 1024  # Approximate size in KB
                                st.info(f"File size: approximately {file_size:.1f} KB")

                            except Exception as e:
                                st.error(f"Error converting file: {str(e)}")

                    # Format information
                    with st.expander("Format Information", expanded=False):
                        st.markdown("""
                        ### CSV (Comma-Separated Values)
                        - **Pros**: Simple, widely supported by all applications
                        - **Cons**: No data type preservation, larger file size
                        - **Best for**: Simple data exchange, spreadsheet applications

                        ### Parquet
                        - **Pros**: Very efficient compression, preserves data types, column-oriented for analytics
                        - **Cons**: Not human-readable, requires specific libraries to read
                        - **Best for**: Big data applications, analytics workflows

                        ### Excel (XLSX)
                        - **Pros**: Native format for Microsoft Excel, supports multiple sheets
                        - **Cons**: Larger file size, limited to ~1M rows
                        - **Best for**: Users who need to work with data in Excel

                        ### JSON (JavaScript Object Notation)
                        - **Pros**: Human-readable, widely used in web applications
                        - **Cons**: Larger file size, less efficient for numeric data
                        - **Best for**: Web applications, API responses, nested data structures
                        """)

                # Tab 10: Report
                with tab10:
                    st.markdown('<h2 class="sub-header">📑 Report Generator</h2>', unsafe_allow_html=True)

                    st.markdown("""
                    <div class="converter-card">
                        <h3>Generate a comprehensive report</h3>
                        <p>Create a downloadable report with all analysis results and visualizations.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Report options
                    st.markdown("### Report Options")

                    col1, col2 = st.columns(2)

                    with col1:
                        report_title = st.text_input("Report title", f"{file_name} Data Quality Report")

                    with col2:
                        include_preview = st.checkbox("Include data preview", value=True)

                    # Report sections selection
                    st.markdown("### Select Sections to Include")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        include_schema = st.checkbox("Schema Information", value=True)
                        include_null = st.checkbox("Null Values Analysis", value=True)
                        include_correlations = st.checkbox("Correlation Analysis", value=True)

                    with col2:
                        include_unique = st.checkbox("Unique Values Analysis", value=True)
                        include_consistency = st.checkbox("Consistency Analysis", value=True)
                        include_distributions = st.checkbox("Distribution Analysis", value=True)

                    with col3:
                        include_issues = st.checkbox("Data Issues", value=True)
                        include_duplicates = st.checkbox("Duplicate Analysis", value=True)
                        include_visualizations = st.checkbox("Include Visualizations", value=True)

                    # Report format
                    st.markdown("### Report Format")
                    st.info(
                        "HTML format is used for reports. It can be opened in any web browser and printed to PDF if needed.")
                    report_format = "HTML"  # Always use HTML

                    # Generate report button
                    if st.button("Generate Report", type="primary"):
                        with st.spinner("Generating report..."):
                            try:
                                # Always generate HTML report
                                html_content = generate_html_report(df, analysis, report_title)

                                if html_content:
                                    if report_format == "HTML":
                                        # Create download link for HTML
                                        st.success("✅ HTML Report successfully generated!")
                                        st.markdown(
                                            get_download_link(html_content, "html", f"{file_name}_data_quality_report"),
                                            unsafe_allow_html=True)

                                        # Show preview
                                        with st.expander("Preview HTML Report", expanded=True):
                                            st.components.v1.html(html_content, height=600, scrolling=True)

                                    else:  # PDF format
                                        try:
                                            # Try to import pdfkit for PDF generation
                                            import pdfkit

                                            # Create a temporary HTML file
                                            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
                                                tmp_file.write(html_content.encode('utf-8'))
                                                html_path = tmp_file.name

                                            # Configure PDF options
                                            options = {
                                                'page-size': 'A4',
                                                'margin-top': '15mm',
                                                'margin-right': '15mm',
                                                'margin-bottom': '15mm',
                                                'margin-left': '15mm',
                                                'encoding': 'UTF-8',
                                                'no-outline': None
                                            }

                                            # Add flag to allow local file access
                                            if hasattr(pdfkit.configuration(), 'get_wkhtmltopdf'):
                                                options['enable-local-file-access'] = None

                                            # Generate PDF
                                            pdf_path = os.path.join(tempfile.gettempdir(),
                                                                    f"{file_name}_data_quality_report.pdf")
                                            pdfkit.from_file(html_path, pdf_path, options=options)

                                            # Read the generated PDF
                                            with open(pdf_path, "rb") as pdf_file:
                                                pdf_content = pdf_file.read()

                                                # Create download link for PDF
                                                b64 = base64.b64encode(pdf_content).decode()
                                                pdf_filename = f"{file_name}_data_quality_report.pdf"
                                                href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" class="download-button">Download PDF Report</a>'

                                                st.success("✅ PDF Report successfully generated!")
                                                st.markdown(href, unsafe_allow_html=True)

                                            # Clean up temporary files
                                            os.remove(html_path)
                                            os.remove(pdf_path)

                                        except ImportError:
                                            st.warning(
                                                "PDF generation requires pdfkit. Please install it with `pip install pdfkit`.")
                                            st.info("Falling back to HTML report format.")
                                            st.markdown(get_download_link(html_content, "html",
                                                                          f"{file_name}_data_quality_report"),
                                                        unsafe_allow_html=True)

                                        except Exception as e:
                                            st.error(f"Error generating PDF: {str(e)}")
                                            st.info("Falling back to HTML report format.")
                                            st.markdown(get_download_link(html_content, "html",
                                                                          f"{file_name}_data_quality_report"),
                                                        unsafe_allow_html=True)

                                            # Show HTML preview
                                            with st.expander("Preview HTML Report", expanded=True):
                                                st.components.v1.html(html_content, height=600, scrolling=True)
                                else:
                                    st.error("Failed to generate report content.")
                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")

                    # Report information
                    with st.expander("About Reports", expanded=False):
                        st.markdown("""
                        ### Report Contents

                        The generated report includes:

                        1. **Overview**: Dataset summary and data quality score
                        2. **Schema Information**: Column types and nullability
                        3. **Data Quality Issues**: Null and blank values analysis
                        4. **Column Cardinality**: Unique values distribution
                        5. **Correlation Analysis**: Relationships between numerical columns
                        6. **Distribution Analysis**: Statistical properties of numerical data
                        7. **Consistency Analysis**: Logical relationships and data integrity
                        8. **Duplicate Analysis**: Exact and key-based duplicate detection

                        All sections include visualizations and tabular data for comprehensive analysis.

                        ### Format Options

                        - **HTML**: Universal format that works in any browser. No additional dependencies required.
                        - **PDF**: More professional format for sharing and printing, but requires wkhtmltopdf to be installed.

                        ### Tips for Best Results

                        - Report generation works best with datasets that are not too large
                        - For very large reports, consider using the individual export options in each tab
                        """)

                # Tab 11: Guide
                with tab11:
                    st.markdown("# 📖 Analysis Guide")

                    st.info("""
                        This guide explains each analysis section in the Data Profiler, why it's important, and when to use it. 
                        Select a topic from the dropdown below to learn more about a specific type of analysis.
                    """)

                    # Create a dropdown to select topics
                    guide_topic = st.selectbox(
                        "Select a section to learn about:",
                        [
                            "Data Quality Overview",
                            "Schema & Null Values Analysis",
                            "Unique Values Analysis",
                            "Content Analysis",
                            "Data Issues Analysis",
                            "Correlation Analysis",
                            "Distribution Analysis",
                            "Consistency & Quality Analysis",
                            "File Conversion",
                            "Report Generation",
                            "AI Assistant"
                        ]
                    )

                    # Data Quality Overview
                    if guide_topic == "Data Quality Overview":
                        st.header("Data Quality Overview")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            The Data Quality Dashboard provides a high-level summary of your dataset's quality through key metrics and 
                            a quality score that quantifies the overall condition of your data.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write("• **Missing Values Count:** Total number of null values across all columns")
                            st.write("• **Blank Values Count:** Total number of empty strings in text columns")
                            st.write("• **Invalid Dates:** Count of improperly formatted dates")
                            st.write("• **Problem Columns:** Number of columns with quality issues")
                            st.write("• **Data Quality Score:** A percentage score based on completeness and validity")
                            st.write(
                                "• **Numerical Distribution:** Visualization of value distributions in numerical columns")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Understanding data quality is crucial because it directly impacts the reliability of any analysis or model built using this data. 
                            Poor quality data leads to unreliable insights, while high-quality data enables confident decision-making.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• As your first step in any data analysis project")
                            st.write("• When deciding if a dataset needs cleaning before use")
                            st.write("• To communicate data quality status to stakeholders")
                            st.write("• To track improvements in data quality over time")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You've received a customer dataset from another department. Before using it for churn analysis, 
                            you need to quickly assess if the data is complete and reliable. The Overview shows a quality score of 65% 
                            with significant missing values in key columns like "Last Purchase Date" and "Customer Segment." 
                            This tells you that data cleaning will be necessary before proceeding with analysis.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Data Quality Score")

                        # Create a gauge chart visualization with pure Streamlit
                        score = 65  # Example quality score

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            # Simple gauge representation
                            st.markdown(f"### {score}%")
                            st.progress(score / 100)

                            if score >= 90:
                                st.markdown("**Status: Excellent**")
                            elif score >= 75:
                                st.markdown("**Status: Good**")
                            elif score >= 60:
                                st.markdown("**Status: Fair**")
                            else:
                                st.markdown("**Status: Poor**")

                        with col2:
                            # Display quality metrics
                            metrics_data = pd.DataFrame({
                                'Metric': ['Missing Values', 'Duplicate Rows', 'Invalid Dates', 'Blank Cells'],
                                'Count': [234, 15, 45, 120],
                                'Percentage': [3.2, 0.5, 1.8, 2.3]
                            })
                            st.dataframe(metrics_data, hide_index=True)

                    # Schema & Null Values Analysis
                    elif guide_topic == "Schema & Null Values Analysis":
                        st.header("Schema & Null Values Analysis")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            This analysis examines your dataset's structure (schema) and identifies missing or blank values.
                            It shows what type of data each column contains and quantifies how many values are missing or empty.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write("• **Schema Information:** Column names, data types, and nullability")
                            st.write(
                                "• **Data Type Distribution:** Visualization of column types (numeric, string, etc.)")
                            st.write("• **Null Value Analysis:** Count and percentage of missing values per column")
                            st.write(
                                "• **Blank Value Analysis:** Count and percentage of empty strings in text columns")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Missing data can significantly impact analysis results. For example, if 30% of customer income data is missing,
                            any income-based segmentation will be biased. Similarly, the wrong data type (e.g., numeric values stored as text)
                            can prevent proper calculations.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• Before any data transformation or modeling")
                            st.write("• When planning a data cleaning strategy")
                            st.write("• To decide on appropriate imputation methods for missing values")
                            st.write("• To identify columns that may need to be excluded from analysis")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You're analyzing sales data and notice that the "Customer ID" column shows 5% null values (150 out of 3000 records).
                            This is concerning because Customer ID should be required. Further investigation shows these null values occurred mostly in 
                            January, indicating a potential system issue that month. You decide to exclude these records from customer-level 
                            analyses but keep them for aggregate sales reporting.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Null Values Analysis")

                        # Sample data for null values visualization
                        null_data = pd.DataFrame({
                            'Column': ['Customer ID', 'Email', 'Phone', 'Address', 'Purchase Date'],
                            'Null Count': [150, 320, 278, 412, 95],
                            'Total Count': [3000, 3000, 3000, 3000, 3000],
                            'Percentage': [5.0, 10.7, 9.3, 13.7, 3.2]
                        })

                        # Create the representation column
                        null_data['Representation'] = null_data.apply(
                            lambda row: f"{row['Null Count']} out of {row['Total Count']} ({row['Percentage']}%)",
                            axis=1
                        )

                        # Display the dataframe
                        st.dataframe(null_data[['Column', 'Null Count', 'Total Count', 'Percentage', 'Representation']],
                                     hide_index=True)

                        # Create a simple bar chart
                        st.bar_chart(null_data.set_index('Column')['Percentage'])

                    # Unique Values Analysis
                    elif guide_topic == "Unique Values Analysis":
                        st.header("Unique Values Analysis")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            The Unique Values Analysis examines the cardinality (number of unique values) in each column of your dataset.
                            It identifies potential primary key columns and allows you to explore the actual values present in each column.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write("• **Column Cardinality:** Count of distinct values in each column")
                            st.write("• **Cardinality Ratio:** Unique values as a percentage of total rows")
                            st.write("• **Potential Primary Keys:** Columns with high uniqueness (>90%)")
                            st.write("• **Value Samples:** Examples of the actual values in each column")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Understanding unique values helps you identify which columns can serve as identifiers, which contain 
                            categorical data, and which might have data quality issues. For example, a "State" column should have 
                            a small number of unique values, while an "Email" column should have high cardinality.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• When exploring a new dataset to understand its structure")
                            st.write("• To identify primary keys for database design or data joining")
                            st.write("• To check for unexpected duplicates in ID columns")
                            st.write("• To find categorical columns for encoding or feature engineering")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You're analyzing a customer transaction dataset and find that the "Transaction_ID" column has only 10,000 unique values 
                            out of 15,000 rows. This is alarming because transaction IDs should be unique. Investigation reveals that some transactions 
                            were duplicated due to a system error, which would have skewed your analysis had you not identified this issue.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Column Cardinality")

                        # Sample data for cardinality visualization
                        cardinality_data = pd.DataFrame({
                            'Column': ['Transaction_ID', 'Customer_ID', 'Product_ID', 'Store_ID', 'Payment_Method'],
                            'Unique Count': [10000, 5200, 420, 45, 6],
                            'Total Rows': [15000, 15000, 15000, 15000, 15000],
                            'Ratio': [0.67, 0.35, 0.028, 0.003, 0.0004]
                        })

                        # Format ratio as percentage
                        cardinality_data['Cardinality %'] = (cardinality_data['Ratio'] * 100).round(1)

                        # Display the dataframe
                        st.dataframe(cardinality_data[['Column', 'Unique Count', 'Total Rows', 'Cardinality %']],
                                     hide_index=True)

                        # Create a simple bar chart
                        st.bar_chart(cardinality_data.set_index('Column')['Cardinality %'])

                        # Show potential primary keys
                        st.markdown("#### Potential Primary Keys")
                        pk_data = cardinality_data[cardinality_data['Ratio'] > 0.9] if any(
                            cardinality_data['Ratio'] > 0.9) else None

                        if pk_data is not None and not pk_data.empty:
                            st.dataframe(pk_data[['Column', 'Unique Count', 'Cardinality %']], hide_index=True)
                        else:
                            st.warning("No potential primary keys found (columns with >90% unique values)")
                            st.markdown(
                                "**Note:** Transaction_ID should be unique but has only 67% unique values, indicating a data issue.")

                    # Content Analysis
                    elif guide_topic == "Content Analysis":
                        st.header("Content Analysis")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            Content Analysis examines the patterns and characteristics within your data values themselves. 
                            It validates date formats, identifies relationships between columns, analyzes string patterns, and detects temporal patterns in time-series data.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write("• **Date Format Validation:** Identification of invalid date formats")
                            st.write("• **Column Relationships:** Connections between ID/reference columns")
                            st.write(
                                "• **String Pattern Analysis:** Character patterns, length stats, and common prefixes/suffixes")
                            st.write("• **Time Series Analysis:** Temporal patterns in date columns")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Understanding the content of your data helps validate its correctness and usability. For example, inconsistent date 
                            formats can break time-based analyses, while unrecognized string patterns might indicate data entry issues.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• Before performing time-series analysis")
                            st.write("• When joining datasets to ensure relationship integrity")
                            st.write("• To standardize text data for processing")
                            st.write("• To identify data entry errors or inconsistencies")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You're analyzing patient records and the "Admission Date" column shows 50 invalid dates. Examining these reveals 
                            that they're in MM/DD/YYYY format instead of the expected YYYY-MM-DD format. Some dates like "02/04/2023" were 
                            interpreted correctly, but others like "13/04/2023" failed because there is no 13th month. This insight helps you create 
                            a proper date parsing function to standardize all dates.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Date Format Validation")

                        # Sample data for date validation
                        date_data = pd.DataFrame({
                            'Column': ['Admission Date', 'Discharge Date', 'Follow-up Date'],
                            'Invalid Count': [50, 35, 22],
                            'Total Count': [1000, 1000, 1000],
                            'Expected Format': ['YYYY-MM-DD', 'YYYY-MM-DD', 'YYYY-MM-DD']
                        })

                        # Calculate percentage
                        date_data['Invalid %'] = (date_data['Invalid Count'] / date_data['Total Count'] * 100).round(1)

                        # Display the dataframe
                        st.dataframe(date_data, hide_index=True)

                        # Show example problematic dates
                        st.markdown("#### Example Invalid Dates")
                        example_dates = pd.DataFrame({
                            'Raw Value': ['13/04/2023', '02/31/2023', '2023/13/01', '04-15-22'],
                            'Issue': [
                                'Invalid month (13)',
                                'Invalid day (31) for month (02)',
                                'Invalid month (13)',
                                'Ambiguous format (MM-DD-YY or DD-MM-YY?)'
                            ]
                        })
                        st.dataframe(example_dates, hide_index=True)

                        # String Pattern Analysis
                        st.markdown("#### String Pattern Analysis Example")

                        pattern_data = pd.DataFrame({
                            'Column': ['Product Code', 'Customer ID', 'Email', 'Phone Number'],
                            'Min Length': [8, 10, 12, 10],
                            'Max Length': [12, 10, 35, 14],
                            'Avg Length': [10.2, 10.0, 24.3, 12.5],
                            'Has Numbers': ['Yes', 'Yes', 'No', 'Yes'],
                            'Has Special Chars': ['Yes', 'No', 'Yes', 'Yes']
                        })

                        st.dataframe(pattern_data, hide_index=True)

                    # Data Issues Analysis
                    elif guide_topic == "Data Issues Analysis":
                        st.header("Data Issues Analysis")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            The Data Issues Analysis identifies problematic rows in your dataset that contain null or invalid values in key columns.
                            It pinpoints exactly which records have issues and provides examples to help you understand the scope of the problem.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write("• **Invalid Rows Summary:** Count and percentage of problematic rows by column")
                            st.write("• **Visual Representation:** Charts showing the distribution of invalid data")
                            st.write("• **Detailed Examples:** Sample records with issues for inspection")
                            st.write("• **Primary Key Reference:** Identifiers for problematic rows")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Invalid data can cause significant problems in analysis and operations. For example, missing customer 
                            information might prevent order fulfillment, while invalid product codes could lead to incorrect sales analytics.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• During data cleaning to identify records needing correction")
                            st.write("• When data quality issues are suspected")
                            st.write("• Before critical analyses where data completeness is essential")
                            st.write("• After data integration to check for mapping problems")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You're preparing a customer dataset for a marketing campaign and find that 200 rows have missing email addresses, 
                            while 150 have missing postal codes. The examples show these are primarily customers who signed up through a specific 
                            channel where these fields weren't required. You decide to exclude these records from email campaigns but include them 
                            in general analytics.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Invalid Data Analysis")

                        # Sample data for invalid rows visualization
                        issues_data = pd.DataFrame({
                            'Column': ['Email', 'Postal Code', 'Phone Number', 'Birth Date', 'Customer ID'],
                            'Invalid Rows': [200, 150, 120, 85, 0],
                            'Total Rows': [1000, 1000, 1000, 1000, 1000]
                        })

                        # Calculate percentage
                        issues_data['Percentage'] = (
                                issues_data['Invalid Rows'] / issues_data['Total Rows'] * 100).round(1)

                        # Display the dataframe
                        st.dataframe(issues_data, hide_index=True)

                        # Show the bar chart
                        st.bar_chart(issues_data.set_index('Column')['Percentage'])

                        # Example records with issues
                        st.markdown("#### Example Problematic Records")

                        example_issues = pd.DataFrame({
                            'Customer ID': ['C1001', 'C1042', 'C1078', 'C1095'],
                            'Issue Column': ['Email', 'Postal Code', 'Phone Number', 'Birth Date'],
                            'Issue Type': ['Missing', 'Invalid Format', 'Missing', 'Invalid Date'],
                            'Channel': ['Web', 'Store', 'Phone', 'Web']
                        })

                        st.dataframe(example_issues, hide_index=True)

                    # Correlation Analysis
                    elif guide_topic == "Correlation Analysis":
                        st.header("Correlation Analysis")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            Correlation Analysis examines the statistical relationships between numerical columns in your dataset.
                            It calculates correlation coefficients (ranging from -1 to 1) that indicate how strongly pairs of variables are related.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write(
                                "• **Correlation Matrix:** Heatmap showing relationships between all numerical columns")
                            st.write("• **Strong Correlations:** Highlighted pairs of highly correlated variables")
                            st.write(
                                "• **Correlation Visualization:** Scatter plots with trend lines for correlated pairs")
                            st.write(
                                "• **Outlier Detection:** Identification of points that don't follow the correlation pattern")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Understanding correlations helps identify redundant features, discover hidden relationships, and build better models.
                            For example, if income and spending are highly correlated (e.g., 0.92), you might only need one of these variables in your model.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• During feature selection for machine learning")
                            st.write("• To identify potential cause-and-effect relationships")
                            st.write("• When looking for redundant variables to simplify analysis")
                            st.write("• To discover hidden factors driving your data")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You're building a predictive model for house prices and discover a strong correlation (0.85) between square footage and price.
                            This confirms your hypothesis that size is a key driver of price. However, you also notice a surprising correlation (0.73) between 
                            ceiling height and price, which wasn't an obvious factor. This insight leads you to include ceiling height as a feature in your model,
                            improving its accuracy.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Correlation Matrix")

                        # Sample data for correlation matrix visualization
                        import numpy as np

                        # Create a correlation matrix
                        np.random.seed(42)  # For reproducibility
                        correlation_matrix = pd.DataFrame(np.random.rand(5, 5),
                                                          columns=['Price', 'Sq Footage', 'Ceiling Height', 'Age',
                                                                   'Bedrooms'],
                                                          index=['Price', 'Sq Footage', 'Ceiling Height', 'Age',
                                                                 'Bedrooms'])

                        # Set diagonal to 1.0
                        np.fill_diagonal(correlation_matrix.values, 1.0)

                        # Set some meaningful correlations
                        correlation_matrix.loc['Price', 'Sq Footage'] = 0.85
                        correlation_matrix.loc['Sq Footage', 'Price'] = 0.85

                        correlation_matrix.loc['Price', 'Ceiling Height'] = 0.73
                        correlation_matrix.loc['Ceiling Height', 'Price'] = 0.73

                        correlation_matrix.loc['Price', 'Age'] = -0.42
                        correlation_matrix.loc['Age', 'Price'] = -0.42

                        correlation_matrix.loc['Sq Footage', 'Bedrooms'] = 0.65
                        correlation_matrix.loc['Bedrooms', 'Sq Footage'] = 0.65

                        # Round to 2 decimal places
                        correlation_matrix = correlation_matrix.round(2)

                        # Display the correlation matrix
                        st.dataframe(correlation_matrix)

                        # Strong correlations
                        st.markdown("#### Strong Correlations")

                        strong_correlations = pd.DataFrame({
                            'Variable 1': ['Price', 'Price', 'Sq Footage'],
                            'Variable 2': ['Sq Footage', 'Ceiling Height', 'Bedrooms'],
                            'Correlation': [0.85, 0.73, 0.65]
                        })

                        st.dataframe(strong_correlations, hide_index=True)

                        # Scatterplot
                        st.markdown("#### Example Correlation Scatter Plot: Price vs. Sq Footage")

                        # Generate some sample data
                        import numpy as np
                        np.random.seed(42)

                        # Sample data points for price vs sq footage
                        sq_footage = np.random.normal(2000, 500, 50)
                        price = 100000 + 200 * sq_footage + np.random.normal(0, 50000, 50)

                        chart_data = pd.DataFrame({
                            'Sq Footage': sq_footage,
                            'Price': price
                        })

                        st.scatter_chart(chart_data, x='Sq Footage', y='Price')

                    # Distribution Analysis
                    elif guide_topic == "Distribution Analysis":
                        st.header("Distribution Analysis")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            Distribution Analysis examines how values are spread across numerical and categorical variables in your dataset.
                            It looks at statistics like mean, median, skewness, and the overall shape of distributions.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write(
                                "• **Distribution Characteristics:** Statistical summary of central tendency and spread")
                            st.write("• **Histograms & Box Plots:** Visual representation of value distributions")
                            st.write(
                                "• **Normality Assessment:** Q-Q plots to check if data follows a normal distribution")
                            st.write("• **Category Balance:** Analysis of imbalance in categorical variables")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            The shape of your data distributions affects what statistical methods are appropriate. For example,
                            skewed income data might require a log transformation before analysis, while imbalanced categories might
                            need resampling techniques for machine learning.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• Before applying statistical tests that assume normality")
                            st.write("• When deciding on data transformations")
                            st.write("• To detect outliers that may skew analysis")
                            st.write("• When preparing classification datasets to check for class imbalance")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You're analyzing employee salary data and find it's highly right-skewed (skewness = 2.3), with a few executives 
                            earning multiples of what average employees make. The mean salary is $75,000, but the median is only $52,000. 
                            This insight tells you to use median rather than mean for typical salary discussions, and to apply a log transformation 
                            before building any models to predict salary.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Salary Distribution")

                        # Sample data for distribution visualization - create a right-skewed distribution
                        import numpy as np

                        # Generate log-normal distribution for salaries
                        np.random.seed(42)
                        salaries = np.random.lognormal(mean=11, sigma=0.5, size=1000)

                        # Calculate statistics
                        mean_salary = np.mean(salaries)
                        median_salary = np.median(salaries)

                        # Create bins for histogram
                        bins = 20
                        hist_values, bin_edges = np.histogram(salaries, bins=bins)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                        # Create DataFrame for the histogram
                        hist_data = pd.DataFrame({
                            'Salary Range': bin_centers,
                            'Count': hist_values
                        })

                        # Display histogram
                        st.bar_chart(hist_data.set_index('Salary Range'))

                        # Display statistics
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Mean Salary", f"${mean_salary:.0f}")

                        with col2:
                            st.metric("Median Salary", f"${median_salary:.0f}")

                        with col3:
                            st.metric("Difference", f"${mean_salary - median_salary:.0f}",
                                      delta=f"{((mean_salary / median_salary) - 1) * 100:.1f}% higher mean")

                        st.markdown("""
                        **Insight:** The mean salary is significantly higher than the median, confirming right skew.
                        This indicates a small number of high earners pulling the mean upward.
                        """)

                        # Categorical Distribution Example
                        st.markdown("#### Categorical Distribution Example: Job Levels")

                        category_data = pd.DataFrame({
                            'Job Level': ['Entry Level', 'Associate', 'Senior', 'Manager', 'Director', 'Executive'],
                            'Count': [350, 280, 220, 100, 40, 10]
                        })

                        # Calculate percentage
                        total = category_data['Count'].sum()
                        category_data['Percentage'] = (category_data['Count'] / total * 100).round(1)

                        # Display the data
                        st.dataframe(category_data, hide_index=True)

                        # Show the chart
                        st.bar_chart(category_data.set_index('Job Level')['Count'])

                        # Show balance analysis
                        st.markdown("""
                        **Category Balance Analysis:** This distribution is moderately imbalanced, with Entry Level employees 
                        representing 35% of the workforce while Executives are only 1%. For machine learning models predicting 
                        job level, you might need to address this imbalance using techniques like oversampling or stratification.
                        """)

                    # Consistency & Quality Analysis
                    elif guide_topic == "Consistency & Quality Analysis":
                        st.header("Consistency & Quality Analysis")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            The Consistency & Quality Analysis checks for logical and structural integrity across your dataset.
                            It examines duplicates, logical relationships between columns, time-series patterns, and overall data completeness.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write(
                                "• **Duplicate Analysis:** Identification of exact duplicate rows and duplicate key values")
                            st.write("• **Logical Consistency:** Detection of inconsistencies between related columns")
                            st.write("• **Time Series Patterns:** Temporal distribution and trends in date columns")
                            st.write("• **Data Completeness:** Overall and column-level completeness metrics")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Logical inconsistencies indicate data quality issues that could lead to incorrect conclusions. For example,
                            an order with a delivery date before its order date is clearly erroneous and should be investigated.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• After data integration to ensure consistency")
                            st.write("• When validating data against business rules")
                            st.write("• To identify process issues in data collection")
                            st.write("• Before using the data for critical business decisions")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You're analyzing a patient treatment dataset and find 23 cases where the "Treatment End Date" is before the 
                            "Treatment Start Date." This logical inconsistency suggests data entry errors. You also discover 45 duplicate 
                            patient records with slightly different spellings of names. These insights lead you to implement data validation 
                            rules in your collection process and create a deduplication strategy.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Logical Inconsistencies")

                        # Duplicate Analysis
                        st.markdown("##### Duplicate Analysis")

                        duplicate_metrics = pd.DataFrame({
                            'Metric': ['Exact Duplicate Rows', 'Duplicate Patient IDs', 'Duplicate Email Addresses'],
                            'Count': [12, 45, 78],
                            'Percentage': [1.2, 4.5, 7.8]
                        })

                        st.dataframe(duplicate_metrics, hide_index=True)

                        # Logical Consistency Issues
                        st.markdown("##### Logical Consistency Issues")

                        consistency_issues = pd.DataFrame({
                            'Issue Type': [
                                'Treatment End Date before Start Date',
                                'Discharge Date before Admission Date',
                                'Follow-up Date before Treatment',
                                'Age inconsistent with Birth Date'
                            ],
                            'Count': [23, 15, 8, 42],
                            'Percentage': [2.3, 1.5, 0.8, 4.2]
                        })

                        st.dataframe(consistency_issues, hide_index=True)

                        # Example inconsistent records
                        st.markdown("##### Example Inconsistent Records")

                        example_inconsistencies = pd.DataFrame({
                            'Patient ID': ['P1001', 'P1042', 'P2053'],
                            'Start Date': ['2023-03-15', '2023-05-20', '2023-04-10'],
                            'End Date': ['2023-02-28', '2023-04-15', '2023-01-20'],
                            'Issue': ['End before Start', 'End before Start', 'End before Start']
                        })

                        st.dataframe(example_inconsistencies, hide_index=True)

                        # Data Completeness
                        st.markdown("##### Data Completeness Overview")

                        # Create a simple gauge for completeness
                        completeness = 84.5  # Example completeness percentage

                        st.markdown(f"### Overall Data Completeness: {completeness}%")
                        st.progress(completeness / 100)

                        completeness_data = pd.DataFrame({
                            'Metric': ['Filled Cells', 'Null Values', 'Blank Values'],
                            'Count': [42250, 6500, 1250],
                            'Percentage': [84.5, 13.0, 2.5]
                        })

                        st.dataframe(completeness_data, hide_index=True)

                    # File Conversion
                    elif guide_topic == "File Conversion":
                        st.header("File Conversion")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            The File Conversion Tool allows you to transform your dataset into different file formats (CSV, Excel, JSON, Parquet),
                            with options to filter the data or select specific columns before conversion.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write("• **Data Filtering:** Options to subset data before conversion")
                            st.write("• **Column Selection:** Ability to include only specific columns")
                            st.write("• **Format Options:** Format-specific settings like separators or compression")
                            st.write("• **Download Links:** Easy access to converted files")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Different file formats serve different purposes in the data ecosystem. For example, CSV files are widely 
                            compatible but inefficient, while Parquet provides excellent compression and performance for big data systems.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• When sharing data with others who use different tools")
                            st.write("• To optimize storage space with compressed formats")
                            st.write("• When transferring data between systems")
                            st.write("• To create subsets of data for specific analyses")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You've completed your data analysis and need to share results with multiple stakeholders. The marketing team 
                            needs an Excel file they can open in Microsoft Office, while the data science team prefers a Parquet file for 
                            efficient processing in their big data pipeline. You use the File Conversion tool to create both formats, selecting 
                            only the columns each team needs, thus providing optimized data for each use case.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: File Conversion Process")

                        # Create a simple workflow diagram using columns
                        st.markdown("##### File Conversion Workflow")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown("### 1. Filter Data")
                            st.markdown("""
                            • Select rows based on criteria
                            • Apply complex filters
                            • Preview filtered results
                            """)

                        with col2:
                            st.markdown("### 2. Select Columns")
                            st.markdown("""
                            • Choose relevant columns
                            • Reorder if needed
                            • Exclude sensitive data
                            """)

                        with col3:
                            st.markdown("### 3. Choose Format")
                            st.markdown("""
                            • CSV (universal)
                            • Excel (for MS Office)
                            • JSON (for web/API)
                            • Parquet (for big data)
                            """)

                        with col4:
                            st.markdown("### 4. Download")
                            st.markdown("""
                            • Generate file
                            • Get download link
                            • Share with stakeholders
                            """)

                        # Format comparison table
                        st.markdown("##### Format Comparison")

                        format_comparison = pd.DataFrame({
                            'Format': ['CSV', 'Excel', 'JSON', 'Parquet'],
                            'Compatibility': ['Universal', 'MS Office', 'Web/API', 'Big Data'],
                            'Size Efficiency': ['Low', 'Medium', 'Low', 'High'],
                            'Human Readable': ['Yes', 'Yes', 'Yes', 'No'],
                            'Best For': ['Simple sharing', 'Reporting', 'Web applications', 'Analytics']
                        })

                        st.dataframe(format_comparison, hide_index=True)

                    # Report Generation
                    elif guide_topic == "Report Generation":
                        st.header("Report Generation")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            The Report Generator creates a comprehensive document containing all the selected analyses from your dataset. 
                            It compiles visualizations, statistics, and insights into a downloadable HTML or PDF format that can be shared with stakeholders.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write("• **Section Selection:** Options to include or exclude specific analyses")
                            st.write("• **Format Options:** Choice between HTML and PDF formats")
                            st.write("• **Visualization Integration:** Inclusion of charts and graphs in the report")
                            st.write("• **Preview Capability:** Ability to review before downloading")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            Comprehensive reports provide documentation of data quality and insights that can be shared with stakeholders
                            who may not have access to the Data Profiler tool.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• For documentation and compliance purposes")
                            st.write("• When presenting data quality assessments to management")
                            st.write("• To create a snapshot of data conditions at a specific point in time")
                            st.write("• For sharing insights with team members or clients")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You've completed a thorough analysis of your company's customer database as part of a data governance initiative.
                            Your findings include several quality issues that need attention, such as missing contact information and duplicate
                            records. You generate a comprehensive PDF report that includes visualizations of these issues, their impact, and
                            recommendations for remediation. This report is shared with the data governance committee and serves as the basis
                            for a data quality improvement project.
                            """)

                        # Visual example using Streamlit components
                        st.success("#### Visual Example: Report Structure")

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.markdown("##### Report Sections")
                            st.markdown("""
                            1. **Executive Summary**
                               - Quality Score
                               - Key Findings
                               - Recommendations

                            2. **Data Profile**
                               - Schema Details
                               - Column Types
                               - Cardinality Analysis

                            3. **Quality Issues**
                               - Missing Values
                               - Invalid Formats
                               - Inconsistencies

                            4. **Visualizations**
                               - Distribution Charts
                               - Issue Heatmaps
                               - Trend Analysis

                            5. **Recommendations**
                               - Data Cleaning Steps
                               - Process Improvements
                               - Validation Rules
                            """)

                        with col2:
                            st.markdown("##### Sample Visualization")

                            # Create a dashboard-like visualization
                            issues_summary = pd.DataFrame({
                                'Issue Type': ['Missing Values', 'Format Errors', 'Duplicates', 'Inconsistencies'],
                                'Count': [1250, 830, 420, 180]
                            })

                            # Create a simple vertical bar chart
                            st.bar_chart(issues_summary.set_index('Issue Type'))

                            # Show quality metrics
                            quality_metrics = pd.DataFrame({
                                'Metric': ['Overall Completeness', 'Format Compliance', 'Consistency Score',
                                           'Duplicate-Free'],
                                'Score (%)': [92.5, 87.3, 94.8, 97.2]
                            })

                            st.dataframe(quality_metrics, hide_index=True)
                            
                    # AI Assistant Guide
                    elif guide_topic == "AI Assistant":
                        st.header("AI Assistant")

                        with st.expander("What is it?", expanded=True):
                            st.write("""
                            The AI Assistant integrates ChatGPT/GPT-4 capabilities into Data Lens, allowing you to ask natural language 
                            questions about your dataset and receive AI-powered insights, analyses, and visualizations.
                            """)

                        with st.expander("Key components", expanded=True):
                            st.write("• **Natural Language Queries:** Ask questions about your data in plain English")
                            st.write("• **Data Context Awareness:** AI understands your dataset's structure and content")
                            st.write("• **Code Generation:** Creates Python code for complex analyses or visualizations")
                            st.write("• **Interactive Execution:** Option to run generated code within the application")
                            st.write("• **Conversation History:** Maintains context across multiple questions")

                        with st.expander("Why it matters", expanded=True):
                            st.write("""
                            AI assistance makes data analysis more accessible and efficient, especially for users who may not be 
                            experts in data science or programming. It can quickly identify patterns, suggest analyses, and generate 
                            visualizations that might otherwise require significant manual effort.
                            """)

                        with st.expander("When to use it", expanded=True):
                            st.write("• When exploring a new dataset to quickly understand its characteristics")
                            st.write("• To get recommendations for the most insightful analyses to perform")
                            st.write("• When you need custom visualizations or analyses tailored to your specific questions")
                            st.write("• To generate data cleaning or transformation code for specific issues")
                            st.write("• When looking for explanations of complex patterns or anomalies in your data")

                        with st.expander("Example scenario", expanded=True):
                            st.write("""
                            You're working with a customer transaction dataset and want to understand purchase patterns. Instead 
                            of manually creating various analyses, you ask the AI: "What are the key purchase patterns in this dataset 
                            and how do they vary by customer segment?" The AI analyzes your data and responds with insights about 
                            purchase frequency, average order value, and product preferences across different customer groups, 
                            complete with visualizations and suggested follow-up analyses.
                            """)

                        # Visual example
                        st.success("#### Visual Example: AI Assistant Interaction")

                        st.markdown("""
                        ##### Example Query
                        "Analyze the relationship between customer age and purchase amount, and suggest ways to increase sales to younger customers."
                        """)

                        st.markdown("""
                        ##### Example Response
                        
                        **Analysis of Age vs. Purchase Amount Relationship:**
                        
                        I've analyzed your customer purchase data and found a significant positive correlation (r=0.42) between customer age and purchase amount. The data shows:
                        
                        - Customers aged 18-25 spend an average of $45 per transaction
                        - Customers aged 26-35 spend an average of $78 per transaction
                        - Customers aged 36+ spend an average of $125 per transaction
                        
                        Here's a visualization of this relationship:
                        
                        ```python
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        # Create scatter plot with trend line
                        plt.figure(figsize=(10, 6))
                        sns.regplot(x='age', y='purchase_amount', data=df, scatter_kws={'alpha':0.5})
                        plt.title('Customer Age vs. Purchase Amount')
                        plt.xlabel('Age')
                        plt.ylabel('Purchase Amount ($)')
                        plt.show()
                        ```
                        
                        **Recommendations to Increase Sales to Younger Customers:**
                        
                        1. **Product Recommendations:** Analysis shows younger customers prefer categories X and Y but rarely purchase from category Z
                        
                        2. **Pricing Strategy:** Consider installment payment options for higher-value items
                        
                        3. **Marketing Channels:** Data indicates younger customers arrived primarily through social media
                        
                        4. **Loyalty Program:** Implement tiered rewards with earlier payoffs for younger segment
                        
                        Would you like me to generate a more detailed analysis of any of these recommendations?
                        """)

                        st.markdown("##### Other Example Questions")
                        example_questions = pd.DataFrame({
                            'Question Type': ['Data Quality', 'Pattern Analysis', 'Visualization', 'Recommendations', 'Code Generation'],
                            'Example': [
                                'What are the most critical data quality issues in this dataset?',
                                'What unusual patterns or anomalies exist in the customer purchase history?',
                                'Create a visualization showing the relationship between X and Y',
                                'What actionable insights can I derive from this sales data?',
                                'Generate a Python function to clean the date formatting issues in this dataset'
                            ]
                        })
                        
                        st.dataframe(example_questions, hide_index=True)

                    # Add print/download option for this guide
                    st.divider()
                    st.info(
                        "📝 **Need a reference copy?** You can print this page or save it as PDF using your browser's print function (Ctrl+P or Cmd+P).")

                # Tab 12: AI Assistant
                with tab12:
                    ai_assistant_tab(tab12, df, analysis, profiler, uploaded_file)
        else:
            st.error("Could not load the file. Please check the file format and try again.")
    else:
        # Show welcome message when no file is uploaded yet
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>Welcome to the Enhanced Data Profiler</h2>
            <p>Upload a data file to begin your analysis.</p>
            <p>Supported formats: CSV, Excel, JSON, Parquet</p>
            <br>
            <img src="https://discover.nyc.gov.sg/omw/-/media/Project/NYC-Corporate/OMW/JobRoles/Images/DATA-ANALYST.png">
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()							
