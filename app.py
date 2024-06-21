import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Load the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def vector_search(question, index="documents"):
    query_embedding = sentence_model.encode(question)
    
    response = es.search(
        index=index,
        body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding.tolist()}
                    }
                }
            },
            "size": 10
        }
    )
    
    retrieved_docs = [hit['_source']['text'] for hit in response['hits']['hits']]
    return retrieved_docs

def hybrid_search(query, index="documents"):
    query_embedding = sentence_model.encode(query)
    
    body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_embedding.tolist()}
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["text"]
                        }
                    }
                ]
            }
        },
        "size": 10
    }
    
    response = es.search(index=index, body=body)
    return response['hits']['hits']

# Initialize session state to store results
if 'vector_results' not in st.session_state:
    st.session_state.vector_results = None

if 'hybrid_results' not in st.session_state:
    st.session_state.hybrid_results = None

# Streamlit app
st.title("Elasticsearch Search App Comparison")

# Split the page into two columns
col1, col2 = st.columns(2)

# Left column for Vector Search
with col1:
    st.header("Vector Search")
    vector_query = st.text_input("Enter your search query for Vector Search", key="vector_query")
    if st.button("Vector Search", key="vector_search_button"):
        st.session_state.vector_results = vector_search(vector_query)
    
    if st.session_state.vector_results:
        st.write(f"Top 10 results for '{vector_query}' using Vector Search:")
        for i, result in enumerate(st.session_state.vector_results[:10]):
            st.write(f"{i+1}. {result}")
    else:
        st.write("No results found.")

# Right column for Hybrid Search
with col2:
    st.header("Hybrid Search")
    hybrid_query = st.text_input("Enter your search query for Hybrid Search", key="hybrid_query")
    if st.button("Hybrid Search", key="hybrid_search_button"):
        st.session_state.hybrid_results = hybrid_search(hybrid_query)
    
    if st.session_state.hybrid_results:
        st.write(f"Top 10 results for '{hybrid_query}' using Hybrid Search:")
        for i, result in enumerate(st.session_state.hybrid_results[:10]):
            st.write(f"{i+1}. {result['_source']['text']}")
    else:
        st.write("No results found.")
