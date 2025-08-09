import streamlit as st
import yaml
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def parse_document(file) -> list:
    """Parse uploaded file and return list of requirement/design items as strings."""
    content = file.read().decode("utf-8")
    # Try JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # Flatten dict values into list of strings
            items = []
            def recurse(d):
                if isinstance(d, dict):
                    for v in d.values():
                        recurse(v)
                elif isinstance(d, list):
                    for x in d:
                        recurse(x)
                else:
                    items.append(str(d))
            recurse(data)
            return items
        elif isinstance(data, list):
            return [str(i) for i in data]
    except json.JSONDecodeError:
        pass

    # Try YAML
    try:
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            items = []
            def recurse(d):
                if isinstance(d, dict):
                    for v in d.values():
                        recurse(v)
                elif isinstance(d, list):
                    for x in d:
                        recurse(x)
                else:
                    items.append(str(d))
            recurse(data)
            return items
        elif isinstance(data, list):
            return [str(i) for i in data]
    except yaml.YAMLError:
        pass

    # Fallback: treat as plain text with one item per line or sentence
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if len(lines) > 1:
        return lines
    else:
        # Use sentence tokenization
        doc = nlp(content)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def semantic_match(reqs, design, threshold=0.3):
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = reqs + design
    X = vectorizer.fit_transform(corpus)

    req_vecs = X[:len(reqs)]
    design_vecs = X[len(reqs):]

    feedback = []
    for i, req_vec in enumerate(req_vecs):
        if design_vecs.shape[0] == 0:
            similarity_scores = []
        else:
            similarity_scores = cosine_similarity(req_vec, design_vecs)[0]
        max_sim = max(similarity_scores) if similarity_scores.size > 0 else 0  # Fix here

        if max_sim >= threshold:
            coverage = "Present"
            issue = ""
        else:
            coverage = "Missing"
            issue = "Requirement not found in design"

        feedback.append({
            "requirement": reqs[i],
            "coverage": coverage,
            "issue": issue
        })
    return feedback


def display_feedback(feedback):
    st.write("### Analysis Feedback")
    st.write("| Requirement | Design Coverage | Issue |")
    st.write("|-------------|-----------------|-------|")
    for item in feedback:
        st.write(f"| {item['requirement'][:40]} | {item['coverage']} | {item['issue']} |")

def main():
    st.title("Release Requirements vs Design Comparison Agent")
    st.write("Upload your release requirements and design documents to get feedback.")

    req_file = st.file_uploader("Upload Requirements File", type=['json', 'yaml', 'yml', 'md', 'txt'])
    design_file = st.file_uploader("Upload Design File", type=['json', 'yaml', 'yml', 'md', 'txt'])
    optional_text = st.text_area("Optional: Additional clarifications or comments")

    if st.button("Analyze"):

        if not req_file or not design_file:
            st.error("Please upload both requirements and design files.")
            return

        with st.spinner("Parsing documents..."):
            requirements = parse_document(req_file)
            design = parse_document(design_file)

            if optional_text.strip():
                # Simple inclusion of optional text into requirements for matching
                requirements.append(optional_text.strip())

            feedback = semantic_match(requirements, design)
        
        display_feedback(feedback)

if __name__ == "__main__":
    main()
