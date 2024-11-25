import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Financial Bot",
    page_icon="💵",
    layout="centered",
)

# App title
st.title("💵 Financial Bot 🤖")

# File upload section
uploaded_file = st.file_uploader("Upload your financial file", type=["csv", "xlsx", "txt", "pdf"])

if uploaded_file:
    # Process the file (placeholder logic for your model)
    st.success("File uploaded successfully and processed.")
    st.info("Your file has been processed. You can now ask finance-related questions below.")

    # Initialize session state for questions and answers
    if "questions" not in st.session_state:
        st.session_state.questions = []  # List to hold user questions
    if "answers" not in st.session_state:
        st.session_state.answers = []  # List to hold model-generated answers

    # Input for user query
    user_query = st.text_input("Ask your financial questions:")

    # When the user submits a question
    if st.button("Submit"):
        if user_query.strip():
            # Add question to session state
            st.session_state.questions.append(user_query)
            
            # Generate answer (placeholder logic; replace with your model's response)
            generated_answer = f"Model's response to: '{user_query}'"
            st.session_state.answers.append(generated_answer)
        else:
            st.warning("Please enter a question before submitting.")

    # Display all questions and their corresponding answers
    for i, question in enumerate(st.session_state.questions):
        st.write(f"**Q{i + 1}:** {question}")
        # Only show answers after the user has asked questions
        st.write(f"**A{i + 1}:** {st.session_state.answers[i]}")

else:
    st.info("Please upload a file to proceed.")
