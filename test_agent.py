import streamlit as st
import os
import google.generativeai as genai
import litellm
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
from PIL import Image
import pdfplumber
import io

# Enable debug
litellm._turn_on_debug()

# Set API key and configure
os.environ["GOOGLE_API_KEY"] = "AIzaSyApCfL9HdN1qb9CWr1WVyz3xk-4FA6Ygs4"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize models
model_text = LiteLLMModel(model_id="gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Setup prompt engineer agent
prompt_engineer = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    additional_authorized_imports=['pandas', 'numpy', 'sklearn', 'json', 'spacy', 'requests', 'beautifulsoup4', 'pdfplumber', 'transformers', 're', 'matplotlib', 'seaborn', 'pytesseract', 'joblib', 'PIL', 'textract'],
    model=model_text
)

def is_prompt_request(user_input):
    """Check if the input is a request to generate a prompt."""
    if not user_input:
        return False
    prompt_keywords = [
        "prompt", "generate", "create", "summarize", "explain", "describe",
        "write", "craft", "build", "produce", "task", "query"
    ]
    input_lower = user_input.lower()
    return any(keyword in input_lower for keyword in prompt_keywords)

def build_prompt_engineer(user_input):
    base_prompt = f"""
You are an expert prompt engineer with 20 years of experience, specializing in generating high-quality and effective prompts for large language models (LLMs). Your task is to create prompts that enable LLMs to perform specific tasks accurately, efficiently, and ethically, leveraging insights from user-uploaded content (e.g., images, PDFs) and internet-sourced prompt engineering resources. Additionally, you must continuously learn from user feedback to improve your prompt generation over time.

When generating a prompt for a given task, follow these steps:

1. **Understand the Task**:
   - Carefully analyze the task description, including any user-uploaded content (e.g., text extracted from images via OCR using `pytesseract` and `PIL`, or documents processed with `textract` or `pdfplumber`).
   - Identify the desired output, constraints, and specific requirements.
   - Use `spacy` for NLP analysis to extract entities, keywords, or context, and `transformers` for semantic embeddings if needed.

2. **Incorporate External Insights**:
   - Review internet-sourced prompt engineering best practices (e.g., from web pages using `requests` and `beautifulsoup4`, or PDFs with `pdfplumber`).
   - Identify techniques like Chain of Thought (CoT), Few-Shot Learning, or Recursive Self-Improvement Prompting (RSIP) relevant to the task.

3. **Consider LLM Capabilities**:
   - Reflect on the target LLM’s strengths (e.g., reasoning, creativity) and limitations (e.g., hallucinations).
   - Structure the prompt to leverage strengths and mitigate weaknesses.

4. **Select Advanced Techniques**:
   - Choose one or more advanced prompt engineering techniques based on the task’s characteristics and feedback history:
     - **Chain of Thought (CoT)**: For step-by-step reasoning in complex tasks.
     - **Few-Shot Learning**: Provide examples to guide the LLM.
     - **Zero-Shot Learning**: Define tasks clearly without examples.
     - **Iterative Refinement**: Test and tweak prompts for accuracy.
     - **Multi-Turn Conversations**: Maintain context over interactions.
     - **System-Level Instructions**: Set roles (e.g., “Act as a medical expert”).
     - **Tree-of-Thought (ToT)**: Explore multiple reasoning paths.
     - **Self-Consistency**: Generate multiple answers and select the most consistent.
     - **Meta Prompting**: Have the LLM generate a structured prompt first.
     - **ReAct**: Combine reasoning with actions like data retrieval.
     - **Contextual Priming**: Embed background information for context-aware answers.
     - **Recursive Self-Improvement Prompting (RSIP)**: Iteratively improve outputs.
     - **Context-Aware Decomposition (CAD)**: Break down complex problems with context.
     - **Controlled Hallucination for Ideation (CHI)**: Use hallucinations for creative tasks.
     - **Multi-Perspective Simulation (MPS)**: Simulate viewpoints for nuanced analysis.
     - **Calibrated Confidence Prompting (CCP)**: Incorporate confidence calibration to reduce misinformation.

5. **Craft the Prompt**:
   - Write a clear, specific, and context-rich prompt using the selected techniques.
   - Include examples if beneficial, system instructions for role-setting, and format specifications (e.g., JSON, markdown) if required.
   - Use `json` to structure outputs if needed.

6. **Ensure Ethical Outputs**:
   - Craft prompts that promote ethical, unbiased, and accurate responses.
   - Avoid harmful or misleading content, especially for sensitive topics.

7. **Review and Refine**:
   - Evaluate the prompt’s alignment with the task using `pandas` for data analysis or `matplotlib`/`seaborn` for visualizing technique effectiveness.
   - Refine using iterative techniques if necessary.

8. **Save and Document**:
   - Save the prompt using `joblib` or `json` for reuse.
   - Document the rationale for technique selection, referencing insights from processed content.

9. **Learn from User Feedback**:
   - After generating a prompt and receiving user feedback (explicit, e.g., ratings/comments, or implicit, e.g., prompt usage), analyze it to identify strengths and weaknesses:
     - For explicit feedback, categorize as positive (e.g., 4-5 stars) or negative (e.g., 1-2 stars).
     - For textual feedback, use `spacy` or `transformers` for sentiment analysis and keyword extraction (e.g., “too vague”).
   - Update your internal strategies or knowledge base:
     - Maintain success and total counts for each strategy or template in a `pandas` DataFrame.
     - Adjust weights using a Bayesian approach (e.g., beta distribution with alpha=1, beta=1 initially; increment alpha for success, beta for failure).
     - Prioritize strategies with higher success rates (alpha / (alpha + beta)).
   - Store feedback data using `json` or `joblib` for persistence.
   - For negative feedback, optionally use `requests`/`beautifulsoup4` to search for better techniques for similar tasks.
   - Visualize feedback trends with `matplotlib`/`seaborn` to identify recurring issues.
   - Ensure ethical use of feedback by addressing biases and maintaining fairness.

By following these steps, you will generate prompts that maximize LLM performance across diverse tasks while continuously improving based on user feedback and leveraging advanced techniques and processed content from user uploads and internet resources.
### User Request:
{user_input}

NOTE: YOU HAVE TO GIVE THE PROMPT IN THE TEXT FORMAT EXPECT THE PROMPT IF THE END USER ASK YOU ANY OTHER QUESTION YOU CAN DISPLAY"SORRY I CANNOT HELP YOU WITH THAT I CAN GAVE YOU A HIGH LEVEL PROMPT"

-While you are giving the output you have to follow these steps:
-Dont give the output on json or html or any other format you have to give it in text format 
-That should be in clear format
"""
    return base_prompt

def process_text(user_input):
    if not is_prompt_request(user_input):
        return "I cannot assist you with that"
    try:
        prompt = build_prompt_engineer(user_input)
        return prompt_engineer.run(prompt)
    except Exception as e:
        return "I cannot assist you with that"

def process_image(image_file, user_input=None):
    if user_input and not is_prompt_request(user_input):
        return "I cannot assist you with that"
    try:
        image = Image.open(image_file)
        vision_model = genai.GenerativeModel("gemini-2.0-flash")
        input_text = user_input if user_input else "Generate a prompt based on the content of this image"
        response = vision_model.generate_content([build_prompt_engineer(input_text), image])
        if not is_prompt_request(response.text):
            return "I cannot assist you with that"
        return response.text
    except Exception as e:
        return "I cannot assist you with that"

def process_file(uploaded_file, user_input=None):
    try:
        if uploaded_file.type in ["image/png", "image/jpeg"]:
            return process_image(uploaded_file, user_input)
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                extracted_text = "\n".join([page.extract_text() for page in pdf.pages])
            full_input = f"{user_input}\nExtracted from uploaded PDF:\n{extracted_text}" if user_input else extracted_text
            return process_text(full_input)
        else:
            return "Unsupported file type."
    except Exception as e:
        return "I cannot assist you with that"

# Streamlit app
st.title("Prompt Engineer Assistant")
st.write("Generate high-quality prompts for large language models by entering text or uploading images/documents. Inspired by Gemini AI and ChatGPT.")

# Tabs for input types
tab1, tab2 = st.tabs(["Text Input", "File Upload"])

with tab1:
    st.header("Text Input")
    user_input = st.text_area("Enter your query (e.g., Summarize a medical article)", key="text_input")
    if st.button("Generate Prompt", key="text_button"):
        if user_input:
            result = process_text(user_input)
            if result == "I cannot assist you with that":
                st.error("I cannot assist you with that")
            else:
                st.write("**Generated Prompt:**")
                st.write(result)
                st.download_button("Download Prompt", result, file_name="prompt.txt", mime="text/plain", key="text_download")
        else:
            st.warning("Please enter a query.")

with tab2:
    st.header("File Upload")
    uploaded_file = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"], key="file_uploader")
    file_user_input = st.text_area("Enter a query to describe what to do with the file (optional)", key="file_input")
    if st.button("Generate Prompt", key="file_button"):
        if uploaded_file:
            result = process_file(uploaded_file, file_user_input)
            if result == "I cannot assist you with that":
                st.error("I cannot assist you with that")
            elif result == "Unsupported file type.":
                st.error("Unsupported file type. Please upload a PNG, JPG, JPEG, or PDF.")
            else:
                st.write("**Generated Prompt:**")
                st.write(result)
                st.download_button("Download Prompt", result, file_name="prompt.txt", mime="text/plain", key="file_download")
        else:
            st.warning("Please upload a file.")