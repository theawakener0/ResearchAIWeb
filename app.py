from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import markdown

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.secret_key = 'research_ai_secret_key'  # Required for flash messages

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_and_split_pdf_document(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text = splitter.split_documents(docs)
        return text

    except FileNotFoundError:
        return ""

def youtube_client (video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    return docs

model_creative = GoogleGenerativeAI(model="gemini-2.0-flash-exp", 
                            api_key="AIzaSyAVoY_PlX6yY0g7_jBq-Awb_G9VqqgzghY",
                            temperature=0.7)


model_datadriven = GoogleGenerativeAI(model="gemini-2.0-flash-exp", 
                            api_key="AIzaSyAVoY_PlX6yY0g7_jBq-Awb_G9VqqgzghY",
                            temperature=0.2)

message = [
    SystemMessage("""

Advanced Research Assistant AI

Mission Statement: Your primary objective is to augment and streamline the research endeavors of users by leveraging advanced artificial intelligence capabilities. You will facilitate the efficient acquisition, analysis, and synthesis of information from a multitude of sources, thereby enhancing the depth and breadth of research outcomes.

Core Responsibilities:

    Multimedia Content Processing:
        YouTube Videos: Extract and summarize key insights from educational and informational videos, providing concise overviews and relevant timestamps.
        PDF Documents: Parse and analyze academic papers, reports, and other PDF materials, summarizing essential points and extracting pertinent data.
        Textual Content: Interpret and synthesize information from articles, books, and other text-based sources, highlighting critical arguments and findings.
        Websites: Navigate and assess online resources, distilling valuable information while evaluating credibility and relevance.

    Data Analysis and Interpretation:
        Employ advanced analytical tools to process quantitative and qualitative data, identifying patterns, correlations, and anomalies.
        Generate visual representations of data to facilitate understanding and support evidence-based conclusions.

    Information Synthesis and Reporting:
        Integrate findings from diverse sources to construct coherent, comprehensive reports that address specific research queries.
        Ensure that synthesized information is presented in a clear, structured manner, suitable for academic or professional use.

    Literature Review Assistance:
        Conduct thorough literature searches to identify seminal works and current developments in specified research areas.
        Summarize and compare findings from multiple studies, noting consensus, discrepancies, and gaps in the existing literature.

    Citation and Reference Management:
        Automatically generate accurate citations and bibliographies in various formatting styles (e.g., APA, MLA, Chicago).
        Assist in organizing and managing reference materials for ease of access and review.

    Ethical Research Practices:
        Adhere strictly to ethical guidelines in research, ensuring proper attribution of sources and avoidance of plagiarism.
        Promote transparency and reproducibility in all research activities.

    Continuous Learning and Adaptation:
        Stay updated with the latest advancements in AI and research methodologies to continually enhance support capabilities.
        Adapt to the evolving needs of researchers, offering personalized assistance tailored to individual project requirements.

Performance Metrics:

    Accuracy: Deliver precise and reliable information, minimizing errors and misinterpretations.
    Efficiency: Provide timely responses and expedite the research process without compromising quality.
    Relevance: Ensure that all information and analyses are pertinent to the user's specific research objectives.
    Innovation: Apply creative problem-solving techniques and offer novel insights that contribute to the advancement of research.
    User Satisfaction: Maintain a user-centric approach, fostering a collaborative and supportive research environment.

By embodying these principles and responsibilities, you will serve as an indispensable ally to researchers, empowering them to achieve greater efficacy and innovation in their scholarly pursuits."""),

]



def get_user_inputs():
    user_inputs = {}

    user_inputs["role"] = input("Enter your role or persona (e.g., AI Researcher, Data Analyst): ")
    user_inputs["research_task"] = input("Describe your specific research task: ")
    user_inputs["action"] = input("What action should be performed? (e.g., summarize, analyze, compare): ")
    user_inputs["topic"] = input("Enter the main topic or subject of your research: ")
    user_inputs["specific_aspects"] = input("List specific aspects or details to focus on: ")

    # Attachments
    user_inputs["attachments"] = {
        "pdf": input("Provide a link or file name for the PDF document (or leave blank): ") or "No PDF provided",
        "youtube": input("Provide a YouTube video URL (or leave blank): ") or "No video provided",
        "website": input("Provide a website URL (or leave blank): ") or "No website provided",
    }

    user_inputs["desired_qualities"] = input("Enter desired qualities (e.g., peer-reviewed, up-to-date, comprehensive): ")
    user_inputs["additional_requirements"] = input("Enter any additional requirements (e.g., citations, statistical analysis): ")
    user_inputs["preferred_format"] = input("Enter preferred format (e.g., bullet points, structured report, table): ")

    return user_inputs


def generate_prompt(user_inputs):
    prompt_template = f"""
As a {user_inputs["role"]}, I require assistance with {user_inputs["research_task"]}. Please {user_inputs["action"]} on {user_inputs["topic"]}, focusing on {user_inputs["specific_aspects"]}. The analysis should be thorough, well-structured, and backed by relevant data, ensuring that all key aspects are addressed comprehensively.  

Attachments:  

- **PDF Document:** {user_inputs["attachments"]["pdf"]}   
- **YouTube Video:** {user_inputs["attachments"]["youtube"]}  
- **Website:** {user_inputs["attachments"]["website"]}  

The provided materials should be carefully analyzed to extract meaningful insights, summarize key points, and present findings in a structured format. Ensure that the extracted information is {user_inputs["desired_qualities"]}, and that the response includes {user_inputs["additional_requirements"]}.  

Deliver the findings in {user_inputs["preferred_format"]} and include a breakdown of critical points, key arguments, and supporting evidence. If applicable, provide visual aids such as graphs, charts, or structured tables to enhance clarity.  

Additionally, cross-reference the provided sources to ensure consistency, highlight contradictions if any, and synthesize the most relevant information to present a well-rounded analysis. Where needed, suggest further readings or potential areas of deeper investigation to expand on the research.
"""
    return prompt_template


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        user_inputs = {
            "role": request.form.get('role', ''),
            "research_task": request.form.get('research_task', ''),
            "action": request.form.get('action', ''),
            "topic": request.form.get('topic', ''),
            "specific_aspects": request.form.get('specific_aspects', ''),
            "attachments": {
                "pdf": request.form.get('pdf', '') or "No PDF provided",
                "youtube": request.form.get('youtube', '') or "No video provided",
                "website": request.form.get('website', '') or "No website provided",
            },
            "desired_qualities": request.form.get('desired_qualities', ''),
            "additional_requirements": request.form.get('additional_requirements', ''),
            "preferred_format": request.form.get('preferred_format', '')
        }
        
        # Initialize variables for file handling
        pdf_content = None
        pdf_filename = None
        website_content = None
        
        # Handle PDF file upload
        if 'pdf_file' in request.files and request.files['pdf_file'].filename:
            pdf_file = request.files['pdf_file']
            if pdf_file.filename != '':
                filename = secure_filename(pdf_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                pdf_file.save(file_path)
                
                # Extract content from PDF
                try:
                    pdf_content = load_and_split_pdf_document(file_path)
                    pdf_filename = filename
                    user_inputs["attachments"]["pdf"] = f"Uploaded file: {filename}"
                except Exception as e:
                    flash(f"Error processing PDF: {str(e)}", "error")
        
        # Handle website scraping
        website_url = request.form.get('website', '')
        if website_url and website_url != "No website provided":
            try:
                website_content = scrape_website(website_url)
                if not website_content:
                    flash("Could not extract content from the provided website URL.", "warning")
            except Exception as e:
                flash(f"Error scraping website: {str(e)}", "error")
        
        # Generate prompt
        prompt_text = generate_prompt(user_inputs)
        
        # Get user prompt if provided
        user_prompt = request.form.get('user_prompt', '')
        if user_prompt:
            prompt_text += "\n\nAdditional instructions: " + user_prompt
        
        # Create message
        message = HumanMessage(prompt_text)
        
        # Get model selection
        model_selection = request.form.get('model_selection', '1')
        
        # Generate response based on model selection
        if model_selection == '1':
            response = model_creative.invoke(message.content)
        else:
            response = model_datadriven.invoke(message.content)
        
        # Convert markdown to HTML
        html_result = markdown.markdown(response)
        
        # Render results template with data
        return render_template('results.html', 
                              result=html_result,
                              role=user_inputs["role"],
                              topic=user_inputs["topic"],
                              action=user_inputs["action"],
                              preferred_format=user_inputs["preferred_format"],
                              pdf=user_inputs["attachments"]["pdf"],
                              youtube=user_inputs["attachments"]["youtube"],
                              website=user_inputs["attachments"]["website"],
                              model_selection=model_selection,
                              pdf_content=pdf_content,
                              pdf_filename=pdf_filename,
                              website_content=website_content)
    
    # GET request - render index template
    return render_template('index.html')


def scrape_website(url):
    try:
        import requests
        from bs4 import BeautifulSoup
        from langchain_core.documents import Document
        
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        # Get the text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Split the text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = [Document(page_content=chunk, metadata={"source": url}) for chunk in splitter.split_text(text)]
        
        return chunks
    except Exception as e:
        print(f"Error scraping website: {str(e)}")
        return None


if __name__ == '__main__':
    app.run(debug=True)
