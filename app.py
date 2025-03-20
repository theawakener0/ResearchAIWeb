from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
from werkzeug.utils import secure_filename
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import markdown
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper  # Add this import
from langchain.memory import ConversationBufferMemory
import uuid
import json
from datetime import datetime
import logging
from research_agent import ResearchAgent

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Jinja2 global variables
app.jinja_env.globals.update(now=datetime.now)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.secret_key = 'research_ai_secret_key'  # Required for flash messages
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # Session lifetime in seconds (24 hours)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API key for Google Gemini
API_KEY = "AIzaSyDAi8T-btBComG9Cs5KrGhbswxNKZNBl7I"

# Initialize the advanced research agent
research_agent = ResearchAgent(api_key=API_KEY)

# Instead, we'll use LangChain's abstractions

def load_and_split_pdf_document(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text = splitter.split_documents(docs)
        return text

    except FileNotFoundError:
        return ""

def youtube_client(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    return docs

# Advanced research agent functions
def generate_research_response(query: str):
    """
    Generates a research response using LangChain's GoogleGenerativeAI wrapper
    """
    try:
        # Use LangChain's wrapper instead of direct Google API
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash-thinking-exp-01-21",
            api_key=API_KEY,
            temperature=0.5,
            top_p=0.95,
            top_k=64,
            max_output_tokens=65536
        )
        
        response = llm.invoke(query)
        return response
    except Exception as e:
        return f"Error generating research response: {str(e)}"



# Chat memory storage - using a dictionary to store conversation history
# Key is session ID, value is a list of messages
chat_sessions = {}

def run_research_agent(query: str):
    """
    Runs the advanced research agent with a given query, utilizing multiple sources.
    This uses the ResearchAgent class which can dynamically select appropriate sources
    (arXiv, PubMed, DuckDuckGo, Wikipedia) based on the query domain.
    """
    try:
        logger.info(f"Running advanced research agent for query: {query}")
        result = research_agent.research(query)
        return result
    except Exception as e:
        logger.error(f"Error running advanced research agent: {str(e)}")
        
        return generate_research_response(query)  # Fallback to direct API call

# Standard models for basic research
model_creative = GoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21", 
    api_key=API_KEY,
    temperature=0.7,
)

model_datadriven = GoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21", 
    api_key=API_KEY,
    temperature=0.2
)

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


# Enhanced research pipeline with multi-iteration and structured output
def enhanced_research_pipeline(query, iterations=3, research_mode="standard", model_selection="1"):
    """
    Performs deep research with multiple iterations to refine results
    
    Args:
        query: The research query
        iterations: Number of research iterations
        research_mode: 'standard' or 'advanced'
        model_selection: '1' for creative, '2' for data-driven
    
    Returns:
        Structured research output with executive summary
    """
    # Initialize research components
    accumulated_research = ""
    research_plan = ""
    
    # Step 1: Generate Research Plan
    planning_prompt = f"""
                You are an expert in research planning with a proven track record in designing comprehensive, actionable research strategies. Based on the research query below, create a detailed research plan that includes the following components:

                1. Research Objectives:  
                - Define clear, concise objectives that outline what the research aims to achieve.

                2. Key Research Questions:  
                - List 3-5 critical questions that need to be answered to address the research query.

                3. Structured Research Outline:  
                Develop an organized outline with the following sections:
                - Introduction:  
                    Provide context, background information, and the significance of the topic.
                - Key Areas to Explore:  
                    Identify major themes or aspects that require further investigation.
                - Methodology:  
                    Detail the approaches, methods, and tools (including any relevant databases or search engines) that will be used to gather and analyze data.
                - Expected Outcomes:  
                    Outline anticipated findings, potential implications, and how the results may impact the field.

                4. Search Strategy:  
                - Recommend specific search terms and keywords tailored to yield high-quality, relevant results across academic databases and general search engines.

                Your final output should be a structured and clearly formatted research plan that serves as a roadmap for investigating the following query:

                Research Query: {query}
    """

    
    # Select model based on user preference
    if model_selection == "1":
        planning_model = model_creative
    else:
        planning_model = model_datadriven
    
    # Generate research plan
    research_plan = planning_model.invoke(planning_prompt)
    accumulated_research += f"## Research Plan\n\n{research_plan}\n\n"
    
    # Step 2: Iterative Research Process
    for i in range(iterations):
        iteration_prompt = f"""
        RESEARCH ITERATION {i+1}/{iterations}
        
        PREVIOUS FINDINGS:
        {accumulated_research if i > 0 else "Initial research iteration"}
        
        RESEARCH PLAN:
        {research_plan}
        
        CURRENT QUERY:
        {query}
        
        Your task for this iteration:
        {
            "Explore broadly and generate creative insights and connections." if model_selection == "1" else 
            "Focus on factual accuracy, data-driven analysis, and critical evaluation of sources."
        }
        
        For this iteration, please:
        1. Focus on a different aspect of the research question
        2. Provide specific insights not covered in previous iterations
        3. Identify any gaps or contradictions in the research so far
        4. Add depth to the most promising areas identified
        
        Format your response as a structured section that can be integrated into the final research document.
        """
        
        # Use appropriate research method based on mode
        if research_mode == "advanced" and research_agent:
            iteration_result = run_research_agent(iteration_prompt)
        else:
            # Use standard model
            if model_selection == "1":
                iteration_result = model_creative.invoke(iteration_prompt)
            else:
                iteration_result = model_datadriven.invoke(iteration_prompt)
        
        # Add iteration results to accumulated research
        accumulated_research += f"## Research Iteration {i+1}\n\n{iteration_result}\n\n"
        
        # Reflection and refinement step
        if i < iterations - 1:
            reflection_prompt = f"""
            REFLECTION ON RESEARCH PROGRESS
            
            CURRENT ACCUMULATED RESEARCH:
            {accumulated_research}
            
            Please analyze the research conducted so far and provide:
            1. What are the most valuable insights discovered?
            2. What important questions remain unanswered?
            3. What should be the focus of the next research iteration?
            4. How should we refine our approach for the next iteration?
            
            Your reflection will guide the next research iteration.
            """
            
            # Use data-driven model for reflection regardless of user selection
            reflection = model_datadriven.invoke(reflection_prompt)
            accumulated_research += f"## Research Reflection\n\n{reflection}\n\n"
    
    # Step 3: Generate Executive Summary and Final Structure
    summary_prompt = f"""
    EXECUTIVE SUMMARY GENERATION
    
    FULL RESEARCH DOCUMENT:
    {accumulated_research}
    
    Please create a comprehensive executive summary of the research findings. Your summary should:
    1. Provide a concise overview of the key findings
    2. Highlight the most important insights
    3. Summarize the methodology used
    4. Suggest potential applications or next steps
    
    Format your response as a professional executive summary section to be placed at the beginning of the research document.
    """
    
    executive_summary = model_datadriven.invoke(summary_prompt)
    
    # Step 4: Final formatting and structure
    final_structure_prompt = f"""
    FINAL RESEARCH DOCUMENT STRUCTURING
    
    EXECUTIVE SUMMARY:
    {executive_summary}
    
    ACCUMULATED RESEARCH:
    {accumulated_research}
    
    Please create a well-structured final research document that integrates all the research findings. Your document should:
    1. Begin with the executive summary
    2. Have a clear introduction
    3. Organize the findings into logical sections with headings
    4. Include a conclusion section
    5. Add citations and references where appropriate
    
    Format your response as a complete, professional research document.
    """
    
    if model_selection == "1":
        final_document = model_creative.invoke(final_structure_prompt)
    else:
        final_document = model_datadriven.invoke(final_structure_prompt)
    
    return final_document


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
        
        # Get research mode and iterations
        research_mode = request.form.get('research_mode', 'standard')
        model_selection = request.form.get('model_selection', '1')
        iterations = int(request.form.get('iterations', '3'))
        
        # Generate response based on research mode and iterations
        if iterations > 1:
            # Use enhanced research pipeline for multi-iteration research
            response = enhanced_research_pipeline(
                prompt_text, 
                iterations=iterations,
                research_mode=research_mode,
                model_selection=model_selection
            )
        else:
            # Use standard research approach for single iteration
            if research_mode == 'advanced':
                # Use the advanced research agent
                response = run_research_agent(prompt_text)
            else:
                # Use standard model
                message = HumanMessage(prompt_text)
                if model_selection == '1':
                    response = model_creative.invoke(message.content)
                else:
                    response = model_datadriven.invoke(message.content)
        
        # Convert markdown to HTML
        html_result = markdown.markdown(response)
        
        # Create a unique session ID for this research session if not exists
        if 'chat_session_id' not in session:
            session['chat_session_id'] = str(uuid.uuid4())
        
        # Initialize chat memory and research metadata for this session
        session_id = session['chat_session_id']
        if session_id not in chat_sessions:
            # Calculate word count
            word_count = len(response.split())
            
            # Calculate sources count
            sources_count = 0
            if pdf_content:
                sources_count += 1
            if website_content:
                sources_count += 1
            if user_inputs["attachments"]["youtube"] != "No video provided":
                sources_count += 1
            
            # Store session start time
            start_time = datetime.now()
            
            chat_sessions[session_id] = {
                'messages': [],
                'context': {
                    'research_result': response,
                    'role': user_inputs["role"],
                    'topic': user_inputs["topic"],
                    'pdf': user_inputs["attachments"]["pdf"],
                    'youtube': user_inputs["attachments"]["youtube"],
                    'website': user_inputs["attachments"]["website"],
                    'start_time': start_time,
                    'word_count': word_count,
                    'sources_count': sources_count
                }
            }
        
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
                              research_mode=research_mode,
                              iterations=iterations,
                              pdf_content=pdf_content,
                              pdf_filename=pdf_filename,
                              website_content=website_content,
                              chat_session_id=session_id,
                              date=datetime.now().strftime('%Y-%m-%d'))
    
    # GET request - render index template
    return render_template('index.html')




@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the chat session ID
        session_id = request.json.get('session_id')
        user_message = request.json.get('message')
        
        if not session_id or not user_message or session_id not in chat_sessions:
            return jsonify({'error': 'Invalid session or message'}), 400
        
        # Get the chat context
        chat_context = chat_sessions[session_id]
        
        # Add user message to history
        chat_context['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Create system message with context
        research_context = f"""You are a helpful AI research assistant. You are discussing research on the topic: {chat_context['context']['topic']}.
        
The user has received a research report that contains the following information:
{chat_context['context']['research_result']}

The user may ask questions about this research or request additional information.

Sources used in the research:
- PDF: {chat_context['context']['pdf']}
- YouTube: {chat_context['context']['youtube']}
- Website: {chat_context['context']['website']}

Please provide helpful, accurate responses based on this research context."""
        
        # Create messages for the AI
        messages = [
            SystemMessage(content=research_context)
        ]
        
        # Add conversation history (last 10 messages to avoid token limits)
        for msg in chat_context['messages'][-10:]:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))
        
        # Generate AI response
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash-thinking-exp-01-21", 
            temperature=0.7,
            api_key=API_KEY
        )
        
        ai_response = llm.invoke(messages)
        
        # Add AI response to history
        chat_context['messages'].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Convert markdown to HTML for display
        html_response = markdown.markdown(ai_response)
        
        return jsonify({
            'response': html_response,
            'raw_response': ai_response,
            'history': chat_context['messages']
        })
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in chat_sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    return jsonify({
        'history': chat_sessions[session_id]['messages']
    })


if __name__ == '__main__':
    app.run(debug=True)
