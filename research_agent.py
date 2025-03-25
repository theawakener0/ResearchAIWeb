# research_agent.py
"""
AI Research Agent

This module implements an advanced research agent that can dynamically retrieve and
synthesize information from arXiv, PubMed, DuckDuckGo, and Wikipedia based on query domains,
using Google's Gemini 2.0 model as the underlying LLM.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import datetime
import os

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import load_tools, Tool, initialize_agent, AgentType
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    An AI research agent that analyzes queries and retrieves information from
    appropriate sources (arXiv, PubMed, DuckDuckGo, Wikipedia).
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the research agent with necessary tools and components.
        
        Args:
            api_key: API key for Google Generative AI
        """
        # Store API key
        self.api_key = api_key
        
        # Initialize LLM using Gemini 2.0
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-pro-exp-03-25",
            google_api_key=api_key,
            temperature=0.3,
            top_p=0.95,
            top_k=64,
            max_output_tokens=4096
        )
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Additional research tool using Gemini directly
        self.research_tool = Tool(
            name="Research AI Agent",
            func=self._generate_research_response,
            description="Uses AI to perform research on a given query."
        )
        
        # Set up LangChain agent
        self.agent = self._initialize_research_agent()
        
        # Domain keywords for tool selection
        self.domain_keywords = {
            'arxiv': [
                'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
                'neural network', 'computer science', 'cs', 'mathematics', 'math', 'physics',
                'quantum', 'algorithm', 'cybersecurity', 'information security', 'cryptography',
                'computer vision', 'nlp', 'natural language processing', 'reinforcement learning',
                'data science', 'robotics', 'statistics', 'optimization', 'computational',
                'transformer', 'gpt', 'llm', 'large language model'
            ],
            'pubmed': [
                'medicine', 'medical', 'health', 'healthcare', 'disease', 'clinical',
                'biology', 'biological', 'gene', 'genetic', 'genomic', 'cell', 'protein',
                'cancer', 'therapy', 'treatment', 'drug', 'pharmaceutical', 'diagnosis',
                'patient', 'hospital', 'doctor', 'physician', 'nurse', 'surgery',
                'virus', 'bacteria', 'infection', 'immune', 'immunology', 'vaccine',
                'neurology', 'brain', 'neuroscience', 'psychiatry', 'psychology'
            ],
            'wikipedia': [
                'history', 'definition', 'concept', 'theory', 'principle', 'law',
                'overview', 'background', 'introduction', 'summary', 'explanation',
                'person', 'biography', 'country', 'location', 'place', 'event',
                'organization', 'institution', 'fundamental', 'basic', 'general knowledge'
            ]
        }
        
        # Initialize chains for analyzing queries and synthesizing results
        self._setup_chains()
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """
        Initialize all required search tools.
        
        Returns:
            Dictionary of initialized tools
        """
        tools = {}
        
        # Initialize arXiv tool
        arxiv_tools = load_tools(["arxiv"])
        tools['arxiv'] = arxiv_tools[0]
        
        # Initialize PubMed tool
        tools['pubmed'] = PubmedQueryRun()
        
        # Initialize DuckDuckGo tool
        tools['duckduckgo'] = DuckDuckGoSearchRun()
        
        # Initialize Wikipedia tool
        try:
            wiki_api_wrapper = WikipediaAPIWrapper()
            tools['wikipedia'] = Tool(
                name="Wikipedia Search",
                func=WikipediaQueryRun(api_wrapper=wiki_api_wrapper).run,
                description="Searches Wikipedia for academic and general knowledge queries."
            )
            logger.info("Wikipedia tool initialized successfully")
        except Exception as e:
            logger.error(f"Wikipedia tool initialization failed: {str(e)}")
            tools['wikipedia'] = None
        
        logger.info("Initialized all search tools successfully")
        return tools
    
    def _initialize_research_agent(self):
        """
        Initialize the LangChain agent with all available research tools.
        
        Returns:
            Initialized agent
        """
        available_tools = [self.research_tool]
        
        # Add database-specific tools
        for tool_name, tool in self.tools.items():
            if tool:  # Only add if the tool was initialized successfully
                available_tools.append(tool)
        
        try:
            agent = initialize_agent(
                tools=available_tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
            )
            logger.info("Research agent initialized successfully")
            return agent
        except Exception as e:
            logger.error(f"Agent initialization failed: {str(e)}")
            return None
    
    def _generate_research_response(self, query: str):
        """
        Generates a research response using Gemini 2.5.
        
        Args:
            query: The research query
            
        Returns:
            Generated research response
        """
        try:
            # Create a specialized version of the LLM for in-depth thinking
            thinking_llm = GoogleGenerativeAI(
                model="gemini-2.5-pro-exp-03-25",
                google_api_key=self.api_key,
                temperature=0.5,
                top_p=0.95,
                top_k=64,
                max_output_tokens=65536
            )
           
            response = thinking_llm.invoke(query)
            return response
        except Exception as e:
            logger.error(f"Error generating research response: {str(e)}")
            return f"Error generating research response: {str(e)}"
    
    def _setup_chains(self):
        """Set up LLM chains for query analysis and result synthesis."""
        
        # Chain for analyzing which tools to use
        tool_selection_template = """
        You are an expert research assistant. Analyze the following research query and determine 
        which information sources would be most appropriate to answer it effectively.
        
        Query: {query}
        
        Please select the most appropriate sources from:
        1. arXiv - For technical and scientific topics (AI, cybersecurity, physics, mathematics, computer science)
        2. PubMed - For biomedical and life sciences research
        3. DuckDuckGo - For general web information and up-to-date news
        4. Wikipedia - For general knowledge, concepts, histories, and foundational information
        
        Think about whether this query relates to:
        - Technical fields like AI, cybersecurity, physics, mathematics (use arXiv)
        - Biomedical or life sciences topics (use PubMed)
        - General background information, definitions, or concepts (use Wikipedia)
        - Recent information or general web content (use DuckDuckGo)
        
        Return a JSON object with the following format:
        {
            "arxiv": true/false,
            "pubmed": true/false,
            "duckduckgo": true/false,
            "wikipedia": true/false,
            "reasoning": "Your step-by-step reasoning"
        }
        
        Only JSON format:
        """
        
        self.tool_selection_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["query"],
                template=tool_selection_template
            )
        )
        
        # Chain for query refinement
        query_refinement_template = """
        You are an expert research assistant helping to refine search queries for specific academic databases.
        
        Original research query: {query}
        Target database: {database}
        
        For arXiv: Focus on technical terms, algorithms, methodologies, and specific scientific concepts.
        For PubMed: Focus on medical terminology, conditions, treatments, and biological mechanisms.
        For DuckDuckGo: Create a web-friendly search that will find recent and relevant information.
        For Wikipedia: Focus on key concepts, proper nouns, and foundational terminology.
        
        Please refine the original query to be more effective for searching in the target database.
        Return only the refined query text with no additional explanations:
        """
        
        self.query_refinement_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["query", "database"],
                template=query_refinement_template
            )
        )
        
        # Chain for synthesizing results
        synthesis_template = """
        You are an expert research analyst tasked with synthesizing information from multiple sources to answer a research query.
        
        Original query: {query}
        
        Information sources:
        {sources}
        
        Please synthesize this information into a comprehensive research summary that answers the original query.
        Your summary should:
        1. Begin with a concise overview of the key findings
        2. Include the most relevant insights from each source
        3. Identify patterns, trends, or contradictions across sources
        4. End with proper citations for each referenced source
        
        Format your response as follows:
        
        # Research Summary: [Brief Title]
        
        ## Overview
        [Concise summary of key findings and insights]
        
        ## Key Insights
        [Detailed analysis of the most important information]
        
        ## Trends and Patterns
        [Analysis of emerging trends, patterns, or contradictions]
        
        ## References
        [Properly formatted citations for all sources used]
        """
        
        self.synthesis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["query", "sources"],
                template=synthesis_template
            )
        )
        
        logger.info("Set up all LLM chains successfully")

    def _analyze_query(self, query: str) -> Dict[str, bool]:
        """
        Analyze the query to determine which tools to use.
        
        Args:
            query: The research query to analyze
            
        Returns:
            Dictionary indicating which tools to use
        """
        try:
            # Use LLM to analyze which tools are appropriate
            result = self.tool_selection_chain.run(query)
            
            # Extract JSON from result (in case LLM adds extra text)
            import json
            import re
            
            # Find JSON pattern in the result
            json_match = re.search(r'\{.*\}', result.replace('\n', ' '), re.DOTALL)
            if json_match:
                tool_selection = json.loads(json_match.group(0))
            else:
                # Fallback to heuristic approach if LLM doesn't return valid JSON
                tool_selection = self._heuristic_tool_selection(query)
                
            logger.info(f"Tool selection for query: {tool_selection}")
            return tool_selection
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            # Fallback to heuristic approach
            return self._heuristic_tool_selection(query)
    
    def _heuristic_tool_selection(self, query: str) -> Dict[str, bool]:
        """
        Fallback method to select tools based on keyword matching.
        
        Args:
            query: The research query
            
        Returns:
            Dictionary indicating which tools to use
        """
        query_lower = query.lower()
        
        # Check for domain-specific keywords
        use_arxiv = any(keyword in query_lower for keyword in self.domain_keywords['arxiv'])
        use_pubmed = any(keyword in query_lower for keyword in self.domain_keywords['pubmed'])
        use_wikipedia = any(keyword in query_lower for keyword in self.domain_keywords['wikipedia'])
        
        # Default to DuckDuckGo for general queries or if no specific domain is detected
        use_duckduckgo = (not (use_arxiv or use_pubmed or use_wikipedia) or 
                         'latest' in query_lower or 
                         'recent' in query_lower or
                         'news' in query_lower)
        
        return {
            'arxiv': use_arxiv,
            'pubmed': use_pubmed,
            'duckduckgo': use_duckduckgo,
            'wikipedia': use_wikipedia,
            'reasoning': "Heuristic selection based on keyword matching"
        }
    
    def _refine_query(self, query: str, database: str) -> str:
        """
        Refine the query for the specific database.
        
        Args:
            query: Original research query
            database: Target database name
            
        Returns:
            Refined query for the specific database
        """
        try:
            refined_query = self.query_refinement_chain.run(query=query, database=database)
            logger.info(f"Refined query for {database}: {refined_query}")
            return refined_query.strip()
        except Exception as e:
            logger.error(f"Error refining query for {database}: {e}")
            return query  # Return original query if refinement fails
    
    def _execute_search(self, query: str, tool_selections: Dict[str, bool]) -> Dict[str, List[Dict]]:
        """
        Execute searches using the selected tools.
        
        Args:
            query: The research query
            tool_selections: Dictionary indicating which tools to use
            
        Returns:
            Dictionary of search results for each tool
        """
        results = {}
        
        try:
            # Execute arXiv search
            if tool_selections.get('arxiv', False) and self.tools.get('arxiv'):
                refined_query = self._refine_query(query, 'arxiv')
                arxiv_results = self.tools['arxiv'].run(refined_query)
                results['arxiv'] = self._parse_arxiv_results(arxiv_results)
                
            # Execute PubMed search
            if tool_selections.get('pubmed', False) and self.tools.get('pubmed'):
                refined_query = self._refine_query(query, 'pubmed')
                pubmed_results = self.tools['pubmed'].run(refined_query)
                results['pubmed'] = self._parse_pubmed_results(pubmed_results)
                
            # Execute DuckDuckGo search
            if tool_selections.get('duckduckgo', False) and self.tools.get('duckduckgo'):
                refined_query = self._refine_query(query, 'duckduckgo')
                duckduckgo_results = self.tools['duckduckgo'].run(refined_query)
                results['duckduckgo'] = self._parse_duckduckgo_results(duckduckgo_results)
                
            # Execute Wikipedia search
            if tool_selections.get('wikipedia', False) and self.tools.get('wikipedia'):
                refined_query = self._refine_query(query, 'wikipedia')
                wikipedia_results = self.tools['wikipedia'].run(refined_query)
                results['wikipedia'] = self._parse_wikipedia_results(wikipedia_results)
                
            logger.info(f"Completed searches for all selected tools")
            return results
            
        except Exception as e:
            logger.error(f"Error executing searches: {e}")
            return results
    
    def _parse_arxiv_results(self, raw_results: str) -> List[Dict]:
        """
        Parse the raw arXiv results into a structured format.
        
        Args:
            raw_results: Raw results from arXiv tool
            
        Returns:
            List of structured paper information
        """
        try:
            papers = []
            # Extract paper blocks from the raw text
            paper_blocks = re.split(r'\d+\.\s', raw_results)
            
            for block in paper_blocks:
                if not block.strip():
                    continue
                    
                paper = {}
                
                # Extract title
                title_match = re.search(r'Title: (.*?)(?:\n|$)', block)
                if title_match:
                    paper['title'] = title_match.group(1).strip()
                    
                # Extract authors
                authors_match = re.search(r'Authors: (.*?)(?:\n|$)', block)
                if authors_match:
                    paper['authors'] = authors_match.group(1).strip()
                    
                # Extract abstract
                abstract_match = re.search(r'Abstract: (.*?)(?:\n|$)', block, re.DOTALL)
                if abstract_match:
                    paper['abstract'] = abstract_match.group(1).strip()
                    
                # Extract URL/link
                url_match = re.search(r'URL: (.*?)(?:\n|$)', block)
                if url_match:
                    paper['url'] = url_match.group(1).strip()
                    
                # Extract published date
                date_match = re.search(r'Published: (.*?)(?:\n|$)', block)
                if date_match:
                    paper['published'] = date_match.group(1).strip()
                
                if paper.get('title'):  # Only add if we have at least a title
                    papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"Error parsing arXiv results: {e}")
            return []
    
    def _parse_pubmed_results(self, raw_results: str) -> List[Dict]:
        """
        Parse the raw PubMed results into a structured format.
        
        Args:
            raw_results: Raw results from PubMed tool
            
        Returns:
            List of structured paper information
        """
        try:
            papers = []
            # Split by numbered entries
            paper_blocks = re.split(r'\d+\.\s', raw_results)
            
            for block in paper_blocks:
                if not block.strip():
                    continue
                    
                paper = {}
                
                # Extract title
                title_match = re.search(r'Title: (.*?)(?:\n|$)', block)
                if title_match:
                    paper['title'] = title_match.group(1).strip()
                    
                # Extract authors
                authors_match = re.search(r'Authors: (.*?)(?:\n|$)', block)
                if authors_match:
                    paper['authors'] = authors_match.group(1).strip()
                    
                # Extract abstract
                abstract_match = re.search(r'Abstract: (.*?)(?:(?:\n\w+:)|$)', block, re.DOTALL)
                if abstract_match:
                    paper['abstract'] = abstract_match.group(1).strip()
                    
                # Extract PMID
                pmid_match = re.search(r'PMID: (.*?)(?:\n|$)', block)
                if pmid_match:
                    paper['pmid'] = pmid_match.group(1).strip()
                    paper['url'] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
                    
                # Extract published date
                date_match = re.search(r'Published: (.*?)(?:\n|$)', block)
                if date_match:
                    paper['published'] = date_match.group(1).strip()
                
                if paper.get('title'):  # Only add if we have at least a title
                    papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"Error parsing PubMed results: {e}")
            return []
    
    def _parse_duckduckgo_results(self, raw_results: str) -> List[Dict]:
        """
        Parse the raw DuckDuckGo results into a structured format.
        
        Args:
            raw_results: Raw results from DuckDuckGo tool
            
        Returns:
            List of structured result information
        """
        try:
            results = []
            # Split by potential result boundaries
            result_blocks = raw_results.split('\n\n')
            
            for block in result_blocks:
                if not block.strip():
                    continue
                
                result = {}
                
                # Try to extract a title-like element (usually the first line)
                lines = block.split('\n')
                if lines:
                    result['title'] = lines[0].strip()
                    
                    # The rest is likely content
                    if len(lines) > 1:
                        result['content'] = '\n'.join(lines[1:]).strip()
                    
                # Try to extract a URL
                url_match = re.search(r'https?://[^\s]+', block)
                if url_match:
                    result['url'] = url_match.group(0)
                
                if result.get('title') or result.get('content'):
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo results: {e}")
            return []
    
    def _parse_wikipedia_results(self, raw_results: str) -> List[Dict]:
        """
        Parse the raw Wikipedia results into a structured format.
        
        Args:
            raw_results: Raw results from Wikipedia tool
            
        Returns:
            List of structured article information
        """
        try:
            articles = []
            # Split content by sections, if any
            sections = raw_results.split('\n\n')
            
            article = {
                'title': 'Wikipedia Information',
                'content': raw_results,
                'url': None  # Wikipedia URLs are not directly available from the tool output
            }
            
            # Try to extract title from the first line if possible
            if sections and sections[0].strip():
                first_line = sections[0].strip().split('\n')[0]
                # If the first line looks like a title (short, possibly with ":" or other patterns)
                if len(first_line) < 100 and not first_line.endswith('.'):
                    article['title'] = first_line
                    # Remove title from content to avoid duplication
                    article['content'] = raw_results.replace(first_line, '', 1).strip()
            
            articles.append(article)
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing Wikipedia results: {e}")
            return []
    
    def _synthesize_results(self, query: str, search_results: Dict[str, List[Dict]]) -> str:
        """
        Synthesize search results into a coherent summary.
        
        Args:
            query: The original research query
            search_results: Dictionary of search results from each tool
            
        Returns:
            Synthesized research summary
        """
        try:
            # Format sources for the synthesis prompt
            sources_text = ""
            
            if 'arxiv' in search_results:
                sources_text += "## arXiv Papers:\n"
                for i, paper in enumerate(search_results['arxiv'][:5], 1):  # Limit to top 5
                    sources_text += f"{i}. Title: {paper.get('title', 'N/A')}\n"
                    sources_text += f"   Authors: {paper.get('authors', 'N/A')}\n"
                    sources_text += f"   Abstract: {paper.get('abstract', 'N/A')}\n"
                    sources_text += f"   URL: {paper.get('url', 'N/A')}\n"
                    sources_text += f"   Published: {paper.get('published', 'N/A')}\n\n"
            
            if 'pubmed' in search_results:
                sources_text += "## PubMed Papers:\n"
                for i, paper in enumerate(search_results['pubmed'][:5], 1):  # Limit to top 5
                    sources_text += f"{i}. Title: {paper.get('title', 'N/A')}\n"
                    sources_text += f"   Authors: {paper.get('authors', 'N/A')}\n"
                    sources_text += f"   Abstract: {paper.get('abstract', 'N/A')}\n"
                    sources_text += f"   PMID: {paper.get('pmid', 'N/A')}\n"
                    sources_text += f"   URL: {paper.get('url', 'N/A')}\n\n"
            
            if 'duckduckgo' in search_results:
                sources_text += "## Web Results:\n"
                for i, result in enumerate(search_results['duckduckgo'][:5], 1):  # Limit to top 5
                    sources_text += f"{i}. Title/Heading: {result.get('title', 'N/A')}\n"
                    sources_text += f"   Content: {result.get('content', 'N/A')}\n"
                    sources_text += f"   URL: {result.get('url', 'N/A')}\n\n"
            
            if 'wikipedia' in search_results:
                sources_text += "## Wikipedia Information:\n"
                for i, article in enumerate(search_results['wikipedia'][:3], 1):  # Limit to top 3
                    sources_text += f"{i}. Title: {article.get('title', 'N/A')}\n"
                    sources_text += f"   Content: {article.get('content', 'N/A')}\n"
                    if article.get('url'):
                        sources_text += f"   URL: {article.get('url')}\n"
                    sources_text += "\n"
            
            # Use direct research if few sources are found
            if not sources_text or sources_text.count('##') < 2:
                # Use the direct research tool as a fallback
                logger.info("Few sources found, using direct research capabilities")
                direct_research = self._generate_research_response(
                    f"Thoroughly research and provide detailed information about: {query}"
                )
                sources_text += f"\n## Additional Research:\n{direct_research}\n\n"
            
            # Run synthesis chain
            synthesis = self.synthesis_chain.run(
                query=query,
                sources=sources_text
            )
            
            logger.info("Successfully synthesized search results")
            return synthesis
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return self._generate_research_response(query)  # Fallback to direct research
    
    def research(self, query: str) -> str:
        """
        Main method to execute the research process.
        
        Args:
            query: The research query to process
            
        Returns:
            Synthesized research summary
        """
        logger.info(f"Starting research for query: {query}")
        
        try:
            # Try using the LangChain agent first if it was initialized successfully
            if self.agent:
                try:
                    # Give a time limit for the agent to respond
                    import threading
                    import time
                    
                    agent_result = None
                    
                    def run_agent():
                        nonlocal agent_result
                        try:
                            agent_result = self.agent.run(
                                f"Research task: {query}. Provide a detailed and structured answer."
                            )
                        except Exception as e:
                            logger.error(f"Agent execution error: {e}")
                    
                    # Execute agent with a timeout
                    agent_thread = threading.Thread(target=run_agent)
                    agent_thread.start()
                    agent_thread.join(timeout=120)  # 120 seconds timeout
                    
                    if agent_result:
                        logger.info("Research completed using LangChain agent")
                        return agent_result
                    else:
                        logger.warning("Agent timed out or failed, falling back to direct approach")
                except Exception as e:
                    logger.error(f"Error using agent: {e}")
            
            # Fallback to direct approach if agent failed or wasn't initialized
            # Step 1: Analyze which tools to use for this query
            tool_selections = self._analyze_query(query)
            
            # Step 2: Execute searches with the selected tools
            search_results = self._execute_search(query, tool_selections)
            
            # Step 3: Synthesize the search results
            synthesis = self._synthesize_results(query, search_results)
            
            logger.info(f"Completed research process for query: {query}")
            return synthesis
        
        except Exception as e:
            logger.error(f"Critical error in research process: {e}")
            # Ultimate fallback: direct research
            return self._generate_research_response(
                f"Please thoroughly research and answer this question: {query}"
            )

# Example usage function
def run_example():
    """
    Example function demonstrating the usage of the ResearchAgent.
    """
    import os
    
    # Get API key from environment variable
    api_key = "AIzaSyDAi8T-btBComG9Cs5KrGhbswxNKZNBl7I"
    if not api_key:
        print("Please set the GOOGLE_API_KEY environment variable")
        return
    
    # Initialize agent
    agent = ResearchAgent(api_key=api_key)
    
    # Example query
    query = "What are the latest advancements in deep learning for cybersecurity applications?"
    
    # Run research
    result = agent.research(query)
    
    print("\n\nRESEARCH RESULTS:")
    print("=" * 80)
    print(result)
    print("=" * 80)

if __name__ == "__main__":
    run_example()
