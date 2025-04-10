import asyncio
import streamlit as st
from typing import Dict, Any, List
from agents import Agent, Runner, trace
from agents import set_default_openai_key
from firecrawl import FirecrawlApp
from agents.tool import function_tool

# Set page configuration
st.set_page_config(
    page_title="Gazai Deep Research Agent",
    page_icon="📘",
    layout="wide"
)

# Custom CSS for vertical alignment
st.markdown("""
    <style>
    [data-testid="column"] {
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stImage"] {
        margin-bottom: 0 !important;
    }
    [data-testid="stMarkdownContainer"] h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize API keys from secrets
try:
    openai_api_key = st.secrets["openai_api_key"]
    firecrawl_api_key = st.secrets["firecrawl_api_key"]
    set_default_openai_key(openai_api_key)
except KeyError:
    st.error("API keys not found in secrets. Please check your Streamlit settings.")
    st.stop()

# Main content with horizontally aligned header
col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.image("gazai.png", width=150)
with col2:
    st.markdown("# Deep Research Agent")
st.markdown("This Agent performs deep research on any topic")

# Research topic input
research_topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest developments in AI")

# Keep the original deep_research tool
@function_tool
async def deep_research(query: str, max_depth: int, time_limit: int, max_urls: int) -> Dict[str, Any]:
    """
    Perform comprehensive web research using Firecrawl's deep research endpoint.
    """
    try:
        # Initialize FirecrawlApp with the API key from secrets
        firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)

        # Define research parameters
        params = {
            "maxDepth": max_depth,
            "timeLimit": time_limit,
            "maxUrls": max_urls
        }

        # Set up a callback for real-time updates
        def on_activity(activity):
            st.write(f"[{activity['type']}] {activity['message']}")

        # Run deep research
        with st.spinner("Performing deep research..."):
            results = firecrawl_app.deep_research(
                query=query,
                params=params,
                on_activity=on_activity
            )

        return {
            "success": True,
            "final_analysis": results['data']['finalAnalysis'],
            "sources_count": len(results['data']['sources']),
            "sources": results['data']['sources']
        }
    except Exception as e:
        st.error(f"Deep research error: {str(e)}")
        return {"error": str(e), "success": False}

# Keep the original agents
research_agent = Agent(
    name="research_agent",
    instructions="""You are a research assistant that can perform deep web research on any topic.

    When given a research topic or question:
    1. Use the deep_research tool to gather comprehensive information
       - Always use these parameters:
         * max_depth: 3 (for moderate depth)
         * time_limit: 180 (3 minutes)
         * max_urls: 10 (sufficient sources)
    2. The tool will search the web, analyze multiple sources, and provide a synthesis
    3. Review the research results and organize them into a well-structured report
    4. Include proper citations for all sources
    5. Highlight key findings and insights
    """,
    tools=[deep_research]
)

elaboration_agent = Agent(
    name="elaboration_agent",
    instructions="""You are an expert content enhancer specializing in research elaboration.

    When given a research report:
    1. Analyze the structure and content of the report
    2. Enhance the report by:
       - Adding more detailed explanations of complex concepts
       - Including relevant examples, case studies, and real-world applications
       - Expanding on key points with additional context and nuance
       - Adding visual elements descriptions (charts, diagrams, infographics)
       - Incorporating latest trends and future predictions
       - Suggesting practical implications for different stakeholders
    3. Maintain academic rigor and factual accuracy
    4. Preserve the original structure while making it more comprehensive
    5. Ensure all additions are relevant and valuable to the topic
    """
)

async def run_research_process(topic: str):
    """Run the complete research process."""
    # Step 1: Initial Research
    with st.spinner("Conducting initial research..."):
        research_result = await Runner.run(research_agent, topic)
        initial_report = research_result.final_output

    # Display initial report in an expander
    with st.expander("View Initial Research Report"):
        st.markdown(initial_report)

    # Step 2: Enhance the report
    with st.spinner("Enhancing the report with additional information..."):
        elaboration_input = f"""
        RESEARCH TOPIC: {topic}

        INITIAL RESEARCH REPORT:
        {initial_report}

        Please enhance this research report with additional information, examples, case studies,
        and deeper insights while maintaining its academic rigor and factual accuracy.
        """

        elaboration_result = await Runner.run(elaboration_agent, elaboration_input)
        enhanced_report = elaboration_result.final_output

    return enhanced_report

# Main research process
if st.button("Start Research", disabled=not research_topic):
    if not research_topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            # Create placeholder for the final report
            report_placeholder = st.empty()

            # Run the research process
            enhanced_report = asyncio.run(run_research_process(research_topic))

            # Display the enhanced report
            report_placeholder.markdown("## Enhanced Research Report")
            report_placeholder.markdown(enhanced_report)

            # Add download button
            st.download_button(
                "Download Report",
                enhanced_report,
                file_name=f"{research_topic.replace(' ', '_')}_report.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by GAZAI.ai")