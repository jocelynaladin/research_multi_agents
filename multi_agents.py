from dotenv import load_dotenv
import os
# from typing import TypedDict, List, Any
from workflow_state import WorkflowState

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

def research_agent(state: WorkflowState) -> WorkflowState:
    # Implement the logic for the research agent
    # For example, it could generate a research question based on the topic
    topic = state['topic']
    
    prompt = f"""You are an expert researcher. 
    Gather comprehensive, factual, and accurate information on the following topic: {topic}.

    Provide:
    - Key facts and statistics
    - Historical context and background
    - Current trends and developments
    - Notable experts and sources in the field
    - Important debates or controversies related to the topic
    - Important considerations or implications of the topic   
    """

    # Genarate research question based on the topic
    research = llm.invoke(prompt).content

    # Return update state
    return {"research": research}

def writer_agent(state: WorkflowState) -> WorkflowState:
    # Tranform research in engaging content
    research = state['research']
    topic = state['topic']

    prompt = f"""You are a professional content writer.
    Create an engaging, well-structured, and informative article about: {topic}

    Base on this research: 
    {research}

    Requirements:
    - Clear and engaging writing style
    - Proper structure with introduction, body, and conclusion
    - Accurate representation of the research information
    - Compelling narrative that captures the reader's interest
    - Use of examples, anecdotes, or case studies to illustrate key points
    - Proper citations or references to the research sources
    """

    draft = llm.invoke(prompt).content
    
    return {"draft": draft}

def critic_agent(state: WorkflowState) -> WorkflowState:
    """Reviews draft and provides constructive feedback"""

    draft = state['draft']
    research = state['research']
    revision_count = state.get('revision_count', 0)

    prompt = f"""You are a tough but fair editor.
    Review the following draft critically:

    {draft}

    Original research:

    {research}

   Evaluate:
    - Clarity and readability of the draft
    - Accuracy (matches the research?)
    - Structure and flow
    - Engagement and interest level
    
    Provide specific, actionable feedback.
    """

    critique = llm.invoke(prompt).content

    return {
        "critique": critique,
        "revision_count": revision_count + 1
    }

def reviser_agent(state: WorkflowState) -> WorkflowState:
    """Implements critic's feedback to improve draft"""

    draft = state['draft']
    critique = state['critique']
    research = state['research']
    revision_count = state['revision_count']

    prompt = f"""Improve this draft based on the critique:

    Current Draft:
    {draft}

    Critique to address:
    {critique}

    Research for reference:
    {research}

    Create an improved version that addresses all feedbacks.
    """

    improved_draft = llm.invoke(prompt).content

    return {"draft": improved_draft,
            "revision_count": revision_count + 1
    }

def should_continue(state: WorkflowState) -> str:
    """Decide whether to continue revising or end"""

    # MAximum number of revisions
    MAX_REVISIONS = 2

    revision_count = state.get('revision_count', 0)
    
    if revision_count >= MAX_REVISIONS:
        print(f"Reached maximum revisions ({MAX_REVISIONS}). Ending workflow.")
        return "end"
    else:
        # Continue improving the draft
        return "revise"

# print(research_agent({"topic": "climate change"}))

# Create the graph
workflow = StateGraph(WorkflowState)

# Add nodes (agents)
workflow.add_node("researcher", research_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("reviser", reviser_agent)

# Define the flow of the graph (edges)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "critic")

# Conditional edge: continue or end?
workflow.add_conditional_edges(
    "critic",
    should_continue,
    {"revise": "reviser", "end": END}
)

# loop back from reviser to critic for multiple rounds of revision
workflow.add_edge("reviser", "critic")

# Compile the graph
app = workflow.compile()

# Define initial state
initial_state = WorkflowState(
    topic="The future of Quantum Computing",
    research="",
    draft="",
    critique="",
    revision_count=0,
    final_output="",
    messages=[]
)

# Run the workflow
result = app.invoke(initial_state)

# Access the final output
final_article = result['draft']
print("Final Article:\n", final_article)