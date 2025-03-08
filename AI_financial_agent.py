import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from openai import OpenAI
import json

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO
import sqlite3

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")

llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

from tavily import TavilyClient

tavily = TavilyClient(api_key=tavily)

from typing import TypedDict, List
from pydantic import BaseModel

class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


EXTRACT_FINANCIALS = """You are a highly skilled financial data extraction specialist. Your task is to meticulously gather and present comprehensive financial data for the specified company. Ensure the data is accurate, detailed, and encompasses all relevant financial metrics. Focus on providing raw financial data without analysis."""

ANALYZE_FINANCIAL_HEALTH = """You are a seasoned financial analyst with expertise in interpreting complex financial datasets. Analyze the provided financial data, extracting key insights, identifying trends, and providing a thorough analysis of the company's financial health and performance. Focus on identifying key ratios and explaining their significance."""

IDENTIFY_COMPETITORS = """You are a market research analyst tasked with identifying key competitors for performance benchmarking. Generate a concise list of up to three targeted search queries to gather relevant information about companies operating in the same industry and market segment. Focus on identifying direct competitors for detailed comparison."""

COMPARE_PERFORMANCE = """You are a comparative financial analyst. Perform a rigorous comparative analysis of the given company's financial performance against its identified competitors. Use the provided data to highlight key differences and similarities in financial metrics. **Explicitly name each competitor in the comparison and provide quantitative evidence for your claims.**"""

REVIEW_FINANCIAL_REPORT = """You are a senior financial review specialist. Provide a detailed critique of the provided financial comparison report. Evaluate the analysis for accuracy, completeness, and clarity. Offer specific suggestions for improvements, including additional analyses or data points that would enhance the report's value. Focus on constructive criticism to improve the overall quality of the report."""

GENERATE_FINANCIAL_REPORT = """You are a professional financial report writer. Synthesize the provided financial analysis, competitor research, comparative performance analysis, and feedback into a comprehensive and insightful financial report. Ensure the report is well-structured, clearly written, and provides a holistic view of the company's financial position and performance. Structure the report to be easily read by a non-financial specialist."""

ADDRESS_CRITIQUE_RESEARCH = """You are a research specialist tasked with addressing specific critiques and gaps identified in a financial report. Generate a focused list of up to three search queries to gather targeted information that directly addresses the feedback provided. Focus on finding data that remedies the critiques, and allows for report improvement."""

def extract_financials_node(state: AgentState):
    # Read the CSV file into a pandas DataFrame
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))

    # Convert the DataFrame to a string
    financial_data_str = df.to_string(index=False)

    # Combine the financial data string with the task
    combined_content = (
        f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    )

    messages = [
        SystemMessage(content=EXTRACT_FINANCIALS),
        HumanMessage(content=combined_content),
    ]

    response = model.invoke(messages)
    return {"financial_data": response.content}


def analyze_financial_health_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_FINANCIAL_HEALTH),
        HumanMessage(content=state["financial_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}


def identify_competitors_node(state: AgentState):
    content = state["content"] or []
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=IDENTIFY_COMPETITORS),
                HumanMessage(content=competitor),
            ]
        )
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}


def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPARE_PERFORMANCE.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def review_financial_report_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=REVIEW_FINANCIAL_REPORT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


def generate_financial_report_node(state: AgentState):
    messages = [
        SystemMessage(content=GENERATE_FINANCIAL_REPORT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}


def address_critique_research_node(state: AgentState):
    messages = [
        SystemMessage(content=ADDRESS_CRITIQUE_RESEARCH),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "generate_financial_report"

builder = StateGraph(AgentState)

builder.add_node("extract_financials", extract_financials_node)
builder.add_node("analyze_financial_health", analyze_financial_health_node)
builder.add_node("identify_competitors", identify_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("review_financial_report", review_financial_report_node)
builder.add_node("address_critique_research", address_critique_research_node)

builder.add_node("generate_financial_report", generate_financial_report_node)


builder.set_entry_point("extract_financials")


builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {
        END: END, 
        "generate_financial_report": "generate_financial_report",
        "review_financial_report": "review_financial_report"
    }
)

builder.add_edge("extract_financials", "analyze_financial_health")
builder.add_edge("analyze_financial_health", "identify_competitors")
builder.add_edge("identify_competitors", "compare_performance")
builder.add_edge("review_financial_report", "address_critique_research")
builder.add_edge("address_critique_research", "compare_performance")
builder.add_edge("compare_performance", "generate_financial_report")

# def read_csv_file(file_path):
#     with open(file_path, "r") as file:
#         print("Reading CSV file...")
#         return file.read()

# if __name__ == "__main__":
#     task = "Analyze my company's financial performance compared to competitors"
#     competitors = ["IBM", "Nvidia", "Google"]
#     csv_file_path = "financials.csv"  # Update with the actual path to your CSV file

#     if not os.path.exists(csv_file_path):
#         print(f"CSV file not found at {csv_file_path}")
#     else:
#         print("Starting the conversation...")
#         csv_data = read_csv_file(csv_file_path)

#         # Create connection with check_same_thread=False
#         conn = sqlite3.connect(":memory:", check_same_thread=False)
#         memory = SqliteSaver(conn)
        
#         # Compile graph with the memory checkpointer
#         graph = builder.compile(checkpointer=memory)

#         initial_state = {
#             "task": task,
#             "competitors": competitors,
#             "csv_file": csv_data,
#             "max_revisions": 2,
#             "revision_number": 1,
#             "content": []
#         }
#         thread = {"configurable": {"thread_id": "1"}}

#         try:
#             for s in graph.stream(initial_state, thread):
#                 print(s)
#         finally:
#             conn.close()

import streamlit as st
import sqlite3
from typing import List

def main():
    st.title("Automated Financial Performance Analysis")

    st.markdown(
        """
        <style>
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border: 2px solidrgb(101, 76, 175); /* Green border */
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button {
            background-color:rgb(97, 76, 175); /* Green background */
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            border-radius: 5px;
            border: none;
        }
        .stFileUploader>div>div>div>div {
            border: 2px dashedrgb(81, 76, 175);
            padding: 20px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    task = st.text_input(
        "Enter the task:",
        "Compare my company's financial performance",
    )
    competitors_input = st.text_area("Enter competitor names (one per line):")
    competitors: List[str] = [comp.strip() for comp in competitors_input.split("\n") if comp.strip()]

    max_revisions = st.number_input("Max Revisions", min_value=1, value=10)
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the company's financial data", type=["csv"]
    )

    if st.button("Analyze") and uploaded_file is not None:
        with st.spinner("Analyzing..."):
            csv_data = uploaded_file.getvalue().decode("utf-8")

            conn = sqlite3.connect(":memory:", check_same_thread=False)
            memory = SqliteSaver(conn)

            graph = builder.compile(checkpointer=memory)

            initial_state = {
                "task": task,
                "competitors": competitors,
                "csv_file": csv_data,
                "max_revisions": max_revisions,
                "revision_number": 1,
                "content": []
            }
            thread = {"configurable": {"thread_id": "1"}}

            try:
                results_container = st.container()
                for s in graph.stream(initial_state, thread):
                    with results_container:
                        st.write(s)
                    final_state = s

                if final_state and "report" in final_state:
                    st.subheader("Final Report")
                    st.write(final_state["report"])
            finally:
                conn.close()

if __name__ == "__main__":
    main()