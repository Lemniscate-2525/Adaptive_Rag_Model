"""Build the adaptive RAG workflow."""

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

from src.config.settings import Config
from src.llms.openai import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.rag.retriever_setup import similarity_search
from src.tools.graph_tools import doc_tool, routing_tool

config = Config()


def query_classifier(state: State):
    """Classify the question first; do not retrieve documents in this node."""
    question = state["messages"][-1].content
    classifier = llm.with_structured_output(RouteIdentifier)
    prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question"],
    )
    result = (prompt | classifier).invoke({"question": question})
    return {
        "route": result.route,
        "original_question": question,
        "latest_query": question,
        "retrieval_attempts": 0,
    }


def general_llm(state: State):
    """Answer conversational and common-knowledge questions directly."""
    result = llm.invoke(state["messages"])
    return {"messages": [result], "context": result.content}


def retriever_node(state: State):
    """Retrieve the top matching chunks from the current user's persisted index."""
    user_id = state.get("user_id") or ""
    query = state.get("latest_query") or state.get("original_question") or ""
    docs = similarity_search(user_id, query, k=4)
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'uploaded document')} | "
        f"Chunk: {doc.metadata.get('chunk_index', 'n/a')}]\n{doc.page_content}"
        for doc in docs
    )
    if not context:
        context = ""
    attempt = state.get("retrieval_attempts", 0) + 1
    return {
        "messages": [AIMessage(content=context or "No matching uploaded-document chunks found.")],
        "context": context,
        "retrieval_attempts": attempt,
    }


def grade(state: State):
    """Check whether the retrieved evidence can answer the original question."""
    context = state.get("context") or ""
    if not context.strip():
        return {"binary_score": "no"}

    grader = llm.with_structured_output(Grade)
    prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"],
    )
    result = (prompt | grader).invoke({
        "question": state.get("original_question", ""),
        "context": context,
    })
    score = result.binary_score.strip().lower()
    return {"binary_score": "yes" if score == "yes" else "no"}


def rewrite_query(state: State):
    """Rewrite only the retrieval query; retain the user's original question."""
    prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"],
    )
    result = (prompt | llm).invoke({"query": state.get("latest_query", "")})
    rewritten = result.content.strip()
    return {"latest_query": rewritten or state.get("latest_query", "")}


def generate(state: State):
    """Answer the original question using the retrieved or searched evidence."""
    context = state.get("context") or ""
    question = state.get("original_question") or state.get("latest_query") or ""
    prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["question", "context"],
    )
    result = (prompt | llm).invoke({"question": question, "context": context})
    return {"messages": [AIMessage(content=result.content)]}


def web_search(state: State):
    """Use Tavily after search routing or when document retrieval is exhausted."""
    question = state.get("original_question") or state.get("latest_query") or ""
    try:
        results = TavilySearchResults().invoke(state.get("latest_query") or question)
        context = "\n\n".join(
            f"[Web source: {item.get('url', 'URL unavailable')}]\n{item.get('content', '')}"
            for item in results if item.get("content")
        )
    except Exception:
        context = ""
    if not context:
        context = "No usable web results were available."
    return {"context": context, "messages": [AIMessage(content=context)]}


graph = StateGraph(State)
graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "query_analysis")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_edge("retriever", "grade")
graph.add_conditional_edges("grade", doc_tool)
graph.add_edge("rewrite", "retriever")
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)
graph.add_edge("general_llm", END)

builder = graph.compile()
