"""
Prompt Generator with State Management - SOLUTION

This solution adds intelligent 2-state workflow to the basic chatbot:
- State 1 (GATHER): Collect requirements through conversation
- State 2 (GENERATE): Create custom prompt based on requirements
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Streamlit: Framework for building web apps
import streamlit as st

# ChatOpenAI: Connects to OpenAI's GPT models
from langchain_openai import ChatOpenAI

# Message types for conversation
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# LangGraph: For building AI workflows
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

# ✨ NEW: For structured output (requirements collection)
from pydantic import BaseModel


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Prompt Generator",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Prompt Generator with State Management")
st.caption("AI that gathers requirements and generates custom prompts")


# =============================================================================
# SESSION STATE
# =============================================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ✨ NEW: Store the 2-state workflow
if "prompt_generator" not in st.session_state:
    st.session_state.prompt_generator = None


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.subheader("🔑 API Keys")
    
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
        
        # ✨ NEW: Show workflow status
        st.divider()
        st.subheader("🔄 Workflow Status")
        if st.session_state.prompt_generator:
            st.info("✅ 2-State Workflow Active")
            st.caption("💬 GATHER → 🎯 GENERATE")
        
        if st.button("Change API Keys"):
            st.session_state.openai_key = ""
            st.session_state.llm = None
            st.session_state.prompt_generator = None  # ✨ NEW
            st.rerun()
    else:
        st.warning("⚠️ Not Connected")


# =============================================================================
# API KEY INPUT
# =============================================================================

if not st.session_state.openai_key:
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-..."
    )
    
    if st.button("Connect"):
        if api_key and api_key.startswith("sk-"):
            st.session_state.openai_key = api_key
            st.rerun()
        else:
            st.error("❌ Invalid API key format")
    
    st.stop()


# =============================================================================
# INITIALIZE AI
# =============================================================================

if not st.session_state.llm:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=st.session_state.openai_key
    )


# =============================================================================
# ✨ NEW: CREATE 2-STATE WORKFLOW
# =============================================================================

if st.session_state.llm and not st.session_state.prompt_generator:
    
    # Step 1: Define requirements structure
    class PromptInstructions(BaseModel):
        """Requirements for prompt generation"""
        objective: str
        variables: List[str]
        constraints: List[str]
        requirements: List[str]
    
    # Step 2: Create LLM with tool binding
    llm_with_tool = st.session_state.llm.bind_tools([PromptInstructions])
    
    # Step 3: Define conversation state
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    # Step 4: STATE 1 - Requirements Gathering Node
    def gather_requirements(state: State):
        """💬 GATHER: Ask questions to collect requirements"""
        system_prompt = """Help the user create a custom AI prompt through friendly conversation.

You need to understand:
1. Purpose: What do they want the AI to help with?
2. Information needed: What details will they provide each time?
3. Things to avoid: What should the AI NOT do?
4. Must include: What should the AI always do?

RULES:
- Ask ONE question at a time in plain language
- No technical terms like "variables" or "parameters"
- Be conversational and friendly

When you have all information, call the tool."""
        
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}
    
    # Step 5: Transition Node (adds tool message)
    def add_tool_message(state: State):
        """Add tool message for state transition"""
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content="Requirements collected! Generating prompt...",
                    tool_call_id=tool_call_id
                )
            ]
        }
    
    # Step 6: STATE 2 - Prompt Generation Node
    def generate_prompt(state: State):
        """🎯 GENERATE: Create custom prompt from requirements"""
        
        # Extract requirements from tool call
        tool_args = None
        post_tool_messages = []
        
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_args = msg.tool_calls[0]["args"]
            elif isinstance(msg, ToolMessage):
                continue
            elif tool_args:
                post_tool_messages.append(msg)
        
        # Create generation prompt
        if tool_args:
            requirements_text = f"""
            Objective: {tool_args.get('objective', 'Not specified')}
            Variables: {', '.join(tool_args.get('variables', []))}
            Constraints: {', '.join(tool_args.get('constraints', []))}
            Requirements: {', '.join(tool_args.get('requirements', []))}
            """
            
            system_msg = SystemMessage(content=f"""Create a prompt template based on:

{requirements_text}

Guidelines:
- Make it clear and specific
- Use {{variable_name}} format for variables
- Address all constraints and requirements
- Use professional prompt engineering techniques""")
            
            messages = [system_msg] + post_tool_messages
        else:
            messages = post_tool_messages
        
        response = st.session_state.llm.invoke(messages)
        return {"messages": [response]}
    
    # Step 7: Router (decides next state)
    def route_conversation(state: State) -> Literal["add_tool_message", "gather", "__end__"]:
        """Route to next state based on current message"""
        last_msg = state["messages"][-1]
        
        # If tool was called, transition to generation
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "add_tool_message"
        
        # If not human message, end
        elif not isinstance(last_msg, HumanMessage):
            return "__end__"
        
        # Continue gathering
        else:
            return "gather"
    
    # Step 8: Build workflow
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("gather", gather_requirements)
    workflow.add_node("add_tool_message", add_tool_message)
    workflow.add_node("generate", generate_prompt)
    
    # Add edges
    workflow.add_edge(START, "gather")
    workflow.add_conditional_edges(
        "gather",
        route_conversation,
        {
            "add_tool_message": "add_tool_message",
            "gather": "gather",
            "__end__": END
        }
    )
    workflow.add_edge("add_tool_message", "generate")
    workflow.add_edge("generate", END)
    
    # Compile
    st.session_state.prompt_generator = workflow.compile()


# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =============================================================================
# HANDLE USER INPUT
# =============================================================================

user_input = st.chat_input("Tell me what kind of prompt you need...")

if user_input:
    # Save and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # ✨ MODIFIED: Use 2-state workflow
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Build message history
            messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Run 2-state workflow
            result = st.session_state.prompt_generator.invoke({"messages": messages})
            response = result["messages"][-1].content
            
            # Display and save response
            st.write(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })