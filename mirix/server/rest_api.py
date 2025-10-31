"""
FastAPI REST API server for Mirix.
This provides HTTP endpoints that wrap the SyncServer functionality,
allowing MirixClient instances to communicate with a cloud-hosted server.
"""

import copy
import json
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Header, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mirix.helpers.message_helpers import prepare_input_message_create
from mirix.llm_api.llm_client import LLMClient
from mirix.log import get_logger
from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.block import Block, BlockUpdate, CreateBlock, Human, Persona
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.environment_variables import (
    SandboxEnvironmentVariable,
    SandboxEnvironmentVariableCreate,
    SandboxEnvironmentVariableUpdate,
)
from mirix.schemas.file import FileMetadata
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.enums import MessageRole
from mirix.schemas.memory import ArchivalMemorySummary, Memory, RecallMemorySummary
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.mirix_response import MirixResponse
from mirix.schemas.organization import Organization
from mirix.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfig,
    SandboxConfigCreate,
    SandboxConfigUpdate,
)
from mirix.schemas.tool import Tool, ToolCreate, ToolUpdate
from mirix.schemas.tool_rule import BaseToolRule
from mirix.schemas.user import User
from mirix.server.server import SyncServer
from mirix.utils import convert_message_to_mirix_message

logger = get_logger(__name__)

# Import queue components
from mirix.queue import initialize_queue
from mirix.queue.manager import get_manager as get_queue_manager
from mirix.queue.queue_util import put_messages

# Initialize server (single instance shared across all requests)
_server: Optional[SyncServer] = None


def get_server() -> SyncServer:
    """Get or create the singleton SyncServer instance."""
    global _server
    if _server is None:
        logger.info("Creating SyncServer instance")
        _server = SyncServer()
    return _server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Mirix REST API server")
    
    # Initialize SyncServer (singleton)
    server = get_server()
    logger.info("SyncServer initialized")
    
    # Initialize queue with server reference
    initialize_queue(server)
    logger.info("Queue service started with SyncServer integration")
    
    yield  # Server runs here
    
    # Shutdown
    logger.info("Shutting down Mirix REST API server")
    
    # Cleanup queue
    queue_manager = get_queue_manager()
    queue_manager.cleanup()
    logger.info("Queue service stopped")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Mirix API",
    description="REST API for Mirix - Memory-augmented AI Agent System",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================


def get_user_and_org(
    x_user_id: Optional[str] = None,
    x_org_id: Optional[str] = None,
) -> tuple[str, str]:
    """
    Get user_id and org_id from headers or use defaults.
    
    Returns:
        tuple[str, str]: (user_id, org_id)
    """
    server = get_server()
    
    if x_user_id:
        user_id = x_user_id
        org_id = x_org_id or server.organization_manager.DEFAULT_ORG_ID
    else:
        user_id = server.user_manager.DEFAULT_USER_ID
        org_id = server.organization_manager.DEFAULT_ORG_ID
    
    return user_id, org_id


def extract_topics_from_messages(messages: List[Dict[str, Any]], llm_config: LLMConfig) -> Optional[str]:
    """
    Extract topics from a list of messages using LLM.

    Args:
        messages: List of message dictionaries (OpenAI format)
        llm_config: LLM configuration to use for topic extraction

    Returns:
        Extracted topics as a string (separated by ';') or None if extraction fails
    """
    try:

        if isinstance(messages, list) and "role" in messages[0].keys():
            # This means the input is in the format of [{"role": "user", "content": [{"type": "text", "text": "..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]

            # We need to convert the message to the format in "content"
            new_messages = []
            for msg in messages:
                new_messages.append({'type': "text", "text": "[USER]" if msg["role"] == "user" else "[ASSISTANT]"})
                new_messages.extend(msg["content"])
            messages = new_messages

        temporary_messages = convert_message_to_mirix_message(messages)
        temporary_messages = [prepare_input_message_create(msg, agent_id="topic_extraction", wrap_user_message=False, wrap_system_message=True) for msg in temporary_messages]

        # Add instruction message for topic extraction
        temporary_messages.append(
            prepare_input_message_create(
                MessageCreate(
                    role=MessageRole.user,
                    content='The above are the inputs from the user, please look at these content and extract the topic (brief description of what the user is focusing on) from these content. If there are multiple focuses in these content, then extract them all and put them into one string separated by ";". Call the function `update_topic` to update the topic with the extracted topics.',
                ),
                agent_id="topic_extraction",
                wrap_user_message=False,
                wrap_system_message=True,
            )
        )

        # Prepend system message
        temporary_messages = [
            prepare_input_message_create(
                MessageCreate(
                    role=MessageRole.system,
                    content="You are a helpful assistant that extracts the topic from the user's input.",
                ),
                agent_id="topic_extraction",
                wrap_user_message=False,
                wrap_system_message=True,
            ),
        ] + temporary_messages

        # Define the function for topic extraction
        functions = [
            {
                "name": "update_topic",
                "description": "Update the topic of the conversation/content. The topic will be used for retrieving relevant information from the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": 'The topic of the current conversation/content. If there are multiple topics then separate them with ";".',
                        }
                    },
                    "required": ["topic"],
                },
            }
        ]

        # Use LLMClient to extract topics
        llm_client = LLMClient.create(
            llm_config=llm_config,
            put_inner_thoughts_first=True,
        )

        if llm_client:
            response = llm_client.send_llm_request(
                messages=temporary_messages,
                tools=functions,
                stream=False,
                force_tool_call="update_topic",
            )
            
            # Extract topics from the response
            for choice in response.choices:
                if (
                    hasattr(choice.message, "tool_calls")
                    and choice.message.tool_calls is not None
                    and len(choice.message.tool_calls) > 0
                ):
                    try:
                        function_args = json.loads(
                            choice.message.tool_calls[0].function.arguments
                        )
                        topics = function_args.get("topic")
                        logger.debug("Extracted topics: %s", topics)
                        return topics
                    except (json.JSONDecodeError, KeyError) as parse_error:
                        logger.warning("Failed to parse topic extraction response: %s", parse_error)
                        continue

    except Exception as e:
        logger.error("Error in extracting topics from messages: %s", e)

    return None


# ============================================================================
# Error Handling
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
        },
    )


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mirix-api"}


# ============================================================================
# Agent Endpoints
# ============================================================================


@app.get("/agents", response_model=List[AgentState])
async def list_agents(
    query_text: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated
    limit: int = 100,
    cursor: Optional[str] = None,
    parent_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all agents for the authenticated user."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    tags_list = tags.split(",") if tags else None
    
    return server.agent_manager.list_agents(
        actor=user,
        tags=tags_list,
        query_text=query_text,
        limit=limit,
        cursor=cursor,
        parent_id=parent_id,
    )


class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""

    name: Optional[str] = None
    agent_type: Optional[AgentType] = AgentType.chat_agent
    embedding_config: Optional[EmbeddingConfig] = None
    llm_config: Optional[LLMConfig] = None
    memory: Optional[Memory] = None
    block_ids: Optional[List[str]] = None
    system: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    tool_rules: Optional[List[BaseToolRule]] = None
    include_base_tools: Optional[bool] = True
    include_meta_memory_tools: Optional[bool] = False
    metadata: Optional[Dict] = None
    description: Optional[str] = None
    initial_message_sequence: Optional[List[Message]] = None
    tags: Optional[List[str]] = None


@app.post("/agents", response_model=AgentState)
async def create_agent(
    request: CreateAgentRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a new agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Create memory blocks if provided
    if request.memory:
        for block in request.memory.get_blocks():
            server.block_manager.create_or_update_block(block, actor=user)
    
    # Prepare block IDs
    block_ids = request.block_ids or []
    if request.memory:
        block_ids.extend([b.id for b in request.memory.get_blocks()])
    
    # Create agent request
    create_params = {
        "description": request.description,
        "metadata_": request.metadata,
        "memory_blocks": [],
        "block_ids": block_ids,
        "tool_ids": request.tool_ids or [],
        "tool_rules": request.tool_rules,
        "include_base_tools": request.include_base_tools,
        "system": request.system,
        "agent_type": request.agent_type,
        "llm_config": request.llm_config,
        "embedding_config": request.embedding_config,
        "initial_message_sequence": request.initial_message_sequence,
        "tags": request.tags,
    }
    
    if request.name:
        create_params["name"] = request.name
    
    agent_state = server.create_agent(CreateAgent(**create_params), actor=user)
    
    return server.agent_manager.get_agent_by_id(agent_state.id, actor=user)


@app.get("/agents/{agent_id}", response_model=AgentState)
async def get_agent(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an agent by ID."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.agent_manager.get_agent_by_id(agent_id, actor=user)


@app.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete an agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    server.agent_manager.delete_agent(agent_id, actor=user)
    return {"status": "success", "message": f"Agent {agent_id} deleted"}


class UpdateAgentRequest(BaseModel):
    """Request model for updating an agent."""

    name: Optional[str] = None
    description: Optional[str] = None
    system: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    llm_config: Optional[LLMConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    message_ids: Optional[List[str]] = None
    memory: Optional[Memory] = None
    tags: Optional[List[str]] = None


@app.patch("/agents/{agent_id}", response_model=AgentState)
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Update an agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # TODO: Implement update_agent in server
    raise HTTPException(status_code=501, detail="Update agent not yet implemented")

# ============================================================================
# Memory Endpoints
# ============================================================================

@app.get("/agents/{agent_id}/memory", response_model=Memory)
async def get_agent_memory(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an agent's in-context memory."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.get_agent_memory(agent_id=agent_id, actor=user)


@app.get("/agents/{agent_id}/memory/archival", response_model=ArchivalMemorySummary)
async def get_archival_memory_summary(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get archival memory summary."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.get_archival_memory_summary(agent_id=agent_id, actor=user)


@app.get("/agents/{agent_id}/memory/recall", response_model=RecallMemorySummary)
async def get_recall_memory_summary(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get recall memory summary."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.get_recall_memory_summary(agent_id=agent_id, actor=user)


@app.get("/agents/{agent_id}/messages", response_model=List[Message])
async def get_agent_messages(
    agent_id: str,
    cursor: Optional[str] = None,
    limit: int = 1000,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get messages from an agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    return server.get_agent_recall_cursor(
        user_id=user_id,
        agent_id=agent_id,
        before=cursor,
        limit=limit,
        reverse=True,
    )


# ============================================================================
# Tool Endpoints
# ============================================================================


@app.get("/tools", response_model=List[Tool])
async def list_tools(
    cursor: Optional[str] = None,
    limit: int = 50,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all tools."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.tool_manager.list_tools(cursor=cursor, limit=limit, actor=user)


@app.get("/tools/{tool_id}", response_model=Tool)
async def get_tool(
    tool_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a tool by ID."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.tool_manager.get_tool_by_id(tool_id, actor=user)


@app.post("/tools", response_model=Tool)
async def create_tool(
    tool: Tool,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a new tool."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.tool_manager.create_tool(tool, actor=user)


@app.delete("/tools/{tool_id}")
async def delete_tool(
    tool_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete a tool."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    server.tool_manager.delete_tool_by_id(tool_id, actor=user)
    return {"status": "success", "message": f"Tool {tool_id} deleted"}


# ============================================================================
# Block Endpoints
# ============================================================================


@app.get("/blocks", response_model=List[Block])
async def list_blocks(
    label: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all blocks."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.block_manager.get_blocks(actor=user, label=label)


@app.get("/blocks/{block_id}", response_model=Block)
async def get_block(
    block_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a block by ID."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.block_manager.get_block_by_id(block_id, actor=user)


@app.post("/blocks", response_model=Block)
async def create_block(
    block: Block,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a block."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.block_manager.create_or_update_block(block, actor=user)


@app.delete("/blocks/{block_id}")
async def delete_block(
    block_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete a block."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    server.block_manager.delete_block(block_id, actor=user)
    return {"status": "success", "message": f"Block {block_id} deleted"}


# ============================================================================
# Configuration Endpoints
# ============================================================================


@app.get("/config/llm", response_model=List[LLMConfig])
async def list_llm_configs(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List available LLM configurations."""
    server = get_server()
    return server.list_llm_models()


@app.get("/config/embedding", response_model=List[EmbeddingConfig])
async def list_embedding_configs(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List available embedding configurations."""
    server = get_server()
    return server.list_embedding_models()


# ============================================================================
# Organization Endpoints
# ============================================================================


@app.get("/organizations", response_model=List[Organization])
async def list_organizations(
    cursor: Optional[str] = None,
    limit: int = 50,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List organizations."""
    server = get_server()
    return server.organization_manager.list_organizations(cursor=cursor, limit=limit)


@app.post("/organizations", response_model=Organization)
async def create_organization(
    name: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create an organization."""
    server = get_server()
    return server.organization_manager.create_organization(
        pydantic_org=Organization(name=name)
    )


@app.get("/organizations/{org_id}", response_model=Organization)
async def get_organization(
    org_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an organization by ID."""
    server = get_server()
    try:
        return server.organization_manager.get_organization_by_id(org_id)
    except Exception:
        # If organization doesn't exist, return default or create it
        return server.get_organization_or_default(org_id)


class CreateOrGetOrganizationRequest(BaseModel):
    """Request model for creating or getting an organization."""
    
    org_id: Optional[str] = None
    name: Optional[str] = None


@app.post("/organizations/create_or_get", response_model=Organization)
async def create_or_get_organization(
    request: CreateOrGetOrganizationRequest,
):
    """
    Create organization if it doesn't exist, or get existing one.
    This endpoint doesn't require authentication as it's used during client initialization.
    
    If org_id is not provided, a random ID will be generated.
    If org_id is provided, it will be used as-is (no prefix constraint).
    """
    server = get_server()
    from mirix.schemas.organization import OrganizationCreate
    
    # Use provided org_id or generate a new one
    if request.org_id:
        org_id = request.org_id
    else:
        # Generate a random org ID
        import uuid
        org_id = f"org-{uuid.uuid4().hex[:8]}"
    
    try:
        # Try to get existing organization
        org = server.organization_manager.get_organization_by_id(org_id)
        if org:
            return org
    except Exception:
        pass
    
    # Create new organization if it doesn't exist
    org_create = OrganizationCreate(
        id=org_id,
        name=request.name or org_id
    )
    org = server.organization_manager.create_organization(
        pydantic_org=Organization(**org_create.model_dump())
    )
    logger.debug("Created new organization: %s", org_id)
    return org


# ============================================================================
# User Endpoints
# ============================================================================


@app.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a user by ID."""
    server = get_server()
    return server.user_manager.get_user_by_id(user_id)


class CreateOrGetUserRequest(BaseModel):
    """Request model for creating or getting a user."""
    
    user_id: Optional[str] = None
    name: Optional[str] = None
    org_id: Optional[str] = None


@app.post("/users/create_or_get", response_model=User)
async def create_or_get_user(
    request: CreateOrGetUserRequest,
):
    """
    Create user if it doesn't exist, or get existing one.
    This endpoint doesn't require authentication as it's used during client initialization.
    
    If user_id is not provided, a random ID will be generated.
    If user_id is provided, it will be used as-is (no prefix constraint).
    """
    server = get_server()
    
    # Use provided user_id or generate a new one
    if request.user_id:
        user_id = request.user_id
    else:
        # Generate a random user ID
        import uuid
        user_id = f"user-{uuid.uuid4().hex[:8]}"
    
    org_id = request.org_id
    if not org_id:
        org_id = server.organization_manager.DEFAULT_ORG_ID
    
    try:
        # Try to get existing user
        user = server.user_manager.get_user_by_id(user_id)
        if user:
            return user
    except Exception:
        pass
    
    from mirix.schemas.user import User as PydanticUser
    
    # Create a User object with all required fields
    user = server.user_manager.create_user(
        pydantic_user=PydanticUser(
            id=user_id,
            name=request.name or user_id,
            organization_id=org_id,
            timezone=server.user_manager.DEFAULT_TIME_ZONE,
            status="active"
        )
    )
    logger.debug("Created new user: %s", user_id)
    return user

# ============================================================================
# Memory API Endpoints (New)
# ============================================================================


class InitializeMetaAgentRequest(BaseModel):
    """Request model for initializing a meta agent."""

    config: Dict[str, Any]
    project: Optional[str] = None
    update_agents: Optional[bool] = False


@app.post("/agents/meta/initialize", response_model=AgentState)
async def initialize_meta_agent(
    request: InitializeMetaAgentRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Initialize a meta agent with configuration.
    
    This creates a meta memory agent that manages specialized memory agents.
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Extract config components
    config = request.config
    llm_config = None
    embedding_config = None
    system_prompts = None
    agents_config = None

    # Build create_params by flattening meta_agent_config
    create_params = {
        "llm_config": LLMConfig(**config["llm_config"]),
        "embedding_config": EmbeddingConfig(**config["embedding_config"]),
    }
    
    # Flatten meta_agent_config fields into create_params
    if "meta_agent_config" in config and config["meta_agent_config"]:
        meta_config = config["meta_agent_config"]
        # Add fields from meta_agent_config directly
        if "agents" in meta_config:
            create_params["agents"] = meta_config["agents"]
        if "system_prompts" in meta_config:
            create_params["system_prompts"] = meta_config["system_prompts"]

    # Check if meta agent already exists for this project
    existing_meta_agents = server.agent_manager.list_agents(actor=user, limit=1000)

    assert len(existing_meta_agents) <= 1, "Only one meta agent can be created for a project"

    if len(existing_meta_agents) == 1:
        meta_agent = existing_meta_agents[0]
        
        # Only update the meta agent if update_agents is True
        if request.update_agents:
            from mirix.schemas.agent import UpdateMetaAgent
            # Update the existing meta agent
            meta_agent = server.agent_manager.update_meta_agent(
                meta_agent_id=meta_agent.id,
                meta_agent_update=UpdateMetaAgent(**create_params),
                actor=user
            )
    else:
        from mirix.schemas.agent import CreateMetaAgent
        meta_agent = server.agent_manager.create_meta_agent(meta_agent_create=CreateMetaAgent(**create_params), actor=user)

    return meta_agent

class AddMemoryRequest(BaseModel):
    """Request model for adding memory."""

    user_id: str
    meta_agent_id: str
    messages: List[Dict[str, Any]]
    chaining: bool = True
    verbose: bool = False


@app.post("/memory/add")
async def add_memory(
    request: AddMemoryRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Add conversation turns to memory (async via queue).
    
    Messages are queued for asynchronous processing by queue workers.
    Processing happens in the background, allowing for fast API response times.
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Get the meta agent by ID
    meta_agent = server.agent_manager.get_agent_by_id(request.meta_agent_id, actor=user)

    message = request.messages

    if isinstance(message, list) and "role" in message[0].keys():
        # This means the input is in the format of [{"role": "user", "content": [{"type": "text", "text": "..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]

        # We need to convert the message to the format in "content"
        new_message = []
        for msg in message:
            new_message.append({'type': "text", "text": "[USER]" if msg["role"] == "user" else "[ASSISTANT]"})
            new_message.extend(msg["content"])
        message = new_message

    input_messages = convert_message_to_mirix_message(message)

    # Queue for async processing instead of synchronous execution
    put_messages(
        actor=user,
        agent_id=meta_agent.id,
        input_messages=input_messages,
        chaining=request.chaining,
        user_id=request.user_id,
        verbose=request.verbose,
    )
    
    logger.debug("Memory queued for processing: %s", meta_agent.id)

    return {
        "success": True,
        "message": "Memory queued for processing",
        "status": "queued",
        "agent_id": meta_agent.id,
        "message_count": len(input_messages),
    }


class RetrieveMemoryRequest(BaseModel):
    """Request model for retrieving memory."""

    user_id: str
    messages: List[Dict[str, Any]]
    limit: int = 10  # Maximum number of items to retrieve per memory type


def retrieve_memories_by_keywords(
    server: SyncServer,
    user: User,
    agent_state: AgentState,
    key_words: str = "",
    limit: int = 10,
) -> dict:
    """
    Helper function to retrieve memories based on keywords using BM25 search.
    
    Args:
        server: The Mirix server instance
        user: The user whose memories to retrieve
        agent_state: Agent state (used as dummy for function signatures, not accessed in BM25)
        key_words: Keywords to search for (empty string returns recent items)
        limit: Maximum number of items to retrieve per memory type
        
    Returns:
        Dictionary containing all memory types with their items
    """
    search_method = "bm25"
    timezone_str = server.user_manager.get_user_by_id(user.id).timezone
    memories = {}
    
    # Get episodic memories (recent + relevant)
    try:
        episodic_manager = server.episodic_memory_manager
        
        # Get recent episodic memories
        recent_episodic = episodic_manager.list_episodic_memory(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=user,
            limit=limit,
            timezone_str=timezone_str,
        )
        
        # Get relevant episodic memories based on keywords
        relevant_episodic = []
        if key_words:
            relevant_episodic = episodic_manager.list_episodic_memory(
                agent_state=agent_state,  # Not accessed during BM25 search
                actor=user,
                query=key_words,
                search_field="details",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
        
        memories["episodic"] = {
            "total_count": episodic_manager.get_total_number_of_items(actor=user),
            "recent": [
                {
                    "id": event.id,
                    "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                    "summary": event.summary,
                    "details": event.details,
                }
                for event in recent_episodic
            ],
            "relevant": [
                {
                    "id": event.id,
                    "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                    "summary": event.summary,
                    "details": event.details,
                }
                for event in relevant_episodic
            ],
        }
    except Exception as e:
        logger.error("Error retrieving episodic memories: %s", e)
        memories["episodic"] = {"total_count": 0, "recent": [], "relevant": []}
    
    # Get semantic memories
    try:
        semantic_manager = server.semantic_memory_manager
        
        semantic_items = semantic_manager.list_semantic_items(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=user,
            query=key_words,
            search_field="details",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
        )
        
        memories["semantic"] = {
            "total_count": semantic_manager.get_total_number_of_items(actor=user),
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "summary": item.summary,
                    "details": item.details,
                }
                for item in semantic_items
            ],
        }
    except Exception as e:
        logger.error("Error retrieving semantic memories: %s", e)
        memories["semantic"] = {"total_count": 0, "items": []}
    
    # Get resource memories
    try:
        resource_manager = server.resource_memory_manager
        
        resources = resource_manager.list_resources(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=user,
            query=key_words,
            search_field="summary",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
        )
        
        memories["resource"] = {
            "total_count": resource_manager.get_total_number_of_items(actor=user),
            "items": [
                {
                    "id": resource.id,
                    "title": resource.title,
                    "summary": resource.summary,
                    "resource_type": resource.resource_type,
                }
                for resource in resources
            ],
        }
    except Exception as e:
        logger.error("Error retrieving resource memories: %s", e)
        memories["resource"] = {"total_count": 0, "items": []}
    
    # Get procedural memories
    try:
        procedural_manager = server.procedural_memory_manager
        
        procedures = procedural_manager.list_procedures(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=user,
            query=key_words,
            search_field="summary",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
        )
        
        memories["procedural"] = {
            "total_count": procedural_manager.get_total_number_of_items(actor=user),
            "items": [
                {
                    "id": procedure.id,
                    "entry_type": procedure.entry_type,
                    "summary": procedure.summary,
                }
                for procedure in procedures
            ],
        }
    except Exception as e:
        logger.error("Error retrieving procedural memories: %s", e)
        memories["procedural"] = {"total_count": 0, "items": []}
    
    # Get knowledge vault items
    try:
        knowledge_vault_manager = server.knowledge_vault_manager
        
        knowledge_items = knowledge_vault_manager.list_knowledge(
            agent_state=agent_state,  # Not accessed during BM25 search
            actor=user,
            query=key_words,
            search_field="caption",
            search_method=search_method,
            limit=limit,
            timezone_str=timezone_str,
        )
        
        memories["knowledge_vault"] = {
            "total_count": knowledge_vault_manager.get_total_number_of_items(actor=user),
            "items": [
                {
                    "id": item.id,
                    "caption": item.caption,
                }
                for item in knowledge_items
            ],
        }
    except Exception as e:
        logger.error("Error retrieving knowledge vault items: %s", e)
        memories["knowledge_vault"] = {"total_count": 0, "items": []}
    
    # Get core memory blocks
    try:
        block_manager = server.block_manager
        
        # Get all blocks for the user (these are the Human and Persona blocks)
        blocks = block_manager.get_blocks(actor=user)
        
        memories["core"] = {
            "total_count": len(blocks),
            "items": [
                {
                    "id": block.id,
                    "label": block.label,
                    "value": block.value,
                }
                for block in blocks
            ],
        }
    except Exception as e:
        logger.error("Error retrieving core memory blocks: %s", e)
        memories["core"] = {"total_count": 0, "items": []}
    
    return memories


@app.post("/memory/retrieve/conversation")
async def retrieve_memory_with_conversation(
    request: RetrieveMemoryRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on conversation context.
    Extracts topics from the conversation messages and uses them to retrieve relevant memories.
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Get all agents for this user
    all_agents = server.agent_manager.list_agents(actor=user, limit=1000)
    
    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this user",
            "topics": None,
            "memories": {},
        }
    
    # Extract topics from the conversation
    # TODO: Consider allowing custom model selection in the future
    llm_config = all_agents[0].llm_config
    topics = extract_topics_from_messages(request.messages, llm_config)
    logger.debug("Extracted topics from conversation: %s", topics)
    
    # Use topics as search keywords
    key_words = topics if topics else ""
    
    # Retrieve memories using the helper function
    memories = retrieve_memories_by_keywords(
        server=server,
        user=user,
        agent_state=all_agents[0],
        key_words=key_words,
        limit=request.limit,
    )
    
    return {
        "success": True,
        "topics": topics,
        "memories": memories,
    }


@app.get("/memory/retrieve/topic")
async def retrieve_memory_with_topic(
    user_id: str,
    topic: str,
    limit: int = 10,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on a topic using BM25 search.
    
    Args:
        user_id: The user ID to retrieve memories for
        topic: The topic/keywords to search for
        limit: Maximum number of items to retrieve per memory type (default: 10)
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Get all agents for this user
    all_agents = server.agent_manager.list_agents(actor=user, limit=1000)
    
    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this user",
            "topic": topic,
            "memories": {},
        }
    
    # Retrieve memories using the helper function
    memories = retrieve_memories_by_keywords(
        server=server,
        user=user,
        agent_state=all_agents[0],
        key_words=topic,
        limit=limit,
    )

    return {
        "success": True,
        "topic": topic,
        "memories": memories,
    }


@app.get("/memory/search")
async def search_memory(
    user_id: str,
    query: str,
    memory_type: str = "all",
    search_field: str = "null",
    search_method: str = "bm25",
    limit: int = 10,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Search for memories using various search methods.
    Similar to the search_in_memory tool function.
    
    Args:
        user_id: The user ID to retrieve memories for
        query: The search query string
        memory_type: Type of memory to search. Options: "episodic", "resource", "procedural", 
                    "knowledge_vault", "semantic", "all" (default: "all")
        search_field: Field to search in. Options vary by memory type:
                     - episodic: "summary", "details"
                     - resource: "summary", "content"
                     - procedural: "summary", "steps"
                     - knowledge_vault: "caption", "secret_value"
                     - semantic: "name", "summary", "details"
                     - For "all": use "null" (default)
        search_method: Search method. Options: "bm25" (default), "embedding"
        limit: Maximum number of results per memory type (default: 10)
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Get all agents for this user
    all_agents = server.agent_manager.list_agents(actor=user, limit=1000)
    
    if not all_agents:
        return {
            "success": False,
            "error": "No agents found for this user",
            "query": query,
            "results": [],
            "count": 0,
        }
    
    agent_state = all_agents[0]
    timezone_str = server.user_manager.get_user_by_id(user.id).timezone
    
    # Validate search parameters
    if memory_type == "resource" and search_field == "content" and search_method == "embedding":
        return {
            "success": False,
            "error": "embedding is not supported for resource memory's 'content' field.",
            "query": query,
            "results": [],
            "count": 0,
        }
    
    if memory_type == "knowledge_vault" and search_field == "secret_value" and search_method == "embedding":
        return {
            "success": False,
            "error": "embedding is not supported for knowledge_vault memory's 'secret_value' field.",
            "query": query,
            "results": [],
            "count": 0,
        }
    
    if memory_type == "all":
        search_field = "null"
    
    # Collect results from requested memory types
    all_results = []
    
    # Search episodic memories
    if memory_type in ["episodic", "all"]:
        try:
            episodic_memories = server.episodic_memory_manager.list_episodic_memory(
                actor=user,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "episodic",
                    "id": x.id,
                    "timestamp": x.occurred_at.isoformat() if x.occurred_at else None,
                    "event_type": x.event_type,
                    "actor": x.actor,
                    "summary": x.summary,
                    "details": x.details,
                }
                for x in episodic_memories
            ])
        except Exception as e:
            logger.error("Error searching episodic memories: %s", e)
    
    # Search resource memories
    if memory_type in ["resource", "all"]:
        try:
            resource_memories = server.resource_memory_manager.list_resources(
                actor=user,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else ("summary" if search_method == "embedding" else "content"),
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "resource",
                    "id": x.id,
                    "resource_type": x.resource_type,
                    "title": x.title,
                    "summary": x.summary,
                    "content": x.content[:200] if x.content else None,  # Truncate content for response
                }
                for x in resource_memories
            ])
        except Exception as e:
            logger.error("Error searching resource memories: %s", e)
    
    # Search procedural memories
    if memory_type in ["procedural", "all"]:
        try:
            procedural_memories = server.procedural_memory_manager.list_procedures(
                actor=user,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "procedural",
                    "id": x.id,
                    "entry_type": x.entry_type,
                    "summary": x.summary,
                    "steps": x.steps,
                }
                for x in procedural_memories
            ])
        except Exception as e:
            logger.error("Error searching procedural memories: %s", e)
    
    # Search knowledge vault
    if memory_type in ["knowledge_vault", "all"]:
        try:
            knowledge_vault_memories = server.knowledge_vault_manager.list_knowledge(
                actor=user,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else "caption",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "knowledge_vault",
                    "id": x.id,
                    "entry_type": x.entry_type,
                    "source": x.source,
                    "sensitivity": x.sensitivity,
                    "secret_value": x.secret_value,
                    "caption": x.caption,
                }
                for x in knowledge_vault_memories
            ])
        except Exception as e:
            logger.error("Error searching knowledge vault: %s", e)
    
    # Search semantic memories
    if memory_type in ["semantic", "all"]:
        try:
            semantic_memories = server.semantic_memory_manager.list_semantic_items(
                actor=user,
                agent_state=agent_state,
                query=query,
                search_field=search_field if search_field != "null" else "summary",
                search_method=search_method,
                limit=limit,
                timezone_str=timezone_str,
            )
            all_results.extend([
                {
                    "memory_type": "semantic",
                    "id": x.id,
                    "name": x.name,
                    "summary": x.summary,
                    "details": x.details,
                    "source": x.source,
                }
                for x in semantic_memories
            ])
        except Exception as e:
            logger.error("Error searching semantic memories: %s", e)
    
    return {
        "success": True,
        "query": query,
        "memory_type": memory_type,
        "search_field": search_field,
        "search_method": search_method,
        "results": all_results,
        "count": len(all_results),
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

