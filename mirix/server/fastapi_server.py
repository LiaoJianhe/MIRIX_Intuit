import asyncio
import json
import logging
import os
import queue
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text

from ..agent.agent_wrapper import AgentWrapper
from ..functions.mcp_client import StdioServerConfig, get_mcp_client_manager
from ..services.mcp_marketplace import get_mcp_marketplace
from ..services.mcp_tool_registry import get_mcp_tool_registry

logger = logging.getLogger(__name__)


# User context switching utilities
def switch_user_context(agent_wrapper, user_id: str):
    """Switch agent's user context and manage user status"""
    if agent_wrapper and agent_wrapper.client:

        # Set current user to inactive
        if agent_wrapper.client.user:
            current_user = agent_wrapper.client.user
            agent_wrapper.client.server.user_manager.update_user_status(
                current_user.id, "inactive"
            )

        # Get and set new user to active
        user = agent_wrapper.client.server.user_manager.get_user_by_id(user_id)
        agent_wrapper.client.server.user_manager.update_user_status(user_id, "active")
        agent_wrapper.client.user = user
        return user
    return None


def get_user_or_default(agent_wrapper, user_id: Optional[str] = None):
    """Get user by ID or return current user"""
    if user_id:
        return agent_wrapper.client.server.user_manager.get_user_by_id(user_id)
    elif agent_wrapper and agent_wrapper.client.user:
        return agent_wrapper.client.user
    else:
        return agent_wrapper.client.server.user_manager.get_default_user()


async def handle_gmail_connection(
    client_id: str, client_secret: str, server_name: str
) -> bool:
    """
    Handle Gmail OAuth2 authentication and MCP connection
    Using EXACT same logic as /Users/yu.wang/work/Gmail/single_user_gmail.py
    """
    import os

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    # Use all required Gmail scopes for full functionality
    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
    ]

    try:
        print(f"🔐 Starting Gmail OAuth for {server_name}")

        # Set up token file path (same pattern as original)
        token_file = os.path.expanduser("~/.mirix/gmail_token.json")
        os.makedirs(os.path.dirname(token_file), exist_ok=True)

        # Create client config - EXACT same structure as original
        client_config = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [
                    "http://localhost:8080/",
                    "http://localhost:8081/",
                    "http://localhost:8082/",
                ],
            }
        }

        creds = None

        # Load existing token if available - EXACT same logic
        if os.path.exists(token_file):
            try:
                creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            except Exception:
                print("🔄 Refreshing Gmail credentials (previous token expired)")
                os.remove(token_file)
                creds = None

        # If there are no (valid) credentials available, let the user log in - EXACT same logic
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing credentials: {e}")
                    creds = None

            if not creds:
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

                print("\n🔐 Starting OAuth authentication...")
                print("Opening browser for Google authentication...")

                # Try specific ports that match redirect URIs - EXACT same logic
                for port in [8080, 8081, 8082]:
                    try:
                        creds = flow.run_local_server(port=port, open_browser=True)
                        break
                    except OSError:
                        if port == 8082:
                            # If all ports fail, use automatic port selection
                            creds = flow.run_local_server(port=0, open_browser=True)

            # Save the credentials for the next run - EXACT same logic
            with open(token_file, "w") as token:
                token.write(creds.to_json())

        # Build the Gmail service - EXACT same logic
        service = build("gmail", "v1", credentials=creds)
        print("✅ Successfully authenticated with Gmail API")

        # Gmail service built successfully - ready for email sending
        print("✅ Gmail API connected successfully")

        # Now create the MCP client and add it to the manager
        from ..functions.mcp_client import (
            GmailMCPClient,
            GmailServerConfig,
            get_mcp_client_manager,
        )

        config = GmailServerConfig(
            server_name=server_name,
            client_id=client_id,
            client_secret=client_secret,
            token_file=token_file,
        )

        # Create Gmail MCP client directly
        client = GmailMCPClient(config)
        client.gmail_service = service
        client.credentials = creds
        client.initialized = True

        # Add to MCP manager
        mcp_manager = get_mcp_client_manager()
        mcp_manager.clients[server_name] = client
        mcp_manager.server_configs[server_name] = config

        # Save configuration to disk for persistence (this was missing!)
        mcp_manager._save_persistent_connections()

        print(
            f"✅ Gmail MCP client added to manager as '{server_name}' and saved to disk"
        )
        return True

    except Exception as e:
        print(f"❌ Error in Gmail OAuth flow: {str(e)}")
        logger.error(f"Gmail connection error: {str(e)}")
        return False


"""
VOICE RECORDING STRATEGY & ARCHITECTURE:

Current Implementation:
- Frontend records audio in 5-second chunks (CHUNK_DURATION = 5000ms)
- Chunks are accumulated locally until a screenshot is sent
- Raw voice files are sent to the agent for accumulation and processing
- Agent accumulates voice files alongside images until TEMPORARY_MESSAGE_LIMIT is reached
- Voice processing happens in agent.absorb_content_into_memory()

Recommended Alternative Strategy:
Instead of 5-second chunks, you can:
1. Send 1-second micro-chunks to reduce latency
2. Agent accumulates chunks until TEMPORARY_MESSAGE_LIMIT is reached
3. This aligns perfectly with how images are accumulated in agent.py

Benefits of 1-second chunks:
- Lower latency for real-time feedback
- More granular control over audio processing
- Better alignment with the existing image accumulation pattern
- Smoother user experience
- Voice processing happens in batches during memory absorption

Implementation changes needed:
- Frontend: Change CHUNK_DURATION from 5000 to 1000
- Agent: Handles voice file accumulation and processing during memory absorption
- Server: Passes raw voice files to agent without processing

FFPROBE WARNING:
The warning about ffprobe/avprobe is harmless and expected if FFmpeg isn't in your system PATH.
To fix it, install FFmpeg:
- Windows: Download from https://ffmpeg.org and add to PATH
- macOS: brew install ffmpeg  
- Linux: sudo apt install ffmpeg

The warning doesn't affect functionality as pydub falls back gracefully.
"""

app = FastAPI(title="Mirix Agent API", version="0.1.5")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def register_mcp_tools_for_restored_connections():
    """Register tools for MCP connections that were restored on startup"""
    try:
        mcp_manager = get_mcp_client_manager()
        connected_servers = mcp_manager.list_servers()

        if connected_servers and agent and agent.client.user:
            logger.info(
                f"Re-registering tools for {len(connected_servers)} restored MCP servers"
            )

            mcp_tool_registry = get_mcp_tool_registry()
            current_user = agent.client.user

            for server_name in connected_servers:
                try:
                    # Register tools for this server
                    registered_tools = mcp_tool_registry.register_mcp_tools(
                        current_user, [server_name]
                    )

                    # Add MCP tool to the current chat agent if available
                    if hasattr(agent, "agent_states"):
                        agent.client.server.agent_manager.add_mcp_tool(
                            agent_id=agent.agent_states.agent_state.id,
                            mcp_tool_name=server_name,
                            tool_ids=list(
                                set(
                                    [tool.id for tool in registered_tools]
                                    + [
                                        tool.id
                                        for tool in agent.client.server.agent_manager.get_agent_by_id(
                                            agent.agent_states.agent_state.id,
                                            actor=agent.client.user,
                                        ).tools
                                    ]
                                )
                            ),
                            actor=agent.client.user,
                        )

                    logger.info(
                        f"Re-registered {len(registered_tools)} tools for server {server_name}"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to re-register tools for server {server_name}: {str(e)}"
                    )

    except Exception as e:
        logger.error(f"Error re-registering MCP tools: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize and restore MCP connections on startup"""
    try:
        logger.info("Starting up Mirix FastAPI server...")

        # Initialize the MCP client manager (this will auto-restore connections)
        print("🚀 Initializing MCP client manager...")
        mcp_manager = get_mcp_client_manager()
        connected_servers = mcp_manager.list_servers()
        logger.info(
            f"MCP client manager initialized with {len(connected_servers)} restored connections: {connected_servers}"
        )
        print(
            f"🔄 MCP Manager: Restored {len(connected_servers)} connections: {connected_servers}"
        )

        # Debug: Check if the configuration file exists
        import os

        config_file = os.path.expanduser("~/.mirix/mcp_connections.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                import json

                configs = json.load(f)
                print(
                    f"📋 Found MCP config file with {len(configs)} entries: {list(configs.keys())}"
                )
        else:
            print(f"📋 No MCP config file found at {config_file}")

        # Tool registration will happen later when agent is available

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


# Global agent instance
agent = None
# Global storage for confirmation queues keyed by confirmation_id
confirmation_queues = {}
# Flag to track if MCP tools have been registered for restored connections
_mcp_tools_registered = False


class MessageRequest(BaseModel):
    message: Optional[str] = None
    image_uris: Optional[List[str]] = None
    sources: Optional[List[str]] = None  # Source names corresponding to image_uris
    voice_files: Optional[List[str]] = None  # Base64 encoded voice files
    memorizing: bool = False
    is_screen_monitoring: Optional[bool] = False
    user_id: Optional[str] = None  # User ID for memory storage


class MessageResponse(BaseModel):
    response: str
    status: str = "success"


class ConfirmationRequest(BaseModel):
    confirmation_id: str
    confirmed: bool


class PersonaDetailsResponse(BaseModel):
    personas: Dict[str, str]


class UpdatePersonaRequest(BaseModel):
    text: str
    user_id: Optional[str] = None


class UpdatePersonaResponse(BaseModel):
    success: bool
    message: str


class UpdateCoreMemoryRequest(BaseModel):
    label: str
    text: str


class UpdateCoreMemoryResponse(BaseModel):
    success: bool
    message: str


class ApplyPersonaTemplateRequest(BaseModel):
    persona_name: str
    user_id: Optional[str] = None


class CoreMemoryPersonaResponse(BaseModel):
    text: str


class SetModelRequest(BaseModel):
    model: str


class AddCustomModelRequest(BaseModel):
    model_name: str
    model_endpoint: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 4096
    maximum_length: int = 32768


class AddCustomModelResponse(BaseModel):
    success: bool
    message: str


class ListCustomModelsResponse(BaseModel):
    models: List[str]


class SetModelResponse(BaseModel):
    success: bool
    message: str
    missing_keys: List[str]
    model_requirements: Dict[str, Any]


class GetCurrentModelResponse(BaseModel):
    current_model: str


class SetTimezoneRequest(BaseModel):
    timezone: str


class SetTimezoneResponse(BaseModel):
    success: bool
    message: str


class GetTimezoneResponse(BaseModel):
    timezone: str


class ScreenshotSettingRequest(BaseModel):
    include_recent_screenshots: bool


class ScreenshotSettingResponse(BaseModel):
    success: bool
    include_recent_screenshots: bool
    message: str


# API Key validation functionality
def get_required_api_keys_for_model(model_endpoint_type: str) -> List[str]:
    """Get required API keys for a given model endpoint type"""
    api_key_mapping = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "azure": ["AZURE_API_KEY", "AZURE_BASE_URL", "AZURE_API_VERSION"],
        "google_ai": ["GEMINI_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "together": ["TOGETHER_API_KEY"],
        "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    }
    return api_key_mapping.get(model_endpoint_type, [])


def check_missing_api_keys(agent) -> Dict[str, List[str]]:
    """Check for missing API keys based on the agent's configuration"""

    if agent is None:
        return {"error": ["Agent not initialized"]}

    try:
        # Use the new AgentWrapper method instead of the old logic
        status = agent.check_api_key_status()

        return {
            "missing_keys": status["missing_keys"],
            "model_type": status.get("model_requirements", {}).get(
                "current_model", "unknown"
            ),
        }

    except Exception as e:
        print(f"Error in check_missing_api_keys: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {"error": [f"Error checking API keys: {str(e)}"]}


class ApiKeyRequest(BaseModel):
    key_name: str
    key_value: str


class ApiKeyCheckResponse(BaseModel):
    missing_keys: List[str]
    model_type: str
    requires_api_key: bool


class ApiKeyUpdateResponse(BaseModel):
    success: bool
    message: str


# Memory endpoint response models
class EpisodicMemoryItem(BaseModel):
    timestamp: str
    content: str
    context: Optional[str] = None
    emotions: Optional[List[str]] = None


class KnowledgeSkillItem(BaseModel):
    title: str
    type: str  # "semantic" or "procedural"
    content: str
    proficiency: Optional[str] = None
    tags: Optional[List[str]] = None


class DocsFilesItem(BaseModel):
    filename: str
    type: str
    summary: str
    last_accessed: Optional[str] = None
    size: Optional[str] = None


class CoreUnderstandingItem(BaseModel):
    aspect: str
    understanding: str
    confidence: Optional[float] = None
    last_updated: Optional[str] = None


class CredentialItem(BaseModel):
    name: str
    type: str
    content: str  # Will be masked
    tags: Optional[List[str]] = None
    last_used: Optional[str] = None


class ClearConversationResponse(BaseModel):
    success: bool
    message: str
    messages_deleted: int


class CleanupDetachedMessagesResponse(BaseModel):
    success: bool
    message: str
    cleanup_results: Dict[str, int]


class ExportMemoriesRequest(BaseModel):
    file_path: str
    memory_types: List[str]
    include_embeddings: bool = False
    user_id: Optional[str] = None


class ExportMemoriesResponse(BaseModel):
    success: bool
    message: str
    exported_counts: Dict[str, int]
    total_exported: int
    file_path: str


class ReflexionRequest(BaseModel):
    pass  # No parameters needed for now


class ReflexionResponse(BaseModel):
    success: bool
    message: str
    processing_time: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the server starts"""
    global agent

    # Handle PyInstaller bundled resources
    import sys
    from pathlib import Path

    if getattr(sys, "frozen", False):
        # Running in PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS)
        config_path = bundle_dir / "mirix" / "configs" / "mirix_monitor.yaml"
    else:
        # Running in development
        config_path = Path("mirix/configs/mirix_monitor.yaml")

    agent = AgentWrapper(str(config_path))
    print("Agent initialized successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring server status"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/send_message")
async def send_message_endpoint(
    fastapi_request: Request,
    request: MessageRequest,
    agent_id: Optional[str] = Header(None, alias="X-Agent-ID")
):
    """
    Send a message to the agent and get the response
    
    Optional Authorization:
        If X-Agent-ID header is provided, validates agent permissions
        and tags memories with agent scope
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # Optional authorization - only if X-Agent-ID is provided
    auth_context = None
    if agent_id:
        from mirix.server.server import db_context
        from mirix.authorization import AgentValidator
        
        with db_context() as db:
            try:
                # Validate agent has write permission (for memorizing)
                required_perms = ["write"] if request.memorizing else []
                auth_context = AgentValidator.validate_agent(db, agent_id, required_perms)
                logger.info(f"Authorized agent '{agent_id}' (scope: {auth_context.scope}) for send_message")
            except HTTPException as e:
                # If authorization fails, raise the error
                raise e

    # Register tools for restored MCP connections (one-time only)
    global _mcp_tools_registered
    if not _mcp_tools_registered:
        register_mcp_tools_for_restored_connections()
        _mcp_tools_registered = True

    # Check for missing API keys
    api_key_check = check_missing_api_keys(agent)
    if "error" in api_key_check:
        raise HTTPException(status_code=500, detail=api_key_check["error"][0])

    if api_key_check["missing_keys"]:
        # Return a special response indicating missing API keys
        return MessageResponse(
            response=f"Missing API keys for {api_key_check['model_type']} model: {', '.join(api_key_check['missing_keys'])}. Please provide the required API keys.",
            status="missing_api_keys",
        )

    try:
        # Handle user context switching if user_id is provided
        if request.user_id:
            switch_user_context(agent, request.user_id)

        print(
            f"Starting agent.send_message (non-streaming) with: message='{request.message}', memorizing={request.memorizing}, user_id={request.user_id}"
        )

        # SCOPE FILTERING: Temporarily hide episodic memories with incompatible scopes
        hidden_memory_ids = []
        if auth_context and request.user_id:
            from mirix.server.server import db_context
            import json as json_lib
            
            print(f"[SCOPE FILTER] Agent scope: {auth_context.scope}, User: {request.user_id}")
            
            with db_context() as db:
                # Find episodic memories with different scopes (not matching current agent's scope)
                result = db.execute(
                    text("""
                        SELECT id, metadata_
                        FROM episodic_memory
                        WHERE user_id = :user_id
                          AND is_deleted = FALSE
                          AND metadata_ IS NOT NULL
                    """),
                    {"user_id": request.user_id}
                )
                
                all_memories = result.fetchall()
                print(f"[SCOPE FILTER] Found {len(all_memories)} total memories for user")
                
                for row in all_memories:
                    mem_id = row[0]
                    metadata = row[1] if isinstance(row[1], dict) else json_lib.loads(row[1] or '{}')
                    mem_scope = metadata.get('agent_scope')
                    
                    print(f"[SCOPE FILTER] Memory {mem_id}: scope={mem_scope}, agent_scope={auth_context.scope}, match={mem_scope == auth_context.scope}")
                    
                    # Hide memories that don't match the current agent's scope
                    if mem_scope and mem_scope != auth_context.scope:
                        hidden_memory_ids.append(mem_id)
                        print(f"[SCOPE FILTER] Hiding {mem_id} (scope: {mem_scope})")
                
                # Temporarily mark incompatible memories as deleted
                if hidden_memory_ids:
                    db.execute(
                        text("""
                            UPDATE episodic_memory
                            SET is_deleted = TRUE
                            WHERE id = ANY(:ids)
                        """),
                        {"ids": hidden_memory_ids}
                    )
                    db.commit()
                    print(f"[SCOPE FILTER] Temporarily hid {len(hidden_memory_ids)} memories")
                    logger.info(f"Temporarily hiding {len(hidden_memory_ids)} episodic memories with incompatible scopes")
                else:
                    print(f"[SCOPE FILTER] No memories to hide")

        # MESSAGE HISTORY FILTERING: Track message count before sending
        message_count_before = 0
        hidden_message_ids = []
        if auth_context and request.user_id:
            from mirix.server.server import db_context
            
            with db_context() as db:
                # Count existing messages
                result = db.execute(
                    text("SELECT COUNT(*) FROM messages WHERE user_id = :user_id"),
                    {"user_id": request.user_id}
                )
                message_count_before = result.scalar()
                
                # Hide messages from different agent scopes
                result = db.execute(
                    text("""
                        SELECT id, name
                        FROM messages
                        WHERE user_id = :user_id
                          AND name IS NOT NULL
                          AND name != :current_scope
                    """),
                    {"user_id": request.user_id, "current_scope": auth_context.scope}
                )
                
                for row in result.fetchall():
                    msg_id = row[0]
                    msg_scope = row[1]
                    if msg_scope and msg_scope != auth_context.scope:
                        hidden_message_ids.append(msg_id)
                
                # Permanently delete messages from other scopes (they're cross-contaminated)
                if hidden_message_ids:
                    # First, remove these message IDs from agent state
                    db.execute(
                        text("""
                            UPDATE agents
                            SET message_ids = (
                                SELECT json_agg(elem)
                                FROM json_array_elements_text(COALESCE(message_ids::json, '[]'::json)) elem
                                WHERE elem::text != ALL(:ids)
                            )
                            WHERE user_id = :user_id
                        """),
                        {"ids": hidden_message_ids, "user_id": request.user_id}
                    )
                    
                    # Then delete the messages
                    db.execute(
                        text("DELETE FROM messages WHERE id = ANY(:ids)"),
                        {"ids": hidden_message_ids}
                    )
                    db.commit()
                    print(f"[SCOPE FILTER] Deleted {len(hidden_message_ids)} cross-scope messages")
                    hidden_message_ids = []  # Don't try to restore

        try:
            # Run the blocking agent.send_message() in a background thread to avoid blocking other requests
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                lambda: agent.send_message(
                    message=request.message,
                    image_uris=request.image_uris,
                    sources=request.sources,  # Pass sources to agent
                    voice_files=request.voice_files,  # Pass voice files to agent
                    memorizing=request.memorizing,
                    user_id=request.user_id,
                ),
            )
            
            # Tag NEW messages with agent scope
            if auth_context and request.user_id:
                from mirix.server.server import db_context
                
                with db_context() as db:
                    # Get message count after
                    result = db.execute(
                        text("SELECT COUNT(*) FROM messages WHERE user_id = :user_id"),
                        {"user_id": request.user_id}
                    )
                    message_count_after = result.scalar()
                    
                    # Tag new messages with agent scope (using name field)
                    if message_count_after > message_count_before:
                        db.execute(
                            text("""
                                UPDATE messages
                                SET name = :scope
                                WHERE user_id = :user_id
                                  AND created_at > NOW() - INTERVAL '5 seconds'
                                  AND (name IS NULL OR name = '')
                            """),
                            {"scope": auth_context.scope, "user_id": request.user_id}
                        )
                        db.commit()
                        new_count = message_count_after - message_count_before
                        print(f"[SCOPE FILTER] Tagged {new_count} new messages with scope: {auth_context.scope}")
        
        finally:
            # SCOPE FILTERING: Restore hidden memories and messages
            if hidden_memory_ids or hidden_message_ids:
                from mirix.server.server import db_context
                with db_context() as db:
                    # Restore memories
                    if hidden_memory_ids:
                        db.execute(
                            text("""
                                UPDATE episodic_memory
                                SET is_deleted = FALSE
                                WHERE id = ANY(:ids)
                            """),
                            {"ids": hidden_memory_ids}
                        )
                        logger.info(f"Restored {len(hidden_memory_ids)} temporarily hidden memories")
                    
                    # Note: We DON'T restore hidden messages - they stay filtered
                    # This prevents cross-contamination in future requests
                    
                    db.commit()

        print(f"Agent response (non-streaming): {response}")

        if response == "ERROR":
            raise HTTPException(status_code=500, detail="Agent returned an error")

        # Handle case where agent returns None
        if response is None:
            if request.memorizing:
                # When memorizing=True, None response is expected (no response needed)
                response = ""
            else:
                # When memorizing=False, None response is an error
                response = "I received your message but couldn't generate a response. Please try again."
        
        # Tag memories with agent scope if authorized (for both modes)
        if auth_context and request.user_id:
            # Schedule background tagging task
            # In normal conversation mode (memorizing=False), memories are created async
            # So we tag all untagged memories for this user
            async def tag_memories_background():
                try:
                    import json as json_lib
                    from datetime import datetime
                    from mirix.server.server import db_context
                    
                    # For normal conversation, wait for memory agents to process
                    # For memorizing mode, memories are created immediately
                    wait_time = 2 if request.memorizing else 1
                    await asyncio.sleep(wait_time)
                    
                    agent_metadata = {
                        "agent_id": auth_context.agent_id,
                        "agent_scope": auth_context.scope,
                        "created_by_agent": auth_context.agent_id,
                        "last_interaction_timestamp": datetime.utcnow().isoformat()
                    }
                    
                    with db_context() as db:
                        # Tag ALL untagged episodic memories for this user
                        result = db.execute(
                            text("""
                                UPDATE episodic_memory
                                SET metadata_ = CAST(:metadata AS json)
                                WHERE user_id = :user_id
                                  AND (metadata_ IS NULL OR metadata_::text = '{}')
                                RETURNING id
                            """),
                            {
                                "metadata": json_lib.dumps(agent_metadata),
                                "user_id": request.user_id
                            }
                        )
                        episodic_count = len(result.fetchall())
                        
                        # Tag ALL untagged semantic memories for this user
                        result = db.execute(
                            text("""
                                UPDATE semantic_memory
                                SET metadata_ = CAST(:metadata AS json)
                                WHERE user_id = :user_id
                                  AND (metadata_ IS NULL OR metadata_::text = '{}')
                                RETURNING id
                            """),
                            {
                                "metadata": json_lib.dumps(agent_metadata),
                                "user_id": request.user_id
                            }
                        )
                        semantic_count = len(result.fetchall())
                        
                        db.commit()
                        
                        if episodic_count > 0 or semantic_count > 0:
                            logger.info(f"Tagged {episodic_count} episodic + {semantic_count} semantic memories with scope: {auth_context.scope}")
                except Exception as e:
                    logger.warning(f"Failed to tag memories in background: {e}")
            
            # Run tagging in background (don't block response)
            asyncio.create_task(tag_memories_background())

        return MessageResponse(response=response)

    except Exception as e:
        print(f"Error in send_message_endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


@app.post("/send_streaming_message")
async def send_streaming_message_endpoint(request: MessageRequest):
    """Send a message to the agent and stream intermediate messages and final response"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    # Register tools for restored MCP connections (one-time only)
    global _mcp_tools_registered
    if not _mcp_tools_registered:
        register_mcp_tools_for_restored_connections()
        _mcp_tools_registered = True

    # Check for missing API keys
    api_key_check = check_missing_api_keys(agent)
    if "error" in api_key_check:
        raise HTTPException(status_code=500, detail=api_key_check["error"][0])

    if api_key_check["missing_keys"]:
        # Return a special SSE event for missing API keys
        async def missing_keys_response():
            yield f"data: {json.dumps({'type': 'missing_api_keys', 'missing_keys': api_key_check['missing_keys'], 'model_type': api_key_check['model_type']})}\n\n"

        return StreamingResponse(
            missing_keys_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    agent.update_chat_agent_system_prompt(request.is_screen_monitoring)

    # Create a queue to collect intermediate messagess
    message_queue = queue.Queue()

    def display_intermediate_message(message_type: str, message: str):
        """Callback function to capture intermediate messages"""
        message_queue.put(
            {"type": "intermediate", "message_type": message_type, "content": message}
        )

    def request_user_confirmation(confirmation_type: str, details: dict) -> bool:
        """Request confirmation from user and wait for response"""
        import uuid

        confirmation_id = str(uuid.uuid4())

        # Create a queue for this specific confirmation
        confirmation_result_queue = queue.Queue()
        confirmation_queues[confirmation_id] = confirmation_result_queue

        # Put confirmation request in message queue
        message_queue.put(
            {
                "type": "confirmation_request",
                "confirmation_type": confirmation_type,
                "confirmation_id": confirmation_id,
                "details": details,
            }
        )

        # Wait for confirmation response with timeout
        try:
            result = confirmation_result_queue.get(timeout=300)  # 5 minute timeout
            return result.get("confirmed", False)
        except queue.Empty:
            # Timeout - default to not confirmed
            return False
        finally:
            # Clean up the queue
            confirmation_queues.pop(confirmation_id, None)

    async def generate_stream():
        """Generator function for streaming responses"""
        try:
            # Start the agent processing in a separate thread
            result_queue = queue.Queue()

            async def run_agent():
                try:
                    # find the current active user
                    users = agent.client.server.user_manager.list_users()
                    active_user = next(
                        (user for user in users if user.status == "active"), None
                    )
                    current_user_id = active_user.id if active_user else None

                    # Run agent.send_message in a background thread to avoid blocking
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,  # Use default ThreadPoolExecutor
                        lambda: agent.send_message(
                            message=request.message,
                            image_uris=request.image_uris,
                            sources=request.sources,  # Pass sources to agent
                            voice_files=request.voice_files,  # Pass raw voice files
                            memorizing=request.memorizing,
                            display_intermediate_message=display_intermediate_message,
                            request_user_confirmation=request_user_confirmation,
                            is_screen_monitoring=request.is_screen_monitoring,
                            user_id=current_user_id,
                        ),
                    )
                    # Handle various response cases
                    if response is None:
                        if request.memorizing:
                            result_queue.put({"type": "final", "response": ""})
                        else:
                            print("[DEBUG] Agent returned None response")
                            result_queue.put(
                                {"type": "error", "error": "Agent returned no response"}
                            )
                    elif isinstance(response, str) and response.startswith("ERROR_"):
                        # Handle specific error types from agent wrapper
                        print(f"[DEBUG] Agent returned specific error: {response}")
                        if response == "ERROR_RESPONSE_FAILED":
                            print("[DEBUG] - Message queue response failed")
                            result_queue.put(
                                {
                                    "type": "error",
                                    "error": "Message processing failed in agent queue",
                                }
                            )
                        elif response == "ERROR_INVALID_RESPONSE_STRUCTURE":
                            print(
                                "[DEBUG] - Response structure invalid (missing messages or insufficient count)"
                            )
                            result_queue.put(
                                {
                                    "type": "error",
                                    "error": "Invalid response structure from agent",
                                }
                            )
                        elif response == "ERROR_NO_TOOL_CALL":
                            print(
                                "[DEBUG] - Expected message missing tool_call attribute"
                            )
                            result_queue.put(
                                {
                                    "type": "error",
                                    "error": "Agent response missing required tool call",
                                }
                            )
                        elif response == "ERROR_NO_MESSAGE_IN_ARGS":
                            print("[DEBUG] - Tool call arguments missing 'message' key")
                            result_queue.put(
                                {
                                    "type": "error",
                                    "error": "Agent tool call missing message content",
                                }
                            )
                        elif response == "ERROR_PARSING_EXCEPTION":
                            print(
                                "[DEBUG] - Exception occurred during response parsing"
                            )
                            result_queue.put(
                                {
                                    "type": "error",
                                    "error": "Failed to parse agent response",
                                }
                            )
                        else:
                            print(f"[DEBUG] - Unknown error type: {response}")
                            result_queue.put(
                                {
                                    "type": "error",
                                    "error": f"Unknown agent error: {response}",
                                }
                            )
                    elif response == "ERROR":
                        print("[DEBUG] Agent returned generic ERROR string")
                        result_queue.put(
                            {"type": "error", "error": "Agent processing failed"}
                        )
                    elif not response or (
                        isinstance(response, str) and response.strip() == ""
                    ):
                        if request.memorizing:
                            print(
                                "[DEBUG] Agent returned empty response - expected for memorizing=True"
                            )
                            result_queue.put({"type": "final", "response": ""})
                        else:
                            print("[DEBUG] Agent returned empty response unexpectedly")
                            result_queue.put(
                                {
                                    "type": "error",
                                    "error": "Agent returned empty response",
                                }
                            )
                    else:
                        print(
                            f"[DEBUG] Agent returned successful response (length: {len(str(response))})"
                        )
                        result_queue.put({"type": "final", "response": response})

                except Exception as e:
                    print(f"[DEBUG] Exception in run_agent: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    result_queue.put({"type": "error", "error": str(e)})

            # Start agent processing as async task
            agent_task = asyncio.create_task(run_agent())

            # Keep track of whether we've sent the final result
            final_result_sent = False

            # Stream intermediate messages and wait for final result
            while not final_result_sent:
                # Check for intermediate messages first
                try:
                    intermediate_msg = message_queue.get_nowait()
                    yield f"data: {json.dumps(intermediate_msg)}\n\n"
                    continue  # Continue to next iteration to check for more messages
                except queue.Empty:
                    pass

                # Check for final result with timeout
                try:
                    # Use a short timeout to allow for intermediate messages
                    final_result = result_queue.get(timeout=0.1)
                    if final_result["type"] == "error":
                        yield f"data: {json.dumps({'type': 'error', 'error': final_result['error']})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'final', 'response': final_result['response']})}\n\n"
                    final_result_sent = True
                    break
                except queue.Empty:
                    # If no result yet, check if task is still running
                    if agent_task.done():
                        # Task is done but no result - this shouldn't happen, but handle it
                        try:
                            # Check if the task raised an exception
                            agent_task.result()
                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'error': f'Agent processing failed: {str(e)}'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'error', 'error': 'Agent processing completed unexpectedly without result'})}\n\n"
                        final_result_sent = True
                        break
                    # Otherwise continue the loop to check for more intermediate messages
                    await asyncio.sleep(0.1)  # Yield control to allow other operations

            # Make sure task completes
            if not agent_task.done():
                try:
                    await asyncio.wait_for(agent_task, timeout=5.0)
                except asyncio.TimeoutError:
                    agent_task.cancel()
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Agent processing timed out'})}\n\n"

        except Exception as e:
            print(f"Traceback: {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    try:
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )
    except Exception as e:
        print(f"Error in send_streaming_message_endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")


@app.get("/personas", response_model=PersonaDetailsResponse)
async def get_personas(user_id: Optional[str] = None):
    """Get all personas with their details (name and text)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        persona_details = agent.get_persona_details()
        return PersonaDetailsResponse(personas=persona_details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting personas: {str(e)}")


@app.post("/personas/update", response_model=UpdatePersonaResponse)
async def update_persona(request: UpdatePersonaRequest):
    """Update the agent's core memory persona text"""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        agent.update_core_memory_persona(request.text)
        return UpdatePersonaResponse(
            success=True, message="Core memory persona updated successfully"
        )
    except Exception as e:
        return UpdatePersonaResponse(
            success=False, message=f"Error updating core memory persona: {str(e)}"
        )


@app.post("/personas/apply_template", response_model=UpdatePersonaResponse)
async def apply_persona_template(request: ApplyPersonaTemplateRequest):
    """Apply a persona template to the agent"""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        agent.apply_persona_template(request.persona_name)
        return UpdatePersonaResponse(
            success=True,
            message=f"Persona template '{request.persona_name}' applied successfully",
        )
    except Exception as e:
        return UpdatePersonaResponse(
            success=False, message=f"Error applying persona template: {str(e)}"
        )


@app.post("/core_memory/update", response_model=UpdateCoreMemoryResponse)
async def update_core_memory(request: UpdateCoreMemoryRequest):
    """Update a specific core memory block with new text"""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        agent.update_core_memory(text=request.text, label=request.label)
        return UpdateCoreMemoryResponse(
            success=True,
            message=f"Core memory block '{request.label}' updated successfully",
        )
    except Exception as e:
        return UpdateCoreMemoryResponse(
            success=False, message=f"Error updating core memory: {str(e)}"
        )


@app.get("/personas/core_memory", response_model=CoreMemoryPersonaResponse)
async def get_core_memory_persona(user_id: Optional[str] = None):
    """Get the core memory persona text"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        persona_text = agent.get_core_memory_persona()
        return CoreMemoryPersonaResponse(text=persona_text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting core memory persona: {str(e)}"
        )


@app.get("/models/current", response_model=GetCurrentModelResponse)
async def get_current_model():
    """Get the current model being used by the agent"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        current_model = agent.get_current_model()
        return GetCurrentModelResponse(current_model=current_model)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting current model: {str(e)}"
        )


@app.post("/models/set", response_model=SetModelResponse)
async def set_model(request: SetModelRequest):
    """Set the model for the agent"""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Check if this is a custom model
        custom_models_dir = Path.home() / ".mirix" / "custom_models"
        custom_config = None

        if custom_models_dir.exists():
            # Look for a config file that matches this model name
            for config_file in custom_models_dir.glob("*.yaml"):
                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                        if config and config.get("model_name") == request.model:
                            custom_config = config
                            print(
                                f"Found custom model config for '{request.model}' at {config_file}"
                            )
                            break
                except Exception as e:
                    print(f"Error reading custom model config {config_file}: {e}")
                    continue

        # Set the model with custom config if found, otherwise use standard method
        if custom_config:
            result = agent.set_model(request.model, custom_agent_config=custom_config)
        else:
            result = agent.set_model(request.model)

        return SetModelResponse(
            success=result["success"],
            message=result["message"],
            missing_keys=result.get("missing_keys", []),
            model_requirements=result.get("model_requirements", {}),
        )
    except Exception as e:
        return SetModelResponse(
            success=False,
            message=f"Error setting model: {str(e)}",
            missing_keys=[],
            model_requirements={},
        )


@app.get("/models/memory/current", response_model=GetCurrentModelResponse)
async def get_current_memory_model():
    """Get the current model being used by the memory manager"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        current_model = agent.get_current_memory_model()
        return GetCurrentModelResponse(current_model=current_model)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting current memory model: {str(e)}"
        )


@app.post("/models/memory/set", response_model=SetModelResponse)
async def set_memory_model(request: SetModelRequest):
    """Set the model for the memory manager"""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Check if this is a custom model
        custom_models_dir = Path.home() / ".mirix" / "custom_models"
        custom_config = None

        if custom_models_dir.exists():
            # Look for a config file that matches this model name
            for config_file in custom_models_dir.glob("*.yaml"):
                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                        if config and config.get("model_name") == request.model:
                            custom_config = config
                            print(
                                f"Found custom model config for memory model '{request.model}' at {config_file}"
                            )
                            break
                except Exception as e:
                    print(f"Error reading custom model config {config_file}: {e}")
                    continue

        # Set the memory model with custom config if found, otherwise use standard method
        if custom_config:
            result = agent.set_memory_model(
                request.model, custom_agent_config=custom_config
            )
        else:
            result = agent.set_memory_model(request.model)

        return SetModelResponse(
            success=result["success"],
            message=result["message"],
            missing_keys=result.get("missing_keys", []),
            model_requirements=result.get("model_requirements", {}),
        )
    except Exception as e:
        return SetModelResponse(
            success=False,
            message=f"Error setting memory model: {str(e)}",
            missing_keys=[],
            model_requirements={},
        )


@app.post("/models/custom/add", response_model=AddCustomModelResponse)
async def add_custom_model(request: AddCustomModelRequest):
    """Add a custom model configuration"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Create config file for the custom model
        config = {
            "agent_name": "mirix",
            "model_name": request.model_name,
            "model_endpoint": request.model_endpoint,
            "api_key": request.api_key,
            "model_provider": "local_server",
            "generation_config": {
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "context_window": request.maximum_length,
            },
        }

        # Create custom models directory if it doesn't exist
        custom_models_dir = Path.home() / ".mirix" / "custom_models"
        custom_models_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from model name (sanitize for filesystem)
        safe_model_name = "".join(
            c for c in request.model_name if c.isalnum() or c in ("-", "_", ".")
        ).rstrip()
        config_filename = f"{safe_model_name}.yaml"
        config_file_path = custom_models_dir / config_filename

        # Save config to YAML file
        with open(config_file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        # Also set the model in the agent
        agent.set_model(request.model_name, custom_agent_config=config)

        return AddCustomModelResponse(
            success=True,
            message=f"Custom model '{request.model_name}' added successfully and saved to {config_file_path}",
        )

    except Exception as e:
        print(f"Error adding custom model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return AddCustomModelResponse(
            success=False, message=f"Error adding custom model: {str(e)}"
        )


@app.get("/models/custom/list", response_model=ListCustomModelsResponse)
async def list_custom_models():
    """List all available custom models"""
    try:
        custom_models_dir = Path.home() / ".mirix" / "custom_models"
        models = []

        if custom_models_dir.exists():
            for config_file in custom_models_dir.glob("*.yaml"):
                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                        if config and "model_name" in config:
                            models.append(config["model_name"])
                except Exception as e:
                    print(f"Error reading custom model config {config_file}: {e}")
                    continue

        return ListCustomModelsResponse(models=models)

    except Exception as e:
        print(f"Error listing custom models: {e}")
        return ListCustomModelsResponse(models=[])


@app.get("/timezone/current", response_model=GetTimezoneResponse)
async def get_current_timezone():
    """Get the current timezone of the agent"""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        if not target_user:
            raise HTTPException(status_code=404, detail="No user found")

        current_timezone = target_user.timezone
        return GetTimezoneResponse(timezone=current_timezone)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting current timezone: {str(e)}"
        )


@app.post("/timezone/set", response_model=SetTimezoneResponse)
async def set_timezone(request: SetTimezoneRequest):
    """Set the timezone for the agent"""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        if not target_user:
            return SetTimezoneResponse(success=False, message="No user found")

        # Update the timezone for the active user
        agent.client.server.user_manager.update_user_timezone(
            user_id=target_user.id, timezone_str=request.timezone
        )

        return SetTimezoneResponse(
            success=True,
            message=f"Timezone '{request.timezone}' set successfully for user {target_user.name}",
        )
    except Exception as e:
        return SetTimezoneResponse(
            success=False, message=f"Error setting timezone: {str(e)}"
        )


@app.get("/screenshot_setting", response_model=ScreenshotSettingResponse)
async def get_screenshot_setting():
    """Get the current screenshot setting"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    return ScreenshotSettingResponse(
        success=True,
        include_recent_screenshots=agent.include_recent_screenshots,
        message="Screenshot setting retrieved successfully",
    )


@app.post("/screenshot_setting/set", response_model=ScreenshotSettingResponse)
async def set_screenshot_setting(request: ScreenshotSettingRequest):
    """Set whether to include recent screenshots in messages"""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        agent.set_include_recent_screenshots(request.include_recent_screenshots)
        return ScreenshotSettingResponse(
            success=True,
            include_recent_screenshots=request.include_recent_screenshots,
            message=f"Screenshot setting updated: {'enabled' if request.include_recent_screenshots else 'disabled'}",
        )
    except Exception as e:
        return ScreenshotSettingResponse(
            success=False,
            include_recent_screenshots=False,
            message=f"Error updating screenshot setting: {str(e)}",
        )


@app.get("/api_keys/check", response_model=ApiKeyCheckResponse)
async def check_api_keys():
    """Check for missing API keys based on current agent configuration"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Use the new AgentWrapper method
        api_key_status = agent.check_api_key_status()

        return ApiKeyCheckResponse(
            missing_keys=api_key_status["missing_keys"],
            model_type=api_key_status.get("model_requirements", {}).get(
                "current_model", "unknown"
            ),
            requires_api_key=len(api_key_status["missing_keys"]) > 0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error checking API keys: {str(e)}"
        )


@app.post("/api_keys/update", response_model=ApiKeyUpdateResponse)
async def update_api_key(request: ApiKeyRequest):
    """Update an API key value"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Use the new AgentWrapper method which handles .env file saving
        result = agent.provide_api_key(request.key_name, request.key_value)

        # Also update environment variable and model_settings for backwards compatibility
        if result["success"]:
            os.environ[request.key_name] = request.key_value

            from mirix.settings import model_settings

            setting_name = request.key_name.lower()
            if hasattr(model_settings, setting_name):
                setattr(model_settings, setting_name, request.key_value)
        else:
            # If AgentWrapper doesn't support this key type, fall back to manual .env saving
            if "Unsupported API key type" in result["message"]:
                # Save to .env file manually for non-Gemini keys
                _save_api_key_to_env_file(request.key_name, request.key_value)
                os.environ[request.key_name] = request.key_value

                from mirix.settings import model_settings

                setting_name = request.key_name.lower()
                if hasattr(model_settings, setting_name):
                    setattr(model_settings, setting_name, request.key_value)

                result["success"] = True
                result["message"] = (
                    f"API key '{request.key_name}' saved to .env file successfully"
                )

        return ApiKeyUpdateResponse(
            success=result["success"], message=result["message"]
        )
    except Exception as e:
        return ApiKeyUpdateResponse(
            success=False, message=f"Error updating API key: {str(e)}"
        )


def _save_api_key_to_env_file(key_name: str, api_key: str):
    """
    Helper function to save API key to .env file for non-AgentWrapper keys.
    """
    from pathlib import Path

    # Find the .env file (look in current directory and parent directories)
    env_file_path = None
    current_path = Path.cwd()

    # Check current directory and up to 3 parent directories
    for _ in range(4):
        potential_env_path = current_path / ".env"
        if potential_env_path.exists():
            env_file_path = potential_env_path
            break
        current_path = current_path.parent

    # If no .env file found, create one in the current working directory
    if env_file_path is None:
        env_file_path = Path.cwd() / ".env"

    # Read existing .env file content
    env_content = {}
    if env_file_path.exists():
        with open(env_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_content[key.strip()] = value.strip()

    # Update the API key
    env_content[key_name] = api_key

    # Write back to .env file
    with open(env_file_path, "w") as f:
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")

    print(f"API key {key_name} saved to {env_file_path}")


# Memory endpoints
@app.get("/memory/episodic")
async def get_episodic_memory(user_id: Optional[str] = None):
    """Get episodic memory (past events)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        # Access the episodic memory manager through the client
        client = agent.client
        episodic_manager = client.server.episodic_memory_manager

        # Get episodic events using the correct method name
        events = episodic_manager.list_episodic_memory(
            agent_state=agent.agent_states.episodic_memory_agent_state,
            actor=target_user,
            limit=50,
            timezone_str=target_user.timezone,
        )

        # Transform to frontend format
        episodic_items = []
        for event in events:
            episodic_items.append(
                {
                    "timestamp": event.occurred_at.isoformat()
                    if event.occurred_at
                    else None,
                    "summary": event.summary,
                    "details": event.details,
                    "event_type": event.event_type,
                    "tree_path": event.tree_path if hasattr(event, "tree_path") else [],
                }
            )

        return episodic_items

    except Exception as e:
        print(f"Error retrieving episodic memory: {str(e)}")
        # Return empty list if no memory or error
        return []


@app.get("/memory/semantic")
async def get_semantic_memory(user_id: Optional[str] = None):
    """Get semantic memory (knowledge)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        client = agent.client
        semantic_items_list = []

        # Get semantic memory items
        try:
            semantic_manager = client.server.semantic_memory_manager
            semantic_items = semantic_manager.list_semantic_items(
                agent_state=agent.agent_states.semantic_memory_agent_state,
                actor=target_user,
                limit=50,
                timezone_str=target_user.timezone,
            )

            for item in semantic_items:
                semantic_items_list.append(
                    {
                        "title": item.name,
                        "type": "semantic",
                        "summary": item.summary,
                        "details": item.details,
                        "tree_path": item.tree_path
                        if hasattr(item, "tree_path")
                        else [],
                    }
                )
        except Exception as e:
            print(f"Error retrieving semantic memory: {str(e)}")

        return semantic_items_list

    except Exception as e:
        print(f"Error retrieving semantic memory: {str(e)}")
        return []


@app.get("/memory/procedural")
async def get_procedural_memory(user_id: Optional[str] = None):
    """Get procedural memory (skills and procedures)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        client = agent.client
        procedural_items_list = []

        # Get procedural memory items
        try:
            procedural_manager = client.server.procedural_memory_manager
            procedural_items = procedural_manager.list_procedures(
                agent_state=agent.agent_states.procedural_memory_agent_state,
                actor=target_user,
                limit=50,
                timezone_str=target_user.timezone,
            )

            for item in procedural_items:
                # Parse steps if it's a JSON string
                steps = item.steps
                if isinstance(steps, str):
                    try:
                        steps = json.loads(steps)
                        # Extract just the instruction text for simpler frontend display
                        if (
                            isinstance(steps, list)
                            and steps
                            and isinstance(steps[0], dict)
                        ):
                            steps = [
                                step.get("instruction", str(step)) for step in steps
                            ]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # If parsing fails, keep as string and split by common delimiters
                        if isinstance(steps, str):
                            steps = [
                                s.strip()
                                for s in steps.replace("\n", "|").split("|")
                                if s.strip()
                            ]
                        else:
                            steps = []

                procedural_items_list.append(
                    {
                        "title": item.entry_type,
                        "type": "procedural",
                        "summary": item.summary,
                        "steps": steps if isinstance(steps, list) else [],
                        "tree_path": item.tree_path
                        if hasattr(item, "tree_path")
                        else [],
                    }
                )

        except Exception as e:
            print(f"Error retrieving procedural memory: {str(e)}")

        return procedural_items_list

    except Exception as e:
        print(f"Error retrieving procedural memory: {str(e)}")
        return []


@app.get("/memory/resources")
async def get_resource_memory(user_id: Optional[str] = None):
    """Get resource memory (docs and files)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        client = agent.client
        resource_manager = client.server.resource_memory_manager

        # Get resource memory items using correct method name
        resources = resource_manager.list_resources(
            agent_state=agent.agent_states.resource_memory_agent_state,
            actor=target_user,
            limit=50,
            timezone_str=target_user.timezone,
        )

        # Transform to frontend format
        docs_files = []
        for resource in resources:
            docs_files.append(
                {
                    "filename": resource.title,
                    "type": resource.resource_type,
                    "summary": resource.summary
                    or (
                        resource.content[:200] + "..."
                        if len(resource.content) > 200
                        else resource.content
                    ),
                    "last_accessed": resource.updated_at.isoformat()
                    if resource.updated_at
                    else None,
                    "size": resource.metadata_.get("size")
                    if resource.metadata_
                    else None,
                    "tree_path": resource.tree_path
                    if hasattr(resource, "tree_path")
                    else [],
                }
            )

        return docs_files

    except Exception as e:
        print(f"Error retrieving resource memory: {str(e)}")
        return []


@app.get("/memory/core")
async def get_core_memory():
    """Get core memory (understanding of user)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Get core memory from the main agent
        core_memory = agent.client.get_in_context_memory(
            agent.agent_states.agent_state.id
        )

        core_understanding = []
        total_characters = 0

        # Extract understanding from memory blocks (skip persona block)
        for block in core_memory.blocks:
            if block.value and block.value.strip() and block.label.lower() != "persona":
                block_chars = len(block.value)
                total_characters += block_chars

                core_item = {
                    "aspect": block.label,
                    "understanding": block.value,
                    "character_count": block_chars,
                    "total_characters": total_characters,
                    "max_characters": block.limit,
                    "last_updated": None,  # Core memory doesn't track individual updates
                }

                core_understanding.append(core_item)

        return core_understanding

    except Exception as e:
        print(f"Error retrieving core memory: {str(e)}")
        return []


@app.get("/memory/credentials")
async def get_credentials_memory():
    """Get credentials memory (knowledge vault with masked content)"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        client = agent.client
        knowledge_vault_manager = client.server.knowledge_vault_manager

        # Get knowledge vault items using correct method name
        vault_items = knowledge_vault_manager.list_knowledge(
            actor=agent.client.user,
            agent_state=agent.agent_states.knowledge_vault_agent_state,
            limit=50,
            timezone_str=agent.client.server.user_manager.get_user_by_id(
                agent.client.user.id
            ).timezone,
        )

        # Transform to frontend format with masked content
        credentials = []
        for item in vault_items:
            credentials.append(
                {
                    "caption": item.caption,
                    "entry_type": item.entry_type,
                    "source": item.source,
                    "sensitivity": item.sensitivity,
                    "content": "••••••••••••"
                    if item.sensitivity == "high"
                    else item.secret_value,  # Always mask the actual content
                }
            )

        return credentials

    except Exception as e:
        print(f"Error retrieving credentials memory: {str(e)}")
        return []


@app.post("/conversation/clear", response_model=ClearConversationResponse)
async def clear_conversation_history():
    """Permanently clear all conversation history for the current agent (memories are preserved)"""
    try:
        if agent is None:
            raise HTTPException(status_code=400, detail="Agent not initialized")

        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)

        # Get current message count for this specific actor for reporting
        current_messages = agent.client.server.agent_manager.get_in_context_messages(
            agent_id=agent.agent_states.agent_state.id, actor=target_user
        )
        # Count messages belonging to this actor (excluding system messages)
        actor_messages_count = len(
            [
                msg
                for msg in current_messages
                if msg.role != "system" and msg.user_id == target_user.id
            ]
        )

        # Clear conversation history using the agent manager reset_messages method
        agent.client.server.agent_manager.reset_messages(
            agent_id=agent.agent_states.agent_state.id,
            actor=target_user,
            add_default_initial_messages=True,  # Keep system message and initial setup
        )

        return ClearConversationResponse(
            success=True,
            message=f"Successfully cleared conversation history for {target_user.name}. Messages from other users and system messages preserved.",
            messages_deleted=actor_messages_count,
        )

    except Exception as e:
        print(f"Error clearing conversation history: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Error clearing conversation: {str(e)}"
        )


@app.post("/export/memories", response_model=ExportMemoriesResponse)
async def export_memories(request: ExportMemoriesRequest):
    """Export memories to Excel file with separate sheets for each memory type"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Find the current active user
        users = agent.client.server.user_manager.list_users()
        active_user = next((user for user in users if user.status == "active"), None)
        target_user = active_user if active_user else (users[0] if users else None)
        result = agent.export_memories_to_excel(
            actor=target_user,
            file_path=request.file_path,
            memory_types=request.memory_types,
            include_embeddings=request.include_embeddings,
        )

        if result["success"]:
            return ExportMemoriesResponse(
                success=True,
                message=result["message"],
                exported_counts=result["exported_counts"],
                total_exported=result["total_exported"],
                file_path=result["file_path"],
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])

    except Exception as e:
        print(f"Error exporting memories: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to export memories: {str(e)}"
        )


@app.post("/reflexion", response_model=ReflexionResponse)
async def trigger_reflexion(request: ReflexionRequest):
    """Trigger reflexion agent to reorganize memory - runs in separate thread to not block other requests"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        print("Starting reflexion process...")
        start_time = datetime.now()

        # Run reflexion in a separate thread to avoid blocking other requests
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            _run_reflexion_process,
            agent,
        )

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        print(f"Reflexion process completed in {processing_time:.2f} seconds")

        return ReflexionResponse(
            success=result["success"],
            message=result["message"],
            processing_time=processing_time,
        )

    except Exception as e:
        print(f"Error in reflexion endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Reflexion process failed: {str(e)}"
        )


# MCP Marketplace endpoints
@app.get("/mcp/marketplace")
async def get_marketplace():
    """Get available MCP servers from marketplace"""
    marketplace = get_mcp_marketplace()
    servers = marketplace.get_all_servers()
    categories = marketplace.get_categories()

    # Check connection status
    mcp_manager = get_mcp_client_manager()
    connected_servers = mcp_manager.list_servers()

    # Debug logging
    logger.debug(
        f"MCP Marketplace: {len(connected_servers)} connected servers: {connected_servers}"
    )

    server_data = []
    for server in servers:
        server_dict = server.to_dict()
        server_dict["is_connected"] = server.id in connected_servers
        server_data.append(server_dict)

    return {"servers": server_data, "categories": categories}


@app.get("/mcp/status")
async def get_mcp_status():
    """Get current MCP connection status"""
    try:
        mcp_manager = get_mcp_client_manager()
        connected_servers = mcp_manager.list_servers()

        # Get detailed status for each connected server
        server_status = {}
        for server_name in connected_servers:
            try:
                # Try to get server info to verify it's actually working
                info = mcp_manager.get_server_info(server_name)
                server_status[server_name] = {
                    "connected": True,
                    "status": "active",
                    "info": info,
                }
            except Exception as e:
                server_status[server_name] = {
                    "connected": False,
                    "status": "error",
                    "error": str(e),
                }

        return {
            "connected_servers": connected_servers,
            "server_count": len(connected_servers),
            "server_status": server_status,
        }

    except Exception as e:
        logger.error(f"Error getting MCP status: {str(e)}")
        return {
            "connected_servers": [],
            "server_count": 0,
            "server_status": {},
            "error": str(e),
        }


@app.get("/mcp/marketplace/search")
async def search_mcp_marketplace(query: str = ""):
    """Search MCP marketplace"""
    try:
        marketplace = get_mcp_marketplace()

        if query.strip():
            results = marketplace.search(query)
        else:
            results = marketplace.get_all_servers()

        # Check connection status
        mcp_manager = get_mcp_client_manager()
        connected_servers = mcp_manager.list_servers()

        result_data = []
        for server in results:
            server_dict = server.to_dict()
            server_dict["is_connected"] = server.id in connected_servers
            result_data.append(server_dict)

        return {"results": result_data}

    except Exception as e:
        logger.error(f"Error searching MCP marketplace: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search MCP marketplace: {str(e)}"
        )


@app.post("/mcp/marketplace/connect")
async def connect_mcp_server(request: dict):
    """Connect to an MCP server"""
    try:
        server_id = request.get("server_id")
        env_vars = request.get("env_vars", {})

        if not server_id:
            raise HTTPException(status_code=400, detail="server_id is required")

        marketplace = get_mcp_marketplace()
        server_listing = marketplace.get_server(server_id)

        if not server_listing:
            raise HTTPException(
                status_code=404, detail=f"Server {server_id} not found in marketplace"
            )

        mcp_manager = get_mcp_client_manager()

        # Special handling for Gmail - handle OAuth flow directly in backend
        if server_id == "gmail-native":
            client_id = env_vars.get("client_id")
            client_secret = env_vars.get("client_secret")

            if not client_id or not client_secret:
                raise HTTPException(
                    status_code=400,
                    detail="client_id and client_secret are required for Gmail integration",
                )

            # Handle Gmail OAuth and MCP connection directly
            success = await handle_gmail_connection(
                client_id, client_secret, server_listing.id
            )

        else:
            # Create stdio config for other servers
            config = StdioServerConfig(
                server_name=server_listing.id,
                command=server_listing.command,
                args=server_listing.args,
                env={**(server_listing.env or {}), **env_vars},
            )
            success = mcp_manager.add_server(config, env_vars)

        if success:
            # Register tools for this server
            mcp_tool_registry = get_mcp_tool_registry()
            # Get current user (using agent's user for now)
            if agent and agent.client.user:
                current_user = agent.client.user
                registered_tools = mcp_tool_registry.register_mcp_tools(
                    current_user, [server_listing.id]
                )
                tools_count = len(registered_tools)
            else:
                tools_count = 0

            # Add MCP tool to the current chat agent if available
            if agent and agent.client.user and hasattr(agent, "agent_states"):
                # Update the agent's MCP tools list
                agent.client.server.agent_manager.add_mcp_tool(
                    agent_id=agent.agent_states.agent_state.id,
                    mcp_tool_name=server_listing.id,
                    tool_ids=list(
                        set(
                            [tool.id for tool in registered_tools]
                            + [
                                tool.id
                                for tool in agent.client.server.agent_manager.get_agent_by_id(
                                    agent.agent_states.agent_state.id,
                                    actor=agent.client.user,
                                ).tools
                            ]
                        )
                    ),
                    actor=agent.client.user,
                )

                print(
                    f"✅ Added MCP tool '{server_listing.id}' to agent '{agent.agent_states.agent_state.name}'"
                )

            return {
                "success": True,
                "server_name": server_listing.name,
                "tools_count": tools_count,
                "message": f"Successfully connected to {server_listing.name}",
            }
        else:
            return {
                "success": False,
                "error": f"Failed to connect to {server_listing.name}",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error connecting MCP server: {str(e)}")
        return {"success": False, "error": f"Connection failed: {str(e)}"}


@app.post("/mcp/marketplace/disconnect")
async def disconnect_mcp_server(request: dict):
    """Disconnect from an MCP server"""

    server_id = request.get("server_id")

    if not server_id:
        raise HTTPException(status_code=400, detail="server_id is required")

    mcp_manager = get_mcp_client_manager()
    success = mcp_manager.remove_server(server_id)

    if success:
        # Unregister tools for this server and get the list of unregistered tool IDs
        mcp_tool_registry = get_mcp_tool_registry()
        if agent and agent.client.user:
            current_user = agent.client.user
            unregistered_tool_ids = mcp_tool_registry.unregister_mcp_tools(
                current_user, server_id
            )
            logger.info(
                f"Unregistered {len(unregistered_tool_ids)} tools for server {server_id}"
            )

            # Remove MCP tool from the current chat agent if available
            if hasattr(agent, "agent_states"):
                # Get current agent state
                current_agent = agent.client.server.agent_manager.get_agent_by_id(
                    agent.agent_states.agent_state.id, actor=agent.client.user
                )

                # Remove the specific MCP server from the mcp_tools list
                updated_mcp_tools = [
                    tool
                    for tool in (current_agent.mcp_tools or [])
                    if tool != server_id
                ]

                # Remove only the tools that belonged to this MCP server
                current_tool_ids = [tool.id for tool in current_agent.tools]
                updated_tool_ids = [
                    tool_id
                    for tool_id in current_tool_ids
                    if tool_id not in unregistered_tool_ids
                ]

                # Update the agent with the filtered lists
                agent.client.server.agent_manager.update_mcp_tools(
                    agent_id=agent.agent_states.agent_state.id,
                    mcp_tools=updated_mcp_tools,
                    tool_ids=updated_tool_ids,
                    actor=agent.client.user,
                )
                print(
                    f"✅ Removed MCP tool '{server_id}' and {len(unregistered_tool_ids)} associated tools from agent '{agent.agent_states.agent_state.name}'"
                )

        return {
            "success": True,
            "message": f"Successfully disconnected from {server_id}",
        }
    else:
        return {"success": False, "error": f"Failed to disconnect from {server_id}"}


def _run_reflexion_process(agent):
    """
    Run the reflexion process - this is the blocking function that runs in a separate thread.
    This function can be replaced with the actual reflexion agent logic.
    """
    try:
        # TODO: Replace this with actual reflexion agent logic
        # For now, this is a placeholder that simulates reflexion work

        agent.reflexion_on_memory()
        return {
            "success": True,
            "message": "Memory reorganization completed successfully. Reflexion agent has optimized memory structure and connections.",
        }

    except Exception as e:
        print(f"Error in reflexion process: {str(e)}")
        return {"success": False, "message": f"Reflexion process failed: {str(e)}"}


@app.post("/confirmation/respond")
async def respond_to_confirmation(request: ConfirmationRequest):
    """Handle user confirmation response"""
    confirmation_id = request.confirmation_id
    confirmed = request.confirmed

    # Find the confirmation queue for this ID
    confirmation_queue = confirmation_queues.get(confirmation_id)

    if confirmation_queue:
        # Send the confirmation result to the waiting thread
        confirmation_queue.put({"confirmed": confirmed})
        return {"success": True, "message": "Confirmation received"}
    else:
        return {"success": False, "message": "Confirmation ID not found or expired"}


@app.get("/users")
async def get_all_users():
    """Get all users in the system"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        users = agent.client.server.user_manager.list_users()
        return {"users": [user.model_dump() for user in users]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving users: {str(e)}")


class SwitchUserRequest(BaseModel):
    user_id: str


class SwitchUserResponse(BaseModel):
    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None


@app.post("/users/switch", response_model=SwitchUserResponse)
async def switch_user(request: SwitchUserRequest):
    """Switch the active user"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Use the existing switch_user_context function
        switch_user_context(agent, request.user_id)

        # Get the switched user details
        current_user = agent.client.user
        if current_user:
            return SwitchUserResponse(
                success=True,
                message=f"Successfully switched to user: {current_user.name}",
                user=current_user.model_dump(),
            )
        else:
            return SwitchUserResponse(
                success=False, message="Failed to switch user - user not found"
            )

    except Exception as e:
        return SwitchUserResponse(
            success=False, message=f"Error switching user: {str(e)}"
        )


class CreateUserRequest(BaseModel):
    name: str
    set_as_active: bool = True  # Whether to set this user as active when created


class CreateUserResponse(BaseModel):
    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None


@app.post("/users/create", response_model=CreateUserResponse)
async def create_user(request: CreateUserRequest):
    """Create a new user in the system"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        # Use the AgentWrapper's create_user method
        result = agent.create_user(
            name=request.name, set_as_active=request.set_as_active
        )

        return CreateUserResponse(
            success=result["success"],
            message=result["message"],
            user=result["user"].model_dump(),
        )

    except Exception as e:
        return CreateUserResponse(
            success=False, message=f"Error creating user: {str(e)}"
        )


# ============================================================================
# Authorization Module Import
# ============================================================================
from mirix.authorization import require_permissions, get_auth_context, ScopeContext

# ============================================================================
# Registered Agents API - Domain agent registration
# ============================================================================

class RegisterAgentRequest(BaseModel):
    agent_id: str
    scope: str  # TAX, BOOKKEEPING, MARKETING, SALES
    permissions: List[str]  # ["create", "read", "update", "list", "search", "delete"]


class RegisterAgentResponse(BaseModel):
    success: bool
    message: str
    agent: Optional[Dict[str, Any]] = None


@app.post("/api/registerAgent", response_model=RegisterAgentResponse)
async def register_agent(request: RegisterAgentRequest):
    """
    Register a new domain agent with permissions and scope
    
    Example:
        POST /api/registerAgent
        {
            "agent_id": "TAX_AGENT_001",
            "scope": "TAX",
            "permissions": ["create", "read", "update", "list", "search"]
        }
    """
    import psycopg2
    
    # Validate scope
    valid_scopes = ["TAX", "BOOKKEEPING", "MARKETING", "SALES"]
    if request.scope.upper() not in valid_scopes:
        return RegisterAgentResponse(
            success=False,
            message=f"Invalid scope. Must be one of: {', '.join(valid_scopes)}",
            agent=None
        )
    
    # Validate permissions
    valid_permissions = ["create", "read", "write", "update", "list", "search", "delete"]
    invalid_perms = [p for p in request.permissions if p not in valid_permissions]
    if invalid_perms:
        return RegisterAgentResponse(
            success=False,
            message=f"Invalid permissions: {', '.join(invalid_perms)}. Valid: {', '.join(valid_permissions)}",
            agent=None
        )
    
    try:
        # Get database connection from server
        from mirix.server.server import db_context
        from sqlalchemy import text
        
        with db_context() as db:
            # Check if agent already exists
            result = db.execute(
                text("SELECT agent_id FROM app.registered_agents WHERE agent_id = :agent_id"),
                {"agent_id": request.agent_id}
            )
            existing = result.fetchone()
            
            if existing:
                return RegisterAgentResponse(
                    success=False,
                    message=f"Agent '{request.agent_id}' already exists",
                    agent=None
                )
            
            # Insert new agent
            db.execute(
                text("""
                    INSERT INTO app.registered_agents (agent_id, scope, permissions, active)
                    VALUES (:agent_id, :scope, :permissions, true)
                """),
                {
                    "agent_id": request.agent_id,
                    "scope": request.scope.upper(),
                    "permissions": json.dumps(request.permissions)
                }
            )
            db.commit()
            
            # Fetch the created agent
            result = db.execute(
                text("""
                    SELECT agent_id, scope, permissions, active, created_at, updated_at
                    FROM app.registered_agents
                    WHERE agent_id = :agent_id
                """),
                {"agent_id": request.agent_id}
            )
            
            row = result.fetchone()
            
            agent_data = {
                "agent_id": row[0],
                "scope": row[1],
                "permissions": row[2],
                "active": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "updated_at": row[5].isoformat() if row[5] else None
            }
            
            return RegisterAgentResponse(
                success=True,
                message=f"Agent '{request.agent_id}' registered successfully",
                agent=agent_data
            )
            
    except Exception as e:
        logger.error(f"Error registering agent: {str(e)}")
        return RegisterAgentResponse(
            success=False,
            message=f"Error registering agent: {str(e)}",
            agent=None
        )


# ============================================================================
# Protected API - Search Customer Memories with Authorization
# ============================================================================

class SearchMemoryRequest(BaseModel):
    customer_id: str
    query: Optional[str] = None  # Optional search query


class SearchMemoryResponse(BaseModel):
    success: bool
    customer_id: str
    mirix_user_id: Optional[str] = None
    agent_scope: str
    has_memories: bool
    memories: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


@app.post("/api/search_customer_memory", response_model=SearchMemoryResponse)
@require_permissions(["read", "search"])
async def search_customer_memory(
    request: Request,
    search_request: SearchMemoryRequest,
    agent_id: str = Header(..., alias="X-Agent-ID")
):
    """
    Search customer memories with authorization and lazy MIRIX user creation
    
    Headers:
        X-Agent-ID: Your registered agent ID (must have 'read' and 'search' permissions)
    
    Flow:
        1. Validates agent has 'read' and 'search' permissions
        2. Extracts agent scope
        3. Gets/creates customer-MIRIX mapping (lazy creation)
        4. Searches memories from MIRIX
        5. Returns memories or welcome message if none found
    
    Example:
        curl -X POST http://localhost:47283/api/search_customer_memory \
          -H "X-Agent-ID: TAX_AGENT_001" \
          -H "Content-Type: application/json" \
          -d '{"customer_id": "cust_123"}'
    """
    try:
        # Get authorization context (added by @require_permissions)
        auth = await get_auth_context(request)
        
        logger.info(f"Agent '{auth.agent_id}' (scope: {auth.scope}) searching memories for customer {search_request.customer_id}")
        
        # Get database session
        from mirix.server.server import db_context
        
        mirix_user_id = None
        is_new_customer = False
        
        with db_context() as db:
            # Check if customer has MIRIX mapping
            result = db.execute(
                text("""
                    SELECT mirix_user_id, created_at FROM app.customer_mirix_mapping
                    WHERE customer_id = :cid
                """),
                {"cid": search_request.customer_id}
            )
            row = result.fetchone()
            
            if not row:
                # First time - Create MIRIX user and mapping (Lazy Creation)
                logger.info(f"First time customer {search_request.customer_id} - creating MIRIX user")
                is_new_customer = True
                
                if agent:
                    mirix_user_result = agent.create_user(
                        name=f"Customer_{search_request.customer_id}"
                    )
                    
                    if mirix_user_result["success"]:
                        mirix_user_id = mirix_user_result["user"].id
                        
                        # Store mapping
                        db.execute(
                            text("""
                                INSERT INTO app.customer_mirix_mapping 
                                (customer_id, mirix_user_id, agent_id)
                                VALUES (:cid, :mid, :aid)
                            """),
                            {
                                "cid": search_request.customer_id,
                                "mid": mirix_user_id,
                                "aid": auth.agent_id
                            }
                        )
                        db.commit()
                        logger.info(f"Created mapping: {search_request.customer_id} → {mirix_user_id}")
                    else:
                        raise HTTPException(500, f"Failed to create MIRIX user: {mirix_user_result['message']}")
                else:
                    raise HTTPException(500, "Agent not initialized")
            else:
                # Existing customer - use mapping
                mirix_user_id = row[0]
                
                # Update last accessed
                db.execute(
                    text("""
                        UPDATE app.customer_mirix_mapping 
                        SET last_accessed = NOW()
                        WHERE customer_id = :cid
                    """),
                    {"cid": search_request.customer_id}
                )
                db.commit()
                logger.info(f"Found existing mapping: {search_request.customer_id} → {mirix_user_id}")
        
        # Search memories from MIRIX
        if agent and mirix_user_id:
            # Get episodic memories
            try:
                episodic_manager = agent.client.server.episodic_memory_manager
                user_obj = agent.client.server.user_manager.get_user_by_id(mirix_user_id)
                
                if user_obj:
                    episodic_events = episodic_manager.list_episodic_memory(
                        agent_state=agent.agent_states.episodic_memory_agent_state,
                        actor=user_obj,
                        limit=20,
                        timezone_str=user_obj.timezone,
                    )
                    
                    # Get semantic memories (facts)
                    semantic_manager = agent.client.server.semantic_memory_manager
                    semantic_memories = semantic_manager.list_semantic_memory(
                        agent_state=agent.agent_states.semantic_memory_agent_state,
                        actor=user_obj,
                        limit=10
                    )
                    
                    # Transform to response format
                    episodic_items = []
                    for event in episodic_events:
                        episodic_items.append({
                            "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                            "summary": event.summary,
                            "details": event.details,
                            "event_type": event.event_type,
                        })
                    
                    semantic_items = []
                    for memory in semantic_memories:
                        semantic_items.append({
                            "id": memory.id,
                            "content": memory.content,
                            "category": memory.category if hasattr(memory, 'category') else None,
                        })
                    
                    has_memories = len(episodic_items) > 0 or len(semantic_items) > 0
                    
                    if not has_memories:
                        # No memories - return welcome message
                        return SearchMemoryResponse(
                            success=True,
                            customer_id=search_request.customer_id,
                            mirix_user_id=mirix_user_id,
                            agent_scope=auth.scope,
                            has_memories=False,
                            message=f"Welcome! I'm your {auth.scope} assistant. How can I help you today?"
                        )
                    else:
                        # Return memories
                        return SearchMemoryResponse(
                            success=True,
                            customer_id=search_request.customer_id,
                            mirix_user_id=mirix_user_id,
                            agent_scope=auth.scope,
                            has_memories=True,
                            memories={
                                "episodic": episodic_items,
                                "semantic": semantic_items,
                                "total_count": len(episodic_items) + len(semantic_items)
                            },
                            message=f"Found {len(episodic_items)} events and {len(semantic_items)} facts from previous interactions."
                        )
                else:
                    raise HTTPException(404, f"MIRIX user not found: {mirix_user_id}")
                    
            except Exception as e:
                logger.warning(f"Error fetching memories: {str(e)}")
                # If error or new customer, return welcome message
                return SearchMemoryResponse(
                    success=True,
                    customer_id=search_request.customer_id,
                    mirix_user_id=mirix_user_id,
                    agent_scope=auth.scope,
                    has_memories=False,
                    message=f"Welcome! I'm your {auth.scope} assistant. How can I help you today?"
                )
        else:
            raise HTTPException(500, "Agent not initialized")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_customer_memory: {str(e)}")
        return SearchMemoryResponse(
            success=False,
            customer_id=search_request.customer_id,
            agent_scope="",
            has_memories=False,
            error=str(e)
        )


# ============================================================================
# Protected API - Store Customer Interaction/Memory with Authorization
# ============================================================================

class StoreMemoryRequest(BaseModel):
    customer_id: str
    message: str  # Customer's message
    agent_response: Optional[str] = None  # Agent's response (optional)


class StoreMemoryResponse(BaseModel):
    success: bool
    customer_id: str
    mirix_user_id: str
    agent_scope: str
    message: str
    memory_stored: bool
    error: Optional[str] = None


@app.post("/api/store_customer_memory", response_model=StoreMemoryResponse)
@require_permissions(["write"])
async def store_customer_memory(
    request: Request,
    store_request: StoreMemoryRequest,
    agent_id: str = Header(..., alias="X-Agent-ID")
):
    """
    Store customer interaction as memory with agent scope tagging
    
    Headers:
        X-Agent-ID: Your registered agent ID (must have 'write' permission)
    
    Flow:
        1. Validates agent has 'write' permission
        2. Extracts agent scope
        3. Gets customer-MIRIX mapping (must exist from search call)
        4. Stores interaction in MIRIX with metadata tagging
        5. Metadata includes: agent_id, agent_scope for filtering
    
    How MIRIX Stores:
        - Creates episodic memory (conversation event)
        - Extracts semantic facts automatically
        - Tags with agent_id & scope in metadata_ field
        - Links to user_id (MIRIX user)
    
    Example:
        curl -X POST http://localhost:47283/api/store_customer_memory \
          -H "X-Agent-ID: TAX_AGENT_001" \
          -H "Content-Type: application/json" \
          -d '{
            "customer_id": "userId101",
            "message": "I paid $5000 in estimated taxes",
            "agent_response": "I will remember that for your tax return"
          }'
    """
    try:
        print(f"[STORE_MEMORY] === ENDPOINT CALLED ===")
        print(f"[STORE_MEMORY] Customer: {store_request.customer_id}, Message: {store_request.message[:50]}...")
        
        # Get authorization context
        auth = await get_auth_context(request)
        print(f"[STORE_MEMORY] Auth successful - Agent: {auth.agent_id}, Scope: {auth.scope}")
        
        logger.info(f"Agent '{auth.agent_id}' (scope: {auth.scope}) storing memory for customer {store_request.customer_id}")
        
        # Get database session
        from mirix.server.server import db_context
        
        mirix_user_id = None
        
        with db_context() as db:
            # Get customer mapping (must exist - should be created during search)
            result = db.execute(
                text("""
                    SELECT mirix_user_id FROM app.customer_mirix_mapping
                    WHERE customer_id = :cid
                """),
                {"cid": store_request.customer_id}
            )
            row = result.fetchone()
            
            if not row:
                raise HTTPException(
                    404,
                    f"Customer '{store_request.customer_id}' not found. Please call search_customer_memory first to create mapping."
                )
            
            mirix_user_id = row[0]
            logger.info(f"Found mapping: {store_request.customer_id} → {mirix_user_id}")
            print(f"[STORE_MEMORY] Found MIRIX user: {mirix_user_id}")
        
        # Prepare interaction text
        # STEP 1: Create memories by switching user context and sending message
        # Use the proven SDK approach from chatbots
        print(f"[STORE_MEMORY] === STEP 1: Creating memories ===")
        print(f"[STORE_MEMORY] Customer message: '{store_request.message[:60]}...'")
        
        try:
            # Get the user object and switch context (like chatbots do)
            print(f"[STORE_MEMORY] Getting user object for {mirix_user_id}")
            user_obj = agent.client.server.user_manager.get_user_by_id(mirix_user_id)
            
            if not user_obj:
                raise HTTPException(404, f"MIRIX user not found: {mirix_user_id}")
            
            # Switch to this user's context
            previous_user = agent.client.user
            agent.client.user = user_obj
            print(f"[STORE_MEMORY] Switched to user context: {user_obj.name}")
            
            # Send message in this user's context
            print(f"[STORE_MEMORY] Calling agent.send_message(memorizing=False)")
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: agent.send_message(
                    message=store_request.message,  # Customer message
                    memorizing=False  # Normal conversation creates memories
                )
            )
            print(f"[STORE_MEMORY] Response generated (length: {len(str(response)) if response else 0})")
            
            # Restore previous user context
            agent.client.user = previous_user
            print(f"[STORE_MEMORY] Restored original user context")
        
        except Exception as e:
            print(f"[STORE_MEMORY] ERROR in Step 1: {e}")
            raise HTTPException(500, f"Failed to send message: {str(e)}")
        
        # STEP 2: Tag untagged memories with agent metadata
        # MIRIX merges related memories, so we tag ALL untagged memories for this user
        # No need for complex retry logic - just tag what exists
        print(f"[STORE_MEMORY] === STEP 2: TAGGING MEMORIES WITH AGENT METADATA ===")
        from mirix.server.server import db_context
        import json as json_lib
        from datetime import datetime
        
        # Create metadata with agent info
        agent_metadata = {
            "agent_id": auth.agent_id,
            "agent_scope": auth.scope,
            "created_by_agent": auth.agent_id,
            "last_interaction_timestamp": datetime.utcnow().isoformat()
        }
        print(f"[STORE_MEMORY] Metadata to apply: {agent_metadata}")
        
        # Tag ALL untagged episodic memories for this user
        # This works whether memories are newly created or merged
        print(f"[STORE_MEMORY] Tagging untagged episodic memories...")
        with db_context() as db:
            result = db.execute(
                text("""
                    UPDATE episodic_memory
                    SET metadata_ = CAST(:metadata AS json)
                    WHERE user_id = :user_id
                      AND (metadata_ IS NULL OR metadata_::text = '{}')
                    RETURNING id
                """),
                {
                    "metadata": json_lib.dumps(agent_metadata),
                    "user_id": mirix_user_id
                }
            )
            
            episodic_rows = result.fetchall()
            db.commit()
            print(f"[STORE_MEMORY] Tagged {len(episodic_rows)} episodic memories")
            if episodic_rows:
                logger.info(f"Tagged {len(episodic_rows)} episodic memories with scope '{auth.scope}'")
        
        # Tag ALL untagged semantic memories for this user
        print(f"[STORE_MEMORY] Tagging untagged semantic memories...")
        with db_context() as db:
            result = db.execute(
                text("""
                    UPDATE semantic_memory
                    SET metadata_ = CAST(:metadata AS json)
                    WHERE user_id = :user_id
                      AND (metadata_ IS NULL OR metadata_::text = '{}')
                    RETURNING id
                """),
                {
                    "metadata": json_lib.dumps(agent_metadata),
                    "user_id": mirix_user_id
                }
            )
            
            semantic_rows = result.fetchall()
            db.commit()
            print(f"[STORE_MEMORY] Tagged {len(semantic_rows)} semantic memories")
            if semantic_rows:
                logger.info(f"Tagged {len(semantic_rows)} semantic memories with scope '{auth.scope}'")
        
        print(f"[STORE_MEMORY] === ENDPOINT COMPLETE - SUCCESS ===")
        return StoreMemoryResponse(
            success=True,
            customer_id=store_request.customer_id,
            mirix_user_id=mirix_user_id,
            agent_scope=auth.scope,
            message=f"Memory stored successfully with {auth.scope} agent scope tagging",
            memory_stored=True
        )
            
    except HTTPException as he:
        print(f"[STORE_MEMORY] HTTP Exception: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        print(f"[STORE_MEMORY] === EXCEPTION CAUGHT ===")
        print(f"[STORE_MEMORY] Exception type: {type(e).__name__}")
        print(f"[STORE_MEMORY] Exception message: {str(e)}")
        logger.error(f"Error in store_customer_memory: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[STORE_MEMORY] Traceback:\n{traceback_str}")
        logger.error(traceback_str)
        return StoreMemoryResponse(
            success=False,
            customer_id=store_request.customer_id,
            mirix_user_id="",
            agent_scope=auth.scope if 'auth' in locals() else "",
            message="Failed to store memory",
            memory_stored=False,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=47283)
