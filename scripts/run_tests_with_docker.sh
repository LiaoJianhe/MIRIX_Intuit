#!/bin/bash
# ============================================================================
# Run tests with Docker/Podman Compose test infrastructure
# ============================================================================
# This script:
#   1. Starts ephemeral PostgreSQL and Redis containers
#   2. Runs pytest with test database configuration
#   3. Stops containers when done (even on failure)
#
# Usage:
#   ./scripts/run_tests_with_docker.sh                    # Run all tests (uses Docker)
#   ./scripts/run_tests_with_docker.sh --podman           # Use Podman instead
#   ./scripts/run_tests_with_docker.sh test_raw_memory.py  # Run specific test file
#   ./scripts/run_tests_with_docker.sh -k "search"        # Run tests matching pattern
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.test.yml"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command-line arguments
USE_PODMAN="0"  # Default to Docker
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --podman)
            USE_PODMAN="1"
            shift
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set up container runtime commands
if [ "$USE_PODMAN" = "1" ]; then
    CONTAINER_RUNTIME="podman"
    # Check for podman-compose (standalone) first - it's more reliable
    if command -v podman-compose &> /dev/null; then
        COMPOSE_CMD="podman-compose"
        COMPOSE_CMD_ARGS=("podman-compose")
        echo -e "${BLUE}Using Podman (podman-compose)${NC}"
    elif command -v podman &> /dev/null; then
        # Check if podman has compose subcommand
        # If it returns "unrecognized command", the plugin doesn't exist
        # Connection errors are OK - we'll start Podman when needed
        # Use || true to prevent set -e from exiting on error
        COMPOSE_CHECK=$(podman compose --help 2>&1 || true)
        if echo "$COMPOSE_CHECK" | grep -q "unrecognized command\|invalid command"; then
            echo -e "${RED}Error: podman found but compose plugin not available.${NC}"
            echo -e "${YELLOW}Please install podman-compose (recommended):${NC}"
            echo -e "  pip install podman-compose"
            echo -e "${YELLOW}Or install docker-compose for podman compose plugin:${NC}"
            echo -e "  brew install docker-compose"
            exit 1
        elif echo "$COMPOSE_CHECK" | grep -q "looking up compose provider failed\|executable file not found"; then
            echo -e "${RED}Error: podman compose requires docker-compose but it's not found.${NC}"
            echo -e "${YELLOW}Please install podman-compose (recommended):${NC}"
            echo -e "  pip install podman-compose"
            echo -e "${YELLOW}Or install docker-compose:${NC}"
            echo -e "  brew install docker-compose"
            exit 1
        else
            # Plugin exists (connection errors are fine)
            COMPOSE_CMD="podman compose"
            COMPOSE_CMD_ARGS=("podman" "compose")
            echo -e "${BLUE}Using Podman (podman compose plugin)${NC}"
        fi
    else
        echo -e "${RED}Error: podman not found. Please install podman.${NC}"
        exit 1
    fi
else
    CONTAINER_RUNTIME="docker"
    # Check for docker-compose (standalone) or docker compose (plugin)
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        COMPOSE_CMD_ARGS=("docker-compose")
    elif command -v docker &> /dev/null; then
        # Check if docker has compose subcommand
        if docker compose --help &> /dev/null 2>&1 || docker compose version &> /dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
            COMPOSE_CMD_ARGS=("docker" "compose")
        else
            echo -e "${RED}Error: docker found but compose plugin not available.${NC}"
            echo -e "${YELLOW}Please install docker-compose or Docker with compose plugin.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: docker not found. Please install docker.${NC}"
        exit 1
    fi
    echo -e "${BLUE}Using Docker${NC}"
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up test infrastructure...${NC}"
    cd "$PROJECT_ROOT"
    "${COMPOSE_CMD_ARGS[@]}" -f "$COMPOSE_FILE" down 2>/dev/null || true
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Check if container runtime is running/available
if [ "$USE_PODMAN" = "1" ]; then
    if ! $CONTAINER_RUNTIME info &> /dev/null; then
        echo -e "${YELLOW}Warning: Podman info check failed, but continuing...${NC}"
        echo -e "${YELLOW}Note: Podman may work without root, continuing anyway${NC}"
    fi
else
    if ! $CONTAINER_RUNTIME info &> /dev/null; then
        echo -e "${RED}Error: Docker is not running. Please start Docker.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Starting test infrastructure...${NC}"
cd "$PROJECT_ROOT"

# Start test infrastructure - show output in real-time
echo -e "${YELLOW}Running: ${COMPOSE_CMD} -f docker-compose.test.yml up -d${NC}"
echo -e "${YELLOW}(This may take a moment to pull images if needed...)${NC}"

# Run the command and show output in real-time
if ! "${COMPOSE_CMD_ARGS[@]}" -f "$COMPOSE_FILE" up -d; then
    COMPOSE_EXIT_CODE=$?
    if [ "$USE_PODMAN" = "1" ]; then
        echo -e "${RED}Error: Failed to start Podman containers (exit code: $COMPOSE_EXIT_CODE).${NC}"
        echo -e "${YELLOW}Common issues:${NC}"
        echo -e "  - podman compose requires docker-compose. Install podman-compose instead:"
        echo -e "    pip install podman-compose"
        echo -e "  - Podman machine not running. Try:"
        echo -e "    podman machine start"
        echo -e "  - Check podman status:"
        echo -e "    podman machine list"
    else
        echo -e "${RED}Error: Failed to start Docker containers (exit code: $COMPOSE_EXIT_CODE).${NC}"
        echo -e "${YELLOW}Make sure Docker is running.${NC}"
    fi
    exit 1
fi

echo -e "${GREEN}✓ Containers started successfully${NC}"

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for test database to be ready...${NC}"
timeout=60  # Increased timeout - database may need more time to initialize
elapsed=0

# Determine how to check container status based on compose tool
if [ "$USE_PODMAN" = "1" ] && [ "$COMPOSE_CMD" = "podman-compose" ]; then
    # podman-compose doesn't support filtering by service name, use podman directly
    CHECK_CMD=("podman" "ps" "--filter" "name=mirix_test_db" "--format" "{{.Names}}\t{{.Status}}")
else
    # docker-compose supports filtering
    CHECK_CMD=("${COMPOSE_CMD_ARGS[@]}" "-f" "$COMPOSE_FILE" "ps" "test_db")
fi

while [ $elapsed -lt $timeout ]; do
    # Show status every 5 seconds
    if [ $((elapsed % 5)) -eq 0 ] && [ $elapsed -gt 0 ]; then
        echo -e "${YELLOW}  Still waiting... (${elapsed}s elapsed)${NC}"
        # Show container status
        echo -e "${YELLOW}  Container status:${NC}"
        if [ "$USE_PODMAN" = "1" ] && [ "$COMPOSE_CMD" = "podman-compose" ]; then
            podman ps --filter "name=mirix_test" --format "table {{.Names}}\t{{.Status}}" 2>&1 || true
        else
            "${COMPOSE_CMD_ARGS[@]}" -f "$COMPOSE_FILE" ps 2>&1 | grep -E "(NAME|test_db)" || true
        fi
    fi
    
    # Check if container is healthy
    DB_STATUS=$("${CHECK_CMD[@]}" 2>&1)
    
    # Check for healthy status (works for both docker-compose and podman)
    if echo "$DB_STATUS" | grep -qi "healthy"; then
        echo -e "${GREEN}✓ Test database is ready!${NC}"
        break
    fi
    
    # Also check if container is running (even if not healthy yet)
    if echo "$DB_STATUS" | grep -qiE "Up|running|starting"; then
        # Container is running, just waiting for health check
        :
    elif echo "$DB_STATUS" | grep -qiE "Exited|stopped|dead"; then
        echo -e "${RED}Error: Database container has stopped!${NC}"
        echo -e "${YELLOW}Container logs:${NC}"
        "${COMPOSE_CMD_ARGS[@]}" -f "$COMPOSE_FILE" logs test_db 2>&1 | tail -30 || true
        exit 1
    elif [ -z "$DB_STATUS" ] || echo "$DB_STATUS" | grep -qiE "error|cannot connect"; then
        # Container might not exist yet or connection issue
        if [ $elapsed -gt 10 ]; then
            echo -e "${YELLOW}  Warning: Cannot get container status, but continuing to wait...${NC}"
        fi
    fi
    
    sleep 1
    elapsed=$((elapsed + 1))
done

if [ $elapsed -eq $timeout ]; then
    echo -e "${RED}Error: Test database failed to become healthy after ${timeout} seconds${NC}"
    echo -e "${YELLOW}Container status:${NC}"
    if [ "$USE_PODMAN" = "1" ] && [ "$COMPOSE_CMD" = "podman-compose" ]; then
        podman ps --filter "name=mirix_test" --format "table {{.Names}}\t{{.Status}}\t{{.Health}}" 2>&1 || true
    else
        "${COMPOSE_CMD_ARGS[@]}" -f "$COMPOSE_FILE" ps 2>&1 || true
    fi
    echo -e "${YELLOW}Container logs (last 30 lines):${NC}"
    "${COMPOSE_CMD_ARGS[@]}" -f "$COMPOSE_FILE" logs test_db 2>&1 | tail -30 || true
    echo -e "${YELLOW}Attempting to check database connectivity manually:${NC}"
    # Try to connect to database directly
    if command -v psql &> /dev/null; then
        PGPASSWORD=test psql -h localhost -p 5433 -U test -d mirix_test -c "SELECT 1;" 2>&1 | head -5 || echo "  psql connection failed"
    else
        echo "  psql not available for manual check"
    fi
    exit 1
fi

# Set environment variables for test database
export MIRIX_PG_URI="postgresql+pg8000://test:test@localhost:5433/mirix_test"
export MIRIX_REDIS_ENABLED="true"
export MIRIX_REDIS_HOST="localhost"
export MIRIX_REDIS_PORT="6380"
# Disable LangFuse during tests to avoid logging errors
export MIRIX_LANGFUSE_ENABLED="false"

# Initialize database schema (tables are created automatically on first connection)
# This is optional - schema will be created when tests connect to the database
echo -e "${YELLOW}Initializing database schema (if needed)...${NC}"
cd "$PROJECT_ROOT"
if python3 -c "
import sys
sys.path.insert(0, '.')
from mirix.server.server import SyncServer
# Create server instance which initializes database schema
server = SyncServer(init_with_default_org_and_user=False)
print('✓ Database schema initialized')
" 2>/dev/null; then
    echo -e "${GREEN}Schema initialization complete${NC}"
else
    echo -e "${YELLOW}Note: Schema will be created automatically on first database connection${NC}"
fi

# Run pytest with provided arguments
echo -e "${GREEN}Running tests...${NC}"
cd "$PROJECT_ROOT"

# Use poetry run pytest if poetry is available, otherwise use pytest directly
if command -v poetry &> /dev/null && [ -f "pyproject.toml" ]; then
    PYTEST_CMD=("poetry" "run" "pytest")
    echo -e "${YELLOW}Using poetry to run pytest${NC}"
else
    PYTEST_CMD=("pytest")
    echo -e "${YELLOW}Using system pytest${NC}"
fi

if [ ${#PYTEST_ARGS[@]} -eq 0 ]; then
    # No arguments - run all tests
    "${PYTEST_CMD[@]}" tests/ -v
else
    # Pass all arguments to pytest
    "${PYTEST_CMD[@]}" "${PYTEST_ARGS[@]}"
fi

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✓ Tests passed!${NC}"
else
    echo -e "\n${RED}✗ Tests failed with exit code $TEST_EXIT_CODE${NC}"
fi

exit $TEST_EXIT_CODE
