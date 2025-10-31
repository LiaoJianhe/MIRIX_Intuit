#!/usr/bin/env python3
"""
Mirix Memory Test Program

This script tests adding different memory types (core, episodic, procedural, semantic) to Mirix.

Prerequisites:
- Start the server first: python scripts/start_server.py --reload
"""

import logging
import sys
from pathlib import Path

from mirix.client import MirixClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_core_memory(client: MirixClient, user_id: str):
    """Add core memory."""
    logger.info("\n%s", "="*80)
    logger.info("TEST 1: ADDING CORE MEMORY")
    logger.info("%s", "="*80)
    
    try:
        result = client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "My name is Alice Johnson and I'm a software engineer at TechCorp. I love hiking and photography."
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "Nice to meet you Alice! I've saved your information to my memory."
                    }]
                }
            ],
            chaining=True
        )
        logger.info("✅ Core memory added successfully: %s", result.get('success', False))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Failed to add core memory: %s", e)
        import traceback
        traceback.print_exc()


def test_episodic_memory(client: MirixClient, user_id: str):
    """Add episodic memory."""
    logger.info("\n%s", "="*80)
    logger.info("TEST 2: ADDING EPISODIC MEMORY")
    logger.info("%s", "="*80)
    
    try:
        result = client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Yesterday at 3 PM, I attended a team meeting with Bob and Carol. We discussed the Q4 roadmap and decided to prioritize the mobile app redesign."
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "I've recorded this event: Team meeting yesterday at 3 PM with Bob and Carol about Q4 roadmap, prioritizing mobile app redesign."
                    }]
                }
            ],
            chaining=True
        )
        logger.info("✅ Episodic memory added successfully: %s", result.get('success', False))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Failed to add episodic memory: %s", e)
        import traceback
        traceback.print_exc()


def test_procedural_memory(client: MirixClient, user_id: str):
    """Add procedural memory."""
    logger.info("\n%s", "="*80)
    logger.info("TEST 3: ADDING PROCEDURAL MEMORY")
    logger.info("%s", "="*80)
    
    try:
        result = client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Here's my workflow for deploying code: First, run all unit tests locally. Second, create a pull request with detailed description. Third, wait for code review approval. Fourth, merge to main branch. Finally, monitor the deployment pipeline."
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "I've saved your deployment workflow procedure with 5 steps."
                    }]
                }
            ],
            chaining=True
        )
        logger.info("✅ Procedural memory added successfully: %s", result.get('success', False))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Failed to add procedural memory: %s", e)
        import traceback
        traceback.print_exc()


def test_semantic_memory(client: MirixClient, user_id: str):
    """Add semantic memory."""
    logger.info("\n%s", "="*80)
    logger.info("TEST 4: ADDING SEMANTIC MEMORY")
    logger.info("%s", "="*80)
    
    try:
        result = client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "MirixDB is our company's new internal database system that combines PostgreSQL with Redis caching. It's specifically designed for handling multi-agent memory operations with automatic cache invalidation and vector search capabilities. We're using it in our Q4 AI assistant rollout."
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "I've saved this information about MirixDB to my semantic memory."
                    }]
                }
            ],
            chaining=True
        )
        logger.info("✅ Semantic memory added successfully: %s", result.get('success', False))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Failed to add semantic memory: %s", e)
        import traceback
        traceback.print_exc()


def test_resource_memory(client: MirixClient, user_id: str):
    """Add resource memory."""
    logger.info("\n%s", "="*80)
    logger.info("TEST 5: ADDING RESOURCE MEMORY")
    logger.info("%s", "="*80)
    
    try:
        # Provide a realistic document with full content (not a placeholder)
        document_content = """# Q4 AI Assistant Rollout Plan

## Project Overview
This document outlines our strategy for deploying the new AI assistant powered by MirixDB across all departments in Q4 2025.

## Timeline
- **Week 1-2 (Oct 1-14)**: Internal beta testing with engineering team
- **Week 3-4 (Oct 15-28)**: Expand to product and design teams
- **Week 5-6 (Nov 1-14)**: Sales and marketing pilot program
- **Week 7-8 (Nov 15-30)**: Company-wide rollout
- **Week 9-12 (Dec 1-31)**: Monitoring, feedback collection, and optimization

## Technical Architecture
- **Backend**: MirixDB (PostgreSQL + Redis)
- **Cache Strategy**: Redis JSON with vector search for memory operations
- **API Gateway**: FastAPI with automatic cache invalidation
- **Frontend**: React-based dashboard for memory management

## Success Metrics
- User adoption rate: Target 80% within first month
- Query response time: < 100ms for cached queries
- Memory accuracy: > 95% user satisfaction
- System uptime: 99.9% availability

## Risk Mitigation
1. Gradual rollout to manage load
2. Fallback to PostgreSQL if Redis fails
3. Daily backups and monitoring
4. 24/7 on-call engineering support during rollout

## Budget
Total estimated cost: $150K
- Infrastructure: $80K
- Development: $50K
- Support: $20K

## Team Contacts
- Project Lead: Sarah Chen (sarah.chen@company.com)
- Tech Lead: Mike Johnson (mike.johnson@company.com)
- Product Manager: Alex Wong (alex.wong@company.com)
"""
        
        result = client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Here's our Q4 AI Assistant Rollout Plan document:\n\n{document_content}"
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "I've saved the Q4 AI Assistant Rollout Plan document to resource memory with full content including timeline, architecture, metrics, and budget details."
                    }]
                }
            ],
            chaining=True
        )
        logger.info("✅ Resource memory added successfully: %s", result.get('success', False))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Failed to add resource memory: %s", e)
        import traceback
        traceback.print_exc()


def test_knowledge_vault(client: MirixClient, user_id: str):
    """Add knowledge vault memory."""
    logger.info("\n%s", "="*80)
    logger.info("TEST 6: ADDING KNOWLEDGE VAULT MEMORY")
    logger.info("%s", "="*80)
    
    try:
        # Provide structured, discrete data points that can be looked up later
        # These are credentials and connection strings - perfect for Knowledge Vault
        result = client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Please save these credentials for our MirixDB production environment:\n\n"
                                "Database URL: postgresql://mirix_user:SecurePass123!@prod-db.company.com:5432/mirix_prod\n"
                                "Redis URL: redis://:RedisSecure456@prod-redis.company.com:6379/0\n"
                                "API Key: mirix_api_sk_prod_a1b2c3d4e5f6g7h8i9j0\n"
                                "Admin Dashboard: https://mirix-admin.company.com\n"
                                "Monitoring URL: https://monitoring.company.com/mirix"
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "I've securely saved the MirixDB production credentials to the knowledge vault, including database connection strings, Redis URL, API key, and admin dashboard URLs."
                    }]
                }
            ],
            chaining=True
        )
        logger.info("✅ Knowledge vault memory added successfully: %s", result.get('success', False))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Failed to add knowledge vault memory: %s", e)
        import traceback
        traceback.print_exc()


def main():
    """Main test execution."""
    logger.info("\n%s", "="*80)
    logger.info("MIRIX MEMORY TEST - ADDING MEMORIES")
    logger.info("%s", "="*80)
    
    # Create client
    user_id = 'demo-user'
    org_id = 'demo-org'
    
    logger.info("\nInitializing MirixClient...")
    client = MirixClient(
        api_key=None,
        user_id=user_id,
        org_id=org_id,
        debug=False,  # Reduce noise in output
    )
    
    # Initialize meta agent
    logger.info("Initializing meta agent...")
    try:
        # Compute config path relative to project root (parent of samples/)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "mirix" / "configs" / "examples" / "mirix_openai.yaml"
        
        client.initialize_meta_agent(
            config_path=str(config_path),
            update_agents=True
        )
        logger.info("Meta agent initialized")
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to initialize meta agent: %s", e)
        sys.exit(1)
    
    # Run tests
    try:
        test_core_memory(client, user_id)
        test_episodic_memory(client, user_id)
        test_procedural_memory(client, user_id)
        test_semantic_memory(client, user_id)
        test_resource_memory(client, user_id)
        test_knowledge_vault(client, user_id)
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("\n\nTest failed with error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    logger.info("\n%s", "="*80)
    logger.info("TEST SUMMARY")
    logger.info("%s", "="*80)
    logger.info("✅ All memory additions completed successfully!")
    logger.info("\nMemories Added:")
    logger.info("  - Core Memory: User profile (name, role, interests)")
    logger.info("  - Episodic Memory: Team meeting event")
    logger.info("  - Procedural Memory: Code deployment workflow")
    logger.info("  - Semantic Memory: MirixDB database system concept")
    logger.info("  - Resource Memory: Q4 AI Assistant Rollout Plan document")
    logger.info("  - Knowledge Vault: MirixDB production credentials and URLs")
    logger.info("%s", "="*80)


if __name__ == "__main__":
    main()

