import os
import logging
import yaml
from flow import coding_agent_flow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coding_agent.log')
    ]
)

logger = logging.getLogger('main')

def load_prompt_from_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)
            if not prompt_data or 'example' not in prompt_data:
                raise ValueError("Invalid prompt file format: missing 'example' field")
            return prompt_data['example']
    except Exception as e:
        logger.error(f"Error loading prompt file: {str(e)}")
        raise

def run_flow(query: str = None, working_dir: str = None) -> None:
    # Set default working directory if not provided
    if working_dir is None:
        working_dir = os.path.join(os.getcwd(), "project")
    
    # Create working directory if it doesn't exist
    if not os.path.exists(working_dir):
        try:
            os.makedirs(working_dir)
            logger.info(f"Created working directory: {working_dir}")
        except Exception as e:
            logger.error(f"Failed to create working directory: {str(e)}")
            raise
    
    # If no query provided, ask for it
    if not query:
        query = input("What would you like me to help you with? ")
    
    # Initialize shared memory
    shared = {
        "user_query": query,
        "working_dir": working_dir,
        "history": [],
        "response": None
    }
    
    logger.info(f"Working directory: {working_dir}")
    
    # Run the flow
    coding_agent_flow.run(shared)

if __name__ == "__main__":
    # Load the prompt files, grouped by category
    prompt_files = [
        # Create category
        "prompts/create/web_apps/create_rest_api.yaml",      # REST API backend
        "prompts/create/web_apps/create_react_frontend.yaml", # React frontend
        "prompts/create/web_apps/create_react_app_structure.yaml", # React app structure
        "prompts/create/documentation/create_project_readme.yaml", # Project documentation
        "prompts/create/documentation/create_requirements.yaml",   # Dependencies
        "prompts/create/devops/create_docker_ci.yaml",       # Docker & CI/CD
        
        # Modify category
        "prompts/modify/refactoring/improve_code_quality.yaml", # Code quality
        "prompts/modify/testing/add_test_coverage.yaml",     # Testing
        
        # Review category
        "prompts/review/security/security_audit.yaml"        # Security audit
    ]
    try:
        query = load_prompt_from_file(prompt_files[1])
        logger.info("Successfully loaded prompt from file")
    except Exception as e:
        logger.error(f"Failed to load prompt: {str(e)}")
        query = ""
    
    working_dir = "project"
    run_flow(query=query, working_dir=working_dir)