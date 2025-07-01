import os
import sys
import logging

# Add project/ to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)
print("Project directory added:", project_dir)

# Debug utils import
try:
    import utils
    print("utils package found at:", utils.__file__)
except ImportError as e:
    print("Import error for utils:", e)

from framework.sa_model_pipeline import SAModelPipeline
from utils.sa_app_config_loader import SAAppConfigLoader

print("Imported all utils modules successfully")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


sa_app_config_loader = SAAppConfigLoader()
sa_model_pipeline = SAModelPipeline(sa_app_config_loader.get_app_config())
sa_model_pipeline.run_model_pipeline()