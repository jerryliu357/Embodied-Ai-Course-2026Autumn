"""Reuse 4.2 in-process; importing this module never opens hardware."""
from pathlib import Path
import sys

HERE43 = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE43.parent / "4_2"))
from agent import Agent, RunLog
from llm_client import SiliconFlowClient, resolve_api_key, strict_json
from robot_tools import RobotTools

