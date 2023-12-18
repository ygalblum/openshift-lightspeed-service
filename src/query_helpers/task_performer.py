import os
import src.constants as constants
from dotenv import load_dotenv
from utils.logger import Logger

load_dotenv()


class TaskPerformer:
    """
    Class for performing tasks based on the given input.
    """

    def __init__(self):
        """
        Initializes the TaskPerformer instance.
        """
        self.logger = Logger("task_performer").logger

    def perform_task(self, conversation, task, **kwargs):
        """
        Performs the specified task and logs the settings and response.

        Args:
        - conversation (str): The identifier for the conversation or task context.
        - task (str): The specific task to be performed.
        - **kwargs: Additional keyword arguments for customization.

        Returns:
        - str: The response generated by performing the task.
        """
        model = kwargs.get(
            "model", os.getenv("TASK_PERFORMER_MODEL", constants.GRANITE_13B_CHAT_V1)
        )
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = f"conversation: {conversation}, task: {task}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        # Determine if this should go to a general LLM, the YAML generator, or elsewhere
        # Send to the right tool

        # Output the response
        response = """
apiVersion: "autoscaling.openshift.io/v1"
kind: "ClusterAutoscaler"
metadata:
  name: "default"
spec:
  resourceLimits:
    maxNodesTotal: 10
  scaleDown:
    enabled: true
"""
        self.logger.info(f"{conversation} response: {response}")
        return response