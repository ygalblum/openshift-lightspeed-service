import os

import llama_index
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage
from llama_index.prompts import PromptTemplate

from src import constants
from utils.logger import Logger

load_dotenv()


class TaskBreakdown:
    """
    Class to handle task breakdowns.
    """

    def __init__(self, model_context):
        """
        Initializes the TaskBreakdown class.

        Args:
        - model_context: Model context to use
        """
        self.logger = Logger("task_breakdown").logger
        self._model_context = model_context

    def breakdown_tasks(self, conversation, query, **kwargs):
        """
        Breakdown tasks based on the given query.

        Args:
            conversation (str): Conversation ID.
            query (str): Query for task breakdown.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: Task breakdown response and referenced documents.
        """
        model = kwargs.get(
            "model", os.getenv("TASK_BREAKDOWN_MODEL", constants.GRANITE_13B_CHAT_V1)
        )
        verbose = kwargs.get("verbose", False)

        # Make llama index show the prompting if verbose is True
        if verbose:
            llama_index.set_global_handler("simple")

        settings_string = f"conversation: {conversation}, query: {query}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        summary_task_breakdown_template = PromptTemplate(
            constants.SUMMARY_TASK_BREAKDOWN_TEMPLATE
        )

        self.logger.info(f"{conversation} Getting service context")
        self.logger.info(f"{conversation} using model: {model}")
        service_context = self._model_context.get_context(model=model)

        storage_context = StorageContext.from_defaults(
            persist_dir=constants.SUMMARY_DOCS_PERSIST_DIR
        )
        self.logger.info(f"{conversation} Setting up index")
        index = load_index_from_storage(
            storage_context=storage_context,
            index_id=constants.SUMMARY_INDEX,
            service_context=service_context,
        )

        self.logger.info(f"{conversation} Setting up query engine")
        query_engine = index.as_query_engine(
            text_qa_template=summary_task_breakdown_template,
            verbose=verbose,
            streaming=False,
            similarity_top_k=1,
        )

        self.logger.info(f"{conversation} Submitting task breakdown query")
        task_breakdown_response = query_engine.query(query)

        referenced_documents = "\n".join(
            [
                source_node.node.metadata["file_name"]
                for source_node in task_breakdown_response.source_nodes
            ]
        )

        for line in str(task_breakdown_response).splitlines():
            self.logger.info(f"{conversation} Task breakdown response: {line}")

        for line in referenced_documents.splitlines():
            self.logger.info(f"{conversation} Referenced documents: {line}")

        return str(task_breakdown_response).splitlines(), referenced_documents
