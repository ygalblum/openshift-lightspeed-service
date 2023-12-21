import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from src import constants
from utils.logger import Logger

load_dotenv()


class YesNoClassifier:
    """
    This class is responsible for classifying a statement as yes, no, or undetermined.
    """

    def __init__(self, model_context):
        """
        Initializes the YesNoClassifier instance.

        Args:
        - model_context: Model context to use
        """
        self.logger = Logger("yes_no_classifier").logger
        self._model_context = model_context

    def classify(self, conversation, statement, **kwargs):
        """
        Classifies a statement as yes, no, or undetermined.

        Args:
        - conversation (str): The identifier for the conversation or task context.
        - statement (str): The statement to be classified.
        - **kwargs: Additional keyword arguments for customization.

        Returns:
        - int: The classification result (1 for yes, 0 for no, 9 for undetermined).
        """
        model = kwargs.get(
            "model", os.getenv("YESNO_MODEL", constants.GRANITE_13B_CHAT_V1)
        )
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = f"conversation: {conversation}, query: {statement}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.YES_OR_NO_CLASSIFIER_PROMPT_TEMPLATE
        )

        self.logger.info(f"{conversation} using model: {model}")
        self.logger.info(f"{conversation} determining yes/no: {statement}")
        query = prompt_instructions.format(statement=statement)

        self.logger.info(f"{conversation} yes/no query: {query}")
        self.logger.info(f"{conversation} using model: {model}")

        bare_llm = self._model_context.get_predictor(model=model)
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        response = llm_chain(inputs={"statement": statement})

        self.logger.info(f"{conversation} bare response: {response}")
        self.logger.info(f"{conversation} yes/no response: {response['text']}")

        if response["text"] not in ["0", "1", "9"]:
            raise ValueError("Returned response not 0, 1, or 9")

        return int(response["text"])
