import os

from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from llama_index import ServiceContext
from llama_index.embeddings import TextEmbeddingsInference


class ModelContext():
    def get_predictor(self, model, **kwargs):
        raise NotImplementedError

    def get_context(self, model, **kwargs):
        raise NotImplementedError


class WatsonXModelContext(ModelContext):
    def get_predictor(self, model, min_new_tokens=1, max_new_tokens=256, **kwargs):
        """
        Get a predictor for WatsonX.

        Args:
            model (str): The model to use.
            min_new_tokens (int): Minimum number of new tokens in the generated output.
            max_new_tokens (int): Maximum number of new tokens in the generated output.
            verbose (bool): Whether to print verbose output.

        Returns:
            LangChainInterface: WatsonX predictor.
        """
        verbose = kwargs.get("verbose", False)

        api_key = os.getenv("BAM_API_KEY", "BOGUS_VALUE")
        api_url = os.getenv("BAM_URL", "http://bogus.url")
        creds = Credentials(api_key, api_endpoint=api_url)

        params = GenerateParams(
            decoding_method="greedy",
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
        )

        predictor = LangChainInterface(
            model=model, params=params, credentials=creds, verbose=verbose
        )
        return predictor


    def get_context(self, model, url="local", tei_embedding_model=None, **kwargs):
        """
        Get a WatsonX service context.

        Args:
            model (str): The model to use.
            url (str): URL for remote service. Default is "local".
            tei_embedding_model (str): TEI embedding model name.
            **kwargs: Additional keyword arguments.

        Returns:
            ServiceContext: WatsonX service context.
        """
        if url != "local":
            embed_model = TextEmbeddingsInference(
                model_name=tei_embedding_model,
                base_url=url,
            )
        else:
            embed_model = "local:BAAI/bge-base-en"

        predictor = self.get_predictor(model)

        service_context = ServiceContext.from_defaults(
            chunk_size=1024, llm=predictor, embed_model=embed_model, **kwargs
        )

        return service_context
