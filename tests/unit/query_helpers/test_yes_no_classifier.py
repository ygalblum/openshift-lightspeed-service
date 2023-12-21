import pytest

import src.query_helpers.yes_no_classifier
from src.query_helpers.yes_no_classifier import YesNoClassifier
from utils.model_context import WatsonXModelContext
from tests.mock_classes.llm_chain import mock_llm_chain


@pytest.fixture
def yes_no_classifier():
    return YesNoClassifier(WatsonXModelContext())


def test_bad_value_response(yes_no_classifier, monkeypatch):
    # response that isn't 1, 0, or 9 should generate a ValueError
    ml = mock_llm_chain({"text": "default"})

    monkeypatch.setattr(src.query_helpers.yes_no_classifier, "LLMChain", ml)

    with pytest.raises(ValueError):
        yes_no_classifier.classify(conversation="1234", statement="The sky is blue.")


def test_good_value_response(yes_no_classifier, monkeypatch):
    # response that is 1, 0, or 9 should return the value
    for x in ["0", "1", "9"]:
        ml = mock_llm_chain({"text": x})

        monkeypatch.setattr(src.query_helpers.yes_no_classifier, "LLMChain", ml)

        assert yes_no_classifier.classify(
            conversation="1234", statement="The sky is blue."
        ) == int(x)
