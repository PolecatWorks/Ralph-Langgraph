from unittest.mock import patch, MagicMock
import pytest
from ralf.llm import get_chain

def test_get_chain_no_api_key():
    # Mock settings.GOOGLE_API_KEY to be None
    with patch("ralf.llm.settings") as mock_settings:
        mock_settings.GOOGLE_API_KEY = None
        with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable is not set"):
            get_chain()

def test_get_chain_success():
    # We patch settings inside ralf.llm
    with patch("ralf.llm.settings") as mock_settings:
        mock_settings.GOOGLE_API_KEY = "dummy_key"

        # We also need to patch ChatGoogleGenerativeAI to verify initialization
        with patch("ralf.llm.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            chain = get_chain()

            # Verify that chain is constructed
            assert chain is not None

            # Verify LLM was initialized with the key
            mock_llm_class.assert_called_once_with(
                model="gemini-pro",
                google_api_key="dummy_key"
            )
