import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dags.api_auth_manager import (
    APIManager,
    AuthConfig,
    AuthenticationError,
    TokenHandler,
)


@pytest.fixture
def auth_config():
    """Configuração fictícia para os testes."""
    return AuthConfig(
        base_url="https://api.fake.com",
        token_endpoint="/auth",
        refresh_endpoint="/refresh",
        username="user",
        password="pass",
    )


@pytest.fixture
def token_handler(auth_config):
    """Inicializa o TokenHandler."""
    return TokenHandler(auth_config)


@pytest.fixture
def api_manager(auth_config):
    """Inicializa o APIManager."""
    return APIManager(auth_config)


def test_get_token_success(token_handler, mocker):
    """Testa autenticação bem-sucedida com username/password."""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_token",
        "refresh_token": "refresh123",
    }

    mocker.patch("requests.post", return_value=mock_response)

    token_handler.get_token()
    assert token_handler.access_token == "new_token"
    assert token_handler._refresh_token == "refresh123"


def test_get_token_failure(token_handler, mocker):
    """Testa falha na autenticação com credenciais erradas."""
    mock_response = mocker.Mock()
    mock_response.status_code = 401

    mocker.patch("requests.post", return_value=mock_response)

    with pytest.raises(AuthenticationError, match="Credenciais inválidas"):
        token_handler.get_token()


def test_refresh_token_success(token_handler, mocker):
    """Testa a renovação do token de acesso usando refresh_token."""
    token_handler._refresh_token = "valid_refresh_token"

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "renewed_token"}

    mocker.patch("requests.post", return_value=mock_response)

    assert token_handler.refresh_token() is True
    assert token_handler.access_token == "renewed_token"


def test_refresh_token_failure(token_handler, mocker):
    """Testa falha na renovação do token (refresh inválido)."""
    token_handler._refresh_token = "invalid_refresh_token"

    mock_response = mocker.Mock()
    mock_response.status_code = 401

    mocker.patch("requests.post", return_value=mock_response)

    assert token_handler.refresh_token() is False


def test_make_request_with_auth(api_manager, mocker):
    """Testa uma requisição bem-sucedida com autenticação."""
    api_manager.token_handler._access_token = "valid_token"

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}

    mocker.patch("requests.request", return_value=mock_response)

    response = api_manager.make_request("/data")
    assert response == {"result": "success"}


def test_pagination(api_manager, mocker):
    """Testa a funcionalidade de paginação."""
    api_manager.token_handler._access_token = "valid_token"

    mock_page_1 = mocker.Mock()
    mock_page_1.status_code = 200
    mock_page_1.json.return_value = [{"id": 1}, {"id": 2}]

    mock_page_2 = mocker.Mock()
    mock_page_2.status_code = 200
    mock_page_2.json.return_value = [{"id": 3}, {"id": 4}]

    mock_page_3 = mocker.Mock()
    mock_page_3.status_code = 200
    mock_page_3.json.return_value = []

    mocker.patch(
        "requests.request", side_effect=[mock_page_1, mock_page_2, mock_page_3]
    )

    results = list(api_manager.paginate("/items", limit=2))
    assert results == [[{"id": 1}, {"id": 2}], [{"id": 3}, {"id": 4}]]
