import logging
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

import requests
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails after maximum attempts."""

    pass


@dataclass
class AuthConfig:
    """Configuração para autenticação na API."""

    base_url: str
    token_endpoint: str
    refresh_endpoint: str
    username: str
    password: str
    token_expiry_minutes: int = 30


class TokenHandler:
    """Gerencia tokens de autenticação para a API."""

    MAX_AUTH_ATTEMPTS = 3

    def __init__(self, config: AuthConfig):
        self.config = config
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_timestamp: Optional[datetime] = None
        self._auth_attempts = 0

    @property
    def access_token(self) -> Optional[str]:
        """Retorna o token de acesso atual."""
        return self._access_token

    @property
    def auth_header(self) -> Dict[str, str]:
        """Retorna o cabeçalho com o token de autenticação."""
        return (
            {"Authorization": f"Bearer {self._access_token}"}
            if self._access_token
            else {}
        )

    def refresh_token(self) -> bool:
        """Tenta renovar o access_token utilizando o refresh_token."""
        if not self._refresh_token:
            return False

        try:
            response = requests.post(
                f"{self.config.base_url}{self.config.refresh_endpoint}",
                headers={"Authorization": f"Bearer {self._refresh_token}"},
                timeout=10,
            )

            if response.status_code == 401:
                return False

            response.raise_for_status()
            self._update_tokens(response.json())
            return True

        except requests.RequestException as e:
            logger.error(f"Erro ao tentar renovar o token: {e}")
            return False

    def get_token(self) -> None:
        """Autenticação completa com username e password."""
        self._auth_attempts += 1
        if self._auth_attempts > self.MAX_AUTH_ATTEMPTS:
            self._auth_attempts = 0
            raise AuthenticationError("Máximo de tentativas de autenticação atingido")

        try:
            response = requests.post(
                f"{self.config.base_url}{self.config.token_endpoint}",
                data={
                    "username": self.config.username,
                    "password": self.config.password,
                    "grant_type": "password",
                },
                timeout=10,
            )

            if response.status_code == 401:
                raise AuthenticationError("Credenciais inválidas")

            response.raise_for_status()
            self._update_tokens(response.json())
            self._auth_attempts = 0  # Reset após sucesso

        except requests.RequestException as e:
            logger.error(f"Erro na autenticação: {e}")
            raise AuthenticationError("Erro ao tentar autenticar na API")

    def _update_tokens(self, token_data: Dict[str, str]) -> None:
        """Atualiza os tokens armazenados com novos valores."""
        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token", self._refresh_token)
        self._token_timestamp = datetime.now()


def handle_auth(func: Callable) -> Callable:
    """
    Decorador que gerencia a autenticação em caso de erro `401 Unauthorized`.
    Tentativas de renovação do token e reexecução da requisição.
    """

    @wraps(func)
    def wrapper(self: "APIManager", *args, **kwargs):
        try:
            kwargs["headers"] = {
                **kwargs.get("headers", {}),
                **self.token_handler.auth_header,
            }
            return func(self, *args, **kwargs)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.info("Recebeu 401. Tentando refresh do token...")

                if self.token_handler.refresh_token():
                    logger.info("Token atualizado com sucesso via refresh.")
                    kwargs["headers"] = {
                        **kwargs.get("headers", {}),
                        **self.token_handler.auth_header,
                    }
                    return func(self, *args, **kwargs)

                logger.info("Refresh falhou. Tentando autenticação completa...")
                self.token_handler.get_token()
                kwargs["headers"] = {
                    **kwargs.get("headers", {}),
                    **self.token_handler.auth_header,
                }
                return func(self, *args, **kwargs)

            raise

    return wrapper


class APIManager:
    """Gerencia requisições à API com autenticação automática e retries."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.token_handler = TokenHandler(config)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.WARNING),
        retry=retry_if_exception_type(
            (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
        )
        | retry_if_exception_type(
            lambda e: isinstance(e, requests.exceptions.HTTPError)
            and e.response.status_code == 500
        ),
    )
    @handle_auth
    def make_request(
        self, endpoint: str, method: str = "GET", **kwargs
    ) -> Dict[str, Any]:
        """Faz uma requisição HTTP à API e retorna a resposta JSON."""
        url = f"{self.config.base_url}{endpoint}"

        try:
            response = requests.request(method, url, timeout=10, **kwargs)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500:
                logger.warning(
                    f"Erro 500 no servidor para {url}, tentando novamente..."
                )
            raise

        except Exception as e:
            logger.error(f"Erro na requisição para {url}: {e!s}")
            raise

    def paginate(
        self, endpoint: str, limit: int = 50
    ) -> Generator[list[dict], None, None]:
        """Itera sobre todas as páginas de um endpoint paginado."""
        skip = 0

        while True:
            logger.info(f"Buscando dados de {endpoint} (skip={skip}, limit={limit})")
            try:
                data = self.make_request(
                    endpoint=endpoint,
                    method="GET",
                    params={"skip": skip, "limit": limit},
                )

                if not data:
                    break

                yield data

                if len(data) < limit:
                    break

                skip += limit  # Atualiza o contador de páginas

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 500:
                    # Caso ocorra erro 500, tenta renovar o token e continua de onde parou
                    logger.warning(
                        f"Erro 500 no servidor para {endpoint}, tentando renovar o token e continuar..."
                    )
                    if self.token_handler.refresh_token():
                        logger.info("Token renovado com sucesso.")
                        continue  # Reinicia o loop sem resetar o skip
                    else:
                        logger.error("Falha ao renovar o token via refresh.")
                        raise AuthenticationError("Falha ao renovar o token.")
                else:
                    # Caso ocorra outro erro HTTP, quebramos o loop
                    logger.error(f"Erro ao acessar {endpoint}: {e!s}")
                    break

            except Exception as e:
                logger.error(f"Erro ao fazer requisição para {endpoint}: {e!s}")
                break
