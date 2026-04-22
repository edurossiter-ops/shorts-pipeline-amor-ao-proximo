"""
Etapa 6: Upload no YouTube.

Refatorado para:
- Credenciais (client_secrets e token) lidas de variáveis de ambiente (conteúdo JSON inteiro)
- privacy_status, tags, made_for_kids, língua configuráveis por canal
- Falha explícita se precisar de login interativo em ambiente headless
- Suporte a multi-canal (onBehalfOfContentOwner / canal específico)

IMPORTANTE: mesmo com privacy='public', o YouTube pode travar o vídeo como privado
enquanto seu projeto API não passar pela auditoria do Google. Isso é esperado.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from .config import PipelineConfig
from .utils import PermanentError, get_logger, load_json, now_iso, save_json


SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _get_credentials(config: PipelineConfig) -> Credentials:
    """
    Carrega credenciais do OAuth.
    Em ambiente cloud (GitHub Actions): lê o conteúdo JSON das env vars.
    Em ambiente local: pode usar arquivos no disco (se ALLOW_INTERACTIVE_AUTH=1).
    """
    logger = get_logger()
    token_env = config.secrets.youtube_token_env
    secrets_env = config.secrets.youtube_client_secrets_env

    token_json_str = os.environ.get(token_env)
    if token_json_str:
        # GitHub Actions path: token vem como string JSON na env var
        token_data = json.loads(token_json_str)
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
    else:
        # Local path: tenta arquivo youtube_token.json no diretório atual
        token_file = Path.cwd() / "youtube_token.json"
        if token_file.exists():
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        else:
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("Token expirado, renovando via refresh_token...")
        creds.refresh(Request())
        return creds

    # Último recurso: fluxo interativo (SÓ em ambiente local com permissão explícita)
    if os.environ.get("ALLOW_INTERACTIVE_AUTH") != "1":
        raise PermanentError(
            "Credenciais do YouTube inválidas e sem refresh token. "
            f"Defina {token_env} com um token válido, ou rode localmente com "
            "ALLOW_INTERACTIVE_AUTH=1 para gerar um novo."
        )

    client_secrets_str = os.environ.get(secrets_env)
    if client_secrets_str:
        # usa um arquivo temporário
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(client_secrets_str)
            temp_secrets = f.name
        try:
            flow = InstalledAppFlow.from_client_secrets_file(temp_secrets, SCOPES)
        finally:
            Path(temp_secrets).unlink(missing_ok=True)
    else:
        local_secrets = Path.cwd() / "client_secrets.json"
        if not local_secrets.exists():
            raise PermanentError(
                f"client_secrets não encontrado nem em env var {secrets_env} "
                f"nem em {local_secrets}"
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(local_secrets), SCOPES)

    creds = flow.run_local_server(port=0)
    # salva para próxima vez
    (Path.cwd() / "youtube_token.json").write_text(creds.to_json(), encoding="utf-8")
    return creds


def _upload(youtube, body: Dict[str, Any], video_path: str) -> Dict[str, Any]:
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    logger = get_logger()
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info(f"Upload: {int(status.progress() * 100)}%")
    return response


def run(cycle_dir: Path, config: PipelineConfig) -> Dict[str, Any]:
    logger = get_logger()
    video_file = cycle_dir / "video-assembly" / "assembled_video.mp4"
    story_file = cycle_dir / "story-generation" / "story_text.json"
    publishing_dir = cycle_dir / "publishing"

    if not video_file.exists():
        raise PermanentError(f"Vídeo final não encontrado: {video_file}")

    story = load_json(story_file) if story_file.exists() else {}

    # Título: adiciona #Shorts no final pra forçar a classificação como Short.
    # Mantém o título dentro do limite de 100 caracteres do YouTube.
    title_base = (story.get("title_hint") or f"História - {cycle_dir.name}").strip()
    SHORTS_TAG = " #Shorts"
    max_title_base_length = 100 - len(SHORTS_TAG)
    title = (title_base[:max_title_base_length] + SHORTS_TAG)

    # Descrição: começa com CTA da história, adiciona hashtags e disclosure de IA.
    cta = (story.get("cta") or "Inscreva-se no canal!").strip()
    hashtags_line = " ".join(f"#{tag.replace(' ', '').replace('-', '')}" for tag in config.youtube.tags)
    # Garante que #Shorts está presente mesmo se não estiver nas tags do config
    if "#shorts" not in hashtags_line.lower():
        hashtags_line = "#Shorts " + hashtags_line

    description = f"{cta}\n\n{hashtags_line}"
    if config.youtube.made_for_kids is False:
        description += "\n\n[Conteúdo produzido com auxílio de ferramentas de IA]"

    tags = config.youtube.tags[:]

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": config.youtube.category_id,
            "defaultLanguage": config.youtube.default_language,
            "defaultAudioLanguage": config.youtube.default_audio_language,
        },
        "status": {
            "privacyStatus": config.youtube.privacy_status,
            "selfDeclaredMadeForKids": config.youtube.made_for_kids,
            "madeForKids": config.youtube.made_for_kids,
            "embeddable": True,
            "license": "youtube",
        },
    }

    save_json(publishing_dir / "publish_request.json", {
        "cycle_id": cycle_dir.name,
        "body": body,
        "video_file_path": str(video_file),
    })

    creds = _get_credentials(config)
    youtube = build("youtube", "v3", credentials=creds)

    logger.info(f"Fazendo upload: {video_file.name} (privacy={config.youtube.privacy_status})")
    api_response = _upload(youtube, body, str(video_file))

    status_info = api_response.get("status", {})
    returned_privacy = status_info.get("privacyStatus")
    if returned_privacy and returned_privacy != config.youtube.privacy_status:
        logger.warning(
            f"YouTube retornou privacy='{returned_privacy}' mas pedimos '{config.youtube.privacy_status}'. "
            f"Isso geralmente indica que o projeto API precisa passar por auditoria do Google."
        )

    result = {
        "cycle_id": cycle_dir.name,
        "platform": "youtube",
        "channel_id": config.channel.youtube_channel_id,
        "publication_status": "uploaded",
        "platform_publication_ids": {"youtube_video_id": api_response.get("id")},
        "privacy_status_requested": config.youtube.privacy_status,
        "privacy_status_returned": returned_privacy,
        "upload_status": status_info.get("uploadStatus"),
        "video_url": f"https://www.youtube.com/watch?v={api_response.get('id')}",
        "status": "success",
        "generated_at": now_iso(),
    }
    save_json(publishing_dir / "publication_result.json", result)
    logger.info(f"YouTube video id: {api_response.get('id')} — {result['video_url']}")
    return result
