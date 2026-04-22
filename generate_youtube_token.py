#!/usr/bin/env python3
"""
Utilitário para gerar o youtube_token.json pela primeira vez.

Rode UMA VEZ localmente (precisa de navegador):

    python generate_youtube_token.py client_secrets.json

Isso vai:
1. Abrir seu navegador para você autorizar
2. Salvar youtube_token.json na pasta atual
3. Imprimir o JSON como string pronta para colar no Secret YOUTUBE_TOKEN_JSON do GitHub

NÃO commite nem o client_secrets.json nem o youtube_token.json gerado.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python generate_youtube_token.py <caminho_para_client_secrets.json>")
        return 1

    client_secrets = Path(sys.argv[1])
    if not client_secrets.exists():
        print(f"Arquivo não encontrado: {client_secrets}")
        return 1

    print("Abrindo navegador para autorização do YouTube...")
    print("Atenção: selecione a conta Google que tem os seus canais.\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
    creds = flow.run_local_server(port=0)

    token_path = Path.cwd() / "youtube_token.json"
    token_path.write_text(creds.to_json(), encoding="utf-8")

    print(f"\n✅ Token salvo em: {token_path}")
    print("\n" + "=" * 70)
    print("PARA O GITHUB ACTIONS:")
    print("=" * 70)
    print("\n1. Vá em Settings > Secrets and variables > Actions > New repository secret")
    print("2. Crie um secret chamado YOUTUBE_TOKEN_JSON com o seguinte valor:\n")

    # imprime compacto, sem quebra de linha — ideal para colar no secret
    compact = json.dumps(json.loads(token_path.read_text(encoding="utf-8")))
    print(compact)

    print("\n3. Crie também um secret YOUTUBE_CLIENT_SECRETS_JSON com o conteúdo compacto")
    print(f"   do arquivo {client_secrets.name}:\n")
    client_compact = json.dumps(json.loads(client_secrets.read_text(encoding="utf-8")))
    print(client_compact)

    print("\n" + "=" * 70)
    print("Lembrete: o token tem refresh_token, então vai se renovar sozinho.")
    print("Se o YouTube revogar (ex: após 90 dias sem uso), rode este script de novo.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
