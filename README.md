# Shorts Pipeline

Pipeline modular para geração e publicação automatizada de YouTube Shorts.

Transforma histórias do Reddit em vídeos verticais completos (com narração, legendas
e vídeo de fundo) e publica no YouTube — tudo rodando automaticamente via GitHub Actions.

## Estrutura do projeto

```
shorts-pipeline/
├── shorts_pipeline/           # código do pipeline
│   ├── orchestrator.py        # executa tudo em sequência (entry point)
│   ├── config.py              # carrega o YAML do canal
│   ├── utils.py               # retry, logging, helpers
│   ├── trend_detection.py     # 1. acha tema no Reddit
│   ├── story_generation.py    # 2. escreve texto via Claude
│   ├── narration.py           # 3. gera áudio via ElevenLabs
│   ├── visual_selection.py    # 4. baixa vídeos do Pexels
│   ├── video_assembly.py      # 5. monta vídeo final (ffmpeg + Whisper)
│   └── publishing.py          # 6. publica no YouTube
├── configs/
│   ├── dramas_reais.yml       # canal 1: histórias de dramas reais
│   └── canal_cristao.yml      # canal 2: conteúdo cristão (exemplo)
├── .github/workflows/
│   └── pipeline.yml           # agendamento no GitHub Actions
├── generate_youtube_token.py  # utilitário: gera token OAuth (rodar 1x local)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Como adicionar um canal novo

1. Copie um dos YAMLs em `configs/` e renomeie.
2. Ajuste:
   - `channel.name` e `channel.youtube_channel_id`
   - `trend.subreddits` (fontes de inspiração)
   - `story.channel_identity` e regras editoriais
   - `visual.queries` (que tipo de fundo combina com o tema)
   - `video.final_speed_multiplier` (aceleração)
   - `narration.voice_id` (voz do ElevenLabs)
3. Rode: `python -m shorts_pipeline.orchestrator --config configs/seu_canal.yml`

**Nenhuma linha de código muda.** Para 10 canais, você tem 10 YAMLs.

## Setup local (primeira vez)

### 1. Clone e instale

```bash
git clone <seu-repo>
cd shorts-pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Instale o FFmpeg no seu sistema

- **Windows**: https://www.gyan.dev/ffmpeg/builds/ (coloque no PATH)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 3. Configure suas chaves

```bash
cp .env.example .env
# edite .env e preencha suas API keys
```

### 4. Gere o token OAuth do YouTube (UMA VEZ)

Baixe `client_secrets.json` do Google Cloud Console, depois:

```bash
python generate_youtube_token.py client_secrets.json
```

Isso abre seu navegador, você autoriza, e o script gera `youtube_token.json` e imprime
as strings prontas para colar nos Secrets do GitHub.

### 5. Rode localmente

```bash
export $(grep -v '^#' .env | xargs)   # carrega as env vars
python -m shorts_pipeline.orchestrator --config configs/dramas_reais.yml
```

Os resultados ficam em `data/<nome_canal>/cycles/<cycle_id>/`.

## Setup no GitHub Actions

### 1. Configure os Secrets

Em **Settings > Secrets and variables > Actions**, crie:

| Secret | Como obter |
|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `ELEVENLABS_API_KEY` | elevenlabs.io/app/settings/api-keys |
| `PEXELS_API_KEY` | pexels.com/api |
| `YOUTUBE_CLIENT_SECRETS_JSON` | conteúdo compacto do `client_secrets.json` |
| `YOUTUBE_TOKEN_JSON` | conteúdo compacto do `youtube_token.json` |

O script `generate_youtube_token.py` imprime os dois últimos já no formato correto.

### 2. Ajuste o cron em `.github/workflows/pipeline.yml`

O arquivo já vem com exemplo rodando 3x por dia. Ajuste horários em UTC conforme
preferir. Lembre: cron do GitHub Actions só aceita horários em UTC.

### 3. Teste manualmente primeiro

No GitHub, vá em **Actions > Shorts Pipeline > Run workflow**. Rode 1x manual antes
de deixar o cron tomar conta.

## Comportamento de retry

Cada etapa tem duas camadas de retry:

1. **Retry de HTTP dentro do módulo** — em caso de 5xx, 429, timeout: backoff exponencial
2. **Retry da etapa inteira** — se todas as tentativas de HTTP falharem, a etapa é
   executada de novo (útil quando uma API está degradada e volta depois de alguns minutos)

Erros classificados como **permanentes** (401, 403, 404, config errada) falham
imediatamente sem retry — não adianta retentar.

## Limitações e avisos importantes

### YouTube pode trancar vídeos como privados

Mesmo com `privacy_status: "public"` no config, **o YouTube pode retornar o vídeo
como `private` se seu projeto API ainda não passou pela auditoria do Google**.

Isso é uma trava do Google para projetos não-verificados criados após julho de 2020.
Para resolver, submeta seu projeto em:
https://support.google.com/youtube/contact/yt_api_form

O campo `privacy_status_returned` em `publication_result.json` mostra o que o YouTube
efetivamente fez com o vídeo. Se for diferente do que você pediu, é sinal de que
você precisa da auditoria.

### Disclosure de IA

O YouTube exige que vídeos com conteúdo gerado/alterado por IA sejam marcados.
O pipeline adiciona uma nota na descrição automaticamente. Dependendo do nicho,
você também pode marcar manualmente no YouTube Studio.

### Política do YouTube sobre conteúdo automatizado

Canais 100% automatizados sem curadoria humana podem ser desmonetizados por
"conteúdo repetitivo/spam". Considere revisar títulos e descrições antes de
publicar em massa.

## Troubleshooting

### "Variável de ambiente 'X' não definida"
→ Você esqueceu de carregar o `.env` local ou configurar o Secret no GitHub.

### "Credenciais do YouTube inválidas e sem refresh token"
→ Seu `youtube_token.json` expirou (90 dias sem uso). Rode `generate_youtube_token.py`
de novo e atualize o Secret.

### "Nenhum post coletado do Reddit"
→ Provavelmente seu IP está rate-limited. Espera uns minutos e tenta de novo.
No GitHub Actions isso costuma não acontecer porque o IP rota.

### Whisper muito lento
→ Mude `video.whisper_model` de `"small"` para `"tiny"` (menos preciso mas muito mais rápido).

### Vídeos ficando cortados no final
→ Aumente `final_speed_multiplier` ou diminua `duration_target_seconds` na config.

## Licença

Uso pessoal. Adapte como quiser.
