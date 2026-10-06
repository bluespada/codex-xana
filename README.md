<p align="center"><strong>Xana</strong> is a fork of <strong>Codex CLI</strong>, the coding agent from OpenAI that runs locally on your computer.
<p align="center"><em>OpenAI Codex (fork) Xana</em></p>
<p align="center">
  <img src="https://github.com/openai/codex/blob/main/.github/codex-cli-splash.png" alt="Codex CLI splash" width="80%" />
</p>
</br>
Upstream: <a href="https://github.com/openai/codex">openai/codex</a>. This fork: <a href="https://github.com/bluespada/codex-xana">bluespada/codex-xana</a>.

---

## Why Xana

Xana started as a personal fork with one goal: keep the agent loop, sandbox, and TUI that already work, and stop treating one vendor's wire format as the only way to talk to a model.

What I want out of it:

- **Multi-provider by default.** One Codex, any endpoint: OpenAI's Responses API, an OpenAI-compatible chat completions server, or an Anthropic Messages endpoint. Switching is a config change, not a different tool.
- **Modern and customizable.** The provider layer is a crate with its own tests instead of protocol branches spread through the client, so adding a protocol or a tool means adding code, not editing core.
- **Tools that do not silently disappear.** Tool discovery, MCP, and web access keep working on every wire protocol, not only on the Responses API.

Upstream harmony still matters. This fork tracks `openai/codex` loosely, keeps upstream behavior as the default, and keeps the divergences listed rather than implied.

## Providers

`wire_api` selects the protocol used against a provider:

- `openai-responses` (default) speaks the Responses API and keeps all upstream behavior, including hosted tools and deferred tool loading.
- `openai-completions` speaks `/chat/completions`, for OpenAI-compatible servers.
- `anthropic-messages` speaks `/messages`, for Anthropic and Anthropic-compatible gateways.

A local gateway that speaks the Messages API:

```toml
model_provider = "local-router"
model = "gpt-5.6-luna"

[model_providers.local-router]
name = "local-router"
base_url = "http://localhost:9990/v1"
wire_api = "anthropic-messages"
env_key = "OPENAI_API_KEY"
```

`base_url` carries the version segment, and the client posts to `/responses`, `/chat/completions`, or `/messages` under it.

## What this fork adds

- `codex-rs/codex-providers`: request bodies and stream decoding for the completions and Messages protocols, built on `async-openai` and `claudius` types and sharing one transcript layer with the Responses path.
- Namespaced tools are flattened into wire-safe function names with an alias map, so a namespaced call comes back named and addressed correctly, and flattened names stay inside the 64-byte function-name limit.
- Reasoning is carried across turns, and the reasoning effort Codex resolves is sent to the provider verbatim.
- Tool discovery works off the Responses API: when the provider is not the Responses API, MCP tools and `spawn_agent` are declared directly instead of deferred, and `tool_search` crosses over as an ordinary function tool.
- `web_fetch`: fetch a URL and return its readable text. It is registered unconditionally, needs no web search, adds no dependencies, and renders as a call row in the transcript.
- Code mode steps down to direct tools when the `codex-code-mode-host` binary is missing, unless `code_mode.disable_in_process_fallback = true` keeps the failure closed.

## Deferred on purpose

- **A Responses module built on the SDK.** Chat Completions and Messages already use vendor SDK types, while Responses still uses its own request body and stream decoding in `codex-api` and `core/src/client.rs`. Nothing blocks the move except risk: that path also carries zstd request compression, `x-openai-*` headers, guardian review metadata, and rollout inference traces. The exit condition is recorded in `codex-rs/tui/src/bottom_pane/AGENTS.md`.
- **`/settings` with a web search provider picker.** Until that pane exists, the hosted web search tool stays hidden and `web_fetch` is how a turn reads a page.

## Building this fork

```shell
cd codex-rs
cargo build --release --bin codex
```

Then run `./target/release/codex` to start it.

Releases for this fork are built from this repository and attached to GitHub Releases as per-platform archives. The install scripts under `scripts/install/` still default to the upstream slug, so parameterizing them is an open item. This fork is not published to npm or Homebrew.

### Upstream Codex

Everything below installs and documents upstream Codex, not Xana. It is kept for reference.

Run the following on Mac or Linux to install Codex CLI:

```shell
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Run the following on Windows to install Codex CLI:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"
```

The standalone installers download from `https://releases.openai.com/codex` by default and fall back to GitHub Releases if a metadata or asset download is unavailable. To force GitHub Releases, set `CODEX_INSTALLER_USE_RELEASES_OPENAI_COM` to `false` (`0` and `no` are also accepted):

```shell
curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_INSTALLER_USE_RELEASES_OPENAI_COM=false sh
```

```powershell
$env:CODEX_INSTALLER_USE_RELEASES_OPENAI_COM='false'; irm https://chatgpt.com/codex/install.ps1 | iex
```

Codex CLI can also be installed via the following package managers:

```shell
# Install using npm
npm install -g @openai/codex
```

```shell
# Install using Homebrew
brew install --cask codex
```

<details>
<summary>You can also go to the <a href="https://github.com/openai/codex/releases/latest">latest GitHub Release</a> and download the appropriate binary for your platform.</summary>

Each GitHub Release contains many executables, but in practice, you likely want one of these:

- macOS
  - Apple Silicon/arm64: `codex-aarch64-apple-darwin.tar.gz`
  - x86_64 (older Mac hardware): `codex-x86_64-apple-darwin.tar.gz`
- Linux
  - x86_64: `codex-x86_64-unknown-linux-musl.tar.gz`
  - arm64: `codex-aarch64-unknown-linux-musl.tar.gz`

Each archive contains a single entry with the platform baked into the name (e.g., `codex-x86_64-unknown-linux-musl`), so you likely want to rename it to `codex` after extracting it.

</details>

### Using Codex with your ChatGPT plan

Run `codex` and select **Sign in with ChatGPT**. We recommend signing into your ChatGPT account to use Codex as part of your Plus, Pro, Business, Edu, or Enterprise plan. [Learn more about what's included in your ChatGPT plan](https://help.openai.com/en/articles/11369540-codex-in-chatgpt).

You can also use Codex with an API key, but this requires [additional setup](https://developers.openai.com/codex/auth#sign-in-with-an-api-key).

## Docs

- [**Codex Documentation**](https://developers.openai.com/codex)
- [**Contributing**](./docs/contributing.md)
- [**Installing & building**](./docs/install.md)
- [**Open source fund**](./docs/open-source-fund.md)

This repository is licensed under the [Apache-2.0 License](LICENSE).
