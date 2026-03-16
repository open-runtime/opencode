import type { Hooks, PluginInput, Plugin as PluginInstance, RuntimeAPI } from "@opencode-ai/plugin"
import { Config } from "../config/config"
import { Bus } from "../bus"
import { Log } from "../util/log"
import { createOpencodeClient } from "@opencode-ai/sdk"
import { Server } from "../server/server"
import { BunProc } from "../bun"
import { Instance } from "../project/instance"
import { Flag } from "../flag/flag"
import { CodexAuthPlugin } from "./codex"
import { Session } from "../session"
import { NamedError } from "@opencode-ai/util/error"
import { CopilotAuthPlugin } from "./copilot"
import { gitlabAuthPlugin as GitlabAuthPlugin } from "@gitlab/opencode-gitlab-auth"
import { LLM } from "../session/llm"
import { Provider } from "../provider/provider"
import { Agent } from "../agent/agent"
import { SessionID, MessageID } from "../session/schema"
import { ProviderID, ModelID } from "../provider/schema"
import * as crypto from "crypto"

export namespace Plugin {
  const log = Log.create({ service: "plugin" })

  // Default third-party auth plugins from upstream OpenCode.
  // Cleared for our fork: these are unnecessary for our embedded runtime use case
  // and cause intermittent ERR_MODULE_NOT_FOUND failures when Bun's compiled
  // binary can't resolve npm packages installed at runtime from /$bunfs/.
  // They also add ~1.5s of startup latency (npm install + module resolution).
  // Our auth is handled by the host Dart process, not by these plugins.
  const BUILTIN: string[] = []

  // Built-in plugins that are directly imported (not installed from npm)
  const INTERNAL_PLUGINS: PluginInstance[] = [CodexAuthPlugin, CopilotAuthPlugin, GitlabAuthPlugin]

  const state = Instance.state(async () => {
    const client = createOpencodeClient({
      baseUrl: "http://localhost:4096",
      directory: Instance.directory,
      headers: Flag.OPENCODE_SERVER_PASSWORD
        ? {
            Authorization: `Basic ${Buffer.from(`${Flag.OPENCODE_SERVER_USERNAME ?? "opencode"}:${Flag.OPENCODE_SERVER_PASSWORD}`).toString("base64")}`,
          }
        : undefined,
      fetch: async (...args) => Server.Default().fetch(...args),
    })
    const config = await Config.get()
    const hooks: Hooks[] = []

    // Create the runtime API for plugin LLM access
    const runtime: RuntimeAPI = {
      stream: async function* (params) {
        const modelSpec = params.model ?? (await Provider.defaultModel?.()) ?? {
          providerID: ProviderID.make("anthropic"),
          modelID: ModelID.make("claude-sonnet-4-20250514"),
        }

        let model: Provider.Model
        try {
          model = await Provider.getModel(modelSpec.providerID, modelSpec.modelID)
        } catch {
          // Create a fallback model with all required fields for SDK loading
          // Map known provider IDs to their SDK npm packages
          const providerNpmPackages: Record<string, string> = {
            openrouter: "@openrouter/ai-sdk-provider",
            anthropic: "@ai-sdk/anthropic",
            openai: "@ai-sdk/openai",
            google: "@ai-sdk/google",
            bedrock: "@ai-sdk/amazon-bedrock",
            azure: "@ai-sdk/azure",
            vertex: "@ai-sdk/google-vertex",
            xai: "@ai-sdk/xai",
            mistral: "@ai-sdk/mistral",
            groq: "@ai-sdk/groq",
            deepinfra: "@ai-sdk/deepinfra",
            cerebras: "@ai-sdk/cerebras",
            cohere: "@ai-sdk/cohere",
            togetherai: "@ai-sdk/togetherai",
            perplexity: "@ai-sdk/perplexity",
          }
          const npm = providerNpmPackages[modelSpec.providerID] ?? "@ai-sdk/openai-compatible"

          // Create a complete fallback model with sensible defaults
          model = {
            providerID: modelSpec.providerID,
            id: modelSpec.modelID,
            name: modelSpec.modelID,
            api: {
              id: modelSpec.modelID,
              url: "", // Will be resolved by the SDK
              npm,
            },
            capabilities: {
              temperature: true,
              reasoning: false,
              attachment: true,
              toolcall: true,
              input: { text: true, audio: false, image: true, video: false, pdf: false },
              output: { text: true, audio: false, image: false, video: false, pdf: false },
              interleaved: true,
            },
            cost: {
              input: 0,
              output: 0,
              cache: { read: 0, write: 0 },
            },
            limit: {
              context: 128000,
              output: 8192,
            },
            status: "active",
            options: {},
            headers: {},
            release_date: new Date().toISOString().split("T")[0],
          } as Provider.Model
        }

        // Use a neutral agent with no prompt to avoid contaminating the
        // system prompt. The "title" agent was previously used here, which
        // prepended title-generation instructions before the caller's
        // system prompt, causing models to generate titles instead of
        // following the actual instruction.
        const agent = {
          name: "runtime",
          mode: "primary",
          permission: [],
          options: {},
        } as Agent.Info

        const requestId = crypto.randomUUID()
        const result = await LLM.stream({
          agent,
          user: {
            id: MessageID.make(crypto.randomUUID()),
            role: "user",
            sessionID: SessionID.make(`plugin-runtime-${requestId}`),
            model: modelSpec,
            agent: "runtime",
            time: { created: Date.now() },
          },
          system: params.systemPrompt ? [params.systemPrompt] : [],
          small: params.small ?? false,
          tools: {},
          model,
          abort: new AbortController().signal,
          sessionID: SessionID.make(`plugin-runtime-${requestId}`),
          retries: 2,
          messages: [{ role: "user", content: params.prompt }],
        })

        let finalText = ""
        let usage = { promptTokens: 0, completionTokens: 0 }

        try {
          for await (const chunk of result.textStream) {
            if (chunk) {
              finalText += chunk
              yield chunk
            }
          }
          usage = (await result.usage) ?? usage
        } catch (e) {
          log.error("runtime stream error", { error: e })
          // Propagate the error so the generated plugin can send an error
          // response to Dart instead of a success with empty text.
          throw e
        }

        return { text: finalText, usage }
      },

      getProviders: async () => {
        // Provider.list() returns Record<string, Info>, not an Array.
        const providers = await Provider.list()
        return Object.values(providers).map((p) => ({
          id: p.id,
          name: p.name ?? p.id,
        }))
      },

      getModels: async (providerID?: string) => {
        // Provider.listModels() does not exist. Iterate Provider.list()
        // and extract models from each provider's Info.models record.
        const providers = await Provider.list()
        const allModels: Array<{ id: string; provider: string; name: string }> = []
        const target = providerID ? { [providerID]: providers[providerID] } : providers
        for (const [pid, provider] of Object.entries(target)) {
          if (!provider) continue
          for (const model of Object.values(provider.models)) {
            allModels.push({
              id: model.id,
              provider: model.providerID ?? pid,
              name: model.name ?? model.id,
            })
          }
        }
        return allModels
      },
    }

    const input: PluginInput = {
      client,
      project: Instance.project,
      worktree: Instance.worktree,
      directory: Instance.directory,
      get serverUrl(): URL {
        return Server.url ?? new URL("http://localhost:4096")
      },
      $: Bun.$,
      runtime,
    }

    for (const plugin of INTERNAL_PLUGINS) {
      log.info("loading internal plugin", { name: plugin.name })
      const init = await plugin(input).catch((err) => {
        log.error("failed to load internal plugin", { name: plugin.name, error: err })
      })
      if (init) hooks.push(init)
    }

    let plugins = config.plugin ?? []
    if (plugins.length) await Config.waitForDependencies()
    if (!Flag.OPENCODE_DISABLE_DEFAULT_PLUGINS) {
      plugins = [...BUILTIN, ...plugins]
    }

    for (let plugin of plugins) {
      // ignore old codex plugin since it is supported first party now
      if (plugin.includes("opencode-openai-codex-auth") || plugin.includes("opencode-copilot-auth")) continue
      log.info("loading plugin", { path: plugin })
      if (!plugin.startsWith("file://")) {
        const lastAtIndex = plugin.lastIndexOf("@")
        const pkg = lastAtIndex > 0 ? plugin.substring(0, lastAtIndex) : plugin
        const version = lastAtIndex > 0 ? plugin.substring(lastAtIndex + 1) : "latest"
        plugin = await BunProc.install(pkg, version).catch((err) => {
          const cause = err instanceof Error ? err.cause : err
          const detail = cause instanceof Error ? cause.message : String(cause ?? err)
          log.error("failed to install plugin", { pkg, version, error: detail })
          Bus.publish(Session.Event.Error, {
            error: new NamedError.Unknown({
              message: `Failed to install plugin ${pkg}@${version}: ${detail}`,
            }).toObject(),
          })
          return ""
        })
        if (!plugin) continue
      }
      // Prevent duplicate initialization when plugins export the same function
      // as both a named export and default export (e.g., `export const X` and `export default X`).
      // Object.entries(mod) would return both entries pointing to the same function reference.
      await import(plugin)
        .then(async (mod) => {
          const seen = new Set<PluginInstance>()
          for (const [_name, fn] of Object.entries<PluginInstance>(mod)) {
            if (seen.has(fn)) continue
            seen.add(fn)
            hooks.push(await fn(input))
          }
        })
        .catch((err) => {
          const message = err instanceof Error ? err.message : String(err)
          log.error("failed to load plugin", { path: plugin, error: message })
          Bus.publish(Session.Event.Error, {
            error: new NamedError.Unknown({
              message: `Failed to load plugin ${plugin}: ${message}`,
            }).toObject(),
          })
        })
    }

    return {
      hooks,
      input,
    }
  })

  export async function trigger<
    Name extends Exclude<keyof Required<Hooks>, "auth" | "event" | "tool">,
    Input = Parameters<Required<Hooks>[Name]>[0],
    Output = Parameters<Required<Hooks>[Name]>[1],
  >(name: Name, input: Input, output: Output): Promise<Output> {
    if (!name) return output
    for (const hook of await state().then((x) => x.hooks)) {
      const fn = hook[name]
      if (!fn) continue
      // @ts-expect-error if you feel adventurous, please fix the typing, make sure to bump the try-counter if you
      // give up.
      // try-counter: 2
      await fn(input, output)
    }
    return output
  }

  export async function list() {
    return state().then((x) => x.hooks)
  }

  export async function init() {
    const hooks = await state().then((x) => x.hooks)
    const config = await Config.get()
    for (const hook of hooks) {
      // @ts-expect-error this is because we haven't moved plugin to sdk v2
      await hook.config?.(config)
    }
    Bus.subscribeAll(async (input) => {
      const hooks = await state().then((x) => x.hooks)
      for (const hook of hooks) {
        hook["event"]?.({
          event: input,
        })
      }
    })
  }
}
