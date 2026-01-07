import type { Context } from "hono"

import consola from "consola"
import { streamSSE, type SSEMessage } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import { getTokenCount } from "~/lib/tokenizer"
import { isNullish } from "~/lib/utils"
import {
  createChatCompletions,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
} from "~/services/copilot/create-chat-completions"
import {
  convertResponseToChatCompletion,
  convertToResponsesPayload,
  createResponses,
  isResponseObject,
} from "~/services/copilot/create-responses"

import { processResponseStream } from "./responses-stream"

// Models that support Responses API (newer GPT models with reasoning)
// gpt-4o, gpt-4o-mini, etc. do NOT support Responses API
// Only gpt-5+, o1, o3, etc. support it
const RESPONSES_API_MODELS = [
  "gpt-5",
  "gpt-5.1",
  "gpt-5.2",
  "gpt-5-mini",
  "gpt-5-codex",
  "gpt-5.1-codex",
  "gpt-5.1-codex-mini",
  "gpt-5.1-codex-max",
  "o1",
  "o3",
  "o4-mini",
]

function supportsResponsesApi(model: string): boolean {
  return RESPONSES_API_MODELS.some(
    (prefix) => model === prefix || model.startsWith(`${prefix}-`),
  )
}

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  let payload = await c.req.json<ChatCompletionsPayload>()
  consola.debug("Request payload:", JSON.stringify(payload).slice(-400))

  // Parse model name suffix for reasoning effort
  // Supports: -thinking, -high, -medium, -low, -minimal, -xhigh, -none
  const effortSuffixes = [
    "thinking",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
  ] as const
  const suffixToEffort: Record<string, string> = {
    thinking: "high", // -thinking is alias for -high
    xhigh: "xhigh",
    high: "high",
    medium: "medium",
    low: "low",
    minimal: "minimal",
    none: "none",
  }

  for (const suffix of effortSuffixes) {
    if (payload.model.endsWith(`-${suffix}`)) {
      const baseModel = payload.model.slice(0, -(suffix.length + 1))
      const effort = suffixToEffort[suffix]
      payload = {
        ...payload,
        model: baseModel,
        reasoning_effort: effort as typeof payload.reasoning_effort,
      }
      consola.debug(
        `Parsed model suffix "-${suffix}" -> model: ${baseModel}, reasoning_effort: ${effort}`,
      )
      break
    }
  }

  // Find the selected model
  const selectedModel = state.models?.data.find(
    (model) => model.id === payload.model,
  )

  // Calculate and display token count
  try {
    if (selectedModel) {
      const tokenCount = await getTokenCount(payload, selectedModel)
      consola.info("Current token count:", tokenCount)
    } else {
      consola.warn("No model selected, skipping token count calculation")
    }
  } catch (error) {
    consola.warn("Failed to calculate token count:", error)
  }

  if (state.manualApprove) await awaitApproval()

  // Set max_tokens first if not provided (needed for reasoning_effort calculation)
  if (isNullish(payload.max_tokens)) {
    payload = {
      ...payload,
      max_tokens: selectedModel?.capabilities.limits.max_output_tokens,
    }
    consola.debug("Set max_tokens to:", JSON.stringify(payload.max_tokens))
  }

  // Convert reasoning_effort to thinking_budget for Claude models
  // (Gemini natively supports reasoning_effort, Claude requires thinking_budget)
  if (
    payload.reasoning_effort
    && !payload.thinking_budget
    && payload.model.startsWith("claude")
  ) {
    // Calculate thinking_budget as a ratio of max_tokens
    // thinking_budget must be less than max_tokens and >= 1024 (Claude minimum)
    const effortToRatio: Record<string, number | undefined> = {
      none: 0,
      minimal: 0.1, // 10%
      low: 0.2, // 20%
      medium: 0.4, // 40%
      high: 0.6, // 60%
      xhigh: 0.8, // 80%
    }
    const ratio = effortToRatio[payload.reasoning_effort]
    if (ratio !== undefined && ratio > 0 && payload.max_tokens) {
      // Claude requires minimum thinking_budget of 1024
      const minBudget = 1024
      const calculatedBudget = Math.floor(payload.max_tokens * ratio)
      const budget = Math.max(minBudget, calculatedBudget)
      payload = { ...payload, thinking_budget: budget }
      consola.debug(
        `Converted reasoning_effort "${payload.reasoning_effort}" to thinking_budget: ${budget} (${ratio * 100}% of max_tokens: ${payload.max_tokens})`,
      )
    }
    // Remove reasoning_effort from payload as Claude doesn't support it
    const { reasoning_effort: _, ...rest } = payload
    payload = rest as typeof payload
  }

  // Check if this model supports Responses API
  if (supportsResponsesApi(payload.model)) {
    consola.debug("Using Responses API for model:", payload.model)
    return handleResponsesApi(c, payload)
  }

  // Use Chat Completions API for all other models
  return handleChatCompletionsApi(c, payload)
}

async function handleChatCompletionsApi(
  c: Context,
  payload: ChatCompletionsPayload,
) {
  const response = await createChatCompletions(payload)

  if (isNonStreaming(response)) {
    consola.debug("Non-streaming response:", JSON.stringify(response))
    return c.json(response)
  }

  consola.debug("Streaming response")
  return streamSSE(c, async (stream) => {
    for await (const chunk of response) {
      // Normalize reasoning_text -> reasoning_content in streaming chunks
      if (chunk.data) {
        normalizeReasoningText(chunk)
      }
      consola.debug("Streaming chunk:", JSON.stringify(chunk))
      await stream.writeSSE(chunk as SSEMessage)
    }
  })
}

interface ChunkData {
  choices?: Array<{
    delta?: { reasoning_text?: string; reasoning_content?: string }
  }>
}

function normalizeReasoningText(chunk: { data?: string }) {
  if (!chunk.data) return

  try {
    const parsed = JSON.parse(chunk.data) as ChunkData
    if (!parsed.choices) return

    for (const choice of parsed.choices) {
      if (choice.delta && "reasoning_text" in choice.delta) {
        choice.delta.reasoning_content = choice.delta.reasoning_text
        delete choice.delta.reasoning_text
      }
    }
    chunk.data = JSON.stringify(parsed)
  } catch {
    // Ignore parse errors
  }
}

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")

async function handleResponsesApi(c: Context, payload: ChatCompletionsPayload) {
  const responsesPayload = convertToResponsesPayload(payload)
  consola.debug("Responses API payload:", JSON.stringify(responsesPayload))

  const response = await createResponses(responsesPayload)

  if (isResponseObject(response)) {
    consola.debug(
      "Non-streaming Responses API response:",
      JSON.stringify(response),
    )
    const chatCompletion = convertResponseToChatCompletion(response)
    return c.json(chatCompletion)
  }

  consola.debug("Streaming Responses API response")
  return streamSSE(c, async (stream) => {
    await processResponseStream(stream, response, payload.model)
  })
}
