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
const RESPONSES_API_PREFIXES = ["o", "gpt-5"]

function supportsResponsesApi(model: string): boolean {
  return RESPONSES_API_PREFIXES.some((prefix) => model.startsWith(prefix))
}

// Effort suffix mappings
const EFFORT_SUFFIXES = [
  "thinking",
  "xhigh",
  "high",
  "medium",
  "low",
  "minimal",
  "none",
] as const

const SUFFIX_TO_EFFORT: Record<string, string> = {
  thinking: "high",
  xhigh: "xhigh",
  high: "high",
  medium: "medium",
  low: "low",
  minimal: "minimal",
  none: "none",
}

const EFFORT_TO_RATIO: Record<string, number | undefined> = {
  none: 0,
  minimal: 0.1,
  low: 0.2,
  medium: 0.4,
  high: 0.6,
  xhigh: 0.8,
}

function parseReasoningEffortSuffix(
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload {
  for (const suffix of EFFORT_SUFFIXES) {
    if (payload.model.endsWith(`-${suffix}`)) {
      const baseModel = payload.model.slice(0, -(suffix.length + 1))
      const effort = SUFFIX_TO_EFFORT[suffix]
      consola.debug(
        `Parsed model suffix "-${suffix}" -> model: ${baseModel}, reasoning_effort: ${effort}`,
      )
      return {
        ...payload,
        model: baseModel,
        reasoning_effort: effort as typeof payload.reasoning_effort,
      }
    }
  }
  return payload
}

function convertReasoningEffortForClaude(
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload {
  if (
    !payload.reasoning_effort
    || payload.thinking_budget
    || !payload.model.startsWith("claude")
  ) {
    return payload
  }

  const ratio = EFFORT_TO_RATIO[payload.reasoning_effort]
  if (ratio !== undefined && ratio > 0 && payload.max_tokens) {
    const minBudget = 1024
    const calculatedBudget = Math.floor(payload.max_tokens * ratio)
    const budget = Math.max(minBudget, calculatedBudget)
    consola.debug(
      `Converted reasoning_effort "${payload.reasoning_effort}" to thinking_budget: ${budget}`,
    )
    const { reasoning_effort: _, ...rest } = payload
    return { ...rest, thinking_budget: budget } as typeof payload
  }

  const { reasoning_effort: _, ...rest } = payload
  return rest as typeof payload
}

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  let payload = await c.req.json<ChatCompletionsPayload>()
  const acceptHeader = c.req.header("accept") ?? ""
  const wantsEventStream = acceptHeader.includes("text/event-stream")

  if (isNullish(payload.stream) && wantsEventStream) {
    payload = { ...payload, stream: true }
  }

  consola.debug("Request payload:", JSON.stringify(payload).slice(-400))

  payload = parseReasoningEffortSuffix(payload)

  const selectedModel = state.models?.data.find((m) => m.id === payload.model)

  try {
    if (selectedModel) {
      const tokenCount = await getTokenCount(payload, selectedModel)
      consola.info("Current token count:", tokenCount)
    }
  } catch (error) {
    consola.warn("Failed to calculate token count:", error)
  }

  if (state.manualApprove) await awaitApproval()

  if (isNullish(payload.max_tokens) && selectedModel) {
    payload = {
      ...payload,
      max_tokens: selectedModel.capabilities.limits.max_output_tokens,
    }
  }

  payload = convertReasoningEffortForClaude(payload)

  if (supportsResponsesApi(payload.model)) {
    consola.debug("Using Responses API for model:", payload.model)
    return handleResponsesApi(c, payload)
  }

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
