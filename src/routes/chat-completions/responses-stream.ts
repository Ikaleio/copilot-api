import type { SSEStreamingApi } from "hono/streaming"

import consola from "consola"

interface StreamState {
  responseId: string
  model: string
  toolCalls: Array<ToolCallState>
  roleEmitted: boolean
}

interface ToolCallState {
  index: number
  id?: string
  type?: "function"
  function?: {
    name?: string
    arguments?: string
  }
}

interface ResponseEvent {
  event?: string
  data?: string
}

function createChunk(
  state: StreamState,
  delta: Record<string, unknown>,
  finishReason: string | null = null,
) {
  return {
    id: state.responseId || `chatcmpl-${Date.now()}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: state.model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason,
        logprobs: null,
      },
    ],
  }
}

async function emitRoleIfNeeded(stream: SSEStreamingApi, state: StreamState) {
  if (state.roleEmitted) return
  state.roleEmitted = true
  // Include content: "" for OpenWebUI compatibility
  const chunk = createChunk(state, { role: "assistant", content: "" })
  await stream.writeSSE({ data: JSON.stringify(chunk) })
}

function handleCreatedEvent(data: Record<string, unknown>, state: StreamState) {
  if (data.id) state.responseId = data.id as string
  if (!state.model && data.model) state.model = data.model as string
}

async function handleReasoningDelta(
  stream: SSEStreamingApi,
  data: Record<string, unknown>,
  state: StreamState,
) {
  const delta = (data.delta as string) || ""
  await emitRoleIfNeeded(stream, state)
  const chunk = createChunk(state, { reasoning_content: delta })
  await stream.writeSSE({ data: JSON.stringify(chunk) })
}

async function handleOutputTextDelta(
  stream: SSEStreamingApi,
  data: Record<string, unknown>,
  state: StreamState,
) {
  const delta = (data.delta as string) || ""
  await emitRoleIfNeeded(stream, state)
  const chunk = createChunk(state, { content: delta })
  await stream.writeSSE({ data: JSON.stringify(chunk) })
}

async function handleFunctionCallDelta(
  stream: SSEStreamingApi,
  data: Record<string, unknown>,
  state: StreamState,
) {
  const toolIndex = state.toolCalls.length > 0 ? state.toolCalls.length - 1 : 0

  if (!state.toolCalls[toolIndex]) {
    state.toolCalls[toolIndex] = {
      index: toolIndex,
      type: "function",
      function: { arguments: "" },
    }
  }

  const toolCall = state.toolCalls[toolIndex]
  toolCall.function = toolCall.function || { arguments: "" }
  toolCall.function.arguments =
    (toolCall.function.arguments || "") + ((data.delta as string) || "")

  await emitRoleIfNeeded(stream, state)
  const chunk = createChunk(state, {
    tool_calls: [
      {
        index: toolIndex,
        function: { arguments: (data.delta as string) || "" },
      },
    ],
  })
  await stream.writeSSE({ data: JSON.stringify(chunk) })
}

async function handleFunctionCallCreated(
  stream: SSEStreamingApi,
  data: Record<string, unknown>,
  state: StreamState,
) {
  const item = data.item as Record<string, unknown>
  const toolIndex = state.toolCalls.length

  state.toolCalls.push({
    index: toolIndex,
    id: item.call_id as string,
    type: "function",
    function: {
      name: item.name as string,
      arguments: "",
    },
  })

  await emitRoleIfNeeded(stream, state)
  const chunk = createChunk(state, {
    tool_calls: [
      {
        index: toolIndex,
        id: item.call_id,
        type: "function",
        function: { name: item.name, arguments: "" },
      },
    ],
  })
  await stream.writeSSE({ data: JSON.stringify(chunk) })
}

async function handleOutputItemDone(
  stream: SSEStreamingApi,
  data: Record<string, unknown>,
  state: StreamState,
) {
  const item = data.item as Record<string, unknown> | undefined
  let finishReason: string | null = null

  if (item?.type === "message") {
    finishReason = "stop"
  } else if (item?.type === "function_call") {
    finishReason = "tool_calls"
  }

  if (finishReason) {
    const chunk = createChunk(state, {}, finishReason)
    await stream.writeSSE({ data: JSON.stringify(chunk) })
  }
}

function isReasoningEvent(
  eventType: string | undefined,
  dataType: string | undefined,
) {
  return (
    eventType === "response.reasoning_summary_text.delta"
    || eventType === "response.reasoning.delta"
    || dataType === "response.reasoning_summary_text.delta"
  )
}

function isOutputTextEvent(
  eventType: string | undefined,
  dataType: string | undefined,
) {
  return (
    eventType === "response.output_text.delta"
    || eventType === "response.content_part.delta"
    || dataType === "response.output_text.delta"
  )
}

/**
 * Process a streaming response from the Responses API and convert to Chat Completions format
 */
export async function processResponseStream(
  stream: SSEStreamingApi,
  events: AsyncIterable<ResponseEvent>,
  model: string,
) {
  const state: StreamState = {
    responseId: "",
    model,
    toolCalls: [],
    roleEmitted: false,
  }

  for await (const event of events) {
    if (!event.data || event.data === "[DONE]") {
      await stream.writeSSE({ data: "[DONE]" })
      continue
    }

    try {
      const data = JSON.parse(event.data) as Record<string, unknown>
      const eventType = event.event
      const dataType = data.type as string | undefined

      consola.debug(
        "Response API event:",
        eventType,
        JSON.stringify(data).slice(0, 200),
      )

      // Handle response created/in_progress
      if (
        eventType === "response.created"
        || eventType === "response.in_progress"
      ) {
        handleCreatedEvent(data, state)
        continue
      }

      // Handle response completed/done
      if (eventType === "response.completed" || eventType === "response.done") {
        await stream.writeSSE({ data: "[DONE]" })
        continue
      }

      // Handle reasoning content delta
      if (isReasoningEvent(eventType, dataType)) {
        await handleReasoningDelta(stream, data, state)
        continue
      }

      // Handle output text delta
      if (isOutputTextEvent(eventType, dataType)) {
        await handleOutputTextDelta(stream, data, state)
        continue
      }

      // Handle function call arguments delta
      if (eventType === "response.function_call_arguments.delta") {
        await handleFunctionCallDelta(stream, data, state)
        continue
      }

      // Handle function call created
      if (eventType === "response.output_item.added") {
        const item = data.item as Record<string, unknown> | undefined
        if (item?.type === "function_call") {
          await handleFunctionCallCreated(stream, data, state)
        }
        continue
      }

      // Handle output item done (finish reason)
      if (eventType === "response.output_item.done") {
        await handleOutputItemDone(stream, data, state)
        continue
      }
    } catch (parseError) {
      consola.warn("Failed to parse Response API event:", parseError)
    }
  }
}
