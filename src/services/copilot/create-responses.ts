import consola from "consola"
import { events } from "fetch-event-stream"

import { copilotHeaders, copilotBaseUrl } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { state } from "~/lib/state"

import type {
  ChatCompletionsPayload,
  ToolCall,
} from "./create-chat-completions"

// Response API input types
export interface ResponseInputItem {
  type?: "message"
  role: "user" | "assistant" | "system" | "developer"
  content: string | Array<ResponseContentPart>
  id?: string
  status?: string
}

export interface ResponseTextPart {
  type: "input_text" | "output_text" | "text"
  text: string
  annotations?: Array<unknown>
}

export interface ResponseImagePart {
  type: "input_image"
  image_url: string
}

export type ResponseContentPart = ResponseTextPart | ResponseImagePart

export interface ResponsesPayload {
  model: string
  input: string | Array<ResponseInputItem>
  instructions?: string
  reasoning?: {
    effort?: "none" | "low" | "medium" | "high"
    summary?: "auto" | "concise" | "detailed"
  }
  max_output_tokens?: number
  temperature?: number
  top_p?: number
  stream?: boolean
  store?: boolean
  tools?: Array<ResponseTool>
  tool_choice?:
    | "none"
    | "auto"
    | "required"
    | { type: "function"; name: string }
  parallel_tool_calls?: boolean
  include?: Array<string>
}

export interface ResponseTool {
  type: "function"
  name: string
  description?: string
  parameters: Record<string, unknown>
  strict?: boolean
}

// Response API output types
export interface ResponseObject {
  id: string
  object: "response"
  created_at: number
  status:
    | "completed"
    | "failed"
    | "in_progress"
    | "cancelled"
    | "queued"
    | "incomplete"
  completed_at?: number
  error?: {
    code: string
    message: string
  } | null
  incomplete_details?: {
    reason: string
  } | null
  instructions?: string | null
  max_output_tokens?: number | null
  model: string
  output: Array<ResponseOutputItem>
  parallel_tool_calls?: boolean
  previous_response_id?: string | null
  reasoning?: {
    effort?: string | null
    summary?: string | null
  }
  store?: boolean
  temperature?: number
  top_p?: number
  truncation?: string
  usage?: {
    input_tokens: number
    input_tokens_details?: {
      cached_tokens: number
    }
    output_tokens: number
    output_tokens_details?: {
      reasoning_tokens: number
    }
    total_tokens: number
  }
}

export type ResponseOutputItem =
  | ResponseMessageItem
  | ResponseReasoningItem
  | ResponseFunctionCallItem

export interface ResponseMessageItem {
  type: "message"
  id: string
  status: "completed" | "in_progress"
  role: "assistant"
  content: Array<ResponseMessageContent>
}

export interface ResponseMessageContent {
  type: "output_text"
  text: string
  annotations?: Array<unknown>
}

export interface ResponseReasoningItem {
  type: "reasoning"
  id: string
  summary?: Array<{
    type: "summary_text"
    text: string
  }>
  encrypted_content?: string
}

export interface ResponseFunctionCallItem {
  type: "function_call"
  id: string
  call_id: string
  name: string
  arguments: string
  status?: "completed" | "in_progress"
}

// Helper functions for payload conversion

function extractInstructions(
  messages: ChatCompletionsPayload["messages"],
): string | undefined {
  let instructions: string | undefined

  for (const msg of messages) {
    if (msg.role === "system" || msg.role === "developer") {
      const content =
        typeof msg.content === "string" ?
          msg.content
        : (msg.content
            ?.filter((p) => p.type === "text")
            .map((p) => (p as { text: string }).text)
            .join("\n") ?? "")
      instructions = instructions ? `${instructions}\n${content}` : content
    }
  }

  return instructions
}

function convertMessageToInputItem(
  msg: ChatCompletionsPayload["messages"][number],
): ResponseInputItem | null {
  if (msg.role === "system" || msg.role === "developer") {
    return null
  }

  return {
    type: "message",
    role: msg.role as "user" | "assistant",
    content:
      typeof msg.content === "string" ?
        msg.content
      : (msg.content?.map((part): ResponseContentPart => {
          if (part.type === "text") {
            return { type: "input_text" as const, text: part.text }
          }
          // Handle image_url type - convert to Responses API format
          return {
            type: "input_image" as const,
            image_url: part.image_url.url,
          }
        }) ?? ""),
  }
}

function buildReasoningConfig(effort: string | undefined) {
  if (!effort || effort === "none") {
    return undefined
  }

  const effortMap: Record<string, "low" | "medium" | "high"> = {
    minimal: "low",
    low: "low",
    medium: "medium",
    high: "high",
    xhigh: "high",
  }

  return {
    effort: effortMap[effort] ?? "medium",
    summary: "auto" as const,
  }
}

function convertTools(
  tools: ChatCompletionsPayload["tools"],
): Array<ResponseTool> | undefined {
  if (!tools || tools.length === 0) {
    return undefined
  }

  return tools.map((tool) => ({
    type: "function" as const,
    name: tool.function.name,
    description: tool.function.description,
    parameters: tool.function.parameters,
  }))
}

function convertToolChoice(
  toolChoice: ChatCompletionsPayload["tool_choice"],
): ResponsesPayload["tool_choice"] {
  if (!toolChoice) {
    return undefined
  }

  if (typeof toolChoice === "string") {
    return toolChoice
  }

  return {
    type: "function",
    name: toolChoice.function.name,
  }
}

/**
 * Convert Chat Completions payload to Responses API payload
 */
export function convertToResponsesPayload(
  payload: ChatCompletionsPayload,
): ResponsesPayload {
  const instructions = extractInstructions(payload.messages)
  const inputMessages = payload.messages
    .map((msg) => convertMessageToInputItem(msg))
    .filter((item): item is ResponseInputItem => item !== null)

  const responsesPayload: ResponsesPayload = {
    model: payload.model,
    input: inputMessages,
    stream: payload.stream ?? false,
    store: false,
  }

  if (instructions) {
    responsesPayload.instructions = instructions
  }

  const reasoning = buildReasoningConfig(payload.reasoning_effort)
  if (reasoning) {
    responsesPayload.reasoning = reasoning
    responsesPayload.include = ["reasoning.encrypted_content"]
  }

  if (payload.max_tokens) {
    responsesPayload.max_output_tokens = payload.max_tokens
  }

  if (payload.temperature !== undefined && payload.temperature !== null) {
    responsesPayload.temperature = payload.temperature
  }

  if (payload.top_p !== undefined && payload.top_p !== null) {
    responsesPayload.top_p = payload.top_p
  }

  const tools = convertTools(payload.tools)
  if (tools) {
    responsesPayload.tools = tools
  }

  const toolChoice = convertToolChoice(payload.tool_choice)
  if (toolChoice) {
    responsesPayload.tool_choice = toolChoice
  }

  return responsesPayload
}

// Helper functions for response conversion

interface ExtractedContent {
  textContent: string
  reasoningContent: string
  toolCalls: Array<ToolCall>
}

function extractMessageContent(
  item: ResponseMessageItem,
  result: ExtractedContent,
) {
  for (const content of item.content) {
    result.textContent += content.text
  }
}

function extractReasoningContent(
  item: ResponseReasoningItem,
  result: ExtractedContent,
) {
  if (!item.summary) return
  for (const summary of item.summary) {
    result.reasoningContent += summary.text
  }
}

function extractFunctionCall(
  item: ResponseFunctionCallItem,
  result: ExtractedContent,
) {
  result.toolCalls.push({
    id: item.call_id,
    type: "function",
    function: {
      name: item.name,
      arguments: item.arguments,
    },
  })
}

function extractOutputContent(
  output: Array<ResponseOutputItem>,
): ExtractedContent {
  const result: ExtractedContent = {
    textContent: "",
    reasoningContent: "",
    toolCalls: [],
  }

  for (const item of output) {
    switch (item.type) {
      case "message": {
        extractMessageContent(item, result)
        break
      }
      case "reasoning": {
        extractReasoningContent(item, result)
        break
      }
      case "function_call": {
        extractFunctionCall(item, result)
        break
      }
      default: {
        // Ignore unknown item types
        break
      }
    }
  }

  return result
}

function determineFinishReason(
  toolCalls: Array<ToolCall>,
  status: ResponseObject["status"],
): "stop" | "length" | "tool_calls" | "content_filter" {
  if (toolCalls.length > 0) {
    return "tool_calls"
  }
  if (status === "incomplete") {
    return "length"
  }
  return "stop"
}

function buildMessage(content: ExtractedContent) {
  const message: {
    role: "assistant"
    content: string | null
    reasoning_content?: string | null
    tool_calls?: Array<ToolCall>
  } = {
    role: "assistant",
    content: content.textContent || null,
  }

  if (content.reasoningContent) {
    message.reasoning_content = content.reasoningContent
  }

  if (content.toolCalls.length > 0) {
    message.tool_calls = content.toolCalls
  }

  return message
}

function buildUsage(usage: ResponseObject["usage"]) {
  if (!usage) return undefined
  return {
    prompt_tokens: usage.input_tokens,
    completion_tokens: usage.output_tokens,
    total_tokens: usage.total_tokens,
  }
}

/**
 * Convert Response API response to Chat Completions format
 */
export function convertResponseToChatCompletion(response: ResponseObject) {
  const content = extractOutputContent(response.output)
  const finishReason = determineFinishReason(content.toolCalls, response.status)
  const message = buildMessage(content)

  return {
    id: response.id,
    object: "chat.completion" as const,
    created: response.created_at,
    model: response.model,
    choices: [
      {
        index: 0,
        message,
        logprobs: null,
        finish_reason: finishReason,
      },
    ],
    usage: buildUsage(response.usage),
  }
}

/**
 * Check if the payload contains any images
 */
function hasImageContent(payload: ResponsesPayload): boolean {
  if (typeof payload.input === "string") return false

  return payload.input.some((item) => {
    if (typeof item.content === "string") return false
    return item.content.some((part) => part.type === "input_image")
  })
}

/**
 * Create a response using the Responses API
 */
export const createResponses = async (payload: ResponsesPayload) => {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  const enableVision = hasImageContent(payload)

  const headers: Record<string, string> = {
    ...copilotHeaders(state, enableVision),
  }

  consola.debug("Responses API payload:", JSON.stringify(payload))

  const response = await fetch(`${copilotBaseUrl(state)}/responses`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const errorText = await response.text()
    consola.error("Failed to create responses", response.status, errorText)
    throw new HTTPError("Failed to create responses", response)
  }

  if (payload.stream) {
    return events(response)
  }

  const result = (await response.json()) as ResponseObject
  return result
}

/**
 * Check if the response is a non-streaming ResponseObject
 */
export const isResponseObject = (
  response: Awaited<ReturnType<typeof createResponses>>,
): response is ResponseObject => {
  return typeof response === "object" && "object" in response
}
