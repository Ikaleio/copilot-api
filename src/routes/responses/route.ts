import { Hono } from "hono"
import { streamSSE, type SSEMessage } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import {
  createResponses,
  isResponseObject,
  type ResponsesPayload,
} from "~/services/copilot/create-responses"

export const responsesRoutes = new Hono()

responsesRoutes.post("/", async (c) => {
  await checkRateLimit(state)

  const payload = await c.req.json<ResponsesPayload>()

  if (state.manualApprove) await awaitApproval()

  const response = await createResponses(payload)

  if (isResponseObject(response)) {
    return c.json(response)
  }

  return streamSSE(c, async (stream) => {
    for await (const event of response) {
      if (!event.data) continue

      if (event.data === "[DONE]") {
        await stream.writeSSE({ data: "[DONE]" })
        continue
      }

      const message: SSEMessage = {
        event: event.event,
        data: event.data,
      }
      await stream.writeSSE(message)
    }
  })
})
