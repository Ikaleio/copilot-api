import { Hono } from 'hono'

import { forwardError } from '~/lib/error'
import {
	createEmbeddings,
	type EmbeddingRequest
} from '~/services/copilot/create-embeddings'

export const embeddingRoutes = new Hono()

embeddingRoutes.post('/', async c => {
	try {
		const payload = await c.req.json<EmbeddingRequest>()

		// Normalize input to array format (Copilot API only accepts array)
		const normalizedPayload: EmbeddingRequest = {
			...payload,
			input: Array.isArray(payload.input) ? payload.input : [payload.input]
		}

		const response = await createEmbeddings(normalizedPayload)

		return c.json(response)
	} catch (error) {
		return await forwardError(c, error)
	}
})
