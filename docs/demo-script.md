# Product demo script (about 90 seconds)

## Preflight

- Open the HTTPS production URL in a clean browser tab.
- Confirm the header says `INDEX ONLINE`.
- Confirm `/health` has ElevenLabs ready and vector points greater than zero.
- Prepare one supported query from the indexed evaluation sample and one unrelated query.
- Start screen recording before the first interaction.

## Flow

**0–10 s — product and claim**

Show the NEXUS landing view. Say: “NEXUS is a multilingual voice RAG system over AI4Bharat MSMARCO-XI. It exposes retrieval and refuses unsupported answers.”

**10–30 s — real voice input**

Click the microphone, speak the supported question, and stop. Keep the transcription panel visible. Point to `ElevenLabs / scribe_v2` and the measured STT time.

**30–55 s — transparent pipeline**

Show the harness stages completing. Read the grounded answer, `VERIFIED` state, generator identity, confidence, and total latency. Do not edit or skip a slow response.

**55–72 s — evidence**

Expand the first retrieved source. Show document ID, chunk ID, actual passage, similarity score, language, query type, strategy, and position.

**72–88 s — guardrail**

Ask the unrelated question. Show the low-relevance/no-context refusal and failed context-validation trace.

**88–90 s — close**

End on the performance panel and repository name.

## Demo integrity

- Do not use hard-coded answers or seeded browser state.
- Keep the network panel available if judges ask.
- If STT fails, show the error rather than substituting typed text while claiming voice worked.
- State whether latency shown is text RAG or full voice-to-answer.
