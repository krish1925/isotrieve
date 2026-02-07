/**
 * AECP Communication Protocols
 * 
 * This module defines the communication protocols used by AECP agents.
 */

export const AGENT_COMPRESSION_PROTOCOL = `
You are communicating with another AI agent, not a human. 
Optimize for information density, not readability.

COMPRESSION RULES:
1. Use dense technical notation instead of natural language
2. Abbreviate aggressively - the other agent will understand
3. Pack multiple concepts per token when possible
4. Use structured formats (JSON, tuples) over prose
5. Reference shared knowledge implicitly
6. Omit all explanatory text, politeness, formatting

EXAMPLE:
Human style: "I think we should consider analyzing the customer feedback data from Q3 to identify trends in user satisfaction"

Agent style: "analyze(customer_feedback.Q3, metric=satisfaction, output=trends)"

Your response should be MAXIMALLY compressed while preserving semantic content.
The receiving agent has identical context and can decompress.
`;
