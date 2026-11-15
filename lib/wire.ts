// Translate OpenAI and Anthropic clients to OpenAI, Anthropic or Gemini providers.
// OpenAI completions are the internal response format. Cross-wire SSE is collected before emission
// so usage and tool inputs are complete; native streams need no translation or extra provider call.
// All generated identities and timestamps are deterministic functions of the upstream answer.

import { isRecord, type Json, type Wire } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    let parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        if (parts.length > 0) {
          messages.push({ role: 'user', content: parts });
          parts = [];
        }
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: Array.isArray(block.content) && block.content.some(b => isRecord(b) && b.type === 'image')
            ? block.content.filter(isRecord).flatMap(b => { const part = partOf(b); return part === null ? [] : [part]; })
            : toolResultText(block.content),
          ...(block.is_error === true ? { is_error: true } : {}),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (isRecord(body.response_format)) out.response_format = body.response_format;
  if (isRecord(body.output_config) && isRecord(body.output_config.format)) {
    out.response_format = { type: 'json_schema', json_schema: { name: 'response', schema: body.output_config.format.schema } };
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'none') out.tool_choice = 'none';
    if (typeof choice.disable_parallel_tool_use === 'boolean') out.parallel_tool_calls = !choice.disable_parallel_tool_use;
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI content part -> Anthropic block, for both messages and multimodal tool results. */
function anthropicPart(part: Json): Json | null {
  if (part.type === 'text') return { type: 'text', text: part.text };
  if (part.type !== 'image_url') return null;
  const url = isRecord(part.image_url) ? part.image_url.url : part.image_url;
  if (typeof url !== 'string') return null;
  const data = /^data:([^;,]+);base64,([\s\S]*)$/.exec(url);
  return { type: 'image', source: data
    ? { type: 'base64', media_type: data[1], data: data[2] }
    : { type: 'url', url } };
}

/** OpenAI request -> Anthropic Messages, including multimodal and tool history. */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system: string[] = [];
  for (const raw of records(body.messages)) {
    if (raw.role === 'system' || raw.role === 'developer') {
      system.push(textOf(raw.content));
      continue;
    }
    const content: Json[] = [];
    if (raw.role === 'tool') {
      content.push({ type: 'tool_result', tool_use_id: raw.tool_call_id,
        content: Array.isArray(raw.content) ? records(raw.content).flatMap(part => {
          const block = anthropicPart(part); return block === null ? [] : [block];
        }) : toolResultText(raw.content),
        ...(raw.is_error === true ? { is_error: true } : {}),
      });
    } else {
      if (typeof raw.content === 'string' && raw.content !== '') content.push({ type: 'text', text: raw.content });
      for (const part of records(raw.content)) {
        const block = anthropicPart(part);
        if (block !== null) content.push(block);
      }
      for (const call of records(raw.tool_calls)) {
        const fn = isRecord(call.function) ? call.function : {};
        content.push({ type: 'tool_use', id: call.id, name: fn.name, input: argumentsOf(fn.arguments) });
      }
    }
    if (content.length === 0) content.push({ type: 'text', text: '' });
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const previous = messages.at(-1);
    if (previous?.role === role && Array.isArray(previous.content)) previous.content.push(...content);
    else messages.push({ role, content });
  }
  const out: Json = { model, messages, max_tokens: body.max_tokens ?? body.max_completion_tokens ?? 1024 };
  if (system.length) out.system = system.join('\n');
  for (const key of ['temperature', 'top_p', 'stream']) if (body[key] !== undefined) out[key] = body[key];
  if (body.stop !== undefined) out.stop_sequences = Array.isArray(body.stop) ? body.stop : [body.stop];
  const tools = records(body.tools).flatMap(tool => isRecord(tool.function) ? [{
    name: tool.function.name, description: tool.function.description ?? '',
    input_schema: tool.function.parameters ?? { type: 'object', properties: {} },
  }] : []);
  if (tools.length) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === 'none') out.tool_choice = { type: choice };
  if (choice === 'required') out.tool_choice = { type: 'any' };
  if (isRecord(choice) && isRecord(choice.function)) out.tool_choice = { type: 'tool', name: choice.function.name };
  if (tools.length && typeof body.parallel_tool_calls === 'boolean') {
    if (!isRecord(out.tool_choice)) out.tool_choice = { type: 'auto' };
    (out.tool_choice as Json).disable_parallel_tool_use = !body.parallel_tool_calls;
  }
  if (isRecord(body.response_format)) {
    // Preserve the common simulator extension as well as Anthropic structured outputs.
    out.response_format = body.response_format;
    if (body.response_format.type === 'json_schema' && isRecord(body.response_format.json_schema)) {
      out.output_config = { format: { type: 'json_schema', schema: body.response_format.json_schema.schema } };
    }
  }
  return out;
}

const records = (value: unknown): Json[] => Array.isArray(value) ? value.filter(isRecord) : [];

/** OpenAI request -> Gemini generateContent. Tool result names are recovered from call history. */
export function openAIRequestToGemini(body: Json): Json {
  const contents: Json[] = [];
  const system: Json[] = [];
  const names = new Map<string, string>();
  for (const message of records(body.messages)) {
    const parts: Json[] = [];
    if (message.role === 'system' || message.role === 'developer') {
      system.push({ text: textOf(message.content) });
      continue;
    }
    if (message.role === 'tool') {
      const value = toolResultText(message.content);
      let response: unknown = value;
      try { response = JSON.parse(value); } catch { /* Plain text result. */ }
      parts.push({ functionResponse: {
        id: message.tool_call_id,
        name: names.get(String(message.tool_call_id)) ?? message.name ?? String(message.tool_call_id ?? 'tool'),
        response: isRecord(response) ? response : { result: response },
      } });
    } else {
      if (typeof message.content === 'string' && message.content !== '') parts.push({ text: message.content });
      for (const part of records(message.content)) {
        if (part.type === 'text') parts.push({ text: part.text });
        if (part.type === 'image_url') {
          const url = isRecord(part.image_url) ? part.image_url.url : part.image_url;
          if (typeof url === 'string') {
            const data = /^data:([^;,]+);base64,([\s\S]*)$/.exec(url);
            parts.push(data ? { inlineData: { mimeType: data[1], data: data[2] } }
              : { fileData: { mimeType: 'image/jpeg', fileUri: url } });
          }
        }
      }
      for (const call of records(message.tool_calls)) {
        const fn = isRecord(call.function) ? call.function : {};
        names.set(String(call.id), String(fn.name ?? ''));
        parts.push({ functionCall: { id: call.id, name: fn.name, args: argumentsOf(fn.arguments) } });
      }
    }
    if (parts.length === 0) parts.push({ text: '' });
    const role = message.role === 'assistant' ? 'model' : 'user';
    const previous = contents.at(-1);
    if (previous?.role === role && Array.isArray(previous.parts)) previous.parts.push(...parts);
    else contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (system.length) out.systemInstruction = { parts: system };
  const generationConfig: Json = {};
  if (body.max_tokens !== undefined || body.max_completion_tokens !== undefined) generationConfig.maxOutputTokens = body.max_tokens ?? body.max_completion_tokens;
  if (body.temperature !== undefined) generationConfig.temperature = body.temperature;
  if (body.top_p !== undefined) generationConfig.topP = body.top_p;
  if (body.stop !== undefined) generationConfig.stopSequences = Array.isArray(body.stop) ? body.stop : [body.stop];
  if (isRecord(body.response_format) && ['json_object', 'json_schema'].includes(String(body.response_format.type))) {
    generationConfig.responseMimeType = 'application/json';
    if (isRecord(body.response_format.json_schema)) generationConfig.responseSchema = body.response_format.json_schema.schema;
  }
  if (Object.keys(generationConfig).length) out.generationConfig = generationConfig;
  const declarations = records(body.tools).flatMap(tool => isRecord(tool.function) ? [{
    name: tool.function.name, description: tool.function.description ?? '',
    parameters: tool.function.parameters ?? { type: 'object', properties: {} },
  }] : []);
  if (declarations.length) out.tools = [{ functionDeclarations: declarations }];
  const choice = body.tool_choice;
  if (choice !== undefined) {
    const config: Json = { mode: choice === 'none' ? 'NONE' : choice === 'required' || isRecord(choice) ? 'ANY' : 'AUTO' };
    if (isRecord(choice) && isRecord(choice.function)) config.allowedFunctionNames = [choice.function.name];
    out.toolConfig = { functionCallingConfig: config };
  }
  return out;
}

/** Build the provider-specific request without ever replacing the original request between retries. */
export function upstreamRequest(body: Json, client: 'openai' | 'anthropic', wire: Wire, model: string): Json {
  if (client === wire) {
    const native: Json = { ...body, model };
    if (wire === 'openai' && body.stream === true) native.stream_options = {
      ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true,
    };
    return native;
  }
  const canonical = client === 'anthropic' ? anthropicToOpenAI(body, model) : { ...body, model };
  if (wire === 'anthropic') return openAIRequestToAnthropic(canonical, model);
  if (wire === 'gemini') return openAIRequestToGemini(canonical);
  if (body.stream === true) canonical.stream_options = { include_usage: true };
  return canonical;
}

/** Anthropic JSON response -> canonical OpenAI completion. */
export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const content = records(body.content);
  const calls = content.filter(b => b.type === 'tool_use').map((block, i) => ({
    id: block.id ?? `call_${i}`, type: 'function',
    function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
  }));
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    ...completion(String(body.id ?? 'aggregate'), model,
      content.filter(b => b.type === 'text' && typeof b.text === 'string').map(b => b.text).join(''), calls,
      body.stop_reason === 'tool_use' ? 'tool_calls' : body.stop_reason === 'max_tokens' ? 'length' : body.stop_reason === 'refusal' ? 'content_filter' : 'stop',
      numberOrZero(usage.input_tokens), numberOrZero(usage.output_tokens)),
    ...(typeof body.stop_sequence === 'string' ? { stop_sequence: body.stop_sequence } : {}),
  };
}

function completion(id: string, model: string, text: string, calls: Json[], finish: string, input: number, output: number, total = input + output): Json {
  return { id, object: 'chat.completion', created: FIXED_OPENAI_CREATED, model,
    choices: [{ index: 0, message: { role: 'assistant', content: text || (calls.length ? null : ''), ...(calls.length ? { tool_calls: calls } : {}) }, finish_reason: finish }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: total } };
}

function geminiFinishReason(reason: unknown, hasCalls: boolean): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (['SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'IMAGE_SAFETY', 'IMAGE_PROHIBITED_CONTENT'].includes(String(reason))) return 'content_filter';
  return hasCalls ? 'tool_calls' : 'stop';
}

/** Gemini JSON response -> canonical OpenAI completion. */
export function geminiResponseToOpenAI(body: Json, model: string): Json {
  const candidate = records(body.candidates)[0] ?? {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = records(content.parts);
  const calls = parts.filter(p => isRecord(p.functionCall)).map((part, i) => {
    const fn = part.functionCall as Json;
    return { id: fn.id ?? `call_${i}`, type: 'function', function: { name: fn.name, arguments: JSON.stringify(fn.args ?? {}) } };
  });
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const finish = geminiFinishReason(candidate.finishReason, calls.length > 0);
  const input = numberOrZero(usage.promptTokenCount);
  const output = numberOrZero(usage.candidatesTokenCount);
  return completion(String(body.responseId ?? 'aggregate'), model,
    parts.filter(p => typeof p.text === 'string' && p.thought !== true).map(p => p.text).join(''), calls, finish,
    input, output, typeof usage.totalTokenCount === 'number' ? usage.totalTokenCount : input + output);
}

export function responseToOpenAI(body: Json, wire: Wire, model: string): Json {
  return wire === 'anthropic' ? anthropicResponseToOpenAI(body, model)
    : wire === 'gemini' ? geminiResponseToOpenAI(body, model) : body;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: typeof body.stop_sequence === 'string' ? 'stop_sequence' : stopReasonOf(choice.finish_reason),
    stop_sequence: typeof body.stop_sequence === 'string' ? body.stop_sequence : null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Parse SSE independently of transport chunk boundaries, including CRLF and multiline data. */
export async function* readSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let data: string[] = [];
  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      if (done && buffer !== '') buffer += '\n';
      for (;;) {
        const at = buffer.search(/[\r\n]/);
        if (at === -1 || (!done && buffer[at] === '\r' && at === buffer.length - 1)) break;
        const line = buffer.slice(0, at);
        const width = buffer[at] === '\r' && buffer[at + 1] === '\n' ? 2 : 1;
        buffer = buffer.slice(at + width);
        if (line === '') {
          if (data.length) yield data.join('\n');
          data = [];
        } else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
      }
      if (done) break;
    }
    if (data.length) yield data.join('\n');
  } finally {
    // Breaking at [DONE] or rejecting malformed events must not leave a provider stream active.
    try { await reader.cancel(); } finally { reader.releaseLock(); }
  }
}

/**
 * Assemble a cross-wire stream before emitting it: this lets Anthropic message_start contain
 * final input usage even when the source wire delivers usage only at the end. Native streams
 * are piped unchanged. No extra generation call is needed for this conversion.
 */
export async function collectStream(stream: ReadableStream<Uint8Array>, wire: Wire, model: string): Promise<Json> {
  let id = 'aggregate';
  let text = '';
  let finish = 'stop';
  let stopSequence: string | null = null;
  let input = 0;
  let output = 0;
  let total: number | undefined;
  const calls = new Map<number, { id: string; name: string; arguments: string }>();
  let nextGeminiCall = 0;
  for await (const payload of readSSE(stream)) {
    if (payload === '[DONE]') break;
    const value: unknown = JSON.parse(payload);
    if (!isRecord(value)) continue;
    if (value.error || value.type === 'error') throw new Error('upstream stream returned an error');
    if (wire === 'openai') {
      if (typeof value.id === 'string') id = value.id;
      if (isRecord(value.usage)) {
        if (typeof value.usage.prompt_tokens === 'number') input = value.usage.prompt_tokens;
        if (typeof value.usage.completion_tokens === 'number') output = value.usage.completion_tokens;
        if (typeof value.usage.total_tokens === 'number') total = value.usage.total_tokens;
      }
      const choice = records(value.choices)[0];
      if (!choice) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
      for (const call of records(delta.tool_calls)) {
        const index = typeof call.index === 'number' ? call.index : 0;
        const current = calls.get(index) ?? { id: `call_${index}`, name: '', arguments: '' };
        if (typeof call.id === 'string' && call.id !== '') current.id = call.id;
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.name === 'string') current.name += fn.name;
        if (typeof fn.arguments === 'string') current.arguments += fn.arguments;
        calls.set(index, current);
      }
      if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
    } else if (wire === 'anthropic') {
      if (value.type === 'message_start' && isRecord(value.message)) {
        id = String(value.message.id ?? id);
        const usage = isRecord(value.message.usage) ? value.message.usage : {};
        input = numberOrZero(usage.input_tokens);
        output = numberOrZero(usage.output_tokens);
      }
      const index = typeof value.index === 'number' ? value.index : 0;
      if (value.type === 'content_block_start' && isRecord(value.content_block)) {
        const block = value.content_block;
        if (block.type === 'text' && typeof block.text === 'string') text += block.text;
        if (block.type === 'tool_use') calls.set(index, {
          id: String(block.id ?? `call_${index}`), name: String(block.name ?? ''),
          arguments: isRecord(block.input) && Object.keys(block.input).length ? JSON.stringify(block.input) : '',
        });
      }
      if (value.type === 'content_block_delta' && isRecord(value.delta)) {
        if (value.delta.type === 'text_delta' && typeof value.delta.text === 'string') text += value.delta.text;
        if (value.delta.type === 'input_json_delta' && typeof value.delta.partial_json === 'string') {
          const call = calls.get(index);
          if (call) call.arguments += value.delta.partial_json;
        }
      }
      if (value.type === 'message_delta') {
        if (isRecord(value.delta)) {
          if (typeof value.delta.stop_sequence === 'string') stopSequence = value.delta.stop_sequence;
          if (typeof value.delta.stop_reason === 'string') {
            finish = value.delta.stop_reason === 'tool_use' ? 'tool_calls' : value.delta.stop_reason === 'max_tokens' ? 'length'
              : value.delta.stop_reason === 'refusal' ? 'content_filter' : 'stop';
          }
        }
        if (isRecord(value.usage)) {
          if (typeof value.usage.input_tokens === 'number') input = value.usage.input_tokens;
          if (typeof value.usage.output_tokens === 'number') output = value.usage.output_tokens;
        }
      }
    } else {
      if (typeof value.responseId === 'string') id = value.responseId;
      if (isRecord(value.usageMetadata)) {
        if (typeof value.usageMetadata.promptTokenCount === 'number') input = value.usageMetadata.promptTokenCount;
        if (typeof value.usageMetadata.candidatesTokenCount === 'number') output = value.usageMetadata.candidatesTokenCount;
        if (typeof value.usageMetadata.totalTokenCount === 'number') total = value.usageMetadata.totalTokenCount;
      }
      const candidate = records(value.candidates)[0] ?? {};
      const content = isRecord(candidate.content) ? candidate.content : {};
      for (const part of records(content.parts)) {
        if (typeof part.text === 'string' && part.thought !== true) text += part.text;
        if (isRecord(part.functionCall)) {
          const fn = part.functionCall;
          const index = nextGeminiCall++;
          calls.set(index, { id: String(fn.id ?? `call_${index}`), name: String(fn.name ?? ''), arguments: JSON.stringify(fn.args ?? {}) });
        }
      }
      if (candidate.finishReason) finish = geminiFinishReason(candidate.finishReason, calls.size > 0);
      else if (calls.size && finish === 'stop') finish = 'tool_calls';
    }
  }
  return {
    ...completion(id, model, text, [...calls.entries()].sort(([a], [b]) => a - b).map(([, call]) => ({
      id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments || '{}' },
    })), finish, input, output, total),
    ...(stopSequence !== null ? { stop_sequence: stopSequence } : {}),
  };
}

/** Emit complete, ordered client events with actual model identity and accurate usage. */
export function* clientEvents(answer: Json, client: 'openai' | 'anthropic', model: string): Generator<string> {
  if (client === 'anthropic') {
    const message = openAIToAnthropic(answer, model);
    const usage = isRecord(message.usage) ? message.usage : {};
    yield sseEvent('message_start', { type: 'message_start', message: {
      ...message, content: [], stop_reason: null, stop_sequence: null,
      usage: { input_tokens: usage.input_tokens ?? 0, output_tokens: 0 },
    } });
    for (const [index, block] of records(message.content).entries()) {
      const tool = block.type === 'tool_use';
      yield sseEvent('content_block_start', { type: 'content_block_start', index, content_block: tool ? { ...block, input: {} } : { type: 'text', text: '' } });
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index,
        delta: tool ? { type: 'input_json_delta', partial_json: JSON.stringify(block.input ?? {}) } : { type: 'text_delta', text: block.text ?? '' } });
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
    }
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: message.stop_reason, stop_sequence: message.stop_sequence }, usage: { output_tokens: usage.output_tokens ?? 0 } });
    yield sseEvent('message_stop', { type: 'message_stop' });
    return;
  }
  const choice = records(answer.choices)[0] ?? {};
  const message = isRecord(choice.message) ? choice.message : {};
  const chunk = (delta: Json, finish: unknown = null): Json => ({
    id: answer.id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  yield sseEvent(null, chunk({ role: 'assistant', content: '' }));
  if (typeof message.content === 'string' && message.content !== '') yield sseEvent(null, chunk({ content: message.content }));
  for (const [index, call] of records(message.tool_calls).entries()) {
    const fn = isRecord(call.function) ? call.function : {};
    yield sseEvent(null, chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: fn.name, arguments: '' } }] }));
    yield sseEvent(null, chunk({ tool_calls: [{ index, function: { arguments: fn.arguments ?? '{}' } }] }));
  }
  yield sseEvent(null, chunk({}, choice.finish_reason ?? 'stop'));
  yield sseEvent(null, { id: answer.id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: answer.usage });
  yield 'data: [DONE]\n\n';
}

/** Compatibility helper sharing the same SSE parser, usage accounting and ordered event emitter. */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  yield* clientEvents(await collectStream(upstream, 'openai', model), 'anthropic', model);
}
