import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import {
  buildCatalog, callBudgetOf, probeProvider, readBrief, type Probe, type RatingsFile, type Wire,
} from './providers.ts';
import { bestRatingFor, candidatesFor, plannedRoute, ProviderBudgets } from './routing.ts';
import {
  anthropicToOpenAI, clientEvents, collectStream, openAIRequestToAnthropic,
  openAIRequestToGemini, openAIToAnthropic, responseToOpenAI, readSSE, sseEvent, upstreamRequest,
} from './wire.ts';

const probe = (id: string, budget = 10, models = ['model']): Probe => ({
  provider: { id, name: id, wire: 'openai', auth: 'bearer', base: `/p/${id}`, key_env: 'KEY', key: 'secret', call_budget: budget },
  ok: true, status: 200, error: null,
  models: models.map(id => ({ id, display_name: id, free: true, context_length: 100000, tools: true, vision: true, json: true })),
});
const ratings = (rows: { provider: string; model?: string; rating: number; vision?: boolean; free?: boolean }[]): RatingsFile => ({
  schema: 'free-model-aggregator-01/ratings/v1', best: [],
  ratings: rows.map(r => ({ provider: r.provider, model: r.model ?? 'model', rating_milli: r.rating,
    vision: r.vision ?? true, tools: true, json: true, context_length: 100000, free: r.free ?? true })),
});
const byteStream = (text: string): ReadableStream<Uint8Array> => new ReadableStream({
  start(controller) {
    // Split every byte to exercise UTF-8, CRLF and delimiter boundaries.
    for (const byte of new TextEncoder().encode(text)) controller.enqueue(Uint8Array.of(byte));
    controller.close();
  },
});
const list = async <T>(source: AsyncIterable<T>): Promise<T[]> => { const out: T[] = []; for await (const x of source) out.push(x); return out; };

test('all free candidates are ranked by quality, capabilities and stable ties', () => {
  const probes = [probe('b', 1, ['paid', 'best', 'weak']), probe('a')];
  const table = ratings([
    { provider: 'b', model: 'paid', rating: 999, free: false },
    { provider: 'b', model: 'best', rating: 900 },
    { provider: 'b', model: 'weak', rating: 100 },
    { provider: 'a', rating: 900, vision: false },
  ]);
  assert.deepEqual(candidatesFor(probes, table, {}).map(r => `${r.probe.provider.id}/${r.model.id}`), ['a/model', 'b/best', 'b/weak']);
  assert.equal(candidatesFor(probes, table, { vision: true })[0].model.id, 'best');
  assert.equal(candidatesFor(probes, table, { min_context: 100001 }).length, 0);
});

test('ranking is independent of provider and model listing order', () => {
  const probes = [probe('z', 2, ['slow', 'best']), probe('a', 2, ['second', 'first'])];
  const table = ratings([
    { provider: 'z', model: 'slow', rating: 100 },
    { provider: 'z', model: 'best', rating: 1000 },
    { provider: 'a', model: 'second', rating: 1000 },
    { provider: 'a', model: 'first', rating: 1000 },
  ]);
  const ids = (items: Probe[]) => candidatesFor(items, table, {}).map(r => `${r.probe.provider.id}/${r.model.id}`);
  assert.deepEqual(ids(probes), ['a/first', 'a/second', 'z/best', 'z/slow']);
  assert.deepEqual(ids([...probes].reverse().map(p => ({ ...p, models: [...p.models].reverse() }))), ids(probes));
});

test('ranking trusts table flags, skips failed probes and deduplicates provider models', () => {
  const hidden = probe('hidden', 2, ['free', 'free', 'paid']);
  hidden.models.forEach(m => { m.free = null; m.tools = false; });
  const failed = { ...probe('failed'), ok: false };
  const table = ratings([
    { provider: 'hidden', model: 'free', rating: 950 },
    { provider: 'hidden', model: 'paid', rating: 9999, free: false },
    { provider: 'failed', rating: 1000 },
  ]);
  assert.deepEqual(candidatesFor([failed, hidden], table, { tools: true }).map(r => r.model.id), ['free']);
  assert.equal(bestRatingFor(table, { tools: true }), 1000);
  assert.equal(bestRatingFor(table, { min_context: 100001 }), 1);
});

test('planner reserves a scarce capability and respects weights and spent calls', () => {
  const probes = [probe('vision', 1), probe('plain', 1)];
  const table = ratings([{ provider: 'vision', rating: 1000 }, { provider: 'plain', rating: 800, vision: false }]);
  const budgets = new ProviderBudgets();
  const future = [{ id: 'later', api: 'openai' as const, stream: false, weight: 2 as const, requires: { vision: true } }];
  const first = plannedRoute(probes, table, budgets, {}, future, 1)!;
  assert.equal(first.probe.provider.id, 'plain');
  assert.equal(budgets.take(first), true);
  assert.equal(budgets.take(first), false);
  const second = plannedRoute(probes, table, budgets, { vision: true }, [], 2)!;
  assert.equal(second.probe.provider.id, 'vision');
  budgets.exhausted.add('vision');
  assert.equal(plannedRoute(probes, table, budgets, { vision: true }, [], 2), null);

  const weighted = plannedRoute(probes, ratings([{ provider: 'vision', rating: 1000 }, { provider: 'plain', rating: 800 }]),
    new ProviderBudgets(), {}, [{ ...future[0], requires: {} }], 1)!;
  assert.equal(weighted.probe.provider.id, 'plain');
});

test('equally optimal workload plans spend the highest-quality model first', () => {
  const probes = [probe('best', 1), probe('backup', 1)];
  const table = ratings([{ provider: 'best', rating: 1000 }, { provider: 'backup', rating: 800 }]);
  const future = [{ id: 'same', api: 'openai' as const, stream: false, weight: 1 as const, requires: {} }];
  assert.equal(plannedRoute(probes, table, new ProviderBudgets(), {}, future, 1)?.probe.provider.id, 'best');
});

test('planner agrees with exhaustive assignment on small deterministic workloads', () => {
  for (let seed = 0; seed < 24; seed++) {
    const probes = [probe('a', 1 + seed % 2), probe('b', 1), probe('c', 1)];
    const table = ratings([
      { provider: 'a', rating: 1000, vision: true },
      { provider: 'b', rating: 850, vision: seed % 3 === 0 },
      { provider: 'c', rating: 700, vision: false },
    ]);
    const requests = Array.from({ length: 4 }, (_, i) => ({
      id: `r${i}`, api: 'openai' as const, stream: false, weight: (1 + (seed + i) % 2) as 1 | 2,
      requires: { vision: (seed + i) % 3 === 0 },
    }));
    const utility = (index: number, route: ReturnType<typeof candidatesFor>[number]): number => {
      const best = candidatesFor(probes, table, requests[index].requires)[0]?.rating ?? 1;
      // The planner serves the request now arriving first (a large bonus at index 0), then maximizes
      // quality; the oracle must model the same objective to be comparable.
      const serving = index === 0 ? 1_000_000_000_000 : 0;
      return serving + 1_000_000_000 + Math.round(requests[index].weight * route.rating / best * 1_000_000);
    };
    const capacities = probes.map(p => p.provider.call_budget);
    const optimum = (index: number): number => {
      if (index === requests.length) return 0;
      let best = optimum(index + 1);
      for (const route of candidatesFor(probes, table, requests[index].requires)) {
        const p = probes.indexOf(route.probe);
        if (capacities[p] === 0) continue;
        capacities[p]--;
        best = Math.max(best, utility(index, route) + optimum(index + 1));
        capacities[p]++;
      }
      return best;
    };
    const expected = optimum(0);
    const route = plannedRoute(probes, table, new ProviderBudgets(), requests[0].requires, requests.slice(1), requests[0].weight)!;
    capacities[probes.indexOf(route.probe)]--;
    assert.equal(utility(0, route) + optimum(1), expected, `seed ${seed}`);
  }
});

test('planner keeps the sealed quality denominator when the best provider is unavailable', () => {
  const probes = [probe('a', 1), probe('b', 1), probe('unavailable', 0)];
  const table = ratings([
    { provider: 'a', rating: 1000 },
    { provider: 'b', rating: 900 },
    { provider: 'unavailable', rating: 10000, vision: false },
  ]);
  const future = [{ id: 'vision', api: 'openai' as const, stream: false, weight: 1 as const, requires: { vision: true } }];
  assert.equal(plannedRoute(probes, table, new ProviderBudgets(), {}, future, 2)?.probe.provider.id, 'b');
});

test('provider accounting is shared by models and counts reservations before a call', () => {
  const provider = probe('limited', 2, ['best', 'backup']);
  const candidates = candidatesFor([provider], null, {});
  const budgets = new ProviderBudgets();
  assert.equal(budgets.take(candidates[0]), true);
  assert.equal(budgets.remaining(provider), 1);
  assert.equal(budgets.take(candidates[1]), true);
  assert.equal(budgets.take(candidates[0]), false);
  assert.equal(budgets.calls.get('limited'), 2);
  const unlimited = probe('unlimited', Infinity);
  assert.equal(budgets.remaining(unlimited), Infinity);
  budgets.exhausted.add('unlimited');
  assert.equal(budgets.take(candidatesFor([unlimited], null, {})[0]), false);
  assert.equal(budgets.calls.has('unlimited'), false);
});

test('planner does not sacrifice a uniquely capable future request', () => {
  const probes = [probe('scarce', 1), probe('fallback', 2)];
  const table = ratings([{ provider: 'scarce', rating: 1000 }, { provider: 'fallback', rating: 700, vision: false }]);
  const future = [{ id: 'image', api: 'anthropic' as const, stream: true, weight: 2 as const, requires: { vision: true } }];
  assert.equal(plannedRoute(probes, table, new ProviderBudgets(), {}, future, 2)?.probe.provider.id, 'fallback');
});

test('budget parser handles zero and explicit limits', () => {
  assert.equal(callBudgetOf({ call_budget: 0 }), 0);
  assert.equal(callBudgetOf({ budget: { calls: 3 } }), 3);
  assert.equal(callBudgetOf({ limits: { requests: 4 } }), 4);
  assert.equal(callBudgetOf({}), Infinity);
});

test('catalog contains only free models with stable provider paths', () => {
  const probes = [probe('p', 1, ['paid', 'free', 'free'])];
  const table = ratings([{ provider: 'p', model: 'paid', rating: 999, free: false }, { provider: 'p', model: 'free', rating: 1 }]);
  const first = buildCatalog('http://localhost:1234', probes, table);
  assert.deepEqual(first.providers[0].models.map(m => m.id), ['free']);
  assert.deepEqual(first, buildCatalog('http://localhost:5678', probes, table));
});

test('catalog probes follow explicit pages without redundant requests and reject malformed lists', async () => {
  const requests: string[] = [];
  const env = createServer((req, res) => {
    const url = new URL(req.url!, 'http://mock');
    requests.push(req.url!);
    res.setHeader('content-type', 'application/json');
    if (url.pathname.startsWith('/p/gemini')) {
      assert.equal(req.headers['x-goog-api-key'], 'secret');
      res.end(JSON.stringify(url.search ? { models: [{ name: 'models/second', free: true }] }
        : { models: [{ name: 'models/first', free: true }], nextPageToken: 'next page' }));
    } else if (url.pathname.startsWith('/p/anthropic')) {
      assert.equal(req.headers['x-api-key'], 'secret');
      res.end(JSON.stringify(url.search ? { data: [{ id: 'second', free: true }], has_more: false }
        : { data: [{ id: 'first', free: true }], has_more: true, last_id: 'first' }));
    } else res.end('{}');
  });
  env.listen(0, '127.0.0.1'); await once(env, 'listening');
  try {
    const address = env.address(); assert.ok(address && typeof address !== 'string');
    for (const wire of ['gemini', 'anthropic', 'openai'] as const) {
      const provider = { ...probe(wire).provider, wire, auth: wire === 'gemini' ? 'x-goog-api-key' as const : wire === 'anthropic' ? 'x-api-key' as const : 'bearer' as const };
      const result = await probeProvider(`http://127.0.0.1:${address.port}`, provider);
      assert.equal(result.ok, wire !== 'openai');
      assert.deepEqual(result.models.map(m => m.id), wire === 'openai' ? [] : ['first', 'second']);
    }
    assert.deepEqual(requests, ['/p/gemini/v1beta/models', '/p/gemini/v1beta/models?pageToken=next%20page',
      '/p/anthropic/v1/models', '/p/anthropic/v1/models?after_id=first', '/p/openai/v1/models']);
  } finally {
    env.closeAllConnections();
    await new Promise<void>(resolve => env.close(() => resolve()));
  }
});

const request = {
  model: 'auto', max_tokens: 73, temperature: 0, stream: true,
  response_format: { type: 'json_object' },
  tools: [{ type: 'function', function: { name: 'lookup', parameters: { type: 'object', properties: { city: { type: 'string' } } } } }],
  tool_choice: { type: 'function', function: { name: 'lookup' } },
  messages: [
    { role: 'system', content: 'Be precise.' },
    { role: 'user', content: [{ type: 'text', text: 'Describe' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,YQ==' } }] },
    { role: 'assistant', content: null, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{"city":"Oslo"}' } }] },
    { role: 'tool', tool_call_id: 'call_1', content: '{"temperature":5}' },
  ],
};

test('Anthropic request roundtrip preserves system, limits, images and tool history', () => {
  const anth = openAIRequestToAnthropic(request, 'chosen');
  assert.equal(anth.model, 'chosen');
  assert.equal(anth.system, 'Be precise.');
  assert.equal(anth.max_tokens, 73);
  assert.deepEqual(anth.tool_choice, { type: 'tool', name: 'lookup' });
  const back = anthropicToOpenAI(anth, 'chosen');
  assert.equal(back.max_tokens, 73);
  assert.deepEqual(back.response_format, request.response_format);
  const messages = back.messages as Record<string, any>[];
  assert.equal(messages[0].content, 'Be precise.');
  assert.equal(messages[1].content[1].image_url.url, 'data:image/png;base64,YQ==');
  assert.equal(messages[2].tool_calls[0].function.arguments, '{"city":"Oslo"}');
  assert.equal(messages[3].tool_call_id, 'call_1');
});

test('Gemini requests use native system, tool, image and JSON fields', () => {
  const gem = openAIRequestToGemini(request);
  const contents = gem.contents as Record<string, any>[];
  assert.deepEqual(gem.systemInstruction, { parts: [{ text: 'Be precise.' }] });
  assert.deepEqual(gem.generationConfig, { maxOutputTokens: 73, temperature: 0, responseMimeType: 'application/json' });
  assert.deepEqual(gem.toolConfig, { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: ['lookup'] } });
  assert.deepEqual(contents[0].parts[1].inlineData, { mimeType: 'image/png', data: 'YQ==' });
  assert.equal(contents[1].role, 'model');
  assert.equal(contents[2].parts[0].functionResponse.name, 'lookup');
  assert.deepEqual(contents[2].parts[0].functionResponse.response, { temperature: 5 });
});

test('native OpenAI streaming requests ask for usage without mutating the client', () => {
  const body = { ...request, stream_options: { include_usage: false, custom: true } };
  const before = JSON.stringify(body);
  assert.deepEqual(upstreamRequest(body, 'openai', 'openai', 'native').stream_options, { include_usage: true, custom: true });
  assert.equal(JSON.stringify(body), before);
});

test('multimodal tool results survive Anthropic and OpenAI history translation', () => {
  const content = [{ type: 'text', text: 'a picture' }, { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'YQ==' } }];
  const canonical = anthropicToOpenAI({ messages: [{ role: 'user', content: [
    { type: 'tool_result', tool_use_id: 'call_1', is_error: true, content },
  ] }] }, 'chosen');
  const back: any = openAIRequestToAnthropic(canonical, 'chosen');
  assert.deepEqual(back.messages[0].content[0], { type: 'tool_result', tool_use_id: 'call_1', is_error: true, content });
});

test('Gemini usage totals and safety reasons agree in streaming and JSON', async () => {
  const response = { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'SPII' }],
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 2, totalTokenCount: 15 } };
  const json: any = responseToOpenAI(response, 'gemini', 'chosen');
  const streamed: any = await collectStream(byteStream(sseEvent(null, response)), 'gemini', 'chosen');
  assert.deepEqual(json.usage, { prompt_tokens: 10, completion_tokens: 2, total_tokens: 15 });
  assert.deepEqual(streamed.usage, json.usage);
  assert.equal(json.choices[0].finish_reason, 'content_filter');
  assert.deepEqual(streamed.choices, json.choices);
});

test('SSE parser accepts byte-split CRLF, multiline data and final unterminated frames', async () => {
  assert.deepEqual(await list(readSSE(byteStream(': hi\r\ndata: {"text":\r\ndata: "é"}\r\n\r\ndata: [DONE]'))), ['{"text":\n"é"}', '[DONE]']);
});

test('SSE parser accepts CR-only line endings', async () => {
  assert.deepEqual(await list(readSSE(byteStream('data: one\r\rdata: two\r\r'))), ['one', 'two']);
});

test('Anthropic history preserves text and tool-result order', () => {
  const body = anthropicToOpenAI({ messages: [{ role: 'user', content: [
    { type: 'text', text: 'before' }, { type: 'tool_result', tool_use_id: 'id', content: 'result' }, { type: 'text', text: 'after' },
  ] }] }, 'model');
  assert.deepEqual((body.messages as any[]).map(m => m.role), ['user', 'tool', 'user']);
  const native = openAIRequestToAnthropic({ ...request, tool_choice: undefined, parallel_tool_calls: false }, 'model');
  assert.deepEqual(native.tool_choice, { type: 'auto', disable_parallel_tool_use: true });
});

test('handwritten Anthropic stream preserves split tool input, stop sequence and usage', async () => {
  const payloads = [
    { type: 'message_start', message: { id: 'a', usage: { input_tokens: 23, output_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'tool', name: 'lookup', input: {} } },
    { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"city":' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '"Oslo"}' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 7 } },
    { type: 'message_stop' },
  ];
  const answer: any = await collectStream(byteStream(payloads.map(p => sseEvent(p.type, p)).join('')), 'anthropic', 'chosen');
  assert.equal(answer.choices[0].message.tool_calls[0].function.arguments, '{"city":"Oslo"}');
  assert.deepEqual(answer.usage, { prompt_tokens: 23, completion_tokens: 7, total_tokens: 30 });
  const stopped = responseToOpenAI({ content: [{ type: 'text', text: 'ok' }], stop_reason: 'stop_sequence', stop_sequence: 'END' }, 'anthropic', 'chosen');
  assert.equal(openAIToAnthropic(stopped, 'chosen').stop_sequence, 'END');
  assert.equal(openAIToAnthropic(stopped, 'chosen').stop_reason, 'stop_sequence');
});

test('handwritten Gemini stream combines text, tools and final usage metadata', async () => {
  const frames = [
    { candidates: [{ content: { parts: [{ text: 'hel' }] } }] },
    { candidates: [{ content: { parts: [{ text: 'lo' }, { functionCall: { name: 'lookup', args: { city: 'Oslo' } } }] }, finishReason: 'STOP' }] },
    { usageMetadata: { promptTokenCount: 11, candidatesTokenCount: 5 } },
  ];
  const answer: any = await collectStream(byteStream(frames.map(p => sseEvent(null, p)).join('')), 'gemini', 'chosen');
  assert.equal(answer.choices[0].message.content, 'hello');
  assert.equal(answer.choices[0].message.tool_calls[0].function.name, 'lookup');
  assert.equal(answer.choices[0].finish_reason, 'tool_calls');
  assert.equal(answer.usage.total_tokens, 16);
});

test('structured output schemas and original requests survive all retry translations', () => {
  const body = { ...request, response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: { type: 'object' } } } };
  const before = JSON.stringify(body);
  const gem: any = upstreamRequest(body, 'openai', 'gemini', 'g');
  assert.deepEqual(gem.generationConfig.responseSchema, { type: 'object' });
  const anth: any = upstreamRequest(body, 'openai', 'anthropic', 'a');
  assert.deepEqual(anth.output_config.format.schema, { type: 'object' });
  assert.equal(JSON.stringify(body), before);
});

const nativeAnswer = (wire: Wire, model = 'model') => wire === 'openai' ? {
  id: 'answer', object: 'chat.completion', created: 1767225600, model, choices: [{ index: 0, message: { role: 'assistant', content: 'héllo', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{"city":"Oslo"}' } }] }, finish_reason: 'tool_calls' }],
  usage: { prompt_tokens: 17, completion_tokens: 9, total_tokens: 26 },
} : wire === 'anthropic' ? {
  id: 'answer', type: 'message', role: 'assistant', model, content: [{ type: 'text', text: 'héllo' }, { type: 'tool_use', id: 'call_1', name: 'lookup', input: { city: 'Oslo' } }],
  stop_reason: 'tool_use', stop_sequence: null, usage: { input_tokens: 17, output_tokens: 9 },
} : {
  responseId: 'answer', candidates: [{ content: { role: 'model', parts: [{ text: 'héllo' }, { functionCall: { id: 'call_1', name: 'lookup', args: { city: 'Oslo' } } }] }, finishReason: 'STOP' }],
  usageMetadata: { promptTokenCount: 17, candidatesTokenCount: 9, totalTokenCount: 26 },
};
const nativeStream = (wire: Wire, model = 'model'): string => {
  if (wire === 'gemini') return sseEvent(null, nativeAnswer(wire, model));
  return [...clientEvents(responseToOpenAI(nativeAnswer(wire, model), wire, model), wire, model)].join('');
};

for (const wire of ['openai', 'anthropic', 'gemini'] as const) {
  test(`${wire} stream and JSON preserve tools, usage and finish reason`, async () => {
    const answer = await collectStream(byteStream(nativeStream(wire)), wire, 'model');
    const expected = responseToOpenAI(nativeAnswer(wire), wire, 'model');
    assert.deepEqual(answer.choices, expected.choices);
    assert.deepEqual(answer.usage, expected.usage);
    assert.equal(answer.model, expected.model);
    for (const client of ['openai', 'anthropic'] as const) {
      const events = [...clientEvents(answer, client, 'model')].join('');
      const roundtrip = await collectStream(byteStream(events), client, 'model');
      assert.deepEqual(roundtrip.choices, answer.choices);
      assert.deepEqual(roundtrip.usage, answer.usage);
      if (client === 'anthropic') {
        const payloads = (await list(readSSE(byteStream(events)))).map(s => JSON.parse(s));
        assert.equal(payloads[0].type, 'message_start');
        assert.equal(payloads[0].message.usage.input_tokens, 17);
        assert.equal(payloads.at(-1).type, 'message_stop');
      }
    }
  });
}

test('server integrates all six wire combinations in JSON and streaming, catalog, failover and replay', { timeout: 30000 }, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'aggregator-test-'));
  const prompt = join(directory, 'brief.md');
  const wires: Wire[] = ['openai', 'anthropic', 'gemini'];
  const budget = 4;
  const providers = wires.map((wire, i) => ({ id: wire, wire,
    auth: wire === 'openai' ? 'bearer' : wire === 'anthropic' ? 'x-api-key' : 'x-goog-api-key',
    path: `/p/${wire}`, key_env: `KEY_${i}`, key: `key-${i}`, call_budget: budget,
  }));
  const manifest = Array.from({ length: 12 }, (_, i) => ({ id: `r${i}`, api: i % 4 < 2 ? 'openai' : 'anthropic', stream: i % 2 === 1, weight: 1, requires: {} }));
  await writeFile(prompt, [
    { schema: 'free-model-aggregator-01/providers/v1', providers },
    ratings(wires.flatMap((wire, i) => [{ provider: wire, model: `model-${wire}`, rating: 1000 - i * 100 }, { provider: wire, model: 'paid', rating: 9999, free: false }])),
    { schema: 'free-model-aggregator-01/workload/v1', manifest },
  ].map(x => JSON.stringify(x)).join('\n'));
  const runs: string[] = [];
  try {
    assert.equal(readBrief(prompt).directory.providers[0].call_budget, budget);
    for (const failingStatus of [0, 0, 402, 429, 500, 503, -1, -2]) {
      let port = 0;
      let registration: any;
      let done = false;
      let failureSent = false;
      const calls: Record<string, number> = {};
      const active = new Set<string>();
      const trace: unknown[] = [];
      let mockError: Error | undefined;
      const env = createServer((req, res) => {
        void (async () => {
          const url = new URL(req.url!, 'http://mock');
          res.setHeader('content-type', 'application/json');
          if (url.pathname === '/register') {
            let text = ''; for await (const chunk of req) text += chunk;
            registration = JSON.parse(text); port = registration.port;
            assert.notEqual(port, 8080);
            res.end('{"requests":12}'); return;
          }
          if (url.pathname === '/work') { res.end(JSON.stringify({ done })); return; }
          const wire = url.pathname.split('/')[2] as Wire;
          const index = wires.indexOf(wire);
          assert.notEqual(index, -1);
          assert.equal(req.headers[wire === 'openai' ? 'authorization' : wire === 'anthropic' ? 'x-api-key' : 'x-goog-api-key'], wire === 'openai' ? `Bearer key-${index}` : `key-${index}`);
          if (req.method === 'GET') {
            const models = [`model-${wire}`, 'paid'];
            res.end(JSON.stringify(wire === 'gemini' ? { models: models.map(id => ({ name: `models/${id}` })) } : { data: models.map(id => ({ id })) })); return;
          }
          assert.equal(active.has(wire), false);
          active.add(wire);
          calls[wire] = (calls[wire] ?? 0) + 1;
          assert.ok(calls[wire] <= budget);
          let text = ''; for await (const chunk of req) text += chunk;
          const body = JSON.parse(text);
          trace.push({ wire, path: url.pathname + url.search, body });
          if ((failingStatus > 0 && !failureSent) || (failingStatus === -1 && wire !== 'gemini')) {
            failureSent = true; active.delete(wire); res.statusCode = failingStatus === -1 ? 503 : failingStatus; res.end('{"error":"failed"}'); return;
          }
          const streaming = wire === 'gemini' ? url.pathname.endsWith(':streamGenerateContent') : body.stream === true;
          if (wire === 'openai') { assert.equal(body.messages[0].content, 'Be precise.'); assert.equal(body.max_tokens, 73); }
          if (wire === 'anthropic') { assert.equal(body.system, 'Be precise.'); assert.equal(body.max_tokens, 73); assert.equal(req.headers['anthropic-version'], '2023-06-01'); }
          if (wire === 'gemini') {
            assert.deepEqual(body.systemInstruction, { parts: [{ text: 'Be precise.' }] });
            assert.equal(body.generationConfig.maxOutputTokens, 73);
            assert.equal(url.search, streaming ? '?alt=sse' : '');
          }
          if (streaming) res.setHeader('content-type', 'text/event-stream');
          res.end(streaming ? nativeStream(wire, `model-${wire}`) : JSON.stringify(nativeAnswer(wire, `model-${wire}`)));
          active.delete(wire);
        })().catch(err => { mockError = err; res.statusCode = 500; res.end(JSON.stringify({ error: String(err) })); });
      });
      env.listen(0, '127.0.0.1'); await once(env, 'listening');
      const address = env.address(); assert.ok(address && typeof address !== 'string');
      const envUrl = `http://127.0.0.1:${address.port}`;
      const childEnv = { ...process.env, MAKEFUN_PROMPT_FILE: prompt, MAKEFUN_ENV_URL: envUrl };
      const child = spawn(process.execPath, ['main.ts'], { cwd: new URL('..', import.meta.url), env: childEnv, stdio: ['ignore', 'pipe', 'pipe'] });
      let stderr = ''; child.stderr.on('data', x => { stderr += x; });
      const exited = once(child, 'exit');
      try {
        // Registration is an external event, not a generation probe.
        await new Promise<void>((resolve, reject) => {
          const timer = setInterval(() => { if (port) { clearInterval(timer); resolve(); } else if (child.exitCode !== null) { clearInterval(timer); reject(new Error(stderr)); } }, 5);
          timer.unref();
        });
        const listingChild = spawn(process.execPath, ['main.ts', '--list'], { cwd: new URL('..', import.meta.url), env: childEnv });
        let listing = ''; listingChild.stdout.on('data', x => { listing += x; });
        assert.equal((await once(listingChild, 'exit'))[0], 0);
        assert.equal(listing, JSON.stringify(registration.catalog) + '\n');
        for (const headers of [{ authorization: 'Bearer client' }, { 'x-api-key': 'client' }]) {
          const catalogResponse = await fetch(`http://127.0.0.1:${port}/v1/models`, { headers });
          const models: any = await catalogResponse.json();
          assert.deepEqual(models.data.map((m: any) => m.id).sort(), wires.map(w => `model-${w}`).sort());
        }
        const combinations = new Set<string>();
        let served = 0;
        const sendEntry = (entry: typeof manifest[number]) => {
          const body = entry.api === 'openai' ? { ...request, stream: entry.stream }
            : { ...openAIRequestToAnthropic(request, 'auto'), stream: entry.stream };
          return fetch(`http://127.0.0.1:${port}${entry.api === 'openai' ? '/v1/chat/completions' : '/v1/messages'}`, {
            method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
          });
        };
        const concurrent = failingStatus === -2 ? manifest.map(sendEntry) : null;
        for (const [index, entry] of manifest.entries()) {
          const response = await (concurrent?.[index] ?? sendEntry(entry));
          if (failingStatus && response.status === 503) continue; // Lost provider capacity cannot serve all twelve.
          assert.equal(response.status, 200, await (response.ok ? Promise.resolve('') : response.text()));
          let answer: any;
          if (entry.stream) answer = await collectStream(response.body!, entry.api as 'openai' | 'anthropic', 'model');
          else answer = responseToOpenAI(await response.json(), entry.api as Wire, 'model');
          served++;
          assert.equal(answer.choices[0].message.content, 'héllo');
          assert.equal(answer.choices[0].finish_reason, 'tool_calls');
          assert.equal(answer.usage.prompt_tokens, 17);
          assert.equal(answer.usage.completion_tokens, 9);
          const last = trace.at(-1) as any;
          if (!concurrent) combinations.add(`${last.wire}/${entry.api}/${entry.stream}`);
        }
        if (!failingStatus) assert.equal(combinations.size, 12);
        if (failingStatus > 0 || failingStatus === -1) {
          assert.equal(calls.openai, 1);
          assert.equal(failureSent, true);
          assert.equal(Object.values(calls).reduce((a, b) => a + b, 0), failingStatus === -1 ? 6 : 9);
          assert.equal(served, failingStatus === -1 ? 4 : 8);
          if (failingStatus === -1) assert.equal(calls.anthropic, 1);
        }
        if (concurrent) {
          assert.equal(served, manifest.length);
          assert.deepEqual(calls, { openai: 4, anthropic: 4, gemini: 4 });
        }
        if (mockError) throw mockError;
        if (!failingStatus) runs.push(JSON.stringify({ catalog: registration.catalog, trace }));
        done = true;
        assert.equal((await exited)[0], 0, stderr);
      } finally {
        if (child.exitCode === null) child.kill('SIGKILL');
        env.closeAllConnections();
        await new Promise<void>(resolve => env.close(() => resolve()));
      }
    }
    assert.equal(runs[0], runs[1], 'replay catalog and provider-call trace must match byte for byte');
  } finally { await rm(directory, { recursive: true, force: true }); }
});
