// free-model-aggregator-01: deterministic free-model routing with workload-aware budgets.
// Read the brief, probe each provider once, and reuse one catalog for --list and /register.
// Serve OpenAI/Anthropic clients through any upstream wire, then exit when /work says done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import { candidatesFor, plannedRoute, ProviderBudgets, type Route } from './lib/routing.ts';
import {
  anthropicError,
  anthropicModelList,
  clientEvents,
  collectStream,
  responseToOpenAI,
  upstreamRequest,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  // Probe in directory order: provider-side request traces must not depend on connection races.
  probes = [];
  for (const provider of brief.directory.providers) probes.push(await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** Derive API listings from the registered catalog, including its exact order and deduplication. */
function servedModels(): ServedModel[] {
  return catalog.providers.flatMap(provider => provider.models.map(model => ({
    id: model.id,
    provider: provider.id,
    display_name: probes.find(p => p.provider.id === provider.id)?.models.find(m => m.id === model.id)?.display_name ?? model.id,
  })));
}

const budgets = new ProviderBudgets();

/** Rank every eligible free model before applying provider capacity or workload reservations. */
function routeFor(requires: Requirement, retry = false): Route | null {
  const ranked = candidatesFor(probes, brief.ratings, requires);
  const best = ranked.find(route => budgets.remaining(route.probe) > 0) ?? null;
  const manifest = brief.workload?.manifest ?? [];
  const future = manifest.slice(manifestIndex);
  // With no future reservations, including failover, always take the next-highest quality.
  if (best === null || retry || future.length === 0) return best;
  return plannedRoute(probes, brief.ratings, budgets, requires, future, manifest[manifestIndex - 1]?.weight ?? 1);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is authoritative; infer common requirements only when it has no matching row.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const structured = isRecord(body.output_config) && isRecord(body.output_config.format);
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' || structured };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Call the route in its own protocol. Gemini selects streaming in the URL, not the body. */
async function callProvider(route: Route, body: Json, stream: boolean): Promise<Response> {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  const path = provider.wire === 'gemini'
    ? `/v1beta/models/${encodeURIComponent(route.model.id)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`
    : provider.wire === 'anthropic' ? '/v1/messages' : '/v1/chat/completions';
  return fetch(`${base}${path}`, {
    method: 'POST', headers, body: JSON.stringify(body), redirect: 'error',
    signal: AbortSignal.timeout(15000),
  });
}

/** Retry only before emitting any client bytes. The caller holds the generation queue. */
async function callWithFailover(requires: Requirement, body: Json, client: 'openai' | 'anthropic'): Promise<{ route: Route; response: Response } | null> {
  let retry = false;
  for (;;) {
    const route = routeFor(requires, retry);
    if (route === null || !budgets.take(route)) return null;
    try {
      const response = await callProvider(route, upstreamRequest(body, client, route.probe.provider.wire, route.model.id), body.stream === true);
      if (response.status !== 402 && response.status !== 429 && response.status < 500) return { route, response };
      await response.body?.cancel();
    } catch {
      // Network failures have no usable answer either. Never spend another call on this provider.
    }
    budgets.exhausted.add(route.probe.provider.id);
    retry = true;
  }
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    // A failed client write must also release the upstream before the FIFO advances.
    try { await reader.cancel(); } finally { reader.releaseLock(); }
    res.end();
  }
}

/** Both client wires share exactly the same routing, accounting and failover path. */
async function generate(req: IncomingMessage, res: ServerResponse, client: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, client, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await callWithFailover(requirementsFor(body), body, client);
  if (result === null) return sendError(res, client, 503, client === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model with remaining budget meets this request');
  const { route, response } = result;
  const wire = route.probe.provider.wire;
  if (!response.ok) {
    const text = await response.text();
    return sendError(res, client, response.status, 'api_error', `the provider answered HTTP ${response.status}: ${text.slice(0, 500)}`);
  }
  const streaming = body.stream === true;
  const eventStream = (response.headers.get('content-type') ?? '').includes('text/event-stream') && response.body !== null;
  if (wire === client && (!streaming || eventStream)) {
    if (streaming && response.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
      await pipeStream(response.body, res);
    } else {
      sendJson(res, 200, await response.json());
    }
    return;
  }
  let answer: Json;
  if (streaming && eventStream && response.body !== null) {
    answer = await collectStream(response.body, wire, route.model.id);
  } else {
    const raw: unknown = await response.json();
    answer = responseToOpenAI(isRecord(raw) ? raw : {}, wire, route.model.id);
  }
  if (streaming) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    for (const event of clientEvents(answer, client, route.model.id)) res.write(event);
    res.end();
  } else {
    sendJson(res, 200, client === 'anthropic' ? openAIToAnthropic(answer, route.model.id) : answer);
  }
}

// A deterministic FIFO also holds the provider slot until its response body is fully consumed.
let generationQueue: Promise<void> = Promise.resolve();
function enqueue(task: () => Promise<void>): Promise<void> {
  const result = generationQueue.then(task);
  generationQueue = result.catch(() => {});
  return result;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await enqueue(() => generate(req, res, 'openai'));
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await enqueue(() => generate(req, res, 'anthropic'));
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      else res.destroy();
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
