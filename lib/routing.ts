import {
  capabilitiesOf, meetsRequirements, modelFree, ratingOfModel,
  type Probe, type ProbedModel, type RatingsFile, type Requirement, type WorkloadRequest,
} from './providers.ts';

export interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Counters are mutated synchronously before fetch, so failed calls also consume capacity. */
export class ProviderBudgets {
  readonly calls = new Map<string, number>();
  readonly exhausted = new Set<string>();

  remaining(probe: Probe): number {
    if (this.exhausted.has(probe.provider.id)) return 0;
    const limit = probe.provider.call_budget;
    const capacity = limit === Infinity ? Infinity : Number.isFinite(limit) ? Math.max(0, Math.floor(limit)) : 0;
    return Math.max(0, capacity - (this.calls.get(probe.provider.id) ?? 0));
  }

  take(route: Route): boolean {
    if (!route.probe.ok || this.remaining(route.probe) <= 0) return false;
    const id = route.probe.provider.id;
    this.calls.set(id, (this.calls.get(id) ?? 0) + 1);
    return true;
  }
}

interface Edge { to: number; reverse: number; capacity: number; cost: number }

/**
 * Maximum-weight assignment of remaining manifest requests to provider slots. Each provider's
 * best eligible model is sufficient for an assignment. Residual edges let a later constrained
 * request displace an earlier flexible one instead of greedily burning scarce capacity.
 */
export function plannedRoute(
  probes: readonly Probe[], ratings: RatingsFile | null, budgets: ProviderBudgets,
  requires: Requirement, future: readonly WorkloadRequest[], weight: number,
): Route | null {
  const requests = [{ requires, weight }, ...future];
  const available = probes.filter(p => p.ok && budgets.remaining(p) > 0);
  const source = 0;
  const providerStart = 1 + requests.length;
  const sink = providerStart + available.length;
  const graph: Edge[][] = Array.from({ length: sink + 1 }, () => []);
  const add = (from: number, to: number, capacity: number, cost: number): Edge => {
    const forward = { to, reverse: graph[to].length, capacity, cost };
    const backward = { to: from, reverse: graph[from].length, capacity: 0, cost: -cost };
    graph[from].push(forward);
    graph[to].push(backward);
    return forward;
  };
  const current: { route: Route; edge: Edge }[] = [];
  requests.forEach((request, i) => {
    add(source, 1 + i, 1, 0);
    const candidates = candidatesFor(available, ratings, request.requires);
    const best = ratings === null ? Math.max(1, candidates[0]?.rating ?? 1) : bestRatingFor(ratings, request.requires);
    const seen = new Set<string>();
    for (const route of candidates) {
      const id = route.probe.provider.id;
      if (seen.has(id)) continue;
      seen.add(id);
      const providerIndex = available.findIndex(p => p.provider.id === id);
      // Serving takes precedence; among equally served plans maximize weighted quality ratio.
      const utility = (i === 0 ? 1_000_000_000_000 : 0)
        + 1_000_000_000 + Math.round(request.weight * route.rating / best * 1_000_000);
      const edge = add(1 + i, providerStart + providerIndex, 1, -utility);
      if (i === 0) current.push({ route, edge });
    }
  });
  available.forEach((p, i) => add(providerStart + i, sink, Math.min(requests.length, budgets.remaining(p)), 0));

  // Bellman-Ford on this tiny graph (normally 20 requests and 10 providers).
  for (;;) {
    const distance = Array(graph.length).fill(Infinity) as number[];
    const previous: { node: number; edge: number }[] = [];
    distance[source] = 0;
    for (let pass = 0; pass < graph.length - 1; pass++) {
      let changed = false;
      for (let node = 0; node < graph.length; node++) {
        if (!Number.isFinite(distance[node])) continue;
        graph[node].forEach((edge, index) => {
          if (edge.capacity > 0 && distance[node] + edge.cost < distance[edge.to]) {
            distance[edge.to] = distance[node] + edge.cost;
            previous[edge.to] = { node, edge: index };
            changed = true;
          }
        });
      }
      if (!changed) break;
    }
    if (!Number.isFinite(distance[sink])) break;
    for (let node = sink; node !== source;) {
      const step = previous[node];
      const edge = graph[step.node][step.edge];
      edge.capacity--;
      graph[node][edge.reverse].capacity++;
      node = step.node;
    }
  }
  const selected = current.find(c => c.edge.capacity === 0)?.route ?? null;
  if (selected === null) return candidatesFor(available, ratings, requires)[0] ?? null;
  // Residual reassignments can swap equally weighted slots and send a weaker model now even
  // when the total quality is tied. Prefer the best current model among optimal assignments.
  for (const candidate of current) {
    if (candidate.route === selected) return selected;
    if (canSwapWithoutCost(graph, candidate.edge)) return candidate.route;
  }
  return selected;
}

/** An unused current edge belongs to another optimum exactly when its residual cycle costs zero. */
function canSwapWithoutCost(graph: Edge[][], edge: Edge): boolean {
  const distance = Array(graph.length).fill(Infinity) as number[];
  distance[edge.to] = 0;
  for (let pass = 0; pass < graph.length - 1; pass++) {
    let changed = false;
    for (let node = 0; node < graph.length; node++) {
      if (!Number.isFinite(distance[node])) continue;
      for (const next of graph[node]) {
        if (next.capacity > 0 && distance[node] + next.cost < distance[next.to]) {
          distance[next.to] = distance[node] + next.cost;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  // Node 1 is the current request. Costs are integral, avoiding floating-point tie ambiguity.
  return distance[1] + edge.cost === 0;
}

/** The scorer compares against the full free table, including providers with no slots left. */
export function bestRatingFor(ratings: RatingsFile | null, requires: Requirement): number {
  let best = 0;
  for (const row of ratings?.ratings ?? []) {
    if (row.free && meetsRequirements(requires, row)) best = Math.max(best, row.rating_milli);
  }
  return Math.max(1, best);
}

const compareText = (a: string, b: string): number => a < b ? -1 : a > b ? 1 : 0;

/** Rank all eligible models, not just one model per provider. Ties never depend on timing. */
export function candidatesFor(probes: readonly Probe[], ratings: RatingsFile | null, requires: Requirement): Route[] {
  const candidates: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row) || !meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      candidates.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return candidates.sort((a, b) => b.rating - a.rating
    || compareText(a.probe.provider.id, b.probe.provider.id) || compareText(a.model.id, b.model.id));
}
