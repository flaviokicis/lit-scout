import { atom, read, update } from 'claude-code'
import type { EngineInterface } from 'claude-code'

import type { LitScoutAnswer, LitScoutBand, LitScoutDeepCheck, LitScoutLogEntry, LitScoutOffer, LitScoutSession, LitScoutUsage } from '../types'

const INITIAL_SESSION: LitScoutSession = {
  band: { kind: 'idle' },
  pending: null,
  ignoredInRow: 0,
  ideasToSkip: 0,
}

export const session = atom({ plugin: 'lit-scout', key: 'session' } as const, INITIAL_SESSION, { shape: 'session-7' })

const OFFERED_LIMIT = 50
/** How many of the session's checked ideas the classifier sees, newest last. */
const CHECKED_SHOWN_LIMIT = 20
const LOG_LIMIT = 2000
const EXPIRY_PROMPTS = 2
const IGNORED_BEFORE_BACKOFF = 3
const MAX_SKIP = 16

export async function readSession($: EngineInterface): Promise<LitScoutSession> {
  return read($, session)
}

/** Every session write goes through `update`: it writes with `ifVersion` and tries again on a miss. */
export async function changeSession($: EngineInterface, change: (value: LitScoutSession) => LitScoutSession): Promise<LitScoutSession> {
  return update($, session, change)
}

/** Changes the session and returns what the change decided, read from the value it was applied to. */
export async function decideSession<T>($: EngineInterface, decide: (value: LitScoutSession) => readonly [LitScoutSession, T]): Promise<T> {
  const decided: { result?: T } = {}

  await update($, session, value => {
    const [next, result] = decide(value)
    decided.result = result

    return next
  })

  return decided.result as T
}

export function setBand(band: LitScoutBand): (value: LitScoutSession) => LitScoutSession {
  return value => ({ ...value, band })
}

/** Frees the band; the queued idea, if any, takes it. Returns the offer that took it. */
export function releaseBand(value: LitScoutSession): readonly [LitScoutSession, LitScoutOffer | null] {
  if (value.pending === null) {
    return [{ ...value, band: { kind: 'idle' } }, null]
  }

  const offer = { ...value.pending, promptsSince: 0 }

  return [{ ...value, band: { kind: 'offer', offer }, pending: null }, offer]
}

/** A search that an earlier load of the module started never finishes in this one: free its band. */
export function clearStale(loadId: string): (value: LitScoutSession) => readonly [LitScoutSession, LitScoutOffer | null] {
  return value => (value.band.kind === 'searching' && value.band.loadId !== loadId ? releaseBand(value) : [value, null])
}

export type PromptTurn = {
  expired: LitScoutOffer | null
  promoted: LitScoutOffer | null
}

/** One user prompt: clears stale work and ages the band (an offer or a report lasts two prompts). */
export function takePrompt(loadId: string): (value: LitScoutSession) => readonly [LitScoutSession, PromptTurn] {
  return value => {
    const [cleared, freedStale] = clearStale(loadId)(value)
    const [aged, expired, freedByAge] = ageBand(cleared)

    return [aged, { expired, promoted: freedByAge ?? freedStale }]
  }
}

function ageBand(value: LitScoutSession): readonly [LitScoutSession, LitScoutOffer | null, LitScoutOffer | null] {
  const band = value.band

  if (band.kind === 'offer') {
    const offer = { ...band.offer, promptsSince: band.offer.promptsSince + 1 }

    if (offer.promptsSince < EXPIRY_PROMPTS) {
      return [{ ...value, band: { kind: 'offer', offer } }, null, null]
    }

    const [released, promoted] = releaseBand(afterAnswer(value, 'expired'))

    return [released, offer, promoted]
  }

  if (band.kind === 'report') {
    const promptsSince = band.promptsSince + 1

    if (promptsSince < EXPIRY_PROMPTS) {
      return [{ ...value, band: { ...band, promptsSince } }, null, null]
    }

    const [released, promoted] = releaseBand(value)

    return [released, null, promoted]
  }

  return [value, null, null]
}

/** One more answered or ignored offer: engaging resets the run, ignoring grows the skip 1, 2, 4, 8, 16 ideas. */
export function afterAnswer(value: LitScoutSession, answer: LitScoutAnswer): LitScoutSession {
  if (answer === 'check' || answer === 'know') {
    return { ...value, ignoredInRow: 0, ideasToSkip: 0 }
  }

  const ignoredInRow = value.ignoredInRow + 1
  const overBy = ignoredInRow - IGNORED_BEFORE_BACKOFF

  return { ...value, ignoredInRow, ideasToSkip: overBy < 0 ? 0 : Math.min(MAX_SKIP, 2 ** overBy) }
}

/*
 * The store has no conditional write, and all of the person's sessions share it. So a session writes only keys
 * of its own (`log:<session id>`), and this load runs its read-change-writes one at a time; readers merge every
 * session's key. A write still in flight when the module hot-reloads is outside the queue of the new load.
 */

type ListName = 'offered' | 'checked' | 'log' | 'deep'

let storeQueue: Promise<unknown> = Promise.resolve()

function serially<T>(work: () => Promise<T>): Promise<T> {
  const run = storeQueue.then(work)
  storeQueue = run.then(
    () => undefined,
    () => undefined,
  )

  return run
}

function changeOwnList<T>($: EngineInterface, name: ListName, change: (items: T[]) => T[]): Promise<void> {
  return serially(async () => {
    const key = `${name}:${await $.session.id()}`
    const stored = await $.store.get(key)
    await $.store.set(key, change(Array.isArray(stored) ? (stored as T[]) : []))
  })
}

async function readAllLists($: EngineInterface, name: ListName): Promise<unknown[]> {
  const keys = (await $.store.keys()).filter(key => key === name || key.startsWith(`${name}:`))
  const lists = await Promise.all(keys.map(key => $.store.get(key)))

  return lists.flatMap(list => (Array.isArray(list) ? (list as unknown[]) : []))
}

async function readList($: EngineInterface, name: 'offered' | 'checked'): Promise<string[]> {
  return [...new Set((await readAllLists($, name)).filter((item): item is string => typeof item === 'string'))]
}

export async function rememberOffered($: EngineInterface, query: string): Promise<void> {
  await changeOwnList<string>($, 'offered', offered => [...offered, query.toLowerCase()].slice(-OFFERED_LIMIT))
}

/** An idea the person needs no check for any more: a check of it ran (any outcome), or they said they know its literature. */
export async function rememberChecked($: EngineInterface, idea: string): Promise<void> {
  await changeOwnList<string>($, 'checked', checked => [...new Set([...checked, idea.toLowerCase()])])
}

/** This session's checked ideas, oldest first, for the classifier to tell a follow-up from a new idea. */
export async function readSessionChecked($: EngineInterface): Promise<string[]> {
  const stored = await $.store.get(`checked:${await $.session.id()}`)

  return (Array.isArray(stored) ? stored.filter((item): item is string => typeof item === 'string') : []).slice(-CHECKED_SHOWN_LIMIT)
}

export async function isDuplicate($: EngineInterface, query: string): Promise<boolean> {
  const seen = [...(await readList($, 'offered')), ...(await readList($, 'checked'))]

  return seen.includes(query.toLowerCase())
}

export async function readLog($: EngineInterface): Promise<LitScoutLogEntry[]> {
  const entries = (await readAllLists($, 'log')) as LitScoutLogEntry[]

  return entries.sort((a, b) => a.time.localeCompare(b.time))
}

export async function appendLog($: EngineInterface, entry: LitScoutLogEntry): Promise<void> {
  await changeOwnList<LitScoutLogEntry>($, 'log', log => [...log, entry].slice(-LOG_LIMIT))
}

export async function amendLog($: EngineInterface, id: string, change: Partial<LitScoutLogEntry>): Promise<void> {
  await changeOwnList<LitScoutLogEntry>($, 'log', log => log.map(entry => (entry.id === id ? { ...entry, ...change } : entry)))
}

export async function readDeepChecks($: EngineInterface): Promise<LitScoutDeepCheck[]> {
  const checks = (await readAllLists($, 'deep')) as LitScoutDeepCheck[]

  return checks.sort((a, b) => a.time.localeCompare(b.time))
}

export async function appendDeepCheck($: EngineInterface, check: LitScoutDeepCheck): Promise<void> {
  await changeOwnList<LitScoutDeepCheck>($, 'deep', checks => [...checks, check].slice(-LOG_LIMIT))
}

export function formatStats(log: readonly LitScoutLogEntry[], deepChecks: readonly LitScoutDeepCheck[]): string {
  const offers = log.filter(entry => entry.outcome === 'shown' || entry.outcome === 'yielded')
  const accepted = offers.filter(entry => entry.answer === 'check').length
  const acceptRate = offers.length === 0 ? 'n/a' : `${Math.round((100 * accepted) / offers.length)}%`
  const byOutcome = countBy(log, entry => entry.outcome)
  const byAnswer = countBy(offers, entry => entry.answer ?? 'pending')
  const skips = log.filter(entry => entry.outcome !== 'shown').slice(-10)

  return [
    `lit-scout: ${log.length} checks, ${offers.length} offers, ${accepted} accepted (accept rate ${acceptRate}).`,
    `Checks by outcome: ${byOutcome}`,
    `Offers by answer: ${byAnswer}`,
    `Haiku ${tokenTotals(log.flatMap(entry => (entry.haikuUsage === undefined ? [] : [entry.haikuUsage])))}`,
    '',
    ...deepCheckStats(deepChecks),
    '',
    'Last 10 checks with no offer shown (time, Haiku label, outcome, prompt start):',
    ...(skips.length === 0
      ? ['(none)']
      : skips.map(entry => `- ${entry.time} ${entry.haikuLabel} ${entry.outcome}${entry.detail === undefined ? '' : ` (${entry.detail})`}: ${entry.promptHead}`)),
  ].join('\n')
}

function deepCheckStats(checks: readonly LitScoutDeepCheck[]): string[] {
  const sum = (pick: (check: LitScoutDeepCheck) => number) => checks.reduce((total, check) => total + pick(check), 0)
  const recent = checks.slice(-10)

  return [
    `Deep checks: ${checks.length}; papers read in full text ${sum(check => check.fullText)}, from the abstract only ${sum(check => check.abstractOnly)}; characters sent ${sum(check => check.charsSent)}.`,
    `Deep checks by reader: ${countBy(checks, check => check.reader)}`,
    `Reader ${tokenTotals(checks.map(check => check.usage))}`,
    '',
    'Last 10 deep checks (time, reader and model, papers in full text / abstract only, characters sent, input / cache read / cache write / output tokens, total time, query):',
    ...(recent.length === 0
      ? ['(none)']
      : recent.map(
          check =>
            `- ${check.time} ${check.reader} ${check.model}, ${check.fullText}/${check.abstractOnly} papers, ${check.charsSent} chars, ` +
            `${check.usage.input_tokens}/${check.usage.cache_read_input_tokens}/${check.usage.cache_creation_input_tokens}/${check.usage.output_tokens} tokens, ` +
            `${check.timings === undefined ? 'time not recorded' : `${Math.round(check.timings.total / 1000)} s`}: ${check.query}`,
        )),
  ]
}

function tokenTotals(usages: readonly LitScoutUsage[]): string {
  const sum = (pick: (usage: LitScoutUsage) => number) => usages.reduce((total, usage) => total + pick(usage), 0)

  return [
    `calls: ${usages.length}; tokens: input ${sum(usage => usage.input_tokens)},`,
    `output ${sum(usage => usage.output_tokens)},`,
    `cache read ${sum(usage => usage.cache_read_input_tokens)},`,
    `cache write ${sum(usage => usage.cache_creation_input_tokens)}.`,
  ].join(' ')
}

function countBy<T>(items: readonly T[], keyOf: (item: T) => string): string {
  const counts = new Map<string, number>()

  for (const item of items) {
    counts.set(keyOf(item), (counts.get(keyOf(item)) ?? 0) + 1)
  }

  return counts.size === 0 ? '(none)' : [...counts].map(([key, count]) => `${key} ${count}`).join(', ')
}
