import type { EngineInterface, ModelUsage, PluginOptions } from 'claude-code'

import type { LitScoutAnswer, LitScoutLogEntry, LitScoutOffer, LitScoutReport, LitScoutSession, LitScoutStage } from '../types'
import { CLASSIFIER_CALL, classifierInput, classifierSystem, parseClassifier, replyFault } from './prompts'
import { appendPriorArt, findingsMarkdown, linksMarkdown, scoutPriorArt } from './scout'
import {
  afterAnswer,
  amendLog,
  appendDeepCheck,
  appendLog,
  changeSession,
  clearStale,
  decideSession,
  isDuplicate,
  releaseBand,
  readSessionChecked,
  rememberChecked,
  rememberOffered,
  setBand,
  takePrompt,
} from './state'

const CLASSIFIER_TIMEOUT_MS = 20_000
const SEARCHING: LitScoutStage = { kind: 'searching' }

/** An empty key leaves the variable unset, so the host's own OPENALEX_API_KEY / S2_API_KEY, if any, still reach the script. */
export type ScoutConfig = { examples: string; isDebug: boolean; keyEnv: Record<string, string> }

/** What one load of the module shares between its hooks: its options, its id, and which offers the band has drawn. */
export type ScoutContext = { config: ScoutConfig; loadId: string; drawnOfferIds: Set<string> }

type Classified = ({ label: 'LIT'; query: string } | { label: 'OTHER' } | { label: 'ERROR'; detail: string }) & { usage: ModelUsage }

/** One prompt's check: the prompt and what Haiku said and cost. */
type CheckRequest = { prompt: string; haikuLabel: Classified['label']; haikuUsage: ModelUsage }

type Placement = { kind: 'shown' } | { kind: 'backoff' } | { kind: 'queued'; holder: LitScoutSession['band']['kind']; superseded: LitScoutOffer | null }

type CheckEntry = Pick<LitScoutLogEntry, 'outcome'> & Partial<Pick<LitScoutLogEntry, 'id' | 'query' | 'detail'>>

export function readConfig(options: PluginOptions): ScoutConfig {
  const keyEnv: Record<string, string> = {}

  for (const [variable, option] of [['OPENALEX_API_KEY', options.openalexKey], ['S2_API_KEY', options.s2Key]] as const) {
    const key = String(option ?? '').trim()

    if (key !== '') {
      keyEnv[variable] = key
    }
  }

  return { examples: String(options.examples ?? ''), isDebug: options.debug === true, keyEnv }
}

export function newLoadId(): string {
  return Math.random().toString(36).slice(2, 10)
}

export function inBackground($: EngineInterface, what: string, work: () => Promise<unknown>): void {
  void work().catch(error => reportFailure($, what, error))
}

function reportFailure($: EngineInterface, what: string, error: unknown): void {
  const text = `lit-scout: ${what} failed: ${messageOf(error)}`
  $.ui.log(text, { to: 'debug' })
  $.ui.toast(text)
}

function say($: EngineInterface, context: ScoutContext, text: string): void {
  $.ui.log(`lit-scout: ${text}`, { to: context.config.isDebug ? 'transcript' : 'debug' })
}

/** A search that the previous load left running never ends in this one, so its band is freed. */
export async function clearStaleWork($: EngineInterface, context: ScoutContext): Promise<void> {
  await announcePromoted($, context, await decideSession($, clearStale(context.loadId)))
}

/** Haiku reads every prompt with the recent turns and the ideas already checked; a LIT answer goes through the anti-spam checks to the band. */
export async function onPrompt($: EngineInterface, context: ScoutContext, text: string): Promise<void> {
  const turn = await decideSession($, takePrompt(context.loadId))

  if (turn.expired !== null) {
    await amendLog($, turn.expired.id, { answer: 'expired', outcome: context.drawnOfferIds.has(turn.expired.id) ? 'shown' : 'yielded' })
  }

  await announcePromoted($, context, turn.promoted)

  const history = (await $.session.messages()).map(message => ({ role: message.role, text: message.text }))
  const classified = await classify($, context, classifierInput(history, text, await readSessionChecked($)))
  const request: CheckRequest = { prompt: text, haikuLabel: classified.label, haikuUsage: classified.usage }

  if (classified.label === 'ERROR') {
    reportFailure($, 'the Haiku classifier', classified.detail)
    return logCheck($, context, request, { outcome: 'error', detail: classified.detail })
  }

  if (classified.label === 'OTHER') {
    return logCheck($, context, request, { outcome: 'none' })
  }

  await offerIdea($, context, request, classified.query)
}

export async function answerOffer($: EngineInterface, context: ScoutContext, offer: LitScoutOffer, answer: LitScoutAnswer): Promise<void> {
  await amendLog($, offer.id, { answer })

  if (answer === 'check' || answer === 'know') {
    await rememberChecked($, offer.query)
  }

  if (answer !== 'check') {
    await announcePromoted($, context, await decideSession($, value => releaseBand(afterAnswer(value, answer))))
    return
  }

  await changeSession($, value => afterAnswer(setBand({ kind: 'searching', id: offer.id, query: offer.query, loadId: context.loadId, stage: SEARCHING })(value), answer))
  inBackground($, 'the prior-art scout', () => runScout($, context, offer.query, offer.id, offer.id))
}

/** Starts a search by hand; false when the band is already taken. */
export async function startManualScout($: EngineInterface, context: ScoutContext, query: string): Promise<boolean> {
  const id = await newId($)
  const isStarted = await decideSession($, value =>
    value.band.kind === 'idle'
      ? ([setBand({ kind: 'searching', id, query, loadId: context.loadId, stage: SEARCHING })(value), true] as const)
      : ([value, false] as const),
  )

  if (isStarted) {
    await rememberChecked($, query)
    inBackground($, 'the prior-art scout', () => runScout($, context, query, id, null))
  }

  return isStarted
}

/**
 * Pastes the findings (verdict, next step, each relevant paper with how it connects), or only the papers' links,
 * into the prompt box after any draft, so the person sees them and sends them to the model with their own words.
 */
export async function addReportToChat($: EngineInterface, context: ScoutContext, report: LitScoutReport, part: 'findings' | 'links'): Promise<void> {
  const text =
    part === 'findings'
      ? `Prior-art check by the lit-scout mod for "${report.query}":\n\n${findingsMarkdown(report)}`
      : `Prior-art papers found by the lit-scout mod for "${report.query}":\n\n${linksMarkdown(report)}`
  const hasDraft = (await $.prompt.read()).text.trim() !== ''
  const filled = await $.prompt.fill(hasDraft ? { text: `\n\n${text}\n\n`, mode: 'append' } : { text: `${text}\n\n`, mode: 'replace' })

  if (!filled.isFilled) {
    throw new Error(`the prompt box did not take the report (${filled.refusal ?? 'refused by a hook'})`)
  }

  await freeBand($, context)
}

/** Dismisses a report, or a check in progress: that check still writes PRIOR_ART.md but no longer takes the band. */
export async function freeBand($: EngineInterface, context: ScoutContext): Promise<void> {
  await announcePromoted($, context, await decideSession($, releaseBand))
}

async function runScout($: EngineInterface, context: ScoutContext, query: string, searchId: string, logId: string | null): Promise<void> {
  const isOwnSearch = (value: LitScoutSession) => value.band.kind === 'searching' && value.band.id === searchId
  const showStage = (stage: LitScoutStage) =>
    changeSession($, value => (value.band.kind === 'searching' && isOwnSearch(value) ? { ...value, band: { ...value.band, stage } } : value)).then(() => undefined)

  try {
    const { report, check } = await scoutPriorArt($, query, context.config.keyEnv, showStage)
    const path = await appendPriorArt($, report)

    if (check !== null) {
      await appendDeepCheck($, check)
      say($, context, `read ${check.fullText} papers in full text and ${check.abstractOnly} from the abstract only with a ${check.reader} on ${check.model}; kept ${report.papers.length}`)
    }

    if (logId !== null) {
      await amendLog($, logId, { detail: `scout: ${report.verdict}` })
    }

    const isShown = await decideSession($, value =>
      isOwnSearch(value) ? ([setBand({ kind: 'report', report, promptsSince: 0 })(value), true] as const) : ([value, false] as const),
    )
    say($, context, `verdict ${report.verdict} for "${query}", appended to ${path}${isShown ? '' : '; not shown, the check was dismissed'}`)
  } catch (error) {
    await announcePromoted($, context, await decideSession($, value => (isOwnSearch(value) ? releaseBand(value) : ([value, null] as const))))

    if (logId !== null) {
      await amendLog($, logId, { detail: `scout error: ${messageOf(error)}` })
    }

    throw error
  }
}

async function classify($: EngineInterface, context: ScoutContext, chat: string): Promise<Classified> {
  const reply = await $.model.complete({ ...CLASSIFIER_CALL, system: classifierSystem(context.config.examples), prompt: chat, timeoutMs: CLASSIFIER_TIMEOUT_MS })

  if (!reply.isAnswered) {
    return { label: 'ERROR', detail: `no answer: ${reply.reason}`, usage: reply.usage }
  }

  const fault = replyFault(reply.text, reply.usage.output_tokens, CLASSIFIER_CALL.maxTokens)

  if (fault !== null) {
    return { label: 'ERROR', detail: fault, usage: reply.usage }
  }

  try {
    return { ...parseClassifier(reply.text), usage: reply.usage }
  } catch (error) {
    return { label: 'ERROR', detail: messageOf(error), usage: reply.usage }
  }
}

/** Dedupe against offered and checked ideas, then back off after ignored offers, then show the offer or queue it. */
async function offerIdea($: EngineInterface, context: ScoutContext, request: CheckRequest, query: string): Promise<void> {
  if (await isDuplicate($, query)) {
    return logCheck($, context, request, { query, outcome: 'deduped' })
  }

  const id = await newId($)
  const offer: LitScoutOffer = { id, query, promptsSince: 0 }
  const placement = await decideSession($, (value): readonly [LitScoutSession, Placement] => {
    if (value.ideasToSkip > 0) {
      return [{ ...value, ideasToSkip: value.ideasToSkip - 1 }, { kind: 'backoff' }]
    }

    return value.band.kind === 'idle'
      ? [{ ...value, band: { kind: 'offer', offer } }, { kind: 'shown' }]
      : [{ ...value, pending: offer }, { kind: 'queued', holder: value.band.kind, superseded: value.pending }]
  })

  if (placement.kind === 'backoff') {
    return logCheck($, context, request, { query, outcome: 'backoff', detail: 'recent offers were ignored' })
  }

  if (placement.kind === 'shown') {
    await rememberOffered($, query)
    return logCheck($, context, request, { id, query, outcome: 'shown' })
  }

  if (placement.superseded !== null) {
    await amendLog($, placement.superseded.id, { outcome: 'superseded', detail: `replaced in the queue by "${query}"` })
  }

  await logCheck($, context, request, { id, query, outcome: 'queued', detail: `the band holds a ${placement.holder}; shown when it frees` })
}

/** A queued idea that took the freed band is an offer now. */
async function announcePromoted($: EngineInterface, context: ScoutContext, offer: LitScoutOffer | null): Promise<void> {
  if (offer === null) {
    return
  }

  await rememberOffered($, offer.query)
  await amendLog($, offer.id, { outcome: 'shown' })
  say($, context, `queued idea shown: ${offer.query}`)
}

async function logCheck($: EngineInterface, context: ScoutContext, request: CheckRequest, entry: CheckEntry): Promise<void> {
  await appendLog($, {
    id: entry.id ?? (await newId($)),
    time: new Date(await $.clock.now()).toISOString(),
    haikuLabel: request.haikuLabel,
    haikuUsage: request.haikuUsage,
    answer: null,
    promptHead: request.prompt.replace(/\s+/g, ' ').slice(0, 80),
    ...entry,
  })
  say($, context, `check (Haiku ${request.haikuLabel}): ${entry.outcome}${entry.detail === undefined ? '' : ` (${entry.detail})`}`)
}

async function newId($: EngineInterface): Promise<string> {
  return `${await $.clock.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
