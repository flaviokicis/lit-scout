import type { EngineInterface, ModelCompleteResult, ModelForkResult, ModelUsage } from 'claude-code'

import type {
  LitScoutAccess,
  LitScoutDeepCheck,
  LitScoutFinding,
  LitScoutHit,
  LitScoutPaper,
  LitScoutQueries,
  LitScoutReader,
  LitScoutReport,
  LitScoutStage,
  LitScoutUsage,
  LitScoutVerification,
} from '../types'
import {
  citingKeptPapers,
  conversationContext,
  parseEntries,
  parseQueries,
  parseReading,
  QUERY_CALL,
  QUERY_KINDS,
  queryPrompt,
  readerPrompt,
  repairPrompt,
  replyFault,
  type Connection,
  type ReadablePaper,
  type Reading,
  type SearchResult,
} from './prompts'

const SEARCH_SCRIPT = 'search_prior_art.py'
const CHECK_SCRIPT = 'check_quotes.py'
const QUERY_TIMEOUT_MS = 30_000
/** The quote check is local string matching: seconds at most. */
const CHECK_TIMEOUT_MS = 60_000
/** The script gives each source 8 s; this only guards against a hung script. */
const SEARCH_TIMEOUT_MS = 60_000
/** Four papers at a time, each PDF download allowed a minute; the host allows ten minutes at most. */
const FETCH_TIMEOUT_MS = 300_000
/** How many of the top search results, by fused rank, are fetched and read; the reader drops the unrelated ones. */
const PAPERS_TO_READ = 10
/** Reading ~100k new tokens closely can take many minutes. */
const READER_TIMEOUT_MS = 20 * 60_000
/** A completion comes back whole, not streamed, and a provider ends one at ten minutes. */
const FALLBACK_READER_CALL = { maxTokens: 32_000, timeoutMs: 10 * 60_000 } as const
/** About half a paper's body once references and appendices are gone: the reader gets each section's start. */
const MAX_PAPER_CHARS = 25_000
const MIN_PAPER_CHARS = 4_000
/** Room left in the window for the reader's instructions, thinking and reply. */
const READER_RESERVE_TOKENS = 40_000
/** Paper text from PDFs runs denser than prose (symbols, numbers), so it is counted at three characters a token. */
const CHARS_PER_TOKEN = 3
const PRIOR_ART_FILE = 'PRIOR_ART.md'
const PRIOR_ART_HEADER = '# Prior art checks\n\nWritten by the lit-scout mod: one entry per check, newest last.\n'

/** Added to a source's status when it ran without an API key. */
const KEYLESS_NOTE = ' [no API key: shared rate pool]'

type SourceStatus = { status: string; detail?: string; count?: number }

type SearchReport = {
  queries: string[]
  /** Sources that take an API key and ran without one. */
  keyless: string[]
  /** One status per query, in the order of QUERY_KINDS. */
  sources: Record<string, SourceStatus[]>
  /** When each source's last answer came, in milliseconds from the start of the search. */
  sourceMs: Record<string, number>
  results: SearchResult[]
}

type FetchedText = { url: string; access: LitScoutAccess; reason?: string; text: string; charsKept: number; truncated: boolean; ms: number }

type ReaderReply = { text: string; usage: ModelUsage; reader: LitScoutReader; charsSent: number }

/** What the check script found for one paper's line. */
type QuoteCheck = { verified: boolean; method: string | null; coverage: number; problems: string[] }

/** A kept paper's line and whether its text backs it; null for a dropped paper. */
type CheckedLine = { connection: Connection; isVerified: boolean; problem?: string } | null

type VerifiedReading = { reading: Reading; lines: CheckedLine[]; verification: LitScoutVerification; repairUsage: ModelUsage | null; verifyMs: number }

export type ScoutResult = { report: LitScoutReport; check: LitScoutDeepCheck | null }

/** Finds candidates, fetches the full text of the top ones, and has the session's model connect them to the researcher's work. */
export async function scoutPriorArt($: EngineInterface, query: string, keyEnv: Record<string, string>, showStage: (stage: LitScoutStage) => Promise<void>): Promise<ScoutResult> {
  const lap = await stopwatch($)
  const queries = await writeQueries($, query)
  const queriesMs = await lap()
  const search = await runScript<SearchReport>($, SEARCH_SCRIPT, ['search', ...QUERY_KINDS.map(kind => queries[kind])], { timeoutMs: SEARCH_TIMEOUT_MS, env: keyEnv })
  const searchMs = await lap()
  const sourceStatus = Object.fromEntries(
    Object.entries(search.sources).map(([source, statuses]) => [source, statusText(statuses) + (search.keyless.includes(source) ? KEYLESS_NOTE : '')]),
  )
  const candidates = search.results.slice(0, PAPERS_TO_READ)

  if (candidates.length === 0) {
    const nextStep = 'Nothing was read, and that is no sign the idea is new: rephrase it with `/lit-scout check <other words>`.'

    return { report: { query, queries, verdict: 'NO MATCHES FOUND', nextStep, goal: '', papers: [], dropped: [], overall: '', sourceStatus }, check: null }
  }

  await showStage({ kind: 'fetching', papers: candidates.length })
  const { papers, paperMs } = await fetchPapers($, candidates, await paperCharCap($, candidates.length))
  const fetchMs = await lap()
  const model = await $.session.model()
  await showStage({ kind: 'reading', papers: papers.length, model })
  const read = await readPapers($, query, papers, model)
  const readerMs = await lap()
  const firstReading = parseReading(read.text, papers.length)
  await showStage({ kind: 'verifying', papers: firstReading.connections.filter(connection => connection !== null).length })
  const checked = await verifyReading($, papers, firstReading, read.reader, model)
  const repairMs = (await lap()) - checked.verifyMs
  const { connections, ...reading } = citingKeptPapers(checked.reading)
  const judged = papers.map(({ body: _body, ...paper }, index) => ({ paper, connection: connections[index] ?? null, line: checked.lines[index] ?? null }))
  const report: LitScoutReport = {
    query,
    queries,
    ...reading,
    papers: judged.flatMap(({ paper, connection, line }) => (connection === null || line === null ? [] : [findingOf(paper, connection, line)])),
    dropped: judged.flatMap(({ paper, connection }) => (connection === null ? [paper] : [])),
    sourceStatus,
  }
  const fullText = papers.filter(paper => paper.access === 'full text').length
  const check: LitScoutDeepCheck = {
    time: new Date(await $.clock.now()).toISOString(),
    query,
    fullText,
    abstractOnly: papers.length - fullText,
    charsSent: read.charsSent,
    reader: read.reader,
    model,
    usage: checked.repairUsage === null ? read.usage : addedUsage(read.usage, checked.repairUsage),
    verification: checked.verification,
    timings: {
      queries: queriesMs,
      search: searchMs,
      sources: search.sourceMs,
      fetch: fetchMs,
      papers: paperMs,
      reader: readerMs,
      verify: checked.verifyMs,
      repair: repairMs,
      total: queriesMs + searchMs + fetchMs + readerMs + checked.verifyMs + repairMs,
    },
  }

  return { report, check }
}

/** A lap timer on the host clock: each call returns the milliseconds since the previous call (or since the start). */
async function stopwatch($: EngineInterface): Promise<() => Promise<number>> {
  let last = await $.clock.now()

  return async () => {
    const now = await $.clock.now()
    const elapsed = now - last
    last = now

    return elapsed
  }
}

/** Haiku writes the plain, jargon and neighbor queries from the idea and the recent turns. */
async function writeQueries($: EngineInterface, idea: string): Promise<LitScoutQueries> {
  const reply = await $.model.complete({ ...QUERY_CALL, prompt: queryPrompt(idea, await recentTurns($)), timeoutMs: QUERY_TIMEOUT_MS })

  if (!reply.isAnswered) {
    throw new Error(`the query writer did not answer: ${missReason(reply)}`)
  }

  const fault = replyFault(reply.text, reply.usage.output_tokens, QUERY_CALL.maxTokens)

  if (fault !== null) {
    throw new Error(`the query writer's reply is unusable: ${fault}`)
  }

  return parseQueries(reply.text)
}

/** One source over the queries: "ok (8/8/7)", one status when all match, else each query's own. */
function statusText(statuses: readonly SourceStatus[]): string {
  const own = (status: SourceStatus) =>
    status.status === 'ok' ? `ok ${status.count ?? 0}` : status.detail === undefined ? status.status : `${status.status}: ${status.detail}`

  if (statuses.every(status => status.status === 'ok')) {
    return `ok (${statuses.map(status => status.count ?? 0).join('/')})`
  }

  const texts = statuses.map(own)

  return new Set(texts).size === 1 ? (texts[0] ?? '') : texts.map((text, index) => `${QUERY_KINDS[index] ?? `query ${index + 1}`} ${text}`).join(', ')
}

async function runScript<T>(
  $: EngineInterface,
  script: string,
  args: readonly string[],
  init: { timeoutMs: number; stdin?: string; env?: Record<string, string> },
): Promise<T> {
  const ran = await $.process.run(['python3', `${$.plugin.root}/scripts/${script}`, ...args], init)
  const what = [script, ...args.slice(0, 1)].join(' ')

  if (ran.exitCode !== 0) {
    throw new Error(`${what} exited ${ran.exitCode}: ${ran.stderr.trim().slice(-300)}`)
  }

  if (ran.isStdoutTruncated) {
    throw new Error(`${what} wrote more than the host reads`)
  }

  return JSON.parse(ran.stdout) as T
}

/** Each paper's share of what the window has left, at most MAX_PAPER_CHARS, so the reader's request fits. */
async function paperCharCap($: EngineInterface, paperCount: number): Promise<number> {
  const { context } = await $.session.usage()
  const roomTokens = context.window - (context.tokens ?? 0) - READER_RESERVE_TOKENS
  const share = Math.floor((roomTokens * CHARS_PER_TOKEN) / paperCount)

  return Math.max(MIN_PAPER_CHARS, Math.min(MAX_PAPER_CHARS, share))
}

async function fetchPapers($: EngineInterface, candidates: readonly SearchResult[], maxChars: number): Promise<{ papers: ReadablePaper[]; paperMs: number[] }> {
  const wanted = candidates.map(({ url, arxivId, pdfUrl }) => ({ url, arxivId, pdfUrl }))
  const fetched = await runScript<FetchedText[]>($, SEARCH_SCRIPT, ['fetch', '--max-chars', String(maxChars)], { timeoutMs: FETCH_TIMEOUT_MS, stdin: JSON.stringify(wanted) })

  if (fetched.length !== candidates.length) {
    throw new Error(`the fetch step returned ${fetched.length} papers for ${candidates.length} asked`)
  }

  const papers = candidates.map((result, index) => {
    const { access, reason, text, charsKept, truncated } = fetched[index] as FetchedText
    const paper: LitScoutPaper = {
      title: result.title,
      year: result.year,
      url: result.url,
      access,
      charsKept,
      truncated,
      foundBy: result.foundBy,
      ...(reason === undefined ? {} : { reason }),
    }

    return { ...paper, body: access === 'full text' ? text : result.abstract || '(no abstract: title only)' }
  })

  return { papers, paperMs: fetched.map(paper => paper.ms) }
}

/**
 * Checks each kept paper's quote and numbers against the text the reader saw. The failed papers alone go back to the
 * same reader once, with what the check found; a rewritten line replaces the old one only when it passes, a DROP
 * drops the paper, and a line that still fails stays, marked unverified.
 */
async function verifyReading($: EngineInterface, papers: readonly ReadablePaper[], reading: Reading, reader: LitScoutReader, model: string): Promise<VerifiedReading> {
  const lap = await stopwatch($)
  const conversation = (await $.session.messages()).map(message => message.text).join('\n')
  const lines = await checkLines($, papers, reading.connections, conversation)
  const verifyMs = await lap()
  const failures = lines.flatMap((line, index) =>
    line === null || line.isVerified ? [] : [{ number: index + 1, paper: papers[index] as ReadablePaper, connection: line.connection, finding: line.problem ?? '' }],
  )
  const verified = lines.filter(line => line?.isVerified === true).length

  if (failures.length === 0) {
    return { reading, lines, verification: { verified, repaired: 0, dropped: 0, unverified: 0 }, repairUsage: null, verifyMs }
  }

  const repair = await askReader($, reader, model, conversationText => repairPrompt(failures, conversationText))
  const rewritten = parseEntries(repair.text, failures.map(failure => failure.number))
  const changed = failures.flatMap((failure, index) => {
    const connection = rewritten[index]

    return connection === null || connection === undefined || isSameLine(connection, failure.connection) ? [] : [{ failure, connection }]
  })
  const rechecked = await checkLines($, changed.map(({ failure }) => failure.paper), changed.map(({ connection }) => connection), conversation)
  const repaired = [...lines]
  failures.forEach((failure, index) => {
    if (rewritten[index] === null) {
      repaired[failure.number - 1] = null
    }
  })
  changed.forEach(({ failure }, index) => {
    const line = rechecked[index] ?? null

    if (line?.isVerified === true) {
      repaired[failure.number - 1] = line
    }
  })
  const dropped = rewritten.filter(connection => connection === null).length
  const unverified = repaired.filter(line => line !== null && !line.isVerified).length

  return {
    reading: { ...reading, connections: repaired.map(line => line?.connection ?? null) },
    lines: repaired,
    verification: { verified, repaired: failures.length - dropped - unverified, dropped, unverified },
    repairUsage: repair.usage,
    verifyMs,
  }
}

function isSameLine(a: Connection, b: Connection): boolean {
  return a.text === b.text && a.quote === b.quote
}

/** One run of the check script over the given papers' lines; a dropped paper is not checked. */
async function checkLines($: EngineInterface, papers: readonly ReadablePaper[], connections: readonly (Connection | null)[], conversation: string): Promise<CheckedLine[]> {
  const kept = connections.flatMap((connection, index) => (connection === null ? [] : [{ index, connection }]))

  if (kept.length === 0) {
    return connections.map(() => null)
  }

  const claims = kept.map(({ index, connection }) => {
    const paper = papers[index] as ReadablePaper

    return { source: paper.body, title: paper.title, year: paper.year, sentence: connection.text, quote: connection.quote }
  })
  const checks = await runScript<QuoteCheck[]>($, CHECK_SCRIPT, [], { timeoutMs: CHECK_TIMEOUT_MS, stdin: JSON.stringify({ context: conversation, papers: claims }) })

  if (checks.length !== kept.length) {
    throw new Error(`the quote check returned ${checks.length} results for ${kept.length} papers`)
  }

  const lines: CheckedLine[] = connections.map(() => null)
  kept.forEach(({ index, connection }, position) => {
    const check = checks[position] as QuoteCheck
    lines[index] = check.verified ? { connection, isVerified: true } : { connection, isVerified: false, problem: check.problems.join('; ') }
  })

  return lines
}

function findingOf(paper: LitScoutPaper, connection: Connection, line: Exclude<CheckedLine, null>): LitScoutFinding {
  return { ...paper, connection: connection.text, quote: connection.quote, isVerified: line.isVerified, ...(line.problem === undefined ? {} : { problem: line.problem }) }
}

function addedUsage(a: ModelUsage, b: ModelUsage): LitScoutUsage {
  return {
    input_tokens: a.input_tokens + b.input_tokens,
    output_tokens: a.output_tokens + b.output_tokens,
    cache_read_input_tokens: a.cache_read_input_tokens + b.cache_read_input_tokens,
    cache_creation_input_tokens: a.cache_creation_input_tokens + b.cache_creation_input_tokens,
  }
}

/** A fork of the session reads the papers over the whole conversation; before the first turn ends, a completion over the recent turns does. */
async function readPapers($: EngineInterface, query: string, papers: readonly ReadablePaper[], model: string): Promise<ReaderReply> {
  const forked = await forkReader($, readerPrompt(query, papers, null))

  if (forked !== null) {
    return forked
  }

  $.ui.log('lit-scout: no main-thread snapshot to fork yet; reading with a completion over the recent turns instead', { to: 'debug' })

  return completeReader($, model, readerPrompt(query, papers, await recentTurns($)))
}

/** Asks the reader that read the papers again: a fork over the conversation, or a completion given the recent turns. */
async function askReader($: EngineInterface, reader: LitScoutReader, model: string, prompt: (conversation: string | null) => string): Promise<ReaderReply> {
  if (reader === 'complete') {
    return completeReader($, model, prompt(await recentTurns($)))
  }

  const forked = await forkReader($, prompt(null))

  if (forked === null) {
    throw new Error('the reader (a fork of the session) found nothing to fork for the repair round')
  }

  return forked
}

/** The fork's reply, or null when the session has no main-thread snapshot to fork yet. */
async function forkReader($: EngineInterface, prompt: string): Promise<ReaderReply | null> {
  const forked = await withinTime($, $.model.fork({ prompt }), READER_TIMEOUT_MS, 'the reader (a fork of the session)')

  if (forked.isAnswered) {
    return { text: forked.text, usage: forked.usage, reader: 'fork', charsSent: prompt.length }
  }

  if (forked.reason !== 'nothing-to-fork') {
    throw new Error(`the reader (a fork of the session) did not answer: ${missReason(forked)}`)
  }

  return null
}

async function recentTurns($: EngineInterface): Promise<string> {
  const history = (await $.session.messages()).map(message => ({ role: message.role, text: message.text }))

  return conversationContext(history)
}

async function completeReader($: EngineInterface, model: string, prompt: string): Promise<ReaderReply> {
  const completed = await $.model.complete({ ...FALLBACK_READER_CALL, model, prompt })

  if (!completed.isAnswered) {
    throw new Error(`the reader (a completion on ${model}) did not answer: ${missReason(completed)}`)
  }

  const fault = replyFault(completed.text, completed.usage.output_tokens, FALLBACK_READER_CALL.maxTokens)

  if (fault !== null) {
    throw new Error(`the reader's reply is unusable: ${fault}`)
  }

  return { text: completed.text, usage: completed.usage, reader: 'complete', charsSent: prompt.length }
}

/** A fork takes no time limit of its own: past `ms` this stops waiting for it and fails. */
function withinTime<T>($: EngineInterface, work: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = $.clock.after(ms, () => reject(new Error(`${what} gave no answer within ${Math.round(ms / 60_000)} minutes`)))
    work.then(
      value => {
        timer.cancel()
        resolve(value)
      },
      (error: unknown) => {
        timer.cancel()
        reject(error)
      },
    )
  })
}

function missReason(result: Exclude<ModelCompleteResult | ModelForkResult, { isAnswered: true }>): string {
  return result.reason === 'api-error' ? `api-error (HTTP ${result.status ?? 'none'}, ${result.error})` : result.reason
}

function paperLine(paper: LitScoutPaper, number: number, isFull: boolean, label = ''): string {
  const access = paper.access === 'full text' ? '' : isFull && paper.reason !== undefined ? `, abstract only: ${paper.reason}` : ', abstract only'
  const foundBy = isFull ? `; found by ${paper.foundBy.map(hitText).join(', ')}` : ''

  return `${number}. [${paper.title}](${paper.url}) (${paper.year ?? 'n.d.'}${access}${label}${foundBy})`
}

/** The paper, how it connects, and the quote that backs it; an unverified one says so (and, in full, why). */
function findingLines(finding: LitScoutFinding, number: number, isFull: boolean): string[] {
  const label = finding.isVerified ? '' : isFull ? `, unverified: ${finding.problem ?? 'the check failed'}` : ', unverified'
  const quote = finding.quote === '' ? [] : [`   - *"${finding.quote}"*${finding.access === 'full text' ? '' : ' (from the abstract)'}`]

  return [paperLine(finding, number, isFull, label), `   - ${finding.connection}`, ...quote]
}

function hitText(hit: LitScoutHit): string {
  return `${hit.source} #${hit.rank} for ${QUERY_KINDS[hit.query] ?? `query ${hit.query + 1}`}`
}

function readCountLine(report: LitScoutReport): string {
  const read = report.papers.length + report.dropped.length

  return report.papers.length === 0 ? `Read ${read} papers, none relevant.` : `Read ${read} papers, ${report.papers.length} relevant:`
}

/** The verdict and next step: the top of the report band, above its buttons, and of what "Add findings to chat" pastes. */
export function findingsHeadMarkdown(report: LitScoutReport): string {
  return `**${report.verdict}**: ${report.nextStep}`
}

/** Each relevant paper with how it connects (or, with no match, the queries and sources): the band's list under its buttons. */
export function findingsBodyMarkdown(report: LitScoutReport): string {
  const found =
    report.verdict === 'NO MATCHES FOUND'
      ? [`Idea: \`${report.query}\``, queriesLine(report), sourcesLine(report)]
      : [readCountLine(report), ...report.papers.flatMap((finding, index) => findingLines(finding, index + 1, false))]

  return found.join('\n')
}

/** What "Add findings to chat" pastes: the verdict and next step, then the findings. */
export function findingsMarkdown(report: LitScoutReport): string {
  return `${findingsHeadMarkdown(report)}\n\n${findingsBodyMarkdown(report)}`
}

/** What "Add links only" pastes: the relevant papers' titles and links. */
export function linksMarkdown(report: LitScoutReport): string {
  return report.papers.length === 0 ? '(no relevant papers)' : report.papers.map(paper => `- [${paper.title}](${paper.url})`).join('\n')
}

/** The whole report for PRIOR_ART.md: the findings with what found each paper, the goal and overall reading, what was dropped, the search. */
export function reportMarkdown(report: LitScoutReport): string {
  const papers = report.papers.flatMap((finding, index) => findingLines(finding, index + 1, true))
  const dropped = report.dropped.map((paper, index) => paperLine(paper, index + 1, true))

  return [
    `**${report.verdict}**: ${report.nextStep}`,
    '',
    `Idea: \`${report.query}\``,
    ...(report.goal === '' ? [] : [`Goal as understood: ${report.goal}`]),
    queriesLine(report),
    ...(report.verdict === 'NO MATCHES FOUND' ? [] : ['', readCountLine(report), ...papers]),
    ...(report.overall === '' ? [] : ['', `Overall: ${report.overall}`]),
    ...(dropped.length === 0 ? [] : ['', 'Read and dropped as unrelated:', ...dropped]),
    '',
    sourcesLine(report),
  ].join('\n')
}

function queriesLine(report: LitScoutReport): string {
  return `Queries: ${QUERY_KINDS.map(kind => `${kind} \`${report.queries[kind]}\``).join('; ')}`
}

function sourcesLine(report: LitScoutReport): string {
  return `Sources: ${Object.entries(report.sourceStatus)
    .map(([source, status]) => `${source} ${status}`)
    .join('; ')}`
}

export async function appendPriorArt($: EngineInterface, report: LitScoutReport): Promise<string> {
  const path = `${await $.session.root()}/${PRIOR_ART_FILE}`
  const existing = (await $.fs.exists(path)) ? await $.fs.read(path) : PRIOR_ART_HEADER
  const stamp = new Date(await $.clock.now()).toISOString()
  const entry = `\n## ${stamp}: ${report.query}\n\n${reportMarkdown(report)}\n`

  await $.fs.write(path, `${existing.trimEnd()}\n${entry}`)

  return path
}
