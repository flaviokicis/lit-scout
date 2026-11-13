import type { EngineInterface, PromptSubmitInput, Register, RenderInput } from 'claude-code'

import type { LitScoutAnswer, LitScoutBand, LitScoutStage } from '../types'
import {
  addReportToChat,
  answerOffer,
  clearStaleWork,
  freeBand,
  inBackground,
  newLoadId,
  onPrompt,
  readConfig,
  startManualScout,
  type ScoutContext,
} from './flow'
import { findingsBodyMarkdown, findingsHeadMarkdown } from './scout'
import { formatStats, readDeepChecks, readLog, readSession } from './state'

const COMMAND = 'lit-scout'
const PERSON_ORIGINS = new Set(['composer', 'bridge'])

export const register: Register = (on, options) => {
  const context: ScoutContext = { config: readConfig(options), loadId: newLoadId(), drawnOfferIds: new Set() }

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: COMMAND,
      description: 'Prior-art scout: show check stats, or "check <idea>" to search by hand',
      argumentHint: '[check <idea>]',
      immediate: true,
    })
    await clearStaleWork($, context)

    return next(e)
  })

  on('command.run', { command: COMMAND }, async ($, e) => {
    const idea = /^check\s+(.+)$/is.exec(e.args.trim())?.[1]?.trim()

    if (idea === undefined) {
      return { text: formatStats(await readLog($), await readDeepChecks($)) }
    }

    return (await startManualScout($, context, idea))
      ? { text: `lit-scout: checking prior art for "${idea}": write three queries, search, fetch the top papers, connect them to this chat. The report shows above the prompt and goes to PRIOR_ART.md.` }
      : { text: 'lit-scout: a check or an offer holds the band now; answer or dismiss it first.' }
  })

  on('prompt.submit', ($, e, next) => {
    if (isPersonPrompt(e)) {
      inBackground($, 'the prompt check', () => onPrompt($, context, e.text))
    }

    return next(e)
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const { band } = await readSession($)

    if (band.kind === 'idle' || e.props.hasSurvey) {
      return next(e)
    }

    const below = await next(e)

    return below.type === 'engine' ? drawBand($, context, e, band) : below
  })
}

function drawBand($: EngineInterface, context: ScoutContext, e: RenderInput<'AbovePrompt'>, band: Exclude<LitScoutBand, { kind: 'idle' }>) {
  const { Box, Button, Markdown, Text } = $.ui.resolve(e)

  if (band.kind === 'searching') {
    return (
      <Box flexDirection="column">
        <Text dimColor>
          {stageText(band.stage)} for: {band.query}...
        </Text>
        <Box flexDirection="row" gap={2}>
          <Button key="stop" hotkey="0" plain role="dismiss" label="Dismiss" onPress={() => inBackground($, 'dismissing the search', () => freeBand($, context))} />
        </Box>
      </Box>
    )
  }

  if (band.kind === 'report') {
    return (
      <Box flexDirection="column">
        <Markdown key="verdict" text={findingsHeadMarkdown(band.report)} />
        <Box flexDirection="row" gap={2}>
          <Button key="add" hotkey="1" plain variant="primary" label="Add findings to chat" onPress={() => inBackground($, 'adding the findings', () => addReportToChat($, context, band.report, 'findings'))} />
          <Button key="links" hotkey="2" plain label="Add links only" onPress={() => inBackground($, 'adding the links', () => addReportToChat($, context, band.report, 'links'))} />
          <Button key="close" hotkey="0" plain role="dismiss" label="Dismiss" onPress={() => inBackground($, 'dismissing the report', () => freeBand($, context))} />
        </Box>
        <Markdown key="papers" text={findingsBodyMarkdown(band.report)} />
      </Box>
    )
  }

  const offer = band.offer
  const press = (answer: LitScoutAnswer) => () => inBackground($, 'answering the offer', () => answerOffer($, context, offer, answer))
  context.drawnOfferIds.add(offer.id)

  return (
    <Box flexDirection="column">
      <Text>
        <Text bold>Prior art?</Text> {offer.query}
      </Text>
      <Text dimColor>Haiku writes plain, jargon and neighbor queries; each searches OpenAlex, arXiv, Semantic Scholar, Hugging Face and GitHub, then your session's model reads the top papers and connects them to this chat.</Text>
      <Box flexDirection="row" gap={2}>
        <Button key="check" hotkey="1" plain variant="primary" label="Check prior art" onPress={press('check')} />
        <Button key="know" hotkey="2" plain label="I know the lit" onPress={press('know')} />
        <Button key="dismiss" hotkey="0" plain role="dismiss" label="Dismiss" onPress={press('dismiss')} />
      </Box>
    </Box>
  )
}

function stageText(stage: LitScoutStage): string {
  if (stage.kind === 'fetching') {
    return `Fetching ${stage.papers} papers`
  }

  if (stage.kind === 'verifying') {
    return `Checking the quotes from ${stage.papers} papers`
  }

  return stage.kind === 'reading' ? `Reading ${stage.papers} papers with ${stage.model}` : 'Searching prior art'
}

function isPersonPrompt(e: PromptSubmitInput): boolean {
  return e.command === undefined && e.isMeta !== true && PERSON_ORIGINS.has(e.origin.kind) && e.text.trim() !== ''
}
