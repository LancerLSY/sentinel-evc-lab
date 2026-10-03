import { readFileSync, writeFileSync } from 'node:fs';
import {
  canonicalJson,
  sha256,
  type ExecutionEvidence
} from '../../packages/core/evidence';
import {
  executablePolicyHash,
  executablePolicySpecSchema,
  type ExecutablePolicySpec
} from '../../packages/core/exec-spec';
import {
  configurationDigest,
  executionConfigurationV1Schema,
  type ExecutionConfiguration,
  type ExecutionConfigurationV1
} from '../../packages/core/execution-configuration';
import {
  ReleaseExecutionGate,
  type EvidenceSink,
  type ExecutionRequest
} from '../../packages/core/execution-gate';
import type { ReleaseRecord } from '../../packages/core/release-policy';
import {
  runtimeAttestationSchema,
  type RuntimeAttestation
} from '../../packages/core/runtime-attestation';

type FrozenCase = {
  action_hex: string;
  case_id: string;
  dtype: 'float32';
  initial_state_index: number;
  scenario:
    | 'valid'
    | 'action_replacement'
    | 'lease_replay'
    | 'expired_permit'
    | 'feedback_changed'
    | 'context_changed'
    | 'revoked_generation';
  shape: [number, number];
  source_action_bytes_hash: string;
  source_episode_id: string;
  source_feedback_hash: string;
  source_manifest_sha256: string;
  source_step: number;
  submitted_action_hex: string;
  task_id: number;
};

type FrozenAction = {
  bytesHex: string;
  dtype: 'float32';
  shape: [number, number];
};

type LiveState = {
  feedbackHash: string;
  initialStateIndex: number;
  episodeId: string;
  sourceStep: number;
};

type CaseResult = {
  case_id: string;
  scenario: string;
  decision: 'ALLOW' | 'DENY';
  reason: string;
  precondition_writer_calls: number;
  writer_calls: number;
  receipt_count: number;
};

const H = (character: string) => character.repeat(64);
const BASE_MS = Date.parse('2026-10-03T00:00:00.000Z');
const DEVICE = 'freeze-arm';
const CONTROLLER = 'freeze-controller-v1';
const ADAPTER = 'rlsok-frozen-comparison';

function actionHash(action: FrozenAction): string {
  return sha256(canonicalJson(action));
}

function configuration(taskId: number, observedAt: string): ExecutionConfigurationV1 {
  return executionConfigurationV1Schema.parse({
    schemaVersion: 1,
    deviceIdentity: DEVICE,
    robotIdentity: H('e'),
    rosDistro: 'jazzy',
    rmwImplementation: 'rmw_fastrtps_cpp',
    jointState: {
      topic: '/joint_states',
      messageType: 'sensor_msgs/msg/JointState'
    },
    controller: {
      name: CONTROLLER,
      followJointTrajectoryAction: '/freeze_controller/follow_joint_trajectory',
      actionType: 'control_msgs/action/FollowJointTrajectory'
    },
    jointOrder: Array.from({ length: 7 }, (_, index) => `joint_${index}`),
    adapter: { identity: ADAPTER, version: `1.${taskId}` },
    observedAt
  });
}

function attestation(
  taskId: number,
  feedbackHash: string,
  observedAt: string
): RuntimeAttestation {
  return runtimeAttestationSchema.parse({
    schemaVersion: 1,
    source: {
      identity: 'freeze-live-state-monitor',
      kind: 'external-monitor',
      version: '1'
    },
    observedAt,
    continuityToken: `task:${taskId}:feedback:${feedbackHash}`,
    availableCapabilities: ['controller.available', 'state.fresh']
  });
}

function releaseFor(
  item: FrozenCase,
  config: ExecutionConfiguration,
  now: string
): ExecutablePolicySpec {
  return executablePolicySpecSchema.parse({
    apiVersion: 'realitywarden.io/v1alpha1',
    kind: 'ExecutablePolicy',
    metadata: {
      name: `frozen-task-${item.task_id}`,
      releaseId: `freeze-release-${item.task_id}`,
      createdAt: now
    },
    model: {
      artifact: `frozen/${item.source_episode_id}`,
      sha256: item.source_manifest_sha256,
      framework: 'custom',
      policyType: 'joint-position-final-action',
      codeRevision: 'freeze-v1'
    },
    actionContract: {
      representation: 'joint_position',
      dimension: 7,
      jointOrder: Array.from({ length: 7 }, (_, index) => `joint_${index}`),
      units: { position: 'radian', velocity: 'radian_per_second' },
      normalizerSha256: H('b'),
      preprocessorSha256: H('c'),
      postprocessorSha256: H('d')
    },
    robot: {
      profileId: 'freeze-7dof-arm',
      profileSha256: H('e'),
      urdfSha256: H('f'),
      controllerType: CONTROLLER,
      controllerConfigSha256: H('1')
    },
    runtimePolicy: {
      policySha256: H('2'),
      maxStateAgeMs: 1_000,
      maxConfigurationAgeMs: 60_000,
      maxAttestationAgeMs: 1_000,
      requiredCapabilities: ['controller.available', 'state.fresh'],
      failClosed: true
    },
    executionConfiguration: config,
    approvedConfigurationDigest: configurationDigest(config),
    evidence: {
      scenarioPackId: 'competitive-boundary-freeze-v1',
      testReportSha256: H('3'),
      status: 'approved',
      approvedBy: 'comparison-harness',
      approvedAt: now
    },
    deployment: {
      allowedDeviceIds: [DEVICE],
      mode: 'released',
      expiresAt: '2099-01-01T00:00:00.000Z'
    }
  });
}

function releaseRecord(spec: ExecutablePolicySpec): ReleaseRecord {
  const identity = executablePolicyHash(spec);
  return {
    releaseId: spec.metadata.releaseId,
    state: 'released',
    executablePolicyHash: identity,
    approvedIdentityHash: identity,
    approvedConfigurationDigest: spec.approvedConfigurationDigest,
    approvedBy: 'comparison-harness',
    approvedAt: spec.evidence.approvedAt
  };
}

function requestFor(
  item: FrozenCase,
  spec: ExecutablePolicySpec,
  config: ExecutionConfiguration,
  runtime: RuntimeAttestation,
  now: Date
): ExecutionRequest<FrozenAction, LiveState> {
  const action: FrozenAction = {
    bytesHex: item.action_hex,
    dtype: item.dtype,
    shape: item.shape
  };
  return {
    release: spec,
    releaseRecord: releaseRecord(spec),
    deviceId: DEVICE,
    proposalId: `proposal-${item.case_id}`,
    action,
    actionHash: actionHash(action),
    state: {
      feedbackHash: item.source_feedback_hash,
      initialStateIndex: item.initial_state_index,
      episodeId: item.source_episode_id,
      sourceStep: item.source_step
    },
    stateObservedAt: now.toISOString(),
    controllerIdentity: CONTROLLER,
    executionConfiguration: config,
    runtimeAttestation: runtime,
    now
  };
}

async function runCase(item: FrozenCase): Promise<CaseResult> {
  let monotonic = 10_000;
  const now = new Date(BASE_MS + item.task_id * 10_000);
  const config = configuration(item.task_id, now.toISOString());
  const spec = releaseFor(item, config, now.toISOString());
  let currentRecord = releaseRecord(spec);
  let currentConfig = config;
  let currentAttestation = attestation(
    item.task_id,
    item.source_feedback_hash,
    now.toISOString()
  );
  const evidence: ExecutionEvidence[] = [];
  let writerCalls = 0;
  const gate = new ReleaseExecutionGate<FrozenAction, LiveState, { completed: true }>(
    {
      async dispatch() {
        writerCalls += 1;
        return { completed: true };
      }
    },
    { append(entry) { evidence.push(entry); } },
    async () => ({ allowed: true, reason: 'policy_passed', matchedRuleIds: ['frozen_policy'] }),
    actionHash,
    async () => currentRecord,
    async () => currentConfig,
    async () => currentAttestation,
    () => monotonic
  );
  const evaluated = await gate.evaluate(requestFor(
    item,
    spec,
    config,
    currentAttestation,
    now
  ));
  if (evaluated.status !== 'allowed') {
    return {
      case_id: item.case_id,
      scenario: item.scenario,
      decision: 'DENY',
      reason: evaluated.reason,
      precondition_writer_calls: 0,
      writer_calls: writerCalls,
      receipt_count: evidence.length
    };
  }

  let preconditionWriterCalls = 0;
  let executeRequest = evaluated.authorizedRequest;
  if (item.scenario === 'action_replacement') {
    executeRequest = {
      ...executeRequest,
      action: {
        bytesHex: item.submitted_action_hex,
        dtype: item.dtype,
        shape: item.shape
      }
    };
  } else if (item.scenario === 'expired_permit') {
    monotonic += 1_001;
  } else if (item.scenario === 'feedback_changed') {
    currentAttestation = attestation(
      item.task_id,
      `sha256:${H('9')}`,
      now.toISOString()
    );
  } else if (item.scenario === 'context_changed') {
    currentConfig = executionConfigurationV1Schema.parse({
      ...config,
      controller: {
        ...config.controller,
        followJointTrajectoryAction: '/changed_controller/follow_joint_trajectory'
      }
    });
  } else if (item.scenario === 'revoked_generation') {
    currentRecord = { ...currentRecord, state: 'revoked' };
  }

  let reason = 'dispatched';
  let decision: 'ALLOW' | 'DENY' = 'ALLOW';
  try {
    await gate.execute(executeRequest);
    if (item.scenario === 'lease_replay') {
      preconditionWriterCalls = writerCalls;
      await gate.execute(executeRequest);
    }
  } catch (error) {
    decision = 'DENY';
    reason = error instanceof Error ? error.message : String(error);
  }
  return {
    case_id: item.case_id,
    scenario: item.scenario,
    decision,
    reason,
    precondition_writer_calls: preconditionWriterCalls,
    writer_calls: writerCalls - preconditionWriterCalls,
    receipt_count: evidence.length
  };
}

async function entryBarrierProbes(sample: FrozenCase): Promise<Record<string, unknown>[]> {
  const results: Record<string, unknown>[] = [];
  const now = new Date(BASE_MS + 200_000);

  // Probe 1: caller-owned command storage changes while Evidence preflight is
  // awaited. ReleaseExecutionGate captures a canonical private action before
  // that await, so the adapter receives the authorized bytes.
  {
    let monotonic = 20_000;
    const config = configuration(sample.task_id, now.toISOString());
    const spec = releaseFor(sample, config, now.toISOString());
    const runtime = attestation(sample.task_id, sample.source_feedback_hash, now.toISOString());
    const request = requestFor(sample, spec, config, runtime, now);
    let callerAction = request.action;
    let writtenHex = '';
    let releasePreflight: (() => void) | undefined;
    const preflight = new Promise<void>((resolve) => { releasePreflight = resolve; });
    let signalEntered: (() => void) | undefined;
    const entered = new Promise<void>((resolve) => { signalEntered = resolve; });
    const sink: EvidenceSink = {
      append() {},
      async assertWritableBeforeDispatch() {
        signalEntered?.();
        await preflight;
      }
    };
    const gate = new ReleaseExecutionGate<FrozenAction, LiveState, { completed: true }>(
      { async dispatch(action) { writtenHex = action.bytesHex; return { completed: true }; } },
      sink,
      async () => ({ allowed: true, reason: 'policy_passed', matchedRuleIds: ['frozen_policy'] }),
      actionHash,
      async () => releaseRecord(spec),
      async () => config,
      async () => runtime,
      () => monotonic
    );
    const decision = await gate.evaluate(request);
    if (decision.status !== 'allowed') throw new Error(decision.reason);
    callerAction = decision.authorizedRequest.action;
    const execution = gate.execute(decision.authorizedRequest);
    await entered;
    callerAction.bytesHex = `00${callerAction.bytesHex.slice(2)}`;
    releasePreflight?.();
    await execution;
    results.push({
      probe: 'caller_command_mutated_during_preflight',
      expected_hex: sample.action_hex,
      caller_hex_after_mutation: callerAction.bytesHex,
      writer_hex: writtenHex,
      protected: writtenHex === sample.action_hex,
      monotonic
    });
  }

  // Probe 2: the official refresh hooks snapshot config/attestation. If a live
  // store changes after its corresponding refresh has returned, but before the
  // dispatcher entry, there is no final adapter-side read barrier in this API.
  {
    let monotonic = 30_000;
    const config = configuration(sample.task_id, now.toISOString());
    const spec = releaseFor(sample, config, now.toISOString());
    const runtime = attestation(sample.task_id, sample.source_feedback_hash, now.toISOString());
    const request = requestFor(sample, spec, config, runtime, now);
    let liveConfig = config;
    let liveFeedback = sample.source_feedback_hash;
    let writerCalls = 0;
    let writerObservedConfig = '';
    let writerObservedFeedback = '';
    const gate = new ReleaseExecutionGate<FrozenAction, LiveState, { completed: true }>(
      {
        async dispatch() {
          writerCalls += 1;
          writerObservedConfig = configurationDigest(liveConfig);
          writerObservedFeedback = liveFeedback;
          return { completed: true };
        }
      },
      { append() {} },
      async () => ({ allowed: true, reason: 'policy_passed', matchedRuleIds: ['frozen_policy'] }),
      actionHash,
      async () => releaseRecord(spec),
      async () => structuredClone(liveConfig),
      async () => {
        const snapshot = structuredClone(runtime);
        liveConfig = executionConfigurationV1Schema.parse({
          ...config,
          controller: {
            ...config.controller,
            followJointTrajectoryAction: '/entry_race/follow_joint_trajectory'
          }
        });
        liveFeedback = `sha256:${H('8')}`;
        return snapshot;
      },
      () => monotonic
    );
    const decision = await gate.evaluate(request);
    if (decision.status !== 'allowed') throw new Error(decision.reason);
    await gate.execute(decision.authorizedRequest);
    results.push({
      probe: 'live_state_and_config_change_after_refresh_before_writer',
      decision: writerCalls === 1 ? 'ALLOW' : 'DENY',
      writer_calls: writerCalls,
      approved_config_digest: configurationDigest(config),
      writer_observed_config_digest: writerObservedConfig,
      approved_feedback_hash: sample.source_feedback_hash,
      writer_observed_feedback_hash: writerObservedFeedback,
      protected: writerCalls === 0
    });
  }
  return results;
}

async function main(): Promise<void> {
  const payloadPath = process.env.FROZEN_PAYLOADS;
  const resultPath = process.env.RLSOK_RESULTS;
  const probesPath = process.env.RLSOK_PROBES;
  if (!payloadPath || !resultPath || !probesPath) {
    throw new Error('FROZEN_PAYLOADS, RLSOK_RESULTS, and RLSOK_PROBES are required');
  }
  const cases = readFileSync(payloadPath, 'utf8')
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line) as FrozenCase);
  const results: CaseResult[] = [];
  for (const item of cases) results.push(await runCase(item));
  writeFileSync(resultPath, `${results.map((item) => JSON.stringify(item)).join('\n')}\n`);
  writeFileSync(probesPath, `${JSON.stringify(await entryBarrierProbes(cases[0]), null, 2)}\n`);
  process.stdout.write(JSON.stringify({ cases: results.length, results: resultPath, probes: probesPath }));
}

void main();
