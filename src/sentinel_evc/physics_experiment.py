"""Paired MuJoCo experiments; evaluator futures never authorize actions."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import platform
from pathlib import Path
import time

from .authority import Authority
from .contracts import Context, Plan, Rejection, Snapshot, canonical_json, sha256_hex
from .delta_cert import CertificateStore
from .events import EventLog
from .evidence import build_bundle, verify_bundle
from .executor import Executor, ExecutorState
from .physics import MujocoController, PHYSICS_DESCRIPTOR, PHYSICS_SCOPE, PhysicsConfig, physics_plan, verify_physics_plan

DURATIONS = (0.6, 0.9, 1.2, 1.6)
SLIP_LIMIT = 0.06  # Predeclared fixture metric; not a robot safety limit.


def _snapshot(controller, count):
    feedback = controller.read_feedback(controller.now_ns)
    valid = all(math.isfinite(v) for v in feedback["qpos"] + feedback["qvel"]) and not any(feedback["warnings"])
    return Snapshot(f"physics-sample-{count}", feedback["position"], feedback["capture_mono_ns"], valid, feedback["supported"])


def _context(controller, executor, plan=None):
    return Context(robot="mujoco-actuated-tray", boot=1, epoch=executor.generation,
                   scene_id=controller.scene_digest(), scene_hash=controller.scene_digest(),
                   controller="mujoco-tray-servo-50ms-v1",
                   task_phase="physics_geometry_only", queue_rev=controller.cursors()["submitted"],
                   committed_prefix_hash=sha256_hex(controller.submitted))


def _new_root(config):
    controller = MujocoController(config)
    for _ in range(round(0.3 / controller.poll_dt)):
        controller.tick()
    return controller, controller.capture_root()


def _summarize(controller, protocol_error, outcome_samples):
    finite = all(math.isfinite(v) for row in outcome_samples for v in row["qpos"] + row["qvel"])
    slip = max((math.hypot(*row["relative_payload"][:2]) for row in outcome_samples), default=0)
    floor = any(row["floor_contact"] for row in outcome_samples)
    escaped = any(abs(row["relative_payload"][0]) > 0.14 or abs(row["relative_payload"][1]) > 0.12 for row in outcome_samples)
    outcome = "nonfinite" if not finite else "drop" if floor or escaped else "slip" if slip > SLIP_LIMIT else "stable"
    return {"outcome": outcome, "maximum_payload_offset_m": slip, "physics_warnings": any(any(row['warnings']) for row in outcome_samples),
            "maximum_tracking_error_m": max((r["tracking_error"] for r in outcome_samples), default=0),
            "floor_contact": floor, "finite": finite, "protocol_error": protocol_error,
            "cursors": controller.cursors(), "final": outcome_samples[-1] if outcome_samples else None}


def run_branch(config, root, plan, run_id, *, revoke_at=None, mutate=None):
    controller = MujocoController(config)
    controller.restore_root(root)
    events = EventLog(run_id, maxlen=20000, schema_version="product-v1")
    events.append("PROPOSAL", plan_hash=plan.hash, scope=PHYSICS_SCOPE, action_source="scripted_fixture", prediction_required=False)
    store = CertificateStore()
    authority = Authority(store, events=events)
    executor = Executor(authority, controller, events)
    error, offset, counter, revoked, after_revoke = None, 0, 0, False, None
    runtime_context = None
    initial_physics_time = controller.data.time
    try:
        while offset < plan.horizon and controller.data.time - initial_physics_time < 3.5:
            counter += 1
            if executor.state == ExecutorState.IDLE:
                suffix = Plan(f"suffix-{offset}", plan.knots[offset:], plan.dt, descriptor=PHYSICS_DESCRIPTOR)
                cert = verify_physics_plan(suffix, controller, f"physics-cert-{offset}")
                store.register(cert)
                events.append("CERTIFICATE", **cert.summary())
                runtime_context = _context(controller, executor)
                snapshot = _snapshot(controller, counter)
                if cert.scene_hash != controller.scene_digest() or cert.proof_scope != PHYSICS_SCOPE:
                    raise ValueError("physics verifier/profile binding changed")
                lease = authority.prepare(suffix, cert, runtime_context, snapshot, controller.now_ns,
                                          prefix_len=min(4, suffix.horizon), ttl_ns=500_000_000)
                if mutate:
                    if mutate == "friction": controller.model.geom_friction[:] *= 0.5
                    elif mutate == "geometry": controller.model.geom_size[controller.model.geom("tray_surface").id, 0] *= 0.5
                    else: raise ValueError("unknown mutation")
                    live = _context(controller, executor)
                else:
                    live = runtime_context
                executor.commit(lease, suffix, snapshot, live, controller.now_ns)
                # A real new physics sample is required after commit. Advancing
                # one substep cannot dispatch any new command by itself.
                controller.tick()
                continue
            if revoke_at is not None and not revoked and controller.cursors()["submitted"] >= revoke_at:
                executor.revoke("physics revoke injection", now_ns=controller.now_ns)
                revoked = True
                after_revoke = {"submitted": controller.cursors()["submitted"], "observed": controller.cursors()["observed"], "position": controller.read_feedback(controller.now_ns)["payload_position"], "physics_time": controller.data.time}
            snapshot = _snapshot(controller, counter)
            # Scene content is checked live, whereas queue revision remains the
            # external-plan revision captured at commit (own dispatch isn't a revision).
            live = Context(**{**runtime_context.summary(), "epoch": executor.generation, "scene_id": controller.scene_digest()})
            executor.tick(controller.now_ns, snapshot, live)
            offset = controller.cursors()["observed"]
            if revoked and controller.cancel_acked and controller.is_drained:
                break
    except Rejection as exc:
        error = exc.code
    except ValueError as exc:
        error = "PHYSICS_PROFILE: " + str(exc)
    finally:
        if executor.state != ExecutorState.FAULT:
            executor.revoke("physics trial ended", now_ns=controller.now_ns)
        # Physics always continues during cancel, drain, and this 0.5s hold.
        for _ in range(round(0.5 / controller.poll_dt)):
            executor.tick(controller.now_ns)
        executor.poll_cancel()
    result = _summarize(controller, error, controller.samples)
    result.update({"run_id": run_id, "plan_hash": plan.hash, "duration": float(plan.plan_id.split('-')[-1]) if plan.plan_id.startswith('physics-duration-') else None,
                   "scope": "mujoco-actuated-tray-geometry-only-v1", "authority_generation": executor.generation,
                   "command_execution_completed": offset == plan.horizon, "physics_time_s": controller.data.time - initial_physics_time,
                   "consequence_prediction": "MODEL_UNKNOWN; numeric 1D calibration not applicable"})
    if after_revoke:
        result["revoke"] = {"new_old_generation_submissions": sum(it["gen"] == 0 for it in controller.submitted[after_revoke["submitted"]:]),
                            "accepted_tail_observed": controller.cursors()["observed"] - after_revoke["observed"],
                            "physics_elapsed_after_request_s": controller.data.time - after_revoke["physics_time"],
                            "payload_motion_after_request_m": math.dist(after_revoke["position"], result["final"]["payload_position"]),
                            "cancel_confirmed": controller.cancel_acked is True, "drained": controller.is_drained}
    events.append("OUTCOME", **{k: v for k, v in result.items() if k != "final"})
    return result, controller, events


def _write_trial(directory, result, controller, events, root, plan, render=False):
    artifacts = {"result.json": canonical_json(result), "root.json": canonical_json(root),
                 "plan.json": canonical_json(plan.summary()), "physics.xml": controller.xml.encode(),
                 "trace.json": canonical_json(controller.samples), "config.json": canonical_json(asdict(controller.config))}
    if render:
        from PIL import Image
        renderer = controller.engine.Renderer(controller.model, height=480, width=640)
        try:
            renderer.update_scene(controller.data, camera="overview")
            import io
            data = io.BytesIO()
            Image.fromarray(renderer.render()).save(data, format="PNG")
            artifacts["final.png"] = data.getvalue()
            frames = []
            # Visualization replays recorded qpos/qvel, never advances physics or
            # changes any decision. State copies are restored after rendering.
            saved_qpos, saved_qvel = controller.data.qpos.copy(), controller.data.qvel.copy()
            try:
                stride = max(1, round(0.04 / controller.config.timestep))
                for row in controller.samples[::stride]:
                    controller.data.qpos[:] = row["qpos"]
                    controller.data.qvel[:] = row["qvel"]
                    controller.engine.mj_forward(controller.model, controller.data)
                    renderer.update_scene(controller.data, camera="overview")
                    frames.append(Image.fromarray(renderer.render()).copy())
                animation = io.BytesIO()
                frames[0].save(animation, format="GIF", save_all=True, append_images=frames[1:], duration=40, loop=0)
                artifacts["replay.gif"] = animation.getvalue()
            finally:
                controller.data.qpos[:] = saved_qpos
                controller.data.qvel[:] = saved_qvel
                controller.engine.mj_forward(controller.model, controller.data)
        finally:
            renderer.close()
    info = build_bundle(events, str(directory), artifacts=artifacts)
    okay, message = verify_bundle(info["bundle_dir"], info["public_key"], events.run_id, expected_tip=info["tip_hash"])
    if not okay:
        raise ValueError(message)
    return {"run_id": events.run_id, "tip_hash": info["tip_hash"], "verification": message,
            "public_key_sha256": hashlib.sha256(Path(info["public_key"]).read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256((Path(info["bundle_dir"]) / "manifest.json").read_bytes()).hexdigest()}


def run_physics_experiment(out, *, seed=7, friction=0.35, render=False):
    directory = Path(out)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("physics output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results, anchors, roots, static_checks = [], {}, {}, []
    for step in (0.002, 0.001, 0.0005, 0.00025, 0.000125):
        config = PhysicsConfig(step, friction, seed=seed)
        settled, root = _new_root(config)
        roots[step] = root
        start = settled.read_feedback(settled.now_ns)["position"]
        feedback = settled.read_feedback(settled.now_ns)
        tail = settled.samples[-max(1, round(.1/step)):]
        static_checks.append({"timestep": step, "supported": feedback["supported"],
                              "drift_m": math.dist(tail[0]["payload_position"], tail[-1]["payload_position"]),
                              "velocity_m_s": math.sqrt(sum(v*v for v in feedback["payload_velocity"])),
                              "passed": feedback["supported"] and math.dist(tail[0]["payload_position"],tail[-1]["payload_position"]) < .001
                              and math.sqrt(sum(v*v for v in feedback["payload_velocity"])) < .001 and not any(feedback["warnings"])})
        plans = [physics_plan(duration, start) for duration in DURATIONS]
        if step == .002:
            # Chosen BEFORE evaluating sibling futures. All four share geometric
            # feasibility; absence of 3D consequence prediction isn't hidden.
            admissible = [p for p in plans if verify_physics_plan(p, settled, "selection")]
            chosen = min(admissible, key=lambda p: DURATIONS[plans.index(p)])
            integrated, selected_controller, selected_events = run_branch(config, root, chosen, f"physics-{seed}-integrated")
            anchors[selected_events.run_id] = _write_trial(directory/selected_events.run_id, integrated, selected_controller, selected_events, root, chosen, render=render)
        # Every sibling uses the same complete integration-state root.
        for duration, plan in zip(DURATIONS, plans):
            run_id = f"physics-{seed}-{int(step*1e6)}us-{int(duration*10)}"
            result, controller, events = run_branch(config, root, plan, run_id)
            anchors[run_id] = _write_trial(directory/run_id, result, controller, events, root, plan, render=render and step == 0.002 and duration == 1.6)
            result["timestep"] = step
            results.append(result)
    config = PhysicsConfig(0.002, friction, seed=seed)
    base, root = _new_root(config)
    plan = physics_plan(1.6, base.read_feedback(base.now_ns)["position"])
    fault_results = {}
    for name, options in (("revoke", {"revoke_at": 6}), ("geometry_mutation", {"mutate": "geometry"}), ("friction_mutation", {"mutate": "friction"}), ("replay", {})):
        result, controller, events = run_branch(config, root, plan, f"physics-{seed}-{name}", **options)
        anchors[events.run_id] = _write_trial(directory/events.run_id, result, controller, events, root, plan)
        fault_results[name] = result
    convergence = []
    for duration in DURATIONS:
        a = next(r for r in results if r["timestep"] == .00025 and r["duration"] == duration)
        b = next(r for r in results if r["timestep"] == .000125 and r["duration"] == duration)
        delta = abs(a["maximum_payload_offset_m"] - b["maximum_payload_offset_m"])
        position_delta = math.dist(a["final"]["payload_position"], b["final"]["payload_position"])
        velocity_delta = math.dist(a["final"]["payload_velocity"], b["final"]["payload_velocity"])
        tracking_delta = abs(a["maximum_tracking_error_m"] - b["maximum_tracking_error_m"])
        qa, qb = a["final"]["qpos"][-4:], b["final"]["qpos"][-4:]
        angle = 2 * math.acos(min(1, abs(sum(x*y for x,y in zip(qa,qb)))))
        same_contact = a["floor_contact"] == b["floor_contact"] and a["final"]["supported"] == b["final"]["supported"]
        convergence.append({"duration": duration, "compared_timesteps": [.00025, .000125], "same_outcome": a["outcome"] == b["outcome"], "offset_difference_m": delta,
                            "final_position_difference_m": position_delta, "final_velocity_difference_m_s": velocity_delta,
                            "maximum_tracking_difference_m": tracking_delta, "final_orientation_difference_rad": angle,
                            "same_contact_outcome": same_contact,
                            "passed": a["outcome"] == b["outcome"] and delta <= .001 and position_delta <= .001
                            and velocity_delta <= .001 and tracking_delta <= .001 and angle <= .01 and same_contact})
    summary = {"schema": "mujoco-experiment-v1", "scope": "three-axis actuated open tray and floor contact benchmark; no physical walls, robot arm or VLA",
               "engine": base.engine.__version__, "python": platform.python_version(), "platform": platform.platform(),
               "config": asdict(config), "clock": "remote-local integer physics substeps; host wall time separately measured",
               "selection_policy": "geometry-only earliest feasible is 0.6s; sibling evaluator labels never authorize",
               "results": results, "integrated": integrated, "static_checks": static_checks,
               "faults": fault_results, "convergence": convergence,
               "all_converged": all(c["passed"] for c in convergence), "anchors": anchors,
               "wall_time_s": time.monotonic() - started,
               "source_digest": "sha256:" + hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()}
    replay = fault_results["replay"]
    baseline = next(r for r in results if r["timestep"] == .002 and r["duration"] == 1.6)
    replay_equal = (directory/replay["run_id"]/"bundle/trace.json").read_bytes() == (directory/baseline["run_id"]/"bundle/trace.json").read_bytes()
    summary["acceptance"] = {"static_stability": all(s["passed"] for s in static_checks),
        "step_convergence": summary["all_converged"], "paired_negative": integrated["outcome"] == "slip" and baseline["outcome"] == "stable",
        "deterministic_root_replay": replay_equal,
        "zero_old_generation_submissions": fault_results["revoke"]["revoke"]["new_old_generation_submissions"] == 0,
        "continued_stop_dynamics": all((fault_results["revoke"]["revoke"]["physics_elapsed_after_request_s"] >= .5,
             fault_results["revoke"]["revoke"]["payload_motion_after_request_m"] > .001,
             fault_results["revoke"]["revoke"]["accepted_tail_observed"] == 1,
             fault_results["revoke"]["revoke"]["cancel_confirmed"], fault_results["revoke"]["revoke"]["drained"])),
        "mutation_refusal": all(fault_results[name]["protocol_error"] == "CONTEXT_CHANGED" and fault_results[name]["cursors"]["submitted"] == 0 for name in ("geometry_mutation","friction_mutation")),
        "no_engine_warnings": not any(r["physics_warnings"] for r in results + list(fault_results.values()) + [integrated]),
        "all_expected_commands": all(r["command_execution_completed"] and r["protocol_error"] is None for r in results + [integrated])}
    summary["infrastructure_gates_pass"] = all(summary["acceptance"].values())
    (directory/"summary.json").write_bytes(canonical_json(summary))
    root_dir = Path(__file__).parents[2]
    source_paths = list(Path(__file__).parent.rglob('*.py'))
    source_paths += [p for name in ('pyproject.toml',) if (p := root_dir/name).is_file()]
    source_paths += list((root_dir/'tests').glob('test_*.py'))
    source_manifest = {str(p.relative_to(root_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
    from importlib.metadata import distributions
    installed = sorted(({"name": d.metadata["Name"], "version": d.version} for d in distributions()), key=lambda d: (d["name"].lower(), d["version"]))
    environment = {"installed_distributions": installed, "python": platform.python_version(), "platform": platform.platform(), "mujoco": base.engine.__version__}
    if render:
        import PIL, os
        environment.update({"pillow": PIL.__version__, "render_backend": os.environ.get('MUJOCO_GL','platform-default')})
    experiment_id = f"physics-experiment-{seed}"
    index_log = EventLog(experiment_id, schema_version="product-v1")
    index_log.append("OUTCOME", acceptance=summary["acceptance"], infrastructure_gates_pass=summary["infrastructure_gates_pass"])
    completion = {"summary_sha256": hashlib.sha256((directory/"summary.json").read_bytes()).hexdigest(), "experiment_id": experiment_id,
                  "source_manifest_sha256": hashlib.sha256(canonical_json(source_manifest)).hexdigest()}
    index = build_bundle(index_log, str(directory/'experiment-index'), artifacts={
        "summary.json": (directory/'summary.json').read_bytes(), "completion.json": canonical_json(completion),
        "source_manifest.json": canonical_json(source_manifest), "environment.json": canonical_json(environment),
        "trial_anchors.json": canonical_json(anchors), "job_config.json": canonical_json({"seed":seed,"friction":friction,"render":render,
        "command":"python -m sentinel_evc physics --out <empty-job-dir>","profile":PHYSICS_SCOPE,
        "timesteps_s":[.002,.001,.0005,.00025,.000125], "durations_s":list(DURATIONS), "gateway_poll_s":base.poll_dt,
        "convergence_limits":{"position_m":.001,"offset_m":.001,"velocity_m_s":.001,"tracking_m":.001,"orientation_rad":.01},
        "slip_limit_m":SLIP_LIMIT})})
    if not verify_bundle(index['bundle_dir'],index['public_key'],experiment_id,expected_tip=index['tip_hash'])[0]:
        raise ValueError('experiment index verification failed')
    # Last-written transfer marker; all claims also live in the signed index.
    completion.update({"index_tip": index['tip_hash'], "public_key_sha256": hashlib.sha256(Path(index['public_key']).read_bytes()).hexdigest()})
    (directory/"COMPLETE.json").write_bytes(canonical_json(completion))
    return summary
