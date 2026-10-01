"""Optional MuJoCo three-axis tray benchmark, with actual contact feedback.

This is an experimental geometry-only profile. Numeric 1D calibration is never
used to authorize this plant, and evaluator rollouts are never online inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .contracts import ActionDescriptor, Certificate, DEFAULT_DESCRIPTOR, Plan, Scene, sha256_hex

PHYSICS_DESCRIPTOR = ActionDescriptor(descriptor_id="mujoco-tray-xyz-position-v1", gripper="absent")
PHYSICS_SCOPE = "mujoco-tray-geometry-v1"


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.002
    friction: float = 0.35
    payload_mass: float = 0.08
    seed: int = 7

    def __post_init__(self):
        if self.timestep not in (0.000125, 0.00025, 0.0005, 0.001, 0.002):
            raise ValueError("unsupported physics refinement timestep")
        if isinstance(self.friction, bool) or not math.isfinite(self.friction) or not 0.01 <= self.friction <= 1:
            raise ValueError("friction must be finite within 0.01..1")
        if isinstance(self.payload_mass, bool) or not math.isfinite(self.payload_mass) or not 0.02 <= self.payload_mass <= 0.2:
            raise ValueError("payload mass must be within 0.02..0.2 kg")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 1_000_000:
            raise ValueError("seed must be a bounded integer")


def model_xml(config: PhysicsConfig) -> str:
    # Fully owned procedural assets: no external robot meshes/checkpoints.
    return f'''<mujoco model="sentinel-three-axis-tray-v1">
      <option timestep="{config.timestep}" integrator="implicitfast" gravity="0 0 -9.81"
              solver="Newton" iterations="100" tolerance="1e-10"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <default><geom friction="{config.friction} 0.005 0.0001" condim="4"
        solref="0.01 1" solimp="0.95 0.99 0.001"/></default>
      <worldbody>
        <light pos="0 -1 3" dir="0 0 -1"/>
        <camera name="overview" pos="1.2 -1.5 1.5" xyaxes="0.78 0.62 0 -0.34 0.43 0.84"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.83 0.87 0.84 1"/>
        <body name="tray" pos="0 0 0.45" gravcomp="1">
          <joint name="slide_x" type="slide" axis="1 0 0" range="-0.6 0.6"/>
          <joint name="slide_y" type="slide" axis="0 1 0" range="-0.6 0.6"/>
          <joint name="slide_z" type="slide" axis="0 0 1" range="-0.1 0.3"/>
          <geom name="tray_surface" type="box" size="0.12 0.10 0.008" mass="1" rgba="0.12 0.40 0.31 1"/>
        </body>
        <body name="payload" pos="0 0 0.482">
          <freejoint name="payload_free"/>
          <geom name="payload_geom" type="box" size="0.02 0.018 0.022" mass="{config.payload_mass}" rgba="0.84 0.48 0.21 1"/>
        </body>
      </worldbody>
      <actuator>
        <position name="x_servo" joint="slide_x" kp="200000" kv="450" ctrlrange="-0.6 0.6" forcerange="-1500 1500"/>
        <position name="y_servo" joint="slide_y" kp="200000" kv="450" ctrlrange="-0.6 0.6" forcerange="-1500 1500"/>
        <position name="z_servo" joint="slide_z" kp="200000" kv="450" ctrlrange="-0.1 0.3" forcerange="-1500 1500"/>
      </actuator>
    </mujoco>'''


def physics_plan(duration: float, start=(0.0, 0.0, 0.45)) -> Plan:
    if duration not in (0.6, 0.9, 1.2, 1.6):
        raise ValueError("unsupported candidate duration")
    # Movements on all three axes; the remaining horizon explicitly holds.
    knots = []
    for step in range(41):
        u = min(step * 0.05 / duration, 1.0)
        s = u ** 3 * (10 - 15 * u + 6 * u * u)
        knots.append(tuple(a + b * s for a, b in zip(start, (0.35, 0.10, 0.08))))
    return Plan(f"physics-duration-{duration}", tuple(knots), 0.05, descriptor=PHYSICS_DESCRIPTOR)


def physics_scene() -> Scene:
    # Ball encloses the tray and a payload only while inside the monitored
    # carried-object envelope. It is NOT a certificate for articulated links.
    return Scene("mujoco-tray-scene-v1", (), (-0.6, -0.6, 0.0), (1.0, 0.7, 1.0), 0.17, 0.005)


def verify_physics_plan(plan, controller, certificate_id):
    """Trusted local verifier for this explicit translational profile only."""
    from .geometry import full_check
    if plan.descriptor != PHYSICS_DESCRIPTOR or plan.gripper_events or plan.dt != 0.05:
        raise ValueError("unsupported physics action profile")
    if any(not (-0.6 <= p[0] <= 0.6 and -0.6 <= p[1] <= 0.6 and 0.35 <= p[2] <= 0.75) for p in plan.knots):
        raise ValueError("target exceeds this fixture's joint ranges")
    # Explicit projection into the analytic translational-ball verifier.
    # Its result is re-bound to the physics Plan and frozen model digest.
    projection = Plan("analytic-tray-projection", plan.knots, plan.dt, descriptor=DEFAULT_DESCRIPTOR)
    okay, margins, _ = full_check(projection, physics_scene())
    if not okay:
        raise ValueError("physics tray geometry violation")
    return Certificate(certificate_id, plan.hash, controller.scene_digest(), plan.dt,
                       plan.horizon, margins, "FULL", proof_scope=PHYSICS_SCOPE)


class MujocoController:
    """One accepted command slot; physics advances even after cancel/drain.

    Observed cursor denotes the completed 50ms command interval, not exact
    target attainment. Feedback is copied from MjData and retains tracking error.
    """
    poll_dt = 0.002  # Fixed protocol cadence, independent of solver refinement.
    def __init__(self, config: PhysicsConfig):
        try:
            import mujoco
        except ImportError as exc:
            raise ValueError("MuJoCo is optional: install sentinel-evc-lab[physics]") from exc
        import random
        self.engine = mujoco
        self.config = config
        self.xml = model_xml(config)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        randomizer = random.Random(config.seed)
        adr = self.model.joint("payload_free").qposadr[0]
        self.data.qpos[adr:adr + 2] = [randomizer.uniform(-0.003, 0.003), randomizer.uniform(-0.003, 0.003)]
        mujoco.mj_forward(self.model, self.data)
        self.submitted, self.accepted, self.observed = [], [], []
        self._command = None
        self._cancel_age = None
        self.cancel_acked = None
        self._target = (0.0, 0.0, 0.45)
        self.samples = []
        self._ticks = 0
        self._tray_id = self.model.body("tray").id
        self._payload_id = self.model.body("payload").id
        self._payload_geom = self.model.geom("payload_geom").id
        self._tray_geom = self.model.geom("tray_surface").id
        self._floor_geom = self.model.geom("floor").id

    @property
    def now_ns(self):
        return 1_000_000_000 + round(self._ticks * self.config.timestep * 1e9)

    def scene_digest(self):
        # Serialize the ENTIRE current compiled model, including all options,
        # joints, collision filters/pairs and actuator metadata. A partial array
        # allowlist would let an omitted physical parameter evade invalidation.
        import hashlib
        import numpy as np
        compiled = np.empty(self.engine.mj_sizeModel(self.model), dtype=np.uint8)
        self.engine.mj_saveModel(self.model, None, compiled)
        return sha256_hex({"xml": self.xml, "compiled": hashlib.sha256(compiled.tobytes()).hexdigest(),
                           "workspace": physics_scene().summary(), "scope": PHYSICS_SCOPE})

    def capture_root(self):
        import numpy as np
        spec = self.engine.mjtState.mjSTATE_INTEGRATION
        state = np.zeros(self.engine.mj_stateSize(self.model, spec))
        self.engine.mj_getState(self.model, self.data, state, spec)
        return {"state_spec": int(spec), "state": state.tolist(), "ticks": self._ticks,
                "target": self._target, "scene_digest": self.scene_digest()}

    def restore_root(self, root):
        if self.submitted or self._command is not None:
            raise ValueError("root restore is allowed only before command execution")
        if root["scene_digest"] != self.scene_digest():
            raise ValueError("root belongs to a different physics model")
        import numpy as np
        spec = self.engine.mjtState.mjSTATE_INTEGRATION
        state = np.asarray(root["state"], dtype=float)
        if root["state_spec"] != int(spec) or len(state) != self.engine.mj_stateSize(self.model, spec) or not np.isfinite(state).all():
            raise ValueError("invalid complete MuJoCo integration state")
        self.engine.mj_setState(self.model, self.data, state, spec)
        self.engine.mj_forward(self.model, self.data)
        self._ticks = root["ticks"]
        self._target = tuple(root["target"])
        self.samples.clear()

    def free_slots(self):
        return int(self._command is None and self._cancel_age is None)

    @property
    def is_drained(self):
        return self._command is None

    def submit(self, action, generation, gripper_event=None):
        if gripper_event is not None:
            raise ValueError("this tray profile has no gripper")
        if len(action) != 3 or any(not math.isfinite(v) for v in action):
            raise ValueError("invalid physics target")
        if not self.free_slots():
            return False
        item = {"action": tuple(action), "gen": generation}
        self.submitted.append(item.copy())
        self._command = {**item, "age": 0, "start": self._target, "accepted": False}
        return True

    def cancel(self):
        self._cancel_age = 0
        self.cancel_acked = None

    def cursors(self):
        return {"submitted": len(self.submitted), "accepted": len(self.accepted), "observed": len(self.observed)}

    def tick(self):
        # Refining physics must not refine the gateway's decision/command
        # schedule as well; otherwise a "convergence" test changes the policy.
        for _ in range(round(self.poll_dt / self.config.timestep)):
            self._substep()

    def _substep(self):
        command = self._command
        if self._cancel_age is not None:
            self._cancel_age += 1
            if self._cancel_age * self.config.timestep >= 0.004 - 1e-12:
                if command is not None and not command["accepted"]:
                    self._command = command = None
                self._cancel_age = None
                self.cancel_acked = True
        if command is not None:
            if not command["accepted"]:
                command["accepted"] = True
                self.accepted.append({"action": command["action"], "gen": command["gen"]})
            command["age"] += 1
            fraction = min(command["age"] * self.config.timestep / 0.05, 1.0)
            self._target = tuple(a + (b - a) * fraction for a, b in zip(command["start"], command["action"]))
        self.data.ctrl[:] = (self._target[0], self._target[1], self._target[2] - 0.45)
        self.engine.mj_step(self.model, self.data)
        # Refresh derived kinematics/contact data at the integrated state.
        self.engine.mj_forward(self.model, self.data)
        self._ticks += 1
        if command is not None and command["age"] * self.config.timestep >= 0.05 - 1e-12:
            self.observed.append({"action": command["action"], "gen": command["gen"], "position": self._position()})
            self._command = None
        self.samples.append(self.read_feedback(self.now_ns))

    def _position(self):
        return tuple(float(x) for x in self.data.xpos[self._tray_id])

    def read_feedback(self, now_ns):
        if now_ns != self.now_ns:
            raise ValueError("feedback must use this simulator's local clock")
        tray = self._position()
        payload = tuple(float(x) for x in self.data.xpos[self._payload_id])
        supported, floor_contact, normal_force = False, False, 0.0
        import numpy as np
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom[0]), int(contact.geom[1])}
            if pair == {self._tray_geom, self._payload_geom}:
                force = np.zeros(6)
                self.engine.mj_contactForce(self.model, self.data, contact_index, force)
                normal_force += abs(float(force[0]))
                supported = supported or force[0] > 0.0
            floor_contact = floor_contact or pair == {self._floor_geom, self._payload_geom}
        return {"capture_mono_ns": now_ns, "physics_time": float(self.data.time), "position": tray,
                "payload_position": payload, "relative_payload": tuple(b - a for a, b in zip(tray, payload)),
                "tray_velocity": tuple(float(x) for x in self.data.qvel[:3]),
                "payload_velocity": tuple(float(x) for x in self.data.qvel[3:6]),
                "supported": bool(supported), "floor_contact": bool(floor_contact),
                "contact_normal_force": normal_force, "contact_count": int(self.data.ncon),
                "tracking_error": math.dist(tray, self._target), "cursors": self.cursors(), "gripper": "absent",
                "qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(),
                "solver_iterations": self.data.solver_niter.tolist(),
                "warnings": [int(w.number) for w in self.data.warning]}
