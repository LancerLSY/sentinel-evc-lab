# A1: explicit native controller input bounding

Declared on 2026-10-04 at 04:53 UTC after the initial eighteen-episode grid,
before examining any A1 outcomes. The initial run and its exact source remain
in the evidence package; they are not overwritten or pooled with A1.

The initial strict native profile rejected model gripper values such as
`-1.00350606441`, outside `[-1,1]`. The installed robosuite 1.4.0
[OSC input scaling](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/base_controller.py)
clips the six Cartesian inputs to its declared input bounds. Its
[Panda gripper](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py)
uses the sign of the gripper input, then clips its internal actuator target.
Moving that request bounding explicitly before certification preserves those
controller semantics and makes the bytes admitted by the strict profile explicit.

A1 enables `native_clip_to_profile=true`: after official LeRobot postprocessing,
both raw-parent and aggregated-final requests are clipped to the unchanged native
profile bounds. Forecasts, geometry certificates, permits and the actual writer
all receive those bounded bytes. The preclip requests, final requests, changed
components and bounds are saved. This is not a widening of the native envelope.

The policy, tasks, initial states, seeds, three branches, predict-50/execute-10
rule, geometry/contact policy, 280-step cap and deadline remain fixed. All six
matched roots are rerun, including roots that failed in the initial grid. A1
includes a fresh excluded one-step engineering preflight before formal episodes.

Results from the initial run and A1 must be labeled separately. This protocol
amendment addresses an observed integration failure; it does not establish
collision benefit, task-success superiority or an independent false-stop rate.
