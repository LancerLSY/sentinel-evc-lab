"""Describe observed wiring patterns without changing qualification verdicts."""
from __future__ import annotations

import math
import struct


def _columns(rows):
    return list(zip(*rows)) if rows and all(len(row) == len(rows[0]) for row in rows) else []


def _numeric_mapping(reference, candidate):
    """Match common wiring transforms on every retained, varying probe."""
    left, right = _columns(reference), _columns(candidate)
    if not left or len(reference) < 3:
        return []
    transforms = (("identity", 1.0, 0.0), ("sign", -1.0, 0.0),
                  ("radians_to_degrees", 180 / math.pi, 0.0),
                  ("degrees_to_radians", math.pi / 180, 0.0),
                  ("minus_one_one_to_zero_one", .5, .5),
                  ("zero_one_to_minus_one_one", 2.0, -1.0))
    result=[]
    for destination, actual in enumerate(right):
        # Changes smaller than the matching tolerance remain BLOCK in qualify.
        if destination < len(left) and actual == left[destination]:
            continue
        matches=[]
        for source, expected in enumerate(left):
            if max(expected) - min(expected) <= 1e-8:
                continue
            for kind, scale, offset in transforms:
                predicted=[scale*x+offset for x in expected]
                if not all(math.isfinite(value) for value in predicted):
                    continue
                errors=[abs(y-x) for x,y in zip(predicted,actual)]
                if not all(math.isfinite(value) for value in errors):
                    continue
                tolerances=[2e-6 * max(1.0,abs(y),abs(x)) for x,y in zip(predicted,actual)]
                if all(error <= limit for error,limit in zip(errors,tolerances)):
                    matches.append({"destination_index":destination,"reference_index":source,
                                    "transform":kind,"scale":scale,"offset":offset,
                                    "max_abs_residual":max(errors)})
        if len(matches) == 1:
            match=matches[0]
            if match['reference_index'] != destination or match['transform'] != 'identity':
                result.append(match)
        elif not matches:
            # Per-component normalization can turn a permutation into an
            # affine relation. Retain it only when one varying source matches
            # every probe; multiple possible sources provide no repair hint.
            affine_matches=[]
            for source, expected in enumerate(left):
                lo=min(range(len(expected)),key=expected.__getitem__)
                hi=max(range(len(expected)),key=expected.__getitem__)
                span=expected[hi]-expected[lo]
                if not math.isfinite(span) or span<=1e-8:
                    continue
                scale=(actual[hi]-actual[lo])/span
                offset=actual[lo]-scale*expected[lo]
                predicted=[scale*x+offset for x in expected]
                errors=[abs(y-x) for x,y in zip(predicted,actual)]
                if all(math.isfinite(value) for value in [scale,offset,*predicted,*errors]) and all(
                    error <= 2e-6*max(1.0,abs(y),abs(x)) for error,x,y in zip(errors,predicted,actual)):
                    affine_matches.append({"destination_index":destination,"reference_index":source,
                                           "transform":"scale_offset","scale":scale,"offset":offset,
                                           "max_abs_residual":max(errors)})
            if len(affine_matches)==1:
                result.append(affine_matches[0])
    return result


def diagnose_bindings(reference: dict, candidate: dict) -> dict:
    """Infer only unambiguous patterns observed across all aligned probes.

    This is a repair hint, not a cause verdict or an automatic transformation.
    Strict byte/value comparison remains the authority for PASS/BLOCK.
    """
    left, right = reference['probes'], candidate['probes']
    patterns=[]
    camera_maps=[]
    for destination in sorted(right[0]['consumed']['cameras']):
        matches=[source for source in sorted(left[0]['consumed']['cameras'])
                 if all(source in l['consumed']['cameras'] and destination in r['consumed']['cameras']
                        and l['consumed']['cameras'][source] == r['consumed']['cameras'][destination]
                        for l,r in zip(left,right))]
        if len(matches)==1 and matches[0]!=destination:
            camera_maps.append({"candidate_key":destination,"reference_key":matches[0]})
    if camera_maps:
        patterns.append({"kind":"camera_route","mapping":camera_maps,
                         "check":"restore the reference input routing, then recapture the same probes"})
    state_map=_numeric_mapping([p['consumed']['state']['values'] for p in left],
                               [p['consumed']['state']['values'] for p in right])
    if state_map:
        patterns.append({"kind":"state_transform","mapping":state_map,
                         "check":"inspect state packing, units and coordinate transforms, then recapture"})
    def actions(rows):
        return [list(struct.unpack('<'+('f' if row['action']['dtype']=='float32' else 'd')*row['action']['shape'][-1],
                                   bytes.fromhex(row['action']['bytes_hex']))) for row in rows]
    action_map=_numeric_mapping(actions(left),actions(right))
    if action_map:
        patterns.append({"kind":"action_transform","mapping":action_map,
                         "check":"inspect output packing, axis signs and gripper normalization, then recapture"})
    cursors={key:[r['cursor'][key]-l['cursor'][key] for l,r in zip(left,right)]
             for key in ('chunk_id','action_index')}
    offsets={key:values[0] for key,values in cursors.items() if len(set(values))==1 and values[0]!=0}
    if offsets:
        patterns.append({"kind":"cursor_offset","offsets":offsets,
                         "check":"inspect chunk reset and action selection indices, then recapture"})
    return {"scope":"observed patterns across every aligned probe; repair hints only",
            "matched_probes":len(left),"patterns":patterns,
            "numeric_matching_tolerance":"2e-6 relative to max(1, observed, expected); qualification remains exact"}
