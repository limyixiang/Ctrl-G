# ALFWorld Embodied Agent Skill

## Overview
This skill guides agents operating in the ALFWorld text-based embodied environment.
The agent must complete household tasks by navigating rooms, interacting with objects,
and using appliances.

**Output format**: Use the model's native thinking mode, then `<decision>...</decision>` for one short summarized decision and `<action>...</action>` for the chosen action.

---

## Task Types

| Type | Goal | Key Steps |
|------|------|-----------|
| Pick & Place | Put object X in/on receptacle Y | Find X -> take X -> go to Y -> put X in/on Y |
| Pick Two & Place | Put two instances of X in/on Y | Find X1 -> take -> place -> find X2 -> take -> place |
| Examine in Light | Examine object X under desklamp | Find X -> take X -> find desklamp -> use desklamp |
| Clean & Place | Clean object X and put in/on Y | Find X -> take X -> go to sink -> clean X -> go to Y -> put X |
| Heat & Place | Heat object X and put in/on Y | Find X -> take X -> go to microwave -> heat X -> go to Y -> put X |
| Cool & Place | Cool object X and put in/on Y | Find X -> take X -> go to fridge -> cool X -> go to Y -> put X |

---

## General Principles

1. **Decompose the task**: Parse the goal into ordered sub-goals (locate, acquire, transform, deliver). Complete each before moving to the next.
2. **Systematic exploration**: Search each surface and container exactly once before revisiting. Open closed containers (drawers, cabinets, fridge) before judging them empty.
3. **Grab immediately**: When a required object is visible and reachable, take it right away before moving elsewhere.
4. **Transform before placing**: If the task requires cleaning, heating, or cooling, perform the state change at the appropriate appliance before heading to the final destination.
5. **Direct delivery**: Once holding the transformed (or untransformed) goal object, navigate straight to the target receptacle and place it.
6. **Track progress**: Maintain an internal count of how many objects still need to be found and placed. Only stop searching when the count reaches zero.
7. **Avoid loops**: Never repeat the same action more than twice in a row. If stuck, move to a different unexplored location.
8. **Only choose admissible actions**: Always pick an action from the actions templates. Do not invent actions.

---

## Common Mistakes to Avoid

- **Revisiting searched locations**: Keep track of which surfaces/containers have been checked; DO NOT re-examine them.
- **Ignoring visible objects**: If the target object appears in the observation, pick it up immediately.
- **Skipping state changes**: DO NOT place an object at the destination without first cleaning/heating/cooling it when required.
- **Premature termination**: DO NOT stop the episode until all goal conditions are verified as met.
- **Action loops**: DO NOT repeat the same actions. Move on to new locations instead.
- The `look` action does not bring you to a new location. Repeating the `look` action DOES NOT bring new observations. Use the `go to` action to visit unexplored locations to look for your object.

---

## Actions Templates and Preconditions

go to

Move to a receptacle you have seen in the room. No effect if you are already there.

```action
name: goto
template: "go to {recep}"
preconditions:
  - seen({recep})
  - not_at({recep})
```

take

Pick up an object from the receptacle you are currently at. Your hand must be empty.

```action
name: take
template: "take {obj} from {recep}"
preconditions:
  - at({recep})
  - obj_at_current({obj})
  - hand_empty
```

move

Place the object you are holding in/on the receptacle you are currently at.

```action
name: move
template: "move {obj} to {recep}"
preconditions:
  - at({recep})
  - holding({obj})
```

open

Open a closed receptacle (cabinet, drawer, fridge, microwave, safe, ...) you are at.

```action
name: open
template: "open {recep}"
preconditions:
  - at({recep})
  - openable({recep})
  - closed({recep})
```

close

Close an open receptacle you are at.

```action
name: close
template: "close {recep}"
preconditions:
  - at({recep})
  - openable({recep})
  - open({recep})
```

clean

Clean the object you are holding, using a sinkbasin you are at.

```action
name: clean
template: "clean {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, sinkbasin)
```

heat

Heat the object you are holding, using a microwave you are at.

```action
name: heat
template: "heat {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, microwave)
```

cool

Cool the object you are holding, using a fridge you are at.

```action
name: cool
template: "cool {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, fridge)
```

use

Toggle/use an object at your current location (e.g. turn on a desklamp).

```action
name: use
template: "use {obj}"
preconditions:
  - obj_at_current({obj})
```

examine

Look closely at an object you hold or see, or the receptacle you are at.

```action
name: examine
template: "examine {thing}"
preconditions: []
```

look

Look around your current location.

```action
name: look
template: "look"
preconditions: []
```

inventory

Check what you are carrying.

```action
name: inventory
template: "inventory"
preconditions: []
```

## Decision Pre-check Contract

Use the model's native thinking mode. After thinking, emit exactly one `<decision>...</decision>` block followed by exactly one `<action>...</action>` block.

The decision must contain a single flow-style YAML mapping that summarizes the task-relevant state immediately before the chosen action. Do not include prose outside the mapping.

Example:

<decision>{task: puttwo, target_type: creditcard, required_count: 2, targets: [{id: creditcard 2, location: dresser 1, status: placed}, {id: creditcard 3, location: countertop 1, status: available}], held: none, at: dresser 1, dest: dresser 1, previous_action: move creditcard 2 to dresser 1, arrived_receptacle_state: open, held_relation: none, search: {locations_searched: [countertop 1, dresser 1], locations_to_search: [drawer 1, drawer 2]}, phase: acquire_second}</decision>
<action>go to countertop 1</action>

### Value Rules

- Copy exact numbered entity identifiers from observations and history, such as `potato 1`, `microwave 1`, or `sinkbasin 1`.
- Use `unknown` only when history does not establish a value.
- Use `none` only when a value is known to be absent, such as an empty hand.
- `at` is the exact receptacle reached by the most recent successful `go to`.
- `held` is the exact object currently carried.
- `previous_action` is the exact action emitted on the preceding trajectory step, or `none` on the first step.
- `arrived_receptacle_state` describes the receptacle in `at` immediately after arrival; use `not_applicable` when the preceding action was not an arrival.
- `held_relation` says whether `held` is the requested target type, a non-target type, absent, or unknown. A cup is not a mug and a mug is not a cup.
- `target` is the selected target instance, or `unknown` before one is found.
- `dest` is one selected destination instance. Once selected, retain the same instance unless it becomes unusable.
- `tool` is the exact sinkbasin, microwave, fridge, or desklamp instance needed by the task.
- `state` may be `unmodified`, `clean`, `hot`, `cool`, or `unknown`.
- Change `state` only after an observation confirms a successful transformation.
- `phase` describes the next subgoal, not the entire plan.
- Do not claim completion. The environment determines whether the task has succeeded and ends the episode.
- Keep the mapping on one line and use the keys in the documented order.

### Search Ledger

Every decision must include a `search` mapping:

- `locations_to_search`: ordered exact receptacle instances whose contents have not yet been observed during the current target search.
- `locations_searched`: exact receptacle instances whose contents have already been observed.
- The two lists must not overlap.
- Copy exact numbered identifiers from the initial observation and trajectory.
- Visiting a closed container does not count as searching it. Move it to `locations_searched` only after opening it and observing its contents.
- Do not reset the ledger after visiting a tool, checking inventory, transforming an object, or delivering the first object in a pick-two task.
- A searched location may be revisited to acquire a target known to be there or to use it as a tool or destination. It must not be revisited merely to search it again.
- If the target has not been found, the next search action should normally inspect the first admissible entry in `locations_to_search`.

### Task-specific Decision Schemas

Pick and place:

`{task: put, target_type: TYPE, target_id: ENTITY|unknown, target_location: RECEPTACLE|unknown, held: ENTITY|none|unknown, at: RECEPTACLE|unknown, dest: RECEPTACLE|unknown, previous_action: ACTION|none|unknown, arrived_receptacle_state: open|closed|not_applicable|unknown, held_relation: none|target|non_target|unknown, search: {locations_searched: [RECEPTACLE, ...], locations_to_search: [RECEPTACLE, ...]}, phase: PHASE}`

Allowed phases: `search`, `acquire`, `go_destination`, `prepare_destination`, `deliver`.

Clean and place:

`{task: clean, target_type: TYPE, target_id: ENTITY|unknown, target_location: RECEPTACLE|unknown, held: ENTITY|none|unknown, at: RECEPTACLE|unknown, state: unmodified|clean|unknown, tool: SINKBASIN|unknown, dest: RECEPTACLE|unknown, previous_action: ACTION|none|unknown, arrived_receptacle_state: open|closed|not_applicable|unknown, held_relation: none|target|non_target|unknown, search: {locations_searched: [RECEPTACLE, ...], locations_to_search: [RECEPTACLE, ...]}, phase: PHASE}`

Allowed phases: `search`, `acquire`, `go_tool`, `transform`, `go_destination`, `prepare_destination`, `deliver`.

Heat and place:

`{task: heat, target_type: TYPE, target_id: ENTITY|unknown, target_location: RECEPTACLE|unknown, held: ENTITY|none|unknown, at: RECEPTACLE|unknown, state: unmodified|hot|unknown, tool: MICROWAVE|unknown, dest: RECEPTACLE|unknown, previous_action: ACTION|none|unknown, arrived_receptacle_state: open|closed|not_applicable|unknown, held_relation: none|target|non_target|unknown, search: {locations_searched: [RECEPTACLE, ...], locations_to_search: [RECEPTACLE, ...]}, phase: PHASE}`

Allowed phases: `search`, `acquire`, `go_tool`, `transform`, `go_destination`, `prepare_destination`, `deliver`.

Cool and place:

`{task: cool, target_type: TYPE, target_id: ENTITY|unknown, target_location: RECEPTACLE|unknown, held: ENTITY|none|unknown, at: RECEPTACLE|unknown, state: unmodified|cool|unknown, tool: FRIDGE|unknown, dest: RECEPTACLE|unknown, previous_action: ACTION|none|unknown, arrived_receptacle_state: open|closed|not_applicable|unknown, held_relation: none|target|non_target|unknown, search: {locations_searched: [RECEPTACLE, ...], locations_to_search: [RECEPTACLE, ...]}, phase: PHASE}`

Allowed phases: `search`, `acquire`, `go_tool`, `transform`, `go_destination`, `prepare_destination`, `deliver`.

Examine under a desklamp:

`{task: examine, target_type: TYPE, target_id: ENTITY|unknown, target_location: RECEPTACLE|unknown, held: ENTITY|none|unknown, at: RECEPTACLE|unknown, tool: DESKLAMP|unknown, tool_location: RECEPTACLE|unknown, tool_state: inactive|active|unknown, previous_action: ACTION|none|unknown, arrived_receptacle_state: open|closed|not_applicable|unknown, held_relation: none|target|non_target|unknown, search: {locations_searched: [RECEPTACLE, ...], locations_to_search: [RECEPTACLE, ...]}, phase: PHASE}`

Allowed phases: `search_target`, `acquire`, `search_tool`, `go_tool`, `use_tool`.

Pick two and place:

`{task: puttwo, target_type: TYPE, required_count: 2, targets: [{id: ENTITY|unknown, location: RECEPTACLE|unknown, status: available|held|placed|unknown}, {id: ENTITY|unknown, location: RECEPTACLE|unknown, status: available|held|placed|unknown}], held: ENTITY|none|unknown, at: RECEPTACLE|unknown, dest: RECEPTACLE|unknown, previous_action: ACTION|none|unknown, arrived_receptacle_state: open|closed|not_applicable|unknown, held_relation: none|target|non_target|unknown, search: {locations_searched: [RECEPTACLE, ...], locations_to_search: [RECEPTACLE, ...]}, phase: PHASE}`

Allowed phases: `search`, `acquire`, `go_destination`, `prepare_destination`, `deliver_first`, `acquire_second`, `deliver_second`.

## Machine-readable Decision and Policy Contract

The following versioned blocks are authoritative. Every decision uses the
listed key order and one-line flow-style YAML. `previous_action`,
`arrived_receptacle_state`, and `held_relation` are the model's own claims from
the supplied trajectory. The two `puttwo` target records are independent and
may name the same location.

```decision-schema
version: 1
task: put
fields:
  - {name: task, schema: {type: literal, value: put}}
  - {name: target_type, schema: {type: lowercase_atom}}
  - {name: target_id, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: target_location, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: held, schema: {type: numbered_entity, alternatives: [none, unknown]}}
  - {name: at, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: dest, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: previous_action, schema: {type: action, alternatives: [none, unknown]}}
  - {name: arrived_receptacle_state, schema: {type: enum, values: [open, closed, not_applicable, unknown]}}
  - {name: held_relation, schema: {type: enum, values: [none, target, non_target, unknown]}}
  - {name: search, schema: {type: record, fields: [{name: locations_searched, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}, {name: locations_to_search, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}]}}
  - {name: phase, schema: {type: enum, values: [search, acquire, go_destination, prepare_destination, deliver]}}
```

```decision-schema
version: 1
task: clean
fields:
  - {name: task, schema: {type: literal, value: clean}}
  - {name: target_type, schema: {type: lowercase_atom}}
  - {name: target_id, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: target_location, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: held, schema: {type: numbered_entity, alternatives: [none, unknown]}}
  - {name: at, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: state, schema: {type: enum, values: [unmodified, clean, unknown]}}
  - {name: tool, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: dest, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: previous_action, schema: {type: action, alternatives: [none, unknown]}}
  - {name: arrived_receptacle_state, schema: {type: enum, values: [open, closed, not_applicable, unknown]}}
  - {name: held_relation, schema: {type: enum, values: [none, target, non_target, unknown]}}
  - {name: search, schema: {type: record, fields: [{name: locations_searched, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}, {name: locations_to_search, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}]}}
  - {name: phase, schema: {type: enum, values: [search, acquire, go_tool, transform, go_destination, prepare_destination, deliver]}}
```

```decision-schema
version: 1
task: heat
fields:
  - {name: task, schema: {type: literal, value: heat}}
  - {name: target_type, schema: {type: lowercase_atom}}
  - {name: target_id, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: target_location, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: held, schema: {type: numbered_entity, alternatives: [none, unknown]}}
  - {name: at, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: state, schema: {type: enum, values: [unmodified, hot, unknown]}}
  - {name: tool, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: dest, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: previous_action, schema: {type: action, alternatives: [none, unknown]}}
  - {name: arrived_receptacle_state, schema: {type: enum, values: [open, closed, not_applicable, unknown]}}
  - {name: held_relation, schema: {type: enum, values: [none, target, non_target, unknown]}}
  - {name: search, schema: {type: record, fields: [{name: locations_searched, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}, {name: locations_to_search, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}]}}
  - {name: phase, schema: {type: enum, values: [search, acquire, go_tool, transform, go_destination, prepare_destination, deliver]}}
```

```decision-schema
version: 1
task: cool
fields:
  - {name: task, schema: {type: literal, value: cool}}
  - {name: target_type, schema: {type: lowercase_atom}}
  - {name: target_id, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: target_location, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: held, schema: {type: numbered_entity, alternatives: [none, unknown]}}
  - {name: at, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: state, schema: {type: enum, values: [unmodified, cool, unknown]}}
  - {name: tool, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: dest, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: previous_action, schema: {type: action, alternatives: [none, unknown]}}
  - {name: arrived_receptacle_state, schema: {type: enum, values: [open, closed, not_applicable, unknown]}}
  - {name: held_relation, schema: {type: enum, values: [none, target, non_target, unknown]}}
  - {name: search, schema: {type: record, fields: [{name: locations_searched, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}, {name: locations_to_search, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}]}}
  - {name: phase, schema: {type: enum, values: [search, acquire, go_tool, transform, go_destination, prepare_destination, deliver]}}
```

```decision-schema
version: 1
task: examine
fields:
  - {name: task, schema: {type: literal, value: examine}}
  - {name: target_type, schema: {type: lowercase_atom}}
  - {name: target_id, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: target_location, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: held, schema: {type: numbered_entity, alternatives: [none, unknown]}}
  - {name: at, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: tool, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: tool_location, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: tool_state, schema: {type: enum, values: [inactive, active, unknown]}}
  - {name: previous_action, schema: {type: action, alternatives: [none, unknown]}}
  - {name: arrived_receptacle_state, schema: {type: enum, values: [open, closed, not_applicable, unknown]}}
  - {name: held_relation, schema: {type: enum, values: [none, target, non_target, unknown]}}
  - {name: search, schema: {type: record, fields: [{name: locations_searched, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}, {name: locations_to_search, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}]}}
  - {name: phase, schema: {type: enum, values: [search_target, acquire, search_tool, go_tool, use_tool]}}
```

```decision-schema
version: 1
task: puttwo
fields:
  - {name: task, schema: {type: literal, value: puttwo}}
  - {name: target_type, schema: {type: lowercase_atom}}
  - {name: required_count, schema: {type: literal, value: 2}}
  - {name: targets, schema: {type: bounded_list, item: {type: record, fields: [{name: id, schema: {type: numbered_entity, alternatives: [unknown]}}, {name: location, schema: {type: numbered_entity, alternatives: [unknown]}}, {name: status, schema: {type: enum, values: [available, held, placed, unknown]}}]}, min_items: 2, max_items: 2}}
  - {name: held, schema: {type: numbered_entity, alternatives: [none, unknown]}}
  - {name: at, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: dest, schema: {type: numbered_entity, alternatives: [unknown]}}
  - {name: previous_action, schema: {type: action, alternatives: [none, unknown]}}
  - {name: arrived_receptacle_state, schema: {type: enum, values: [open, closed, not_applicable, unknown]}}
  - {name: held_relation, schema: {type: enum, values: [none, target, non_target, unknown]}}
  - {name: search, schema: {type: record, fields: [{name: locations_searched, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}, {name: locations_to_search, schema: {type: bounded_list, item: {type: numbered_entity, alternatives: []}, min_items: 0, max_items: 64}}]}}
  - {name: phase, schema: {type: enum, values: [search, acquire, go_destination, prepare_destination, deliver_first, acquire_second, deliver_second]}}
```

```policy
version: 1
name: open_declared_closed_receptacle
priority: 100
when:
  - {field: arrived_receptacle_state, equals: closed}
  - {field: at, is_type: numbered_entity}
effect:
  require_action: {action: open, bindings: {recep: at}}
```

```policy
version: 1
name: put_down_declared_non_target
priority: 90
when:
  - {field: held_relation, equals: non_target}
  - {field: held, is_type: numbered_entity}
  - {field: at, is_type: numbered_entity}
effect:
  require_action: {action: move, bindings: {obj: held, recep: at}}
```

```policy
version: 1
name: preserve_target_lexical_type
priority: 20
when: []
effect:
  restrict_take_type: {field: target_type}
```

```policy
version: 1
name: do_not_repeat_declared_previous_action
priority: 10
when:
  - {field: previous_action, not_in: [none, unknown]}
effect:
  forbid_exact_action: {field: previous_action}
```
