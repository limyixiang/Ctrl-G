# AlfWorld Skills

This file is (a) injected verbatim into the agent's context so the model knows
which tools exist, and (b) parsed by the harness to build per-step generation
constraints for Ctrl-G.

Each skill has a fenced ```tool block with:

- `template`: the action surface form. Placeholders `{obj}` / `{recep}` are
  grounded at each step against the tracked world state.
- `preconditions`: predicates that must ALL hold for a grounded action to be
  admissible. Available predicates (implemented in `state.py`):
  `at({recep})`, `not_at({recep})`, `holding({obj})`, `hand_empty`,
  `visible({obj})`, `seen({recep})`, `openable({recep})`, `closed({recep})`,
  `open({recep})`, `recep_type({recep}, <type>)`, `obj_at_current({obj})`.

Precondition semantics are *sound but not complete*: an action is only pruned
when the tracker can prove a precondition false. Unknown facts default to
admissible, so imperfect state tracking can never rule out the correct action.

At each step the harness grounds every template, keeps the bindings whose
preconditions hold, and compiles the resulting finite set of action strings
into a token-trie DFA for Ctrl-G's `ConstraintLogitsProcessor`.

The model must emit exactly one action per turn as `<tool>action</tool>`.

---

## go to

Move to a receptacle you have seen in the room. No effect if you are already there.

```tool
name: goto
template: "go to {recep}"
preconditions:
  - seen({recep})
  - not_at({recep})
```

## take

Pick up an object from the receptacle you are currently at. Your hand must be empty.

```tool
name: take
template: "take {obj} from {recep}"
preconditions:
  - at({recep})
  - obj_at_current({obj})
  - hand_empty
```

## put

Put the object you are holding in/on the receptacle you are currently at.

```tool
name: put
template: "put {obj} in/on {recep}"
preconditions:
  - at({recep})
  - holding({obj})
```

## open

Open a closed receptacle (cabinet, drawer, fridge, microwave, safe, ...) you are at.

```tool
name: open
template: "open {recep}"
preconditions:
  - at({recep})
  - openable({recep})
  - closed({recep})
```

## close

Close an open receptacle you are at.

```tool
name: close
template: "close {recep}"
preconditions:
  - at({recep})
  - openable({recep})
  - open({recep})
```

## clean

Clean the object you are holding, using a sinkbasin you are at.

```tool
name: clean
template: "clean {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, sinkbasin)
```

## heat

Heat the object you are holding, using a microwave you are at.

```tool
name: heat
template: "heat {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, microwave)
```

## cool

Cool the object you are holding, using a fridge you are at.

```tool
name: cool
template: "cool {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, fridge)
```

## use

Toggle/use an object at your current location (e.g. turn on a desklamp).

```tool
name: use
template: "use {obj}"
preconditions:
  - obj_at_current({obj})
```

## examine

Look closely at an object you hold or see, or the receptacle you are at.

```tool
name: examine
template: "examine {thing}"
preconditions: []
```

## look

Look around your current location.

```tool
name: look
template: "look"
preconditions: []
```

## inventory

Check what you are carrying.

```tool
name: inventory
template: "inventory"
preconditions: []
```
