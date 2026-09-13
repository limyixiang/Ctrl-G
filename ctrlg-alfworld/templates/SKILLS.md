# ALFWorld Embodied Agent Skill

## Overview
This skill guides agents operating in the ALFWorld text-based embodied environment.
The agent must complete household tasks by navigating rooms, interacting with objects,
and using appliances. Actions must be chosen from the admissible action list provided
at each step.

**Output format**: Always output `<think>...</think>` for reasoning, then `<decision>...</decision>` for one short summarized decision and `<action>...</action>` for the chosen action.

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
8. **Only choose admissible actions**: Always pick an action from the admissible action list. Do not invent actions.

---

## Common Mistakes to Avoid

- **Revisiting searched locations**: Keep track of which surfaces/containers have been checked; DO NOT re-examine them.
- **Ignoring visible objects**: If the target object appears in the observation, pick it up immediately.
- **Skipping state changes**: DO NOT place an object at the destination without first cleaning/heating/cooling it when required.
- **Premature termination**: DO NOT stop the episode until all goal conditions are verified as met.
- **Action loops**: DO NOT repeat the same actions. Move on to new locations instead.

---

## Actions Templates and Preconditions

### go to

Move to a receptacle you have seen in the room. No effect if you are already there.

```action
name: goto
template: "go to {recep}"
preconditions:
  - seen({recep})
  - not_at({recep})
```

### take

Pick up an object from the receptacle you are currently at. Your hand must be empty.

```action
name: take
template: "take {obj} from {recep}"
preconditions:
  - at({recep})
  - obj_at_current({obj})
  - hand_empty
```

### put

Put the object you are holding in/on the receptacle you are currently at.

```action
name: put
template: "put {obj} in/on {recep}"
preconditions:
  - at({recep})
  - holding({obj})
```

### open

Open a closed receptacle (cabinet, drawer, fridge, microwave, safe, ...) you are at.

```action
name: open
template: "open {recep}"
preconditions:
  - at({recep})
  - openable({recep})
  - closed({recep})
```

### close

Close an open receptacle you are at.

```action
name: close
template: "close {recep}"
preconditions:
  - at({recep})
  - openable({recep})
  - open({recep})
```

### clean

Clean the object you are holding, using a sinkbasin you are at.

```action
name: clean
template: "clean {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, sinkbasin)
```

### heat

Heat the object you are holding, using a microwave you are at.

```action
name: heat
template: "heat {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, microwave)
```

### cool

Cool the object you are holding, using a fridge you are at.

```action
name: cool
template: "cool {obj} with {recep}"
preconditions:
  - at({recep})
  - holding({obj})
  - recep_type({recep}, fridge)
```

### use

Toggle/use an object at your current location (e.g. turn on a desklamp).

```action
name: use
template: "use {obj}"
preconditions:
  - obj_at_current({obj})
```

### examine

Look closely at an object you hold or see, or the receptacle you are at.

```action
name: examine
template: "examine {thing}"
preconditions: []
```

### look

Look around your current location.

```action
name: look
template: "look"
preconditions: []
```

### inventory

Check what you are carrying.

```action
name: inventory
template: "inventory"
preconditions: []
```
