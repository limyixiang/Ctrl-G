prechecks:
  - name: object-delivery
    task_type: pick_and_place
    fields:
      - name: target_visible
        values: ["yes", "no", "unknown"]
      - name: target_location_known
        values: ["yes", "no"]
      - name: holding_target
        values: ["yes", "no", "unknown"]
      - name: destination_known
        values: ["yes", "no"]
      - name: at_destination
        values: ["yes", "no", "unknown"]
      - name: destination_ready
        values: ["yes", "no", "unknown"]
      - name: goal_complete
        values: ["yes", "no", "unknown"]
      - name: next_phase
        values: ["search", "acquire", "prepare_destination", "deliver", "complete"]
