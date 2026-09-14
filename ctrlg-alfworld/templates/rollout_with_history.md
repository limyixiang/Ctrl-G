You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Your initial observation was: {initial_observation}
Below are all {history_length} trajectory steps taken since that observation:
{action_history}
You are now at step {current_step} and your current observation is: {current_observation}
{admissible_actions_section}

Now it's your turn to take an action.
Choose the next action that makes progress on the task. The action will be checked against the environment's current admissible commands.
{decision_instruction}
