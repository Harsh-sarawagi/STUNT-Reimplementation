# STUNT Reproduction

Independent reproduction of the core ideas from the STUNT paper for few-shot tabular learning.

## Current status
- [x] Project structure
- [x] Basic ProtoNet implementation
- [ ] Exact STUNT task generation
- [ ] Meta-training
- [ ] Few-shot evaluation
- [ ] Baseline comparison

## Workflow
1. Verify the paper's exact task-generation algorithm.
2. Implement and test `src/task_generator.py`.
3. Train the tabular encoder.
4. Implement the complete ProtoNet training loop.
5. Evaluate on a real few-shot tabular task.
6. Compare against supervised baselines.
7. Add experiments and documentation.

This is an independent educational/research reproduction, not the official STUNT implementation.
