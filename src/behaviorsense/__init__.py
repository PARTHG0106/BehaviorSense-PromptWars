"""BehaviorSense AI - identity-aware behaviour monitoring for elderly care.

Package layout::

    behaviorsense/
      schemas.py            inter-agent data contracts (Pydantic, versioned)
      agents/
        perception.py       Agent 1: detect -> track -> ReID -> pose -> role
        activity.py         Agent 2: skeleton windows -> 20-class ADL + smoothing
        behaviour.py        Agent 3: baseline -> deviation -> drift -> alerts
        reasoning/
          verifier.py       deterministic claim-level faithfulness verification
          reporter.py       Agent 4: schema-constrained LLM report generation
      data/
        simulator.py        longitudinal behaviour simulator with injected ground truth
      eval/                 metrics for each protocol

Agents exchange validated schema objects; the service persists records for replay.
Each stage is independently testable without rerunning the video pipeline.
"""

__version__ = "0.1.0"
