"""
AdvisorSwarm — Advisor Strategy Pattern

Implements the advisor strategy described in Anthropic's research
(April 2026): pair a cheaper executor model that drives the task
end-to-end with a powerful advisor model consulted only at key
decision points.

The advisor never calls tools or produces user-facing output — it
only provides strategic guidance to the executor.

Architecture:
    task --> [ Advisor: plan ] --> [ Executor: work ]
                                       ^         |
                                       |  loop   v
                                  [ Executor ] <-- [ Advisor: review ]

Flow:
    1. Advisor receives the task and produces a strategic plan
    2. Executor works on the task using the advisor's guidance
    3. Advisor reviews the executor's output (budget permitting)
    4. If not satisfactory, executor refines based on feedback
    5. Repeat 3-4 until advisor says SATISFACTORY or budget exhausted

Reference: "The advisor strategy: Give agents an intelligence
boost" (Anthropic, April 2026)
"""

import re
from typing import Any, Callable, List, Optional

from loguru import logger

from swarms.prompts.advisor_swarm_prompts import (
    ADVISOR_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.swarm_id import swarm_id
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="advisor_swarm")


class AdvisorSwarm:
    """Implements the Advisor Strategy: pairs a cheaper executor model
    with a powerful advisor model consulted at key decision points.

    The advisor provides strategic guidance before work begins, then
    reviews executor output in an iterative refinement loop. The
    advisor never calls tools or produces user-facing output.

    This is provider-agnostic — any model supported by LiteLLM works
    for either role.

    Args:
        id: Unique identifier for this swarm instance.
        name: Human-readable name.
        description: Description of the swarm's purpose.
        executor_model_name: Model for the executor agent.
        advisor_model_name: Model for the advisor agent.
        executor_system_prompt: System prompt for the executor.
        advisor_system_prompt: System prompt for the advisor.
        max_advisor_uses: Max advisor calls per run() invocation.
            Budget: 1 for planning + up to (N-1) for reviews.
        max_loops: Outer iteration count (each loop is a full
            plan-execute-review cycle).
        output_type: Format for conversation history output.
        verbose: Enable detailed logging.
        executor_agent: Optional pre-configured Agent for execution
            (e.g., with tools or MCP configs).
        advisor_agent: Optional pre-configured Agent for advising.
        tools: Tools available to the executor agent only.

    Examples:
        >>> swarm = AdvisorSwarm(
        ...     executor_model_name="claude-sonnet-4-6",
        ...     advisor_model_name="claude-opus-4-6",
        ... )
        >>> result = swarm.run("Write a Python function to merge two sorted lists")

        >>> # With custom executor that has tools
        >>> from swarms import Agent
        >>> executor = Agent(
        ...     agent_name="Executor",
        ...     model_name="claude-sonnet-4-6",
        ...     max_loops=1,
        ...     tools=[my_tool],
        ... )
        >>> swarm = AdvisorSwarm(
        ...     executor_agent=executor,
        ...     advisor_model_name="claude-opus-4-6",
        ... )
        >>> result = swarm.run("Refactor the auth module for performance")
    """

    def __init__(
        self,
        id: str = None,
        name: str = "AdvisorSwarm",
        description: str = "An executor-advisor swarm implementing the advisor strategy pattern",
        executor_model_name: str = "claude-sonnet-4-6",
        advisor_model_name: str = "claude-opus-4-6",
        executor_system_prompt: str = EXECUTOR_SYSTEM_PROMPT,
        advisor_system_prompt: str = ADVISOR_SYSTEM_PROMPT,
        max_advisor_uses: int = 3,
        max_loops: int = 1,
        output_type: OutputType = "dict-all-except-first",
        verbose: bool = False,
        executor_agent: Optional[Agent] = None,
        advisor_agent: Optional[Agent] = None,
        tools: Optional[List[Callable]] = None,
        *args,
        **kwargs,
    ) -> None:
        self.id = id or swarm_id()
        self.name = name
        self.description = description
        self.executor_model_name = executor_model_name
        self.advisor_model_name = advisor_model_name
        self.executor_system_prompt = executor_system_prompt
        self.advisor_system_prompt = advisor_system_prompt
        self.max_advisor_uses = max_advisor_uses
        self.max_loops = max_loops
        self.output_type = output_type
        self.verbose = verbose
        self.tools = tools

        self.reliability_check()

        self.conversation = Conversation()

        self.executor_agent = executor_agent or self._create_executor()
        self.advisor_agent = advisor_agent or self._create_advisor()

    def reliability_check(self):
        """Validate swarm configuration."""
        if self.max_advisor_uses < 1:
            raise ValueError(
                f"max_advisor_uses must be >= 1 (need at least 1 for the planning call), got {self.max_advisor_uses}"
            )
        if self.max_loops < 1:
            raise ValueError(
                f"max_loops must be >= 1, got {self.max_loops}"
            )
        if not self.executor_model_name:
            raise ValueError("executor_model_name must be provided")
        if not self.advisor_model_name:
            raise ValueError("advisor_model_name must be provided")

        if self.verbose:
            logger.info(
                f"AdvisorSwarm initialized: "
                f"executor={self.executor_model_name}, "
                f"advisor={self.advisor_model_name}, "
                f"max_advisor_uses={self.max_advisor_uses}, "
                f"max_loops={self.max_loops}"
            )

    def _create_executor(self) -> Agent:
        """Create the executor agent with tools."""
        return Agent(
            agent_name="Executor",
            agent_description="Executes tasks using advisor strategic guidance",
            system_prompt=self.executor_system_prompt,
            model_name=self.executor_model_name,
            max_loops=1,
            temperature=1.0,
            output_type="final",
            verbose=self.verbose,
            tools=self.tools,
        )

    def _create_advisor(self) -> Agent:
        """Create the advisor agent. No tools — guidance only."""
        return Agent(
            agent_name="Advisor",
            agent_description="Provides strategic guidance to the executor",
            system_prompt=self.advisor_system_prompt,
            model_name=self.advisor_model_name,
            max_loops=1,
            temperature=1.0,
            output_type="final",
            verbose=self.verbose,
        )

    def _is_satisfactory(self, review: str) -> bool:
        """Parse the advisor's review to determine if output is satisfactory.

        Checks for VERDICT: SATISFACTORY using regex to handle
        variations in whitespace and casing.
        """
        return bool(
            re.search(r"verdict\s*:\s*satisfactory", review, re.IGNORECASE)
        )

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute the advisor-executor orchestration flow.

        Args:
            task: The task to accomplish.
            img: Optional single image input.
            imgs: Optional list of image inputs.

        Returns:
            Formatted conversation history per output_type.
        """
        if not task:
            raise ValueError("A task is required")

        self.conversation.add(role="User", content=task)

        for loop in range(self.max_loops):
            if self.verbose:
                logger.info(
                    f"[AdvisorSwarm] Loop {loop + 1}/{self.max_loops}"
                )

            advisor_uses = 0

            # --- Step 1: Advisor plans ---
            # Advisor sees the task and full conversation history
            # (history matters for multi-loop runs)
            history = self.conversation.get_str()
            planning_prompt = (
                f"You are in PLANNING MODE.\n\n"
                f"Task: {task}\n\n"
                f"--- CONVERSATION HISTORY ---\n{history}\n"
                f"--- END CONVERSATION HISTORY ---\n\n"
                f"Provide a strategic plan for the Executor to follow."
            )

            advice = self.advisor_agent.run(task=planning_prompt)
            advisor_uses += 1
            self.conversation.add(
                role="Advisor",
                content=f"[Strategic Plan]\n{advice}",
            )

            if self.verbose:
                logger.info(
                    f"[AdvisorSwarm] Advisor plan "
                    f"({advisor_uses}/{self.max_advisor_uses} calls used)"
                )

            # --- Step 2: Executor works ---
            # Executor sees the task and the advisor's plan
            executor_prompt = (
                f"Task: {task}\n\n"
                f"Strategic guidance from your Advisor:\n{advice}\n\n"
                f"Execute this task following the Advisor's guidance. "
                f"Produce complete, concrete output."
            )

            executor_output = self.executor_agent.run(
                task=executor_prompt, img=img, imgs=imgs
            )
            self.conversation.add(
                role="Executor", content=executor_output
            )

            if self.verbose:
                logger.info("[AdvisorSwarm] Executor produced output")

            # --- Step 3-4: Review-refine loop ---
            while advisor_uses < self.max_advisor_uses:
                # Step 3: Advisor reviews
                # Advisor sees the full conversation history including
                # all prior exchanges — mirrors the Anthropic pattern
                # where the advisor sees the full transcript
                history = self.conversation.get_str()
                review_prompt = (
                    f"You are in REVIEW MODE.\n\n"
                    f"Original task: {task}\n\n"
                    f"--- CONVERSATION HISTORY ---\n{history}\n"
                    f"--- END CONVERSATION HISTORY ---\n\n"
                    f"Review the Executor's most recent output against "
                    f"the original task. Start your response with "
                    f"VERDICT: SATISFACTORY or VERDICT: NEEDS_REVISION."
                )

                review = self.advisor_agent.run(task=review_prompt)
                advisor_uses += 1
                self.conversation.add(
                    role="Advisor",
                    content=f"[Review]\n{review}",
                )

                if self.verbose:
                    logger.info(
                        f"[AdvisorSwarm] Advisor review "
                        f"({advisor_uses}/{self.max_advisor_uses} calls used)"
                    )

                if self._is_satisfactory(review):
                    if self.verbose:
                        logger.info(
                            "[AdvisorSwarm] Advisor verdict: SATISFACTORY"
                        )
                    break

                # Step 4: Executor refines
                # Executor sees: original task, its own output, and
                # the advisor's specific feedback
                refinement_prompt = (
                    f"Original task: {task}\n\n"
                    f"Your previous output:\n{executor_output}\n\n"
                    f"Advisor feedback:\n{review}\n\n"
                    f"Revise your output to address each point in the "
                    f"Advisor's feedback. Produce the complete revised output."
                )

                executor_output = self.executor_agent.run(
                    task=refinement_prompt, img=img, imgs=imgs
                )
                self.conversation.add(
                    role="Executor", content=executor_output
                )

                if self.verbose:
                    logger.info(
                        "[AdvisorSwarm] Executor refined output"
                    )

            if self.verbose:
                logger.info(
                    f"[AdvisorSwarm] Loop {loop + 1} complete. "
                    f"Advisor calls used: {advisor_uses}/{self.max_advisor_uses}"
                )

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def batched_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """Run multiple tasks sequentially.

        Args:
            tasks: List of task strings.

        Returns:
            List of results, one per task.
        """
        return [self.run(task=t, *args, **kwargs) for t in tasks]

    def __call__(self, task: str, *args, **kwargs) -> Any:
        """Make the swarm callable."""
        return self.run(task=task, *args, **kwargs)
