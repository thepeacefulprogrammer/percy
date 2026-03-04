from pydantic import BaseModel, Field
from ventures_agent_framework import get_custom_client, logger
from pathlib import Path


class HandOverAgentUpdate(BaseModel):
    message: str = Field(description="Response to the user")
    report_update: str | None = Field(
        None,
        description="A report update created by the Agent to capture important notes between prompts",
    )
    lessons_learned_update: str | None = Field(
        None, description="A lesson learned to be added to the lessons learned registry"
    )


class HandOver:
    def __init__(self, output_dir: Path, _compaction_trigger: int = 40):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.report: str = ""
        self.compaction_trigger: int = _compaction_trigger
        self.output_dir = output_dir
        self.message_history: list[str] = []
        self.message_history_file = output_dir / "message_history.md"
        self.load_message_history()
        self.report_file = output_dir / "report.md"
        self.lessons_learned: str = ""
        self.lessons_learned_file = output_dir / "lessons_learned.md"
        self.load_lessons_learned()
        client = get_custom_client("gpt-4.1-nano")
        self.report_update_agent = client.as_agent(
            name="report_update_agent",
            instructions="You are a report update agent. You receive a report and a report update and you combine them together, removing duplication. You distill the information down to succinct points without losing information or details. You combine similar concepts together and organize the report in logical sections. The end result is a report that is coherent and can be used by future AI Agents to perform their tasks more efficiently and effectively. You respond with only the report string with no other commentary. The report must be in markdown format with logical sections, headers and standard formatting.",
        )
        self.compact_message_history_agent = client.as_agent(
            name="compact_message_history_agent",
            instructions="You are a message history compaction agent. You receive a series of messages between a user and an agent. Your job is to extract import information from the message history that the agent would need to know in order to maintain its conversation and do its job. You will summarize the conversation. Note: this will replace the messages in the message history so be sure that your summarization maintains key information, but removes any superfulous text.",
        )
        self.lessons_learned_agent = client.as_agent(
            name="lessons_learned_agent",
            instructions="You maintain a lessons learned registry. You will recieve requests to add a new lesson learned. Your job is to review the request and consider it in relation to the existing lessons learned. You will maintain the lessons learned in logical sections and will add the update to the appropraite section (creating new sections as necessary). Be consise - these lessons learned will be used by AI Agents to ensure they can do their job more effectively and efficiently. Respond with the complete, updated lessons learned registry.",
        )

    async def add_report_update(self, report_update: str):
        logger.debug(f"Received report update: {report_update}")
        response = await self.report_update_agent.run(
            f"<report_update>{report_update}</report_update><report>{self.report}</report>"
        )
        self.report = str(response)
        with open(self.report_file, "w") as f:
            f.write(self.report)

        logger.debug(f"Report updated and saved in {self.output_dir}")

    async def add_prompt(self, prompt: str):
        await self._update_message_history("USER", prompt)

    async def add_response(self, handover_update: HandOverAgentUpdate):
        logger.debug(handover_update)
        await self._update_message_history("AGENT", handover_update.message)
        if handover_update.lessons_learned_update:
            await self.add_lessons_learned_update(
                handover_update.lessons_learned_update
            )
        if handover_update.report_update:
            await self.add_report_update(handover_update.report_update)

    async def _update_message_history(self, sender: str, message: str):
        self.message_history.append(f"[{sender}]: {message}")
        await self.compact_message_history()
        self.save_message_history()

    async def compact_message_history(self):
        total_chars_before_compaction = sum(len(s) for s in self.message_history)

        if len(self.message_history) >= self.compaction_trigger:
            logger.debug("Compacting message history")
            response = await self.compact_message_history_agent.run(
                f"Compact this message history into a summary. Return only the summary.<message_history>{self.message_history}</message_history>"
            )
            # keep the last 10 messages
            last_messages = self.message_history[-10:]
            self.message_history = [
                f"# Message History\n\nPrevious messages have been removed and replaced with this conversation summary:\n\n{response}.",
                *last_messages,
            ]
            total_chars_after_compaction = sum(len(s) for s in self.message_history)
            logger.debug(
                f"Message history reduced from {total_chars_before_compaction} to {total_chars_after_compaction} chars. Reduction of {(1 - total_chars_after_compaction / total_chars_before_compaction) * 100}%."
            )

    def save_message_history(self):
        with open(self.message_history_file, "w") as f:
            f.writelines(s + "\n" for s in self.message_history)

    def save_lessons_learned(self):
        with open(self.lessons_learned_file, "w") as f:
            f.write(self.lessons_learned)

    def load_message_history(self):
        if self.message_history_file.exists():
            with open(self.message_history_file, "r") as f:
                self.message_history = [line.strip() for line in f]

    def load_lessons_learned(self):
        if self.lessons_learned_file.exists():
            with open(self.lessons_learned_file, "r") as f:
                self.lessons_learned = f.read()

    async def add_lessons_learned_update(self, lessons_learned_update: str):
        response = await self.lessons_learned_agent.run(f"""
        {"No lessons have yet been captured, create the lessons learned registry based on this first lesson learned" if len(self.lessons_learned) == 0 else f"<lessons_learned>{self.lessons_learned}</lessons_learned>"}
         <lesson_learned_update>{lessons_learned_update}</lessons_learned_update>""")

        self.lessons_learned = str(response)
        self.save_lessons_learned()

    async def get_handover_prompt(self, prompt: str) -> str:
        response = f"""
            # Handover Report
            {self.report}
    
            # Lessons learned
            {self.lessons_learned}
    
            # Message history
            {self.message_history}

            # Users prompt
            {prompt}
            """
        await self.add_prompt(prompt)
        return response
