from string import Template
from importlib import resources

from local_agent.llm.base import Message


class PromptRenderer:
    def render(self, template_name: str, context: dict[str, object]) -> str:
        template_text = _read_template(template_name)
        template = Template(template_text)
        return template.safe_substitute(
            {key: str(value) for key, value in context.items()}
        )

    def render_messages(
        self,
        system_template: str,
        user_template: str,
        context: dict[str, object],
    ) -> list[Message]:
        return [
            Message("system", self.render(system_template, context)),
            Message("user", self.render(user_template, context)),
        ]


def _read_template(template_name: str) -> str:
    try:
        return (
            resources.files("local_agent.agent.prompt_templates")
            .joinpath(template_name)
            .read_text()
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt template not found: {template_name}") from exc
