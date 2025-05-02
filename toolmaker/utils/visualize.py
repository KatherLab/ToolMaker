"""
This module contains the code for visualizing the logs of a Toolmaker run.
Note that this is mostly LLM-generated code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, TypedDict

import typer
from jinja2 import Template
from markdown import markdown
from markupsafe import Markup

app = typer.Typer()


class LogEntry(TypedDict):
    type: str
    time: str
    name: str
    content: Any
    metadata: Mapping[str, Any]
    children: list[LogEntry]
    result: LogEntry | None
    content_text: str | None
    content_html: Markup | None


html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Toolmaker Logs</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            .log-entry {
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
            }
            .indented {
                margin-left: 30px;
            }
            .log-header {
                display: flex;
                align-items: center;
                cursor: pointer;
            }
            .log-type {
                font-weight: bold;
            }
            .log-time {
                color: #666;
                font-size: 0.9em;
                margin-left: 10px;
            }
            .log-content {
                background: #f8f8f8;
                padding: 10px;
                border-radius: 4px;
                white-space: pre-wrap;
                font-family: monospace;
                font-size: 14px;
                line-height: 1.4;
            }
            .expand-button {
                margin-right: 10px;
                width: 20px;
                height: 20px;
                border: none;
                background: #eee;
                border-radius: 3px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .expand-button:hover {
                background: #ddd;
            }
            .log-subtype {
                color: #0066cc;
                margin-left: 10px;
            }
            .markdown-content {
                background: #f8f8f8;
                padding: 10px;
                border-radius: 4px;
                font-size: 14px;
                line-height: 1.6;
            }
            .markdown-content p {
                margin: 0 0 1em 0;
            }
            .markdown-content ul, .markdown-content ol {
                margin: 0 0 1em 0;
                padding-left: 2em;
            }
            .markdown-content code {
                background: #e8e8e8;
                padding: 2px 4px;
                border-radius: 3px;
            }
            .log-entry[data-name="update_code"] {
                border-color: orange;
                background-color: #fff5e6;
            }
            .log-entry[data-name="action_call"] {
                border-color: green;
                background-color: #f0fff0;
            }
            .log-entry[data-name="llm_call"] {
                border-color: #6b46c1;
                background-color: #f5f3ff;
            }
            .log-entry[data-name="function_execution_result"] {
                border-color: #FF1493;  /* Deep Pink */
                background-color: #FFF0F5;  /* Lavender Blush */
            }
            .llm-messages {
                margin-top: 10px;
            }
            .llm-message {
                padding: 8px 12px;
                margin: 4px 0;
                border-radius: 4px;
                max-width: 100%;
                overflow-wrap: break-word;
            }
            .llm-message pre {
                white-space: pre-wrap;
                overflow-wrap: break-word;
                background: rgba(0, 0, 0, 0.03);
                padding: 8px;
                border-radius: 4px;
                margin: 4px 0;
                max-width: 100%;
            }
            .llm-message.user {
                background-color: #e9ecef;
                margin-right: 20%;
            }
            .llm-message.assistant {
                background-color: #d9f2e6;
                margin-left: 20%;
            }
            .llm-message.system {
                background-color: #fff3cd;
            }
            .llm-message.tool {
                background-color: #cff4fc;
            }
            .llm-message-role {
                font-weight: bold;
                margin-bottom: 4px;
                color: #666;
            }
            .llm-stats {
                margin-top: 8px;
                padding: 8px;
                background-color: #edf2f7;
                border-radius: 4px;
                font-size: 0.9em;
            }
            .expandable-content {
                display: none;  /* Start hidden */
                padding: 10px;
                border-radius: 4px;
            }
            .controls {
                margin-bottom: 20px;
            }
            .controls button {
                margin-right: 10px;
                padding: 5px 10px;
                font-size: 14px;
                border-radius: 4px;
                border: 1px solid #ccc;
                background: #f8f8f8;
                cursor: pointer;
            }
            .controls button:hover {
                background: #e8e8e8;
            }
            .log-result {
                margin-top: 10px;
                padding: 10px;
                background-color: #f8f8f8;
                border-left: 3px solid #666;
            }

            .log-result h4 {
                margin: 0 0 10px 0;
                color: #333;
            }

            .log-result h5 {
                margin: 5px 0;
                color: #666;
            }

            .result-metadata, .result-content {
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div style="display: flex; align-items: center; gap: 20px;">
            <h1>Toolmaker Logs - Total Cost: ${{ "%.2f"|format(total_cost) }}</h1>
            <div class="controls">
                <button onclick="expandAll()">Expand All</button>
                <button onclick="expandNested()">Expand Nested</button>
                <button onclick="collapseAll()">Collapse All</button>
            </div>
        </div>

        {% macro render_llm_message(message) %}
            <div class="llm-message {{ message.role }}">
                <div class="llm-message-role">{{ message.role | title }}</div>
                {% if message.content %}
                    <pre>{{ message.content | escape }}</pre>
                {% endif %}
                {% if message.tool_calls %}
                    <div class="tool-calls">
                        {% for tool_call in message.tool_calls %}
                            <div class="tool-call">
                                <strong>Tool:</strong> {{ tool_call.function.name }}<br>
                                <strong>Arguments:</strong> <pre class="to-json">{{ tool_call.function.arguments | escape }}</pre>
                            </div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
        {% endmacro %}

        {% macro render_log_content(log) %}
            {% if log.metadata %}
                <pre class="log-content to-json">{{ log.metadata | tojson(indent=2) }}</pre>
            {% endif %}
            {% if log.content is defined %}
                {% if log.content is none %}
                    <em>No content.</em>
                {% elif log.name == 'llm_call' %}
                    <div class="llm-messages">
                        {% if log.content is mapping %}
                            {{ render_llm_message(log.content) }}
                        {% else %}
                            {% for message in log.content %}
                                {{ render_llm_message(message) }}
                            {% endfor %}
                        {% endif %}
                    </div>
                {% elif log.content_text %}
                    <pre class="log-content">{{ log.content_text | escape }}</pre>
                {% else %}
                    <h4>Content:</h4>
                    <pre class="log-content to-json">{{ log.content | tojson(indent=2) }}</pre>
                {% endif %}
            {% endif %}

            {% if log.children %}
                {% for child in log.children %}
                    {{ render_log(child) }}
                {% endfor %}
            {% endif %}

            {% if log.result is defined and log.result is not none and log.result.content is defined %}
                <h4>Result:</h4>
                <div class="log-result">
                    {{ render_log_content(log.result) }}
                </div>
            {% endif %}
        {% endmacro %}

        {% macro render_log(log) %}
            <div class="log-entry" data-type="{{ log.type }}" data-name="{{ log.name }}">
                <div class="log-header" onclick="toggleContent(this)">
                    <button class="expand-button">+</button>
                    <div class="log-type">
                        {{ log.name }}  {% if log.type == 'end' %} - END{% endif %}
                        <span class="log-subtype">
                            {% if log.type == 'start' %}
                                {% if log.name == 'llm_call' %}
                                    {{ log.metadata.model }}
                                    (${{ '%.2f' | format(log.metadata.get(_call_cost, 0)) }})
                                {% else %}
                                    {% for key, value in log.metadata.items() %}
                                        {{ key }}={{ value }}{% if not loop.last %}, {% endif %}
                                    {% endfor %}
                                {% endif %}
                            {% endif %}
                        </span>
                        <span class="log-time">{{ log.time }}</span>
                    </div>
                </div>
                
                <div class="expandable-content">
                    {{ render_log_content(log) }}
                </div>
            </div>
        {% endmacro %}

        {% for log in logs %}
            {{ render_log(log) }}
        {% endfor %}

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const contents = document.querySelectorAll('.to-json');
                contents.forEach(content => {
                    try {
                        const jsonData = JSON.parse(content.textContent);
                        content.textContent = JSON.stringify(jsonData, null, 2);
                    } catch (e) {
                        console.warn('Failed to parse JSON content:', e);
                    }
                });
            });

            function toggleContent(header) {
                const parent = header.closest('.log-entry');
                const button = header.querySelector('.expand-button');
                const content = parent.querySelector('.expandable-content');

                if (content.style.display === 'block') {
                    content.style.display = 'none';
                    button.textContent = '+';
                } else {
                    content.style.display = 'block'; 
                    button.textContent = '-';
                }
            }

            function expandAll() {
                const expandButtons = document.querySelectorAll('.expand-button');
                const expandableContents = document.querySelectorAll('.expandable-content');

                expandButtons.forEach(button => {
                    button.textContent = '-';
                });

                expandableContents.forEach(content => {
                    content.style.display = 'block';
                });
            }

            function collapseAll() {
                const expandButtons = document.querySelectorAll('.expand-button');
                const expandableContents = document.querySelectorAll('.expandable-content');

                expandButtons.forEach(button => {
                    button.textContent = '+';
                });

                expandableContents.forEach(content => {
                    content.style.display = 'none';
                });
            }

            function expandNested() {
                collapseAll();
                const logEntries = document.querySelectorAll('.log-entry');
                
                logEntries.forEach(entry => {
                    // Check if this entry has any child log entries
                    const hasChildren = entry.querySelector('.expandable-content .log-entry') !== null;
                    
                    if (hasChildren) {
                        const button = entry.querySelector('.expand-button');
                        const content = entry.querySelector('.expandable-content');
                        
                        button.textContent = '-';
                        content.style.display = 'block';
                    }
                });
            }
        </script>
    </body>
    </html>
    """


@app.command()
def visualize_logs(
    log_file: Annotated[Path, typer.Argument(help="The path to the log file")],
    output_file: Annotated[
        Path, typer.Option("-o", help="The path to the output file")
    ] = Path("toolmaker.html"),
):
    logs: list[LogEntry] = []
    stack: list[LogEntry] = []
    total_cost: float = 0.0

    # Read logs and build hierarchy
    with open(log_file, "r") as f:
        for line in f:
            log_entry: LogEntry = json.loads(line)
            log_entry["children"] = []  # Initialize children list for all entries

            # Code updates will be rendered as text
            if log_entry["name"] in (
                "update_code",
                "tool_code",
                "installed_repository_bash",
            ):
                log_entry["content_text"] = log_entry.get("content", "")
            # Convert string content to HTML markdown
            elif "content" in log_entry and isinstance(
                log_entry.get("content", None), str
            ):
                log_entry["content_html"] = Markup(markdown(log_entry["content"]))

            # Handle indentation levels
            if log_entry["type"] == "start":
                if stack:
                    stack[-1]["children"].append(log_entry)
                else:
                    logs.append(log_entry)
                stack.append(log_entry)
            elif log_entry["type"] == "end":
                if stack:
                    parent = stack.pop()
                    assert parent["name"] == log_entry["name"]
                    parent["result"] = log_entry
                    if log_entry["name"] == "llm_call":
                        total_cost += log_entry["metadata"].get("cost", 0)
                        parent["metadata"]["_call_cost"] = log_entry["metadata"].get(
                            "cost", 0
                        )
            else:
                if stack:
                    stack[-1]["children"].append(log_entry)
                else:
                    logs.append(log_entry)

    template = Template(html)
    rendered_html = template.render(
        logs=logs,
        total_cost=total_cost,
        markdown=lambda x: Markup(markdown(x)) if x is not None else "",
    )

    with open(output_file, "w") as f:
        f.write(rendered_html)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--log_file", type=str, default="toolmaker.jsonl")
    parser.add_argument("-o", "--output_file", type=str, default="toolmaker.html")
    args = parser.parse_args()
    visualize_logs(args.log_file, args.output_file)


if __name__ == "__main__":
    app()
