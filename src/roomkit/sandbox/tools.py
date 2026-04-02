"""Shared constants and reference tool schemas for Sandbox integration."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Tool name constants
# ---------------------------------------------------------------------------

TOOL_SANDBOX_READ = "sandbox_read"
TOOL_SANDBOX_LS = "sandbox_ls"
TOOL_SANDBOX_GREP = "sandbox_grep"
TOOL_SANDBOX_FIND = "sandbox_find"
TOOL_SANDBOX_GIT = "sandbox_git"
TOOL_SANDBOX_WRITE = "sandbox_write"
TOOL_SANDBOX_EDIT = "sandbox_edit"
TOOL_SANDBOX_DELETE = "sandbox_delete"
TOOL_SANDBOX_DIFF = "sandbox_diff"
TOOL_SANDBOX_BASH = "sandbox_bash"

SANDBOX_TOOL_PREFIX = "sandbox_"

# ---------------------------------------------------------------------------
# System prompt preamble
# ---------------------------------------------------------------------------

SANDBOX_PREAMBLE = (
    "You have access to a sandboxed development environment with file reading, "
    "search, and git capabilities. All commands run in an isolated container "
    "with token-optimized output. Prefer specific tools (sandbox_read, "
    "sandbox_grep, sandbox_git) over sandbox_bash for common operations — "
    "they produce more structured, compact output."
)

# ---------------------------------------------------------------------------
# Reference tool schemas
# ---------------------------------------------------------------------------

SANDBOX_READ_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_READ,
    "description": (
        "Read file contents from the sandbox. Returns numbered lines. "
        "Use offset and limit to read specific portions of large files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (0-based).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read.",
            },
        },
        "required": ["path"],
    },
}

SANDBOX_LS_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_LS,
    "description": "List directory contents with compact, token-optimized output.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Defaults to working directory.",
            },
        },
    },
}

SANDBOX_GREP_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_GREP,
    "description": "Search file contents using regex patterns with compact output.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in.",
            },
            "type": {
                "type": "string",
                "description": "File type filter (e.g. 'py', 'js', 'rs').",
            },
        },
        "required": ["pattern"],
    },
}

SANDBOX_FIND_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_FIND,
    "description": "Find files by name pattern with compact tree output.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to search in.",
            },
            "name": {
                "type": "string",
                "description": "Filename pattern to match.",
            },
            "type": {
                "type": "string",
                "description": "File type filter: 'f' for files, 'd' for directories.",
            },
        },
    },
}

SANDBOX_GIT_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_GIT,
    "description": (
        "Run any git command with compact, token-optimized output. "
        "Supports all git subcommands: status, diff, log, show, clone, "
        "checkout, fetch, pull, push, blame, branch, stash, etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "args": {
                "type": "string",
                "description": (
                    "Git arguments as a single string. "
                    "Examples: 'status', 'diff HEAD~3', 'log --oneline -10', "
                    "'clone https://github.com/org/repo.git', "
                    "'checkout -b feature', 'blame src/main.py'."
                ),
            },
        },
        "required": ["args"],
    },
}

SANDBOX_WRITE_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_WRITE,
    "description": (
        "Write content to a file in the sandbox. Creates the file if it "
        "doesn't exist, or overwrites it if it does."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "The full content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
}

SANDBOX_EDIT_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_EDIT,
    "description": (
        "Edit a file in the sandbox by replacing an exact string match. "
        "Use sandbox_read first to see the current content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace (must be unique in the file).",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
}

SANDBOX_DELETE_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_DELETE,
    "description": "Delete a file or empty directory in the sandbox.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or empty directory to delete.",
            },
        },
        "required": ["path"],
    },
}

SANDBOX_DIFF_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_DIFF,
    "description": "Compare two files with ultra-condensed diff output.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_a": {
                "type": "string",
                "description": "Path to the first file.",
            },
            "file_b": {
                "type": "string",
                "description": "Path to the second file.",
            },
        },
        "required": ["file_a", "file_b"],
    },
}

SANDBOX_BASH_SCHEMA: dict[str, Any] = {
    "name": TOOL_SANDBOX_BASH,
    "description": (
        "Execute a bash command in the sandbox. Output is token-optimized. "
        "Prefer specific tools (sandbox_read, sandbox_grep, sandbox_git) "
        "over bash equivalents for common operations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 30.",
            },
        },
        "required": ["command"],
    },
}

# Ordered reference catalog — implementations can use this directly
# via ``tool_definitions()`` or pick a subset.
SANDBOX_TOOL_SCHEMAS: list[dict[str, Any]] = [
    SANDBOX_READ_SCHEMA,
    SANDBOX_WRITE_SCHEMA,
    SANDBOX_EDIT_SCHEMA,
    SANDBOX_LS_SCHEMA,
    SANDBOX_GREP_SCHEMA,
    SANDBOX_FIND_SCHEMA,
    SANDBOX_GIT_SCHEMA,
    SANDBOX_DIFF_SCHEMA,
    SANDBOX_DELETE_SCHEMA,
    SANDBOX_BASH_SCHEMA,
]
