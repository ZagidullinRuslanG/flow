# Prompts Directory

This directory contains example prompts for different software development scenarios. These prompts can be used as templates or inspiration when working with the coding assistant.

## Directory Structure

- `create/` - Prompts for creating new software projects
  - `web_apps/` - Web application creation prompts
  - `cli_tools/` - Command-line tools creation prompts
  - `libraries/` - Library/package creation prompts
  - `scripts/` - Utility scripts creation prompts

- `modify/` - Prompts for modifying existing code
  - `refactoring/` - Code refactoring prompts
  - `bug_fixes/` - Bug fixing prompts
  - `feature_add/` - Feature addition prompts
  - `optimization/` - Code optimization prompts

- `review/` - Prompts for code review and analysis
  - `security/` - Security review prompts
  - `performance/` - Performance analysis prompts
  - `best_practices/` - Code quality review prompts

## How to Use

1. Browse the appropriate category for your needs
2. Find a prompt that matches your scenario
3. Use the prompt as a template, customizing it for your specific case
4. Submit the prompt to the coding assistant

## Prompt Format

Each prompt file should follow this structure:

```yaml
title: "Brief description of what the prompt does"
category: "create/modify/review"
subcategory: "specific category"
difficulty: "beginner/intermediate/advanced"
description: |
  Detailed description of what this prompt is for
  and what kind of response to expect

example: |
  Example prompt text that can be used as a template

expected_outcome: |
  Description of what to expect from the assistant
  when using this prompt

notes: |
  Additional tips or considerations when using this prompt
```

## Contributing

Feel free to add new prompts or improve existing ones. When adding new prompts:

1. Follow the established directory structure
2. Use the standard prompt format
3. Include clear examples and expected outcomes
4. Add appropriate difficulty level
5. Provide helpful notes for users 