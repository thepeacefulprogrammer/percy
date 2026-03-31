## Percy

Percy is a local orchestration assistant built on `ventures_agent_framework`.

### TODO Tracking

Percy uses the shared TODO store from the framework. TODOs persist in
`/home/randy/local/percy-output/todos.json` and are injected into the handover
prompt every turn.

#### Text mode slash commands

- `/todo` — list active TODOs
- `/todo-add <title> [project] [priority]`
- `/todo-start <id>`
- `/todo-done <id>`
- `/todo-note <id> <note>`
- `/todo-show <id>`
- `/todo-next`

#### Agent tools

The agent has access to TODO tools (`todo_list`, `todo_add`, `todo_update`,
`todo_done`, `todo_note`, `todo_next`, `todo_show`) for automatic updates.