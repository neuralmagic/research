---
name: breakdown-epic
description: Breaks down a Jira epic into smaller story tickets through a conversational planning process. Identifies the epic, suggests a high-level breakdown, then collaboratively creates individual story tickets with all necessary details.
---

# Breakdown Epic into Stories Skill

This skill guides the user through breaking down a Jira epic into smaller, actionable story tickets that can be completed in hours or days.

## Phase 1: Identify the Epic

Start by explaining that you'll help break down a Jira epic into story tickets through a collaborative planning conversation.

### Epic Identification

1. **Ask for the Epic**: Ask the user which epic needs to be broken down. They can provide:
   - A Jira ticket ID
   - The epic name
   - A brief description

2. **Search and Identify**: Based on the information provided, search for the epic using `acli jira workitem search`. Use appropriate JQL queries to find matching epics in the INFERENG project.

   Example search command:
   ```bash
   acli jira workitem search --jql "project = INFERENG AND type = Epic AND summary ~ '<search-term>'" --json
   ```

3. **Confirm the Epic**: Present the epic you found (with its key, summary, and brief description) and confirm with the user that this is the correct epic before proceeding.

## Phase 2: Retrieve Epic Details and Existing Stories

Once the user confirms the epic:

1. **Retrieve the Epic**: Use `acli jira workitem view` to get the full epic details including description, goals, and all fields.

   Example command:
   ```bash
   acli jira workitem view <EPIC-KEY> --fields '*all' --json
   ```

2. **Find Existing Stories**: Search for any existing stories linked to this epic using JQL. Read those stories to understand what work has already been planned or completed.

   Example command:
   ```bash
   acli jira workitem search --jql "project = INFERENG AND type = Story AND parent = <EPIC-KEY>" --json
   ```

## Phase 3: Suggest High-Level Breakdown

Based on the epic's content and any existing stories:

1. **Propose a Breakdown**: Suggest a high-level breakdown of the epic into stories. For each story, provide:
   - A proposed name/title
   - A short description of the scope and what should be accomplished
   - If it's an existing story, clearly note that it already exists

2. **Be Thoughtful**: Consider logical groupings, dependencies, and natural work boundaries. Stories should be achievable in hours or days, not weeks.

3. **Include Context**: Reference specific goals, LLMs, evaluations, or technical details from the epic that each story addresses.

4. **Iterate on the Breakdown**: Work with the user to refine the high-level breakdown. Add, remove, or modify stories based on their feedback. Continue until they approve the overall breakdown.

## Phase 4: Create Individual Stories

Once the high-level breakdown is approved, go through each NEW story one at a time (skip existing stories):

### For Each Story:

#### Step 1: Propose Story Details

**Suggest a summary (name) and detailed description** for the story based on the high-level breakdown. The description should include:
- What needs to be done
- Specific tasks or deliverables
- Any relevant technical details from the epic
- Success criteria or expected outcomes

#### Step 2: Iterate on Story Content

Work with the user to refine the summary and description until they're satisfied.

#### Step 3: Gather Required Fields

Once the story content is confirmed, ask about (one at a time):

1. **Assignee**: Who should be assigned to this story?
2. **Priority**: What's the Priority level?
   1. Blocker
   2. Critical
   3. Major
   4. Normal
   5. Minor
3. **Components**: Which components does this story belong to? (multiple allowed)
   1. Optimized Models
   2. Speculators
   3. Inference Research
4. **Activity Type**: What type of activity is this?
   1. Tech Debt & Quality
   2. New Features
   3. Learning & Enablement

### Default Fields

Inform the user of these defaults (they can override if needed):
- **Reporter**: Alexandre Marques (almarque@redhat.com)
- **Team**: INFERENG MLR

### Parent

The story must have the epic ticket assigned as a parent.


#### Step 4: Preview and Confirm

Present a complete summary of the story with all fields:
- Summary
- Description
- Assignee
- Priority
- Components
- Activity Type
- Reporter
- Team
- Parent Epic

Ask the user to review and confirm before creation.

#### Step 5: Create the Story

**CRITICAL**: Only create the story after explicit user approval.

Once approved, use the Atlassian CLI (`acli`) to create the story:

```bash
acli jira workitem create \
  --project INFERENG \
  --type Story \
  --summary "<story summary>" \
  --description "<detailed description>" \
  --assignee "<assignee-email>" \
  --priority <priority-id> \
  --components <component-id> \
  --field customfield_10464="<activity-type-id>" \
  --field customfield_10001="<team-id>" \
  --parent <EPIC-KEY> \
  --json
```

**Important Notes:**
- Always use the INFERENG project key
- Use email addresses for assignee (e.g., "almarque@redhat.com")
- Use numeric IDs for priority, components, and custom fields (see Field IDs Reference below)
- For multiple components, repeat the `--components` flag: `--components 33675 --components 33677`
- Set Activity Type using customfield_10464 with the appropriate ID
- Set Team using customfield_10001 with the team ID
- Link to parent epic using `--parent <EPIC-KEY>`
- Always include `--json` to get structured output with the created issue key
- Confirm to the user that the story has been created with the issue key

#### Step 6: Move to Next Story

After successful creation, move on to the next story in the breakdown and repeat Steps 1-5.

## Communication Style

- Be conversational, collaborative, and proactive
- **Help fill in the gaps**: Actively propose story names and descriptions based on the epic's content
- Draw from the epic's goals, technical details, and context when suggesting stories
- Ask focused questions to clarify and refine
- Help the user think through logical work boundaries and story scope
- Don't rush - allow time for thoughtful planning and iteration
- Be genuinely helpful in breaking down the work, not just documenting it
- Keep stories focused and achievable in hours or days

## Field IDs Reference

These are the Jira field IDs for creating stories in the INFERENG project. Use these when calling `acli jira workitem create`:

### Priority IDs
- Blocker: `10000`
- Critical: `10001`
- Major: `10002`
- Normal: `10003`
- Minor: `10004`
- Undefined: `10005`

### Activity Type (customfield_10464)
- Tech Debt & Quality: `12228`
- New Features: `12229`
- Learning & Enablement: `12230`

### Components
- Optimized Models: `33675`
- Speculators: `33677`
- Inference Research: `33682`

### Team (customfield_10001)
- INFERENG MLR: `ec74d716-af36-4b3c-950f-f79213d08f71-639`

**Important:** Use the plain string ID value for the Team field (not wrapped in an object):
```json
"customfield_10001": "ec74d716-af36-4b3c-950f-f79213d08f71-639"
```

## Notes

- All epics and stories are in the INFERENG project
- Stories should be linked to their parent epic
- Existing stories should be acknowledged but not recreated
- The breakdown should cover the full scope of the epic
