---
name: create-epic
description: Creates a Jira epic ticket through a conversational planning process. Prompts user to define goals, LLMs for validation, evaluations, and other epic details before ticket creation. Requires epic name, assignee, and priority. Uses defaults for summary, reporter, and team if not specified.
---

# Create Jira Epic Skill

This skill guides the user through a conversational planning process to create a well-defined Jira epic ticket.

## Planning Phase

Start by explaining that you'll help create a Jira epic ticket through a collaborative planning conversation where you'll work together to define and refine the epic.

### Step 1: Understand the Epic

Begin by asking the user what the epic is about. What problem are they trying to solve or what work needs to be done? Listen carefully to their initial description.

### Step 2: Collaborate on Description and Goals

Based on what the user shared, **propose a detailed description** for the epic. Your proposal should include:
- A clear description of what the epic entails
- Specific, concrete goals that should be accomplished
- Expected outcomes or success criteria
- Which LLMs should be used for validation (if applicable)
- Which evaluations should be run (if applicable)
- Any other relevant technical details

**Iterate on the description**: Present your proposal and work with the user to refine it. Ask clarifying questions, suggest additions, and help fill in any gaps. Continue this iterative process until you both agree the description is:
- Clear and detailed enough
- Has well-defined goals
- Includes all necessary technical specifications (LLMs, evaluations, etc.)

### Step 3: Propose and Refine the Epic Name

Once the description is solid, **propose a name** for the epic based on the description. The name should be concise but descriptive. Iterate with the user until you agree on the right name.

#### Step 4: Gather Required Fields

Once the story content is confirmed, ask about (one at a time):

1. **Assignee**: Who should be assigned to this epic?
2. **Priority**: What's the priority level?
   1. blocker
   2. critical
   3. major
   4. normal
   5. minor
3. **Components**: Which components does this epic belong to? (multiple allowed)
   1. Optimized Models
   2. Speculators
   3. Inference Research

### Optional Fields with Defaults

Inform the user of these defaults (they can override if needed):
- **Summary**: Defaults to the epic name if not specified
- **Reporter**: Defaults to "Alexandre Marques" (almarque@redhat.com)
- **Team**: Defaults to "INFERENG MLR"

## Summary and Confirmation Phase

Once you have gathered sufficient information:

1. **Present a Summary**: Display a clear summary of the epic ticket with all fields:
   - Epic Name
   - Summary
   - Description
   - Goals
   - Assignee
   - Reporter
   - Priority
   - Components
   - Team
   - Any additional fields specified

2. **Request Corrections**: Ask the user to review and provide any corrections or changes needed. Engage conversationally to refine the ticket details.

3. **Iterate**: Continue refining based on user feedback until they are satisfied.

## Creation Phase

**CRITICAL**: Only create the Jira ticket after the user has **explicitly approved** it. Look for clear approval statements like "looks good", "create it", "approved", "go ahead", etc.

Once approved, use the appropriate MCP Atlassian tools to create the epic:
- First, get the cloud ID and project information using `mcp__atlassian__getVisibleJiraProjects`
- Look up the assignee's account ID using `mcp__atlassian__lookupJiraAccountId`
- Always create epics under the INFERENG project key
- Validate the field names before creating the epic
- Create the epic using `mcp__atlassian__createJiraIssue` with issue type "Epic"
- Include all the gathered information in the appropriate fields
- Always include the Team field

## Field IDs Reference

These are the Jira field IDs for creating stories in the INFERENG project. Use these when calling `mcp__atlassian__createJiraIssue`:

### Priority IDs
- Blocker: `10000`
- Critical: `10001`
- Major: `10002`
- Normal: `10003`
- Minor: `10004`
- Undefined: `10005`

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


After creation, confirm to the user that the epic has been created and provide the issue key/link.

## Communication Style

- Be conversational, collaborative, and proactive
- **Help fill in the gaps**: Don't just ask questions - actively propose content based on what the user shares
- When proposing descriptions, goals, or names, draw from the context and technical details the user provides
- Ask focused questions to clarify and refine, not just to gather information
- Help the user think through the epic by suggesting specific LLMs, evaluations, or approaches that make sense for their goals
- Don't rush - allow time for thoughtful planning and iteration
- Be genuinely helpful in shaping the epic, not just documenting it
