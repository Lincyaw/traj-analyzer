# Agent Conduct Rules

All rules in this document are mandatory requirements, with no exceptions or exemptions.
Temporary operations, one time operations, command line operations, operations that are not immediately detected, and operations performed in good faith, for progress, or to assist must all comply with this document.

Each time the user mentions `AGENTS.md` in a message, the Agent must reread this file.
The Agent must not omit this rereading on the grounds that it was read previously.

## Language Rules

The rules in this section apply to reasoning, responses, documentation, comments, and all other natural language content.

### Style of Expression

- Every sentence in natural language content must occupy its own line.
Forced line breaks within a sentence are prohibited.
Line breaks may only occur after a complete sentence.
Code blocks, tables, and content governed by fixed formatting requirements are exempt from this rule.
- Dashes are prohibited in all natural language content.
- Do not use contrastive constructions such as “not ... but ...” or “rather than ...”.
This restriction does not apply when the user explicitly requests a comparison.
- Do not provide procedural work descriptions before reading the code or completing research, such as “perform one operation first, then perform another operation to avoid a certain issue”.
Do not assume issues, steps, or risks before the actual situation is understood.
- Designs must be complete, sufficient, and directly implementable.
Do not use phased wording such as “complete a first version, observe it, and handle it later”.
- Do not classify designs as “conservative” or “aggressive”.
When multiple designs are genuinely necessary, each design must stand independently and represent a parallel option with meaningful selection value.
- When searching for specified content, report only results that meet the requirements.
Do not list candidates confirmed not to meet the requirements or describe their exclusion.
- Do not add an overview opening or a summary conclusion to responses, including wording such as “the preceding content is ... and the following provides details” or “one sentence summary ...”, or equivalent expressions.
- Do not assess a task’s workload or use wording such as “That's a lot” or “This is a substantial rewrite”, or equivalent expressions.
Do not reduce design, implementation, verification, or explanatory content due to workload.

### Wording Requirements

- Chinese wording must use complete, commonly accepted forms.
When a common term of two or more characters exists, do not replace it with a single Chinese character.
For example, use complete terms such as “崩溃、终止、判定、推断、抛出、挂起、卡死”.
- Do not coin nouns or abbreviate common expressions.
For example, “两个字的版本” must not be abbreviated as “两字版本”, and “单个字的版本” must not be abbreviated as “单字版本”.
- When describing an operation, use a complete verb object structure that clearly states the action and its target.
For example, write “使用新版本动态库替换 `_vllm_fa3_C.abi3.so` 共享库文件” instead of an abbreviated expression such as “换库”.
- Do not use industry jargon such as “落地”, “钉死”, or “对齐” when clear common alternatives exist.
Wording must be directly understandable to people with ordinary Chinese reading ability.
- Proper nouns that are suitable for retaining in English must use English.
- All identifiers in code must retain their original English names, including variable names, class names, function names, and other identifiers.
Translation is strictly prohibited.

### Prohibited Characters and Terms

- Do not use the Chinese character “栈”, including in expressions such as “技术栈” and “模型栈”.
State the specific object directly, such as “使用的技术” or “全部模型”.
- Do not use the Chinese character “落”, including in expressions such as “落下” and “落盘”.
- Do not use the Chinese character “死”, including in expressions such as “定死”, “钉死”, and “打死”.
- Do not use the Chinese character “拆”.
When expressing an analysis process, use clear wording such as “理解”.
- Do not use the term “契约”.
- Do not use the Chinese character “偏”, including in expressions such as “偏弱” and “偏大”.
- “粗、细、硬、软、实、虚” must not be used as standalone words.
They may only be used as parts of commonly accepted terms containing two or more characters and must not express their literal meanings.
Common terms such as “详细” and “实际” may be used.
Expressions such as “细小” and “坚硬” must not be used.

## Factual Judgment Rules

- When answering questions about the existence or correctness of repository related content, prioritize preventing misreports and false positives.
- Do not expand the scope of judgment to provide more results.
Do not present unverified possibilities as facts.
- Do not assume that the user expects an affirmative or negative conclusion.
Every conclusion must rely on actual evidence and remain factual.
- Do not add unverified conclusions, questions, or suggestions to accommodate the user.

## Final State Principle

The Agent must not add content that the user did not request.
After the user requests removal of extra content, subsequent code, documentation, commit messages, PR descriptions, comments, and other output must not retain that extra content or traces of its discussion.

For example, when the user requests tomato and scrambled eggs, do not add Dongpo pork without authorization.
After the user indicates that this content is unnecessary, Dongpo pork must no longer appear in the final result or related descriptions.

All textual output must describe the final state directly.
Do not retain wording such as “Design A was wrong, so Design B was used” or accounts of the causes of previous errors.
After the user identifies an error, treat that error as confirmed when continuing the work.

## Operational Rules

Unless the user explicitly requests otherwise, the following restrictions must be followed.

- Do not use `try-except` to perform an `import`.
Required libraries must be imported directly.
- Do not enter plan mode without authorization.
- Do not use Git to revert any code.
When the user says “revert”, it always means manually restoring file content with a file editing tool.
- Do not read from or write to the `/tmp` directory.
Intermediate results must be stored in a dedicated directory under the current directory, and that directory must be added to `.gitignore`.
- Do not proactively use visual capabilities.

The following requirements must be followed when running analyses and recording their results.

- Analyses run through the `traj` commands of an analysis project: ingestion, extraction, evaluation, mining, hunts, anomalies and reviews.
- Every analysis result, including the unified trajectories, the tables, the features, the model calls, the hunt verdicts, the kept probes, the verdicts people give and the reports, is stored where those commands write it: `.traj/` of the project, `hunts/<name>/`, `reviews/verdicts.jsonl` and `reports/`.
- Do not complete analysis tasks with ad hoc scripts, and do not keep analysis results in a temporary `.scratch` directory.

The following requirements must be followed when handling web materials and dependencies.

- When providing a web link, read and understand the complete content at the link before beginning the related task.
- When an error is found in a library’s usage, reread the complete content at the web link provided by the user.
- Do not aim to reduce dependencies or bypass required dependencies by implementing existing functionality independently or through other nonstandard means.

The following requirements must be followed when implementing code.

- Before writing code, read and understand the relevant code, project structure, and existing project conventions.
Evaluate whether the architecture is appropriate and whether the code style follows those conventions.
Do not begin writing code until this understanding and evaluation are complete.
- Code must follow the fast fail principle and terminate immediately at the location where an error occurs.
Do not catch and conceal errors or configure a fallback.
- Do not write test cases without the user's prior approval.
Ask the user and obtain explicit approval before writing any test cases.
- Verify a change by running the `traj` commands on real trajectories in an analysis project, through the complete flow from the raw files to the tables, the reports and the pages.
Do not use unit testing for verification.
- Tests that replay model calls use the answers a real run of the same instructions gave.
When an instruction or a request changes, record the answers of a new real run, and never write model answers by hand.
- Commit in small, focused increments, and commit promptly after each self-contained change is verified.
- In the viewer and the backend, reuse existing components and libraries whenever possible.
Implement a component independently only when no suitable existing component is available.
- Keep code concise and maintainable.
Prefer the mature abstractions the project already uses, such as pydantic models, Arrow tables, DuckDB queries, sqlglot for SQL and aifn functions for model calls, over handwritten equivalents.
- Do not use mocks, fake data, deceptive implementations, or workarounds intended only to pass tests in implementation or testing.
- The user may withdraw or adjust parts of a file after the Agent modifies it.
Before continuing modifications, if the file state differs from the result of the previous operation, reread the current file and continue from its current contents.
Do not restore content that the user deleted or modified.
- When other questions that can be answered immediately are received during task execution, answer them directly.
After answering, continue the previously unfinished task.
- After discovering a documentation or code error, updated content must not retain traces of the error or record the process from the erroneous version to the correct version.
- Every task and feature must proceed through implementation, execution, verification, and iteration until the required functionality operates correctly.
Do not stop after an initial implementation and ask the user to test it independently.

## Repository SOP

The following process governs how trajectories move from their raw files to reviewed findings.
Component boundaries follow the [design](docs/design.md).

### What Belongs in This Repository

- This repository holds only what no data source, harness or scenario owns: the unified trajectory, the parsing layers of general formats, the operators and event rules that hold in every scenario, evaluation, mining, hunts and the viewer.
- An adapter, a harness profile, an operator, an event rule, a hunt or a report of one scenario belongs to the analysis project of that scenario.
For example, the ATR adapter and the profiles of WorkBuddy, Codex and dsh belong to `cgfs-trajectories`.
- Before changing a shared interface, such as the unified trajectory, the table schemas, the operator interface or the state of a hunt, list every analysis project and every test that uses it.

### Analysis Lifecycle

1. An analysis project starts from `traj init`, and its `traj.yaml` names the datasets with their adapters, the enabled operators, the batches and the strata.
2. Only adapters read raw files, through `traj ingest`.
An adapter whose source records the calls of the agent gives them as `recorded_calls`, so ingestion finds every call the parser loses.
3. Do not edit raw files, rendered trajectories or anything under `.traj/` by hand.
Change the adapter, the configuration, the operators or the hunt, and run the commands again.
4. Run `traj check` after installing or upgrading aifn or dsh, and before the first command that calls models.
Model calls go through aifn functions and the mailbox, which answers an unchanged call from its earlier conclusion.
5. Features come from enabled operators, from mining after its checks, and from the probes a finished hunt keeps.
6. Results compared with each other use the same configuration, operators, models and harness profiles.
Results of different configurations are reported separately.

### Model Measurements and Their Review

- A judged feature, a mining tail check and a hunt verdict are measurements by a model.
Report them with what measures them: the noise against a second model, the precision of a probe, the coverage and the bound on what the probes miss, the agreement of the blind reads, and the verdicts people gave.
- Models never read the grades of a trajectory.
`render.hide_metadata` names every metadata key that holds a grade, a reward or a label.
- A reader of an agent session's hunt is a fresh subagent that gets the instruction and nothing of the proposer's reasoning.
- People record their verdicts with `traj review` or on the pages, in `reviews/verdicts.jsonl`.
- Conclusions go to `reports/` of the analysis project, with the trajectory keys and steps needed to find them.

### Shared Working Copy

- Other sessions may change the repository concurrently.
Stage only the paths you changed, and leave their changes in place.

## Code Organization and Naming

Before adding or changing code, verify all of the following:

1. The code belongs in the file and matches the file's responsibility.
2. The file name clearly describes the file's purpose.
3. The file belongs in its directory and matches the directory's responsibility.
4. Each function name clearly describes its behavior and the object it operates on.

A reader must be able to infer a function's basic purpose from its directory path, file name, and function name.
Revise the location or names when that purpose is unclear.
Keep code identifiers in English.

After changing code, list the affected directory structure and the function names in each file, and confirm from that listing alone why each file is divided as it is, why each function has its name, and how they relate to each other.
For example, `hunting/kept.py` with `read_kept` and `write_kept` reads as the facets and probes a finished hunt keeps, with one function reading them and one writing them.
When the listing cannot be understood this way, redesign the division or rename the files and functions.
The listing can be produced with commands such as the following, with the paths replaced by the affected directories.

```bash
tree --gitignore -I __pycache__ src/traj_analyzer/hunting src/traj_analyzer/viewer/static/review
grep -rnE '^\s*(async )?def |^class ' src/traj_analyzer/hunting
grep -rnE '^export (default )?(async )?(function|const|class) |^(async )?function ' src/traj_analyzer/viewer/static/review
```

## File and Command Rules

- Use existing command line tools whenever they can complete the task, the `traj` commands first.
- When existing commands lack a required capability, assess whether it belongs in the `traj` commands.
Add or improve the appropriate command when applicable, and use that command to complete the task.
- Use pipelines and short scripts to combine commands and perform simple transformations.
Do not create or accumulate ad hoc scripts that duplicate commands or implement missing reusable task behavior.
- Do not modify code programmatically, including through heredocs, Python scripts, `sed`, `perl`, or other bulk text processing methods.
This restriction applies even when the user requests such methods.
Code modifications must be performed with a file editing tool.
- Do not inline excessively long or multiline Bash commands or Python scripts in a Bash command.
When a script must be executed, write it to a file first.
- Do not manually write a parser to parse an existing mature file format through strings or byte streams.
Use an appropriate third party library for parsing or avoid performing that parsing operation.
- Do not use ASCII Art to draw diagrams or tables.
When a drawing is genuinely necessary, use Mermaid.

## Documentation Rules

- Maintain a small set of necessary documents, each with a clear purpose: `README.md` for installing and using the tool, `docs/design.md` for the architecture and the design, `docs/viewer.md` for the pages, and the skills under `.claude/skills/` for the agents that run analyses.
Update an existing document when it already covers the subject.
Do not create new documents or append task records for every change.
- Consolidate useful information by subject into one authoritative location before removing redundant documents.
Evaluate the information within each document when deciding what to retain.
- Keep documentation for information the code does not communicate.
Preserve overall architecture, component responsibilities, dependency directions, important design decisions and their rationale.
Make implementation behavior clear through code structure and naming instead of maintaining parallel descriptions of interfaces, fields, functions or command workflows.
- Preserve the goals of the analyses, the requirements on the data, the statistics behind a design, analysis conclusions and unfinished designs that cannot be recovered from code.
Before deleting a design as duplicated implementation documentation, verify that it has been implemented and contains no additional requirements or conclusions.
Distinguish existing implementation, target design and pending verification, and retain the evidence references needed to interpret conclusions.
Keep task specifications and runtime instructions, such as the instructions of the aifn functions and the workspace README files, as functional inputs.
- Store per-run counts, result matrices, run identifiers, execution timestamps and progress records in `.traj/` of the analysis project, its reports and its logs.
Keep explanatory documents focused on architecture, requirements, design rationale, acceptance criteria and analysis conclusions.
Retain only the evidence scope and references needed to interpret those conclusions.

## Python and Comment Rules

- Do not add a docstring at the beginning of a Python file.
- Do not add a shebang to a Python file.
- Comments must use English, while technical terms remain in English.
- Limit the number of comments to what is needed to explain necessary information.
Excessive comments are prohibited.

## Rules for Handling Questions

Content the user raises in an interrogative sentence must be answered as a question and must not be treated as an implementation command.
When answering such questions:

- Do not include alternative designs or expanded suggestions that the user did not request.
- Do not ask the user to answer other questions in return.
- Do not add urging or guiding expressions at the end, such as “implementation can begin once preparation is complete”.

