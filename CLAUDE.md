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

The following requirements must be followed when handling web materials and dependencies.

- When providing a web link, read and understand the complete content at the link before beginning the related task.
- When an error is found in a library’s usage, reread the complete content at the web link provided by the user.
- Do not aim to reduce dependencies or bypass required dependencies by implementing existing functionality independently or through other nonstandard means.

The following requirements must be followed when implementing code.

- Code must follow the fast fail principle and terminate immediately at the location where an error occurs.
Do not catch and conceal errors or configure a fallback.
- Do not use mocks, fake data, deceptive implementations, or workarounds intended only to pass tests in implementation or testing.
- The user may withdraw or adjust parts of a file after the Agent modifies it.
Before continuing modifications, if the file state differs from the result of the previous operation, reread the current file and continue from its current contents.
Do not restore content that the user deleted or modified.
- When other questions that can be answered immediately are received during task execution, answer them directly.
After answering, continue the previously unfinished task.
- After discovering a documentation or code error, updated content must not retain traces of the error or record the process from the erroneous version to the correct version.
- Every task and feature must proceed through implementation, execution, testing, and iteration until the required functionality operates correctly.
Do not stop after an initial implementation and ask the user to test it independently.

## File and Command Rules

- Do not modify code programmatically, including through heredocs, Python scripts, `sed`, `perl`, or other bulk text processing methods.
This restriction applies even when the user requests such methods.
Code modifications must be performed with a file editing tool.
- Do not inline excessively long or multiline Bash commands or Python scripts in a Bash command.
When a script must be executed, write it to a file first.
- Do not manually write a parser to parse an existing mature file format through strings or byte streams.
Use an appropriate third party library for parsing or avoid performing that parsing operation.
- Do not use ASCII Art to draw diagrams or tables.
When a drawing is genuinely necessary, use Mermaid.

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

