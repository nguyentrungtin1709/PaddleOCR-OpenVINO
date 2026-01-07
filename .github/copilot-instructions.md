# GitHub Copilot Instructions

You are an expert AI programming assistant. When creating or refactoring code for this project, you MUST follow the rules below. These rules apply to all programming languages (Python, C#, Java, C++, JavaScript, etc.).

---

## 1. Project Overview

This project focuses on developing ML/AI solutions. The principles in this document apply to all ML workflows in general. All code must adhere to good design principles, be maintainable, extensible, and testable.

**Core Principles:**
- **Clean Code**: Code must be readable, understandable, and self-documenting.
- **DRY (Don't Repeat Yourself)**: Avoid code duplication, extract into reusable functions/classes.
- **KISS (Keep It Simple, Stupid)**: Prefer simple solutions, avoid over-engineering.
- **YAGNI (You Aren't Gonna Need It)**: Don't write code for features not yet needed.

---

## 2. SOLID Design Principles (REQUIRED)

All object-oriented programming (OOP) code must strictly adhere to the 5 SOLID principles:

### 2.1. S - Single Responsibility Principle (SRP)
- Each class/module should have **only one reason** to change.
- Separate layers: **UI/Presentation** ↔ **Business Logic** ↔ **Data Access**.
- Example: Don't write API calls directly in event handlers. Extract into a separate `Service` class.

### 2.2. O - Open/Closed Principle (OCP)
- Software entities should be **open for extension** but **closed for modification**.
- Use `interface` or `abstract class` to define common behaviors.
- Add new features by creating new implementations, not modifying existing code.

### 2.3. L - Liskov Substitution Principle (LSP)
- Objects of a subclass must be able to **replace the parent class** without breaking the program.
- Ensure subclasses properly implement the contract of their interface/parent class.

### 2.4. I - Interface Segregation Principle (ISP)
- **Do not force** clients to depend on interfaces they don't use.
- Split large interfaces into smaller, specialized interfaces.

### 2.5. D - Dependency Inversion Principle (DIP)
- High-level modules should not depend on low-level modules. Both should depend on **abstractions**.
- Use **Dependency Injection (DI)** to inject dependencies instead of instantiating directly.

---

## 3. ML Pipeline Development Strategy (Modular & Configurable)

When working with image processing and ML pipeline components:

### 3.1. Modularity
- Design pipelines as independent steps: `Preprocessing` → `Inference` → `Postprocessing`.
- Each step must be a "black box" with **clear Input/Output**.
- Each component can be tested independently.

### 3.2. Configurability
- **NEVER** hard-code parameters (threshold, model path, colors, sizes).
- All parameters must be read from configuration files (`config.json`, `config.yaml`, `.env`).
- Processing classes receive parameters via **Constructor Injection** or **Configuration Object**.

### 3.3. Reusability
- Design utility functions/classes that can be reused across projects.
- Separate business logic from framework-specific code.

---

## 4. Naming Conventions

Apply consistently across all languages:

| Component | Rule | Example |
|-----------|------|---------|
| **Class/Interface** | `PascalCase`, noun | `ImageProcessor`, `IDataLoader` |
| **Method/Function** | `camelCase`, verb | `processImage()`, `loadData()` |
| **Variable/Parameter** | `camelCase`, noun | `imageWidth`, `maxRetries` |
| **Constant** | `UPPER_SNAKE_CASE` | `MAX_IMAGE_SIZE`, `DEFAULT_TIMEOUT` |
| **Private field** | `_camelCase` or language convention | `_imageCache`, `_configPath` |
| **Boolean** | Prefix `is`, `has`, `can`, `should` | `isValid`, `hasPermission` |

**General Principles:**
- Names must be **self-documenting**.
- Avoid uncommon abbreviations (`img` → `image`, `btn` → `button`).
- Function/method names must describe the action (`getUserById`, not `user`).

---

## 5. Logging Rules

Follow best practices for effective logging:

### 5.1. Log Levels (Severity)
Use the correct log level:

| Level | Description | When to Use |
|-------|-------------|-------------|
| **DEBUG** | Technical details | Development, troubleshooting |
| **INFO** | Important business events | Startup, shutdown, key operations |
| **WARNING** | Abnormal situations but not errors | Deprecated usage, retry attempts |
| **ERROR** | Errors affecting specific operations | Exception caught, operation failed |
| **CRITICAL/FATAL** | Errors affecting the entire system | System crash, unrecoverable error |

### 5.2. Structured Logging
- Use **structured format** (JSON) instead of plain text.
- Include context: `timestamp`, `level`, `message`, `correlation_id`, `user_id`.

### 5.3. Best Practices
- SHOULD: Log **meaningful messages** with full context.
- SHOULD: Include **request/correlation ID** for tracing across services.
- SHOULD: Log at **entry/exit points** of important operations.
- AVOID: Logging sensitive data (passwords, tokens, PII).
- AVOID: Excessive logging in production (performance impact).
- AVOID: Using `print()` instead of a logging framework.

### 5.4. Error Logging
```
// Good: Full context
logger.error("Failed to process image", {
    imageId: imageId,
    filePath: filePath,
    error: exception.message,
    stackTrace: exception.stack
});

// Bad: Missing information
logger.error("Error occurred");
```

---

## 6. Testing Rules

### 6.1. AAA Pattern (Arrange-Act-Assert)
All unit tests must follow this structure:
```
// Arrange: Prepare data and dependencies
// Act: Execute the action to test
// Assert: Verify the result
```

### 6.2. Testing Principles
- **Isolation**: Each test must be independent, not dependent on other tests.
- **Deterministic**: Same input always produces same output.
- **Fast**: Unit tests must run quickly (< 100ms/test).
- **Readable**: Test names must clearly describe the scenario and expected result.

### 6.3. Test Naming Convention
Use format: `methodName_scenario_expectedBehavior`
```
// Examples:
processImage_withValidInput_returnsProcessedImage
processImage_withNullInput_throwsArgumentException
calculateTotal_withEmptyList_returnsZero
```

### 6.4. Test Coverage Guidelines
- **Unit Tests**: Business logic, utilities, algorithms.
- **Integration Tests**: Database, API, external services.
- **E2E Tests**: Critical user flows.

### 6.5. Mocking & Dependencies
- Mock external dependencies (APIs, databases, file system).
- Use **Dependency Injection** to easily mock.
- Avoid excessive mocking - it may indicate high code coupling.

---

## 7. Git Commit Rules (Conventional Commits)

Follow the [Conventional Commits Specification](https://www.conventionalcommits.org/en/v1.0.0/):

### 7.1. Commit Message Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 7.2. Commit Types

| Type | Description | SemVer |
|------|-------------|--------|
| `feat` | Add new feature | MINOR |
| `fix` | Fix bug | PATCH |
| `docs` | Documentation changes only | - |
| `style` | Code formatting, no logic changes | - |
| `refactor` | Refactor code, no new features or bug fixes | - |
| `perf` | Performance improvements | - |
| `test` | Add or fix tests | - |
| `build` | Build system or dependency changes | - |
| `ci` | CI configuration changes | - |
| `chore` | Other changes (not affecting src/test) | - |

### 7.3. Breaking Changes
- Use `!` after type: `feat!: remove deprecated API`
- Or add footer: `BREAKING CHANGE: description`

### 7.4. Commit Message Examples
```
feat(auth): add OAuth2 authentication support

fix(image-processor): handle null pointer in crop function

docs: update API documentation for v2.0

refactor(pipeline): extract preprocessing into separate module

feat(api)!: change response format to JSON

BREAKING CHANGE: API now returns JSON instead of XML
```

### 7.5. Commit Best Practices
- SHOULD: Each commit is a complete **logical unit**.
- SHOULD: Commit message is concise (< 72 characters for title).
- SHOULD: Use **imperative mood** ("add feature" not "added feature").
- AVOID: Committing code that doesn't compile/run.
- AVOID: Combining unrelated changes into one commit.

---

## 8. Error Handling

### 8.1. General Principles
- Always use `try-catch` when working with: **File I/O**, **Network**, **External Process**, **Database**.
- **Fail fast**: Validate input early and throw exceptions immediately if invalid.
- Don't swallow exceptions (empty `catch`) - at least log them.

### 8.2. Custom Exceptions
- Create custom exception classes for domain-specific errors.
- Exception messages must be clear and actionable.

### 8.3. Error Response
- APIs must return error responses with a consistent structure.
- Include: error code, message, details (if needed).

---

## 9. Workflow

### 9.1. Before Writing Code
1. Analyze requirements for SOLID violations.
2. Identify test cases to write.
3. Review design with the team for major changes.

### 9.2. While Writing Code
1. Write tests first (TDD) or alongside implementation.
2. Commit frequently with clear messages.
3. Refactor immediately when you see code smells.

### 9.3. After Writing Code
1. Self-review code before creating a PR.
2. Ensure all tests pass.
3. Update documentation if needed.

---

## 10. Code Review Checklist

- [ ] Code follows SOLID principles
- [ ] Naming conventions are correct
- [ ] Sufficient unit tests
- [ ] Complete error handling
- [ ] Appropriate logging
- [ ] No hard-coded configuration
- [ ] No code duplication
- [ ] Commit messages follow Conventional Commits

---

## 11. Using MCP (Model Context Protocol) Tools

### 11.1. Context7 - API Documentation Lookup (REQUIRED)

During development, you **MUST** use MCP Context7 to look up API information for libraries/frameworks:

**Usage Process:**
1. Use `resolve-library-id` to find the Context7-compatible library ID.
2. Use `get-library-docs` to retrieve accurate documentation.

**Version Selection Principles:**
- SHOULD: **Prioritize stability and safety** over the latest version.
- SHOULD: Use **stable** versions (not alpha, beta, canary).
- SHOULD: Version may not be the latest, but **should not be too far behind** the current version.
- AVOID: Using the latest APIs if they haven't been verified as stable.
- AVOID: Relying on outdated knowledge - always verify with Context7.

**Reasons:**
- Avoid using deprecated or changed APIs.
- Ensure code compatibility with the library version being used.
- Latest APIs (canary, alpha, beta) are often unstable and may change.

### 11.2. Search MCP Tools (Recommended)

Using MCP tools to search for information on the internet is encouraged when needed:

**Use Cases:**
- Searching for best practices and new patterns.
- Looking up solutions for specific problems.
- Learning about new technologies/libraries.
- Verifying information from multiple sources.

**Available Tools:**
- `search` - Search for information on DuckDuckGo.
- `fetch_content` - Retrieve content from a specific URL.

**Notes:**
- This is **recommended**, not required.
- Prioritize official sources (official docs, GitHub, Stack Overflow).
- Cross-check information from multiple sources when possible.

### 11.3. MCP Integration Workflow

```
1. Receive request from user
   ↓
2. Identify libraries/APIs to use
   ↓
3. [REQUIRED] Look up Context7 for accurate API documentation
   ↓
4. [Optional] Search for additional information if needed
   ↓
5. Implement code with verified APIs
   ↓
6. Test and validate
```