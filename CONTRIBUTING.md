# Contributing to RKNS

Thank you for considering contributing to RKNS! 
This document outlines the guidelines and workflows we use to make contributing to this project as smooth as possible.

## Getting Started

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/rekonas-com/rkns-python`
3. Set up the development environment:
   ```bash
   # Open the project in VS Code with devcontainer
   code rkns
   # In VS Code: Reopen in Container
   
   # Once the container is running, you can start a bash within the devcontainer.
   pytest # to verify everything works
   ```

## Git Workflow

We follow a pull request-based workflow where all changes must be merged to `main` via pull requests.

### Branch Naming Convention

Create branches with clear, descriptive names following this format:
`<type>/<description>`

Where `<type>` is one of:
- **feature/** - New features or enhancements
- **fix/** - Bug fixes
- **hotfix/** - Urgent fixes for production issues
- **docs/** - Documentation changes
- **chore/** - Maintenance tasks, dependency updates, etc.
- **refactor/** - Code refactoring without functional changes
- **test/** - Adding or updating tests
- **ci/** - CI/CD pipeline changes
- **style/** - Code style/formatting changes (no functional changes)
- **dev/** - Changes to the development environment (e.g. devcontainer)

### Creating a Branch

Always create branches from the latest `main`:
```bash
git checkout main
git pull
git checkout -b feature/new-feature-name
```

### Keeping Your Branch Updated

Regularly pull changes from `main` into your branch:
```bash
git checkout your-branch-name
git pull origin main
```

### Commit Guidelines

We follow the [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages.

#### Commit Message Format
```markdown
<type>(optional scope): <description>
```

#### Types
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools
- **ci**: Changes to our CI configuration files and scripts
- **dev** - Changes to the development environment (e.g. devcontainer)

### Pull Request Process

1. If applicable, update the documentation with details of any changes to the interface.
2. Ensure all tests pass and add new tests for new functionality.
3. Update the version numbers to the new version.
4. Make sure your code follows the project's coding standards (run linters).
5. Submit the pull request with a clear, descriptive title following the conventional commits format.

### Pull Request Title

Follow the Conventional Commits format for the PR title:
```markdown
<type>(optional scope): <description>
```

Example:
feat(auth): implement OAuth authentication 
fix(validation): correct email validation regex

### Pull Request Template

Your pull request description should answer:
- What changes have you made?
- Why did you make these changes?
- How did you test these changes?
- Any additional information that reviewers should know?

## Code Style

We use Ruff for code linting and formatting. Ensure your code follows these standards:
```bash
# Check style
ruff check .

# Format code
ruff format .