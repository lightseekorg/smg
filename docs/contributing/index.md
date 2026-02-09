---
title: Contributing
---

# Contributing to SMG

Thank you for your interest in contributing to Shepherd Model Gateway! This guide will help you get started.

---

## Ways to Contribute

<div class="grid cards" markdown>

-   :material-bug:{ .lg .middle } **Report Bugs**

    ---

    Found a bug? Open an issue with a clear description, steps to reproduce, and expected vs actual behavior.

    [:octicons-arrow-right-24: Open an issue](https://github.com/lightseekorg/smg/issues/new)

-   :material-lightbulb:{ .lg .middle } **Suggest Features**

    ---

    Have an idea? Open a feature request describing the problem you're solving and your proposed solution.

    [:octicons-arrow-right-24: Request a feature](https://github.com/lightseekorg/smg/issues/new)

-   :material-code-tags:{ .lg .middle } **Contribute Code**

    ---

    Ready to code? Follow our development guide to set up your environment and submit a pull request.

    [:octicons-arrow-right-24: Development guide](development.md)

-   :material-file-document:{ .lg .middle } **Improve Docs**

    ---

    Documentation improvements are always welcome! Fix typos, clarify explanations, or add examples.

    [:octicons-arrow-right-24: Edit on GitHub](https://github.com/lightseekorg/smg/tree/main/docs)

</div>

---

## Quick Start

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/smg.git
cd smg
```

### 2. Set Up Development Environment

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cargo build

# Run tests
cargo test
```

### 3. Make Changes

```bash
# Create a branch
git checkout -b feature/my-feature

# Make your changes
# ...

# Run tests and linting
cargo test
cargo clippy
cargo fmt --check
```

### 4. Submit a Pull Request

```bash
# Push your branch
git push origin feature/my-feature

# Open a pull request on GitHub
```

---

## Contribution Guidelines

### Code Quality

- [ ] All tests pass (`cargo test`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] New code has tests
- [ ] Documentation updated if needed

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:

```
feat(routing): add weighted round-robin policy
fix(health): handle timeout in health checks
docs(readme): update installation instructions
```

### Pull Request Process

1. **Open draft PR early** for complex changes to get feedback
2. **Link related issues** using "Fixes #123" or "Relates to #123"
3. **Update documentation** for user-facing changes
4. **Add tests** for new functionality
5. **Keep PRs focused** - one feature or fix per PR

---

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/lightseekorg/smg/discussions)
- **Bugs**: Open an [Issue](https://github.com/lightseekorg/smg/issues)
- **Chat**: Join our community on [Discord](https://discord.gg/wkQ73CVTvR) or [Slack](https://join.slack.com/t/lightseekorg/shared_invite/zt-3py6mpreo-XUGd064dSsWeQizh3YKQrQ)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our [Code of Conduct](https://github.com/lightseekorg/smg/blob/main/CODE_OF_CONDUCT.md).

---

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
