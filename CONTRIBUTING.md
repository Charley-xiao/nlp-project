Thank you for considering contributing to **VeriScribbi: The Text Authenticator**! We’re excited to have you on board as we build this tool to distinguish human-written text from machine-generated content.

This guide will walk you through the process of contributing to **VeriScribbi**, whether you're fixing bugs, adding features, or improving the documentation.

## How Can You Contribute?

We welcome contributions in the form of:
- **Bug reports**: Let us know if you encounter issues while using the tool.
- **Bug fixes**: Help us improve **VeriScribbi** by fixing any reported bugs.
- **New features**: Suggest and implement new features to enhance the capabilities of the project.
- **Documentation improvements**: Help us improve the readability and accuracy of our documentation.
- **Testing**: Run tests to ensure everything is working as expected and report issues with test coverage.

## Getting Started

Before you start working on **VeriScribbi**, please follow these steps:

1. **Fork the repository**  
   Click the "Fork" button in the top right of the repository page to create your own copy of the **VeriScribbi** repository.

2. **Clone your fork locally**  
   Clone your fork of the repository to your local machine using:

   ```bash
   git clone https://github.com/Charley-xiao/nlp-project.git
   ```

3. **Set up the development environment**  
   Install the necessary dependencies by running:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Create a new branch**  
   It’s a good practice to create a new branch for each feature or bug fix. Use a descriptive name for your branch:

   ```bash
   git checkout -b feature-name
   ```

5. **Make your changes**  
   Work on your changes in your branch. Whether you're adding a new feature, fixing a bug, or improving documentation, make sure to test your changes locally.

6. **Commit your changes**  
   After making your changes, commit them with a clear and concise message describing the changes you’ve made:

   ```bash
   git add .
   git commit -m "Add feature to improve text classification"
   ```

7. **Push your changes to your fork**  
   Push your changes to your fork on GitHub:

   ```bash
   git push origin feature-name
   ```

8. **Create a Pull Request**  
   After pushing your branch, go to the **VeriScribbi** repository on GitHub and open a pull request to merge your changes. Make sure to provide a clear description of the changes and reference any relevant issues.

## Pull Request Guidelines

When submitting a pull request, please follow these guidelines:

- **Provide a clear description** of what your pull request does.
- **Link to any relevant issues** (e.g., Fixes #123).
- **Include tests** if applicable (we encourage adding tests for new features or bug fixes).
- **Ensure that your code follows the style** of the existing codebase (e.g., consistent naming conventions, indentation).
- **Run all tests** and ensure they pass before submitting your PR.
- **Be responsive**: Be open to feedback from the maintainers, and be ready to make any changes necessary to get your PR merged.

## Code of Conduct

By contributing to **VeriScribbi**, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md). We strive to maintain a positive, inclusive environment for all contributors.

## Reporting Bugs

If you find a bug or issue, please open a GitHub issue with the following information:

1. **Steps to reproduce** the issue.
2. **Expected behavior** and **actual behavior**.
3. **Environment details** (e.g., OS, Python version, dependencies).
4. **Error logs** or **screenshot** if applicable.

## License

By contributing to **VeriScribbi**, you agree that your contributions will be licensed under the MIT License, which is included in the [LICENSE](LICENSE) file.

## Thank You!

Thank you for taking the time to contribute to **VeriScribbi**. Whether you’re reporting bugs, suggesting new features, or submitting code, your efforts are invaluable in helping us improve the project.

We look forward to your contributions!
