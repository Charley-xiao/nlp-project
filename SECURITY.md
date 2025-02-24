# VeriScribbi Security Guide

## General Security Considerations

While we work on improving **VeriScribbi**, we ask users and contributors to be mindful of the following points:

### Data Privacy
- **VeriScribbi** does **not** store or collect any personal data from users. Any text entered into the application for classification purposes is used only within the session and is not saved to a server.
- **No tracking**: We do not track user activity on the platform, nor do we use any third-party tracking services.

### Input Validation
- Input text is checked for basic validity (non-empty strings). However, there may be edge cases where input handling could be further improved for security, especially around unexpected characters or malicious input.
  
### Dependencies
- We use common libraries like **transformers**, **torch**, **spaCy**, and **Streamlit**. As with any project, it's important to regularly update dependencies to ensure we are using the latest, secure versions.
- Always run `pip install -r requirements.txt` to install dependencies and ensure you're working with the most up-to-date, stable versions of the libraries.

## Security Measures for Development

### Secure Coding Practices
- **Error handling:** While **VeriScribbi** provides basic error messages, we aim to prevent the exposure of sensitive details in production environments. Detailed error messages are logged internally, but they do not get exposed to users.
- **External Libraries:** We make use of well-established libraries and models (e.g., **roberta-base**, **Qwen2.5-0.5B-Instruct**). All external resources and models are accessed over secure HTTPS connections.

### Authentication and Authorization
- **VeriScribbi** does not currently have any user authentication or authorization mechanisms, as it is a course project. However, if you plan to deploy or expand the app, consider adding proper user authentication (e.g., OAuth or JWT) if you plan to allow user-specific functionality.

## Security Risks and Mitigation

1. **Model Misuse**:
   - **Risk**: There could be potential for misuse of the classification model (e.g., generating machine-written text that passes as human-written).
   - **Mitigation**: We recommend that the tool is used responsibly. Users should be aware that the model is not 100% accurate and that it should be considered as an aid rather than a definitive tool.

2. **Dependency Vulnerabilities**:
   - **Risk**: Using third-party libraries always comes with the risk of vulnerabilities.
   - **Mitigation**: Ensure that all dependencies are up to date by using pip-tools or other dependency management tools to track versions, and avoid using deprecated or unsupported libraries.

## Reporting Security Issues

If you discover any security issues related to the **VeriScribbi** project, please report them responsibly by opening a GitHub issue or directly reaching out to the project maintainers at [charleyxiao057@gmail.com](mailto:charleyxiao057@gmail.com).

We take all security reports seriously and will do our best to resolve any vulnerabilities promptly.