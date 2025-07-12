# Contributing to CEDANet

Thank you for your interest in contributing to CEDANet! This document outlines the guidelines and requirements for contributing to this research project.

## Before You Contribute

### 1. Read the License
Please carefully read the [LICENSE](LICENSE) file to understand the restrictions and requirements for this project.

### 2. Understand Code Protection
Review [CODE_PROTECTION.md](CODE_PROTECTION.md) to understand our intellectual property protection measures.

### 3. Academic vs Commercial Use
This project is intended for academic research. Commercial use requires separate licensing agreements.

## Contribution Guidelines

### Acceptable Contributions
- ✅ Bug fixes that don't expose proprietary algorithms
- ✅ Documentation improvements
- ✅ Code optimizations that maintain IP protection
- ✅ Test cases for public functionality
- ✅ Installation and setup improvements

### Restricted Contributions
- ❌ Exposing proprietary algorithm details
- ❌ Adding commercial features without permission
- ❌ Including third-party proprietary code
- ❌ Bypassing or weakening code protection measures

## Code Contribution Process

### 1. Fork and Clone
```bash
git fork https://github.com/Tao-Yida/CEDANet
git clone https://github.com/YourUsername/CEDANet
```

### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Follow existing code style and structure
- Add copyright headers to new files (see existing files for format)
- Ensure no sensitive data is included
- Test your changes thoroughly

### 4. Commit Guidelines
- Use clear, descriptive commit messages
- Reference issues when applicable
- Keep commits focused and atomic

### 5. Pull Request
- Fill out the pull request template completely
- Include detailed description of changes
- Reference any related issues
- Ensure all checks pass

## Code Standards

### Copyright Headers
All new Python files must include the standard CEDANet copyright header:
```python
#!/usr/bin/env python3
"""
CEDANet: Citizen-engaged Domain Adaptation Net
Copyright (c) 2024 Tao Yida, University of Amsterdam
All rights reserved.

This file is part of CEDANet and is protected under the terms of the
CEDANet License Agreement. See LICENSE file for details.

Unauthorized commercial use, distribution, or modification is prohibited.
For research use only under academic license terms.
"""
```

### Sensitive Information
Never commit:
- Model weights or trained parameters
- Real datasets or annotations
- API keys or credentials
- Configuration files with sensitive data
- Private research notes or proprietary algorithm details

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Include docstrings for public functions
- Keep algorithm implementation details minimal in comments

## Review Process

### Academic Review
- Code review by project maintainer
- Verification of license compliance
- IP protection assessment
- Technical accuracy validation

### Approval Criteria
- ✅ Maintains code protection standards
- ✅ Follows contribution guidelines
- ✅ Adds value to academic research
- ✅ Doesn't compromise intellectual property
- ✅ Includes proper attribution and headers

## Legal Requirements

### Contributor Agreement
By contributing to CEDANet, you agree that:
- Your contributions are your original work
- You grant the project maintainer rights to use your contributions
- Your contributions comply with the project license
- You will not claim ownership of the overall CEDANet system

### Attribution
Contributors will be acknowledged in:
- CONTRIBUTORS.md file (if contribution is substantial)
- Academic publications (for significant algorithmic contributions)
- Release notes (for important fixes or features)

## Questions and Support

### Getting Help
- Review existing documentation first
- Check existing issues for similar questions
- Create an issue for technical questions
- Contact maintainer for licensing questions

### Contact Information
- **Project Maintainer**: Tao Yida
- **Institution**: University of Amsterdam
- **For Commercial Licensing**: [Contact information to be provided]

## Recognition

### Types of Recognition
- **Code Contributors**: Listed in CONTRIBUTORS.md
- **Research Contributors**: Co-authorship consideration for academic papers
- **Bug Reporters**: Mentioned in release notes
- **Documentation Contributors**: Listed in documentation credits

### Criteria for Recognition
- Meaningful contribution to project goals
- Adherence to code protection guidelines
- Professional collaboration approach
- Respect for intellectual property rights

---

**Note**: This project balances open collaboration with intellectual property protection. All contributions must respect both academic research goals and commercial licensing rights.

**Last Updated**: 2024