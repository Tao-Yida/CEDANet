# CEDANet Code Protection Guidelines

## Overview
This document outlines the code protection measures implemented in the CEDANet repository to safeguard intellectual property while maintaining open research collaboration.

## Protection Measures Implemented

### 1. License Protection
- **File**: `LICENSE`
- **Type**: Restrictive Academic License
- **Key Features**:
  - Prohibits commercial use without explicit permission
  - Allows academic and research use with attribution
  - Prevents redistribution and derivative commercial products
  - Requires proper citation in academic work

### 2. Copyright Headers
All Python source files include copyright headers that:
- Establish ownership (Tao Yida, University of Amsterdam)
- Reference the license agreement
- Warn against unauthorized commercial use
- Clarify research-only permissions

### 3. Comprehensive .gitignore
Protects against accidental exposure of:
- Model weights and trained parameters
- Training data and datasets
- Configuration files with sensitive information
- API keys and credentials
- Research notes and proprietary documentation
- Large media files and industrial monitoring data

### 4. Sensitive Data Exclusion
The following types of data are excluded from the repository:
- Industrial surveillance footage
- Citizen science volunteer data
- Trained model weights (.pth, .pt, .ckpt files)
- Configuration files that might contain sensitive parameters
- Research notes with proprietary algorithm details

## Best Practices for Contributors

### For Maintainers
1. **Never commit** model weights or trained parameters
2. **Review all pull requests** for potential IP exposure
3. **Sanitize comments** that might reveal proprietary algorithms
4. **Use placeholder values** in configuration examples
5. **Document public APIs only** - keep implementation details private

### For Contributors
1. **Read the LICENSE** before making contributions
2. **Sign contributor agreements** if implementing new features
3. **Avoid detailed algorithm comments** in public commits
4. **Use generic variable names** for proprietary components
5. **Submit clean code** without debug information or private notes

## What's Protected vs. What's Public

### Public (Safe to Share)
- General framework structure
- Standard ML utilities (dataloaders, metrics)
- Example scripts with dummy data
- Installation and basic usage instructions
- Academic citations and references

### Protected (Not in Repository)
- Trained model weights and parameters
- Industrial monitoring datasets
- Citizen science volunteer annotations
- Detailed algorithmic innovations
- Performance benchmarks on real data
- Configuration files with real API endpoints

## Commercial Use Guidelines

### Prohibited Without Permission
- Using CEDANet models in commercial products
- Training derivative models for commercial sale
- Integrating CEDANet algorithms in proprietary systems
- Reverse engineering the domain adaptation methodology

### Permitted with License
- Academic research and publications
- Educational use in courses and workshops
- Non-commercial prototype development
- Open source research collaborations

## Reporting Security Issues

If you discover potential IP exposure or security vulnerabilities:

1. **Do NOT create public issues** for security concerns
2. **Contact the maintainer directly** at [contact information]
3. **Provide details privately** about the potential exposure
4. **Allow time for resolution** before public disclosure

## Legal Compliance

### Attribution Requirements
When using CEDANet in academic work:
```
@software{cedanet2024,
  title={CEDANet: Citizen-engaged Domain Adaptation Net},
  author={Tao Yida},
  institution={University of Amsterdam},
  year={2024},
  license={Restrictive Academic License}
}
```

### Commercial Licensing
For commercial licensing inquiries:
- Contact: [To be provided]
- Include: Intended use case, commercial application details
- Expect: Evaluation period, licensing terms discussion

## Enforcement

### Violations May Result In
- DMCA takedown requests
- Legal action for copyright infringement
- Termination of research collaboration
- Academic misconduct reporting

### Monitoring
- Regular repository scans for license compliance
- Automated detection of protected content exposure
- Community reporting of potential violations

---

**Note**: This protection strategy balances open research collaboration with intellectual property protection. It allows academic advancement while preserving commercial value and preventing unauthorized exploitation.

**Last Updated**: 2024
**Version**: 1.0