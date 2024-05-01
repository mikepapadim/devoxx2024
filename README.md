# Devoxx2024 TornadoVM Tensor API Demo
![img.png](img.png)

## Overview
This repository contains the source code and resources for the Devoxx2024 conference's Java-based applications. It's structured to facilitate the development, testing, and deployment of Java applications, incorporating various models and utilities to support these processes.
## Getting Started
To use this project, you'll need Java and Maven installed on your machine. Clone this repository and navigate to the project directory.

### Prerequisites
- Java JDK 21+
- Maven
- TornadoVM 1.0.5-dev

### Installation
1. Clone the repo: `git clone https://github.com/mikepapadim/devoxx2024.git`
2. Navigate to the project directory: `cd devoxx2024`
3. modify `setpaths` and `source setpaths`
4. Install dependencies: `mvn clean package -DskipTests`

### Running the Demo
Run the application using the provided shell script:
```bash
╰─cmd ➜ ./runner.sh 
Select which DemoTensorAPI version to run:
    0) Build Demo
    1) Basic TensorAPI
    2) TensorAPI with TornadoVM
    3) TensorAPI with Vector API
    4) TensorAPI with OnnxRT
    5) Exit
    Enter your choice [0-5]: 

```

