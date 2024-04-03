#!/bin/bash

# Define ANSI colors and styles for pretty print
BOLD=$(tput bold)
UNDERLINE=$(tput smul)
NORMAL=$(tput sgr0)
GREEN=$(tput setaf 2)

# Define the JAR file path
JAR_PATH="target/devoxx2024-1.0-SNAPSHOT.jar"

buildProject() {
    echo "${GREEN}Compiling the project with Maven...${NORMAL}"
    mvn clean package -DskipTests
    echo "${BOLD}${GREEN}Compilation finished.${NORMAL}"
}

# Function for running the basic DemoTensorAPI
runDemoTensorAPI() {
    echo "Running the basic DemoTensorAPI..."
    tornado -cp $JAR_PATH com.mikepapadim.devoxx.DemoTensorAPI
}

# Function for running the DemoTensorAPI with TornadoVM
runDemoTensorTornado() {
    echo "Running the DemoTensorAPI with TornadoVM..."
    tornado --threadInfo -cp $JAR_PATH com.mikepapadim.devoxx.DemoTensorTornado
}

# Function for running the DemoTensorAPI with Vector API
runDemoTensorVectorAPI() {
    echo "Running the DemoTensorAPI with Vector API..."
    tornado -cp $JAR_PATH com.mikepapadim.devoxx.DemoTensorVectorAPI
}

# Function for running the DemoTensorAPI with Onnx RT
runDemoTensorAPIOnnxRT() {
    echo "Running the DemoTensorAPI with Onnx RT..."
    tornado -cp $JAR_PATH com.mikepapadim.devoxx.DemoTensorAPIOnnxRT
}

# Main menu function
# Main menu function
showMenu() {
    echo "${UNDERLINE}Select which DemoTensorAPI version to run:${NORMAL}"
    echo "0) Build Demo"
    echo "1) Basic TensorAPI"
    echo "2) TensorAPI with TornadoVM"
    echo "3) TensorAPI with Vector API"
    echo "4) TensorAPI with OnnxRT"
    echo "5) Exit"
    read -p "Enter your choice [0-5]: " choice

    case $choice in
        0) buildProject ;;
        1) runDemoTensorAPI ;;
        2) runDemoTensorTornado ;;
        3) runDemoTensorVectorAPI ;;
        4) runDemoTensorAPIOnnxRT ;;
        5) echo "${BOLD}Exiting...${NORMAL}"; exit 0 ;;
        *) echo "${BOLD}Invalid option selected. Please try again.${NORMAL}" ;;
    esac
}

# Loop the menu until the user decides to exit
#while true; do
    showMenu
#done

