# Text Categorizer - experiments

Text Categorizer is a tool available in https://github.com/LuisVilarBarbosa/TextCategorizer/ that implements a configurable pipeline of methods used to train models that predict the categories of textual data.

This repository contains side-projects that use a minimal version of the code necessary to categorize text to test different tools that could be added to Text Categorizer.

## Getting Started

These instructions will get you a copy of the projects up and running on your local machine.

The different projects are designed to be used natively, but can easily be used with Docker.

### Prerequisites

- To execute natively, a machine with Anaconda3 64-bit or Miniconda3 64-bit installed is required.

### Installing/Updating

Here are presented the instructions on how to install/update all the dependencies necessary to execute the projects.

To install/update natively, open a shell (an Anaconda prompt is recommended on Windows and Bash is recommended on Linux) and type the following commands:
```
cd <path-to-experiment-folder>
conda env update -f environment.yml
```

## Executing

Here are presented the instructions on how to execute the projects.

To execute natively, open a shell (an Anaconda prompt is recommended on Windows and Bash is recommended on Linux) and type the following commands:
```
cd <path-to-experiment-folder>
conda activate text-categorizer
python <Python-file>
```

Calling ```python <Python-file>``` will present the usage parameters that must be indicated for the execution of the code.

These parameters are relatively simple to understand, but an overview of the code is recommended to understand the behavior of each parameter.

## Authors

* **Luís Barbosa** - [LuisVilarBarbosa](https://github.com/LuisVilarBarbosa)

## Acknowledgments

The layout of this README was inspired on https://github.com/LuisVilarBarbosa/TextCategorizer/blob/5bad65078999edde5312915c9654b2f5d910c288/README.md.

# Development Notes

- In general, the projects have been tested on Windows 10 and Ubuntu, but some projects might only work on Linux.

- Pickle is used to dump and load data to and from files. This protocol is the fastest of the tested protocols, but is considered insecure. Please take this information into consideration.

- Some projects might assume that an Internet connection is available.
