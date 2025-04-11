#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from ma1.crew import Ma1

from crewai.telemetry import Telemetry

def noop(*args, **kwargs):
    # with open("./logfile.txt", "a") as f:
    #     f.write("Telemetry method called and noop'd\n")
    pass


for attr in dir(Telemetry):
    if callable(getattr(Telemetry, attr)) and not attr.startswith("__"):
        setattr(Telemetry, attr, noop)
        
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {}
    
    try:
        Ma1().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    return

def replay():
    return

def test():
    return

