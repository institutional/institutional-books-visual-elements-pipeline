# NOTE: This command's goal is to help us have a look at some sample data to make sure the pipeline behaves as expected

import click

click.command("peek")
click.option("step")  # values [detect, classify]
click.option("n")  # values [integer>0, "all"]
click.option("")


def peek(step, n):
    # make sure n is less than or equal to total number of samples in database for corresponding step
    # if step is detect, grab the entire page image from cache and draw the box, with associated confidence score
    # if step is classify, show the crop (using the crop utility function in models.detection), the associated predicted class, and the confidence score
    pass


def peek_detect():
    pass


def peek_cls():
    pass


def draw_box():
    pass


def peek_dedupe():
    pass
