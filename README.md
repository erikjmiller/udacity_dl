
Part 4 of the Udacity Data Engineering Nano-Degree - Data Lakes

# Project Overview

As an employee at a fictional company called Sparkify which operates a music streaming service, 
I must create an analytics solution to determine which songs are played most frequently.

Data is stored in s3 as json files.  There are 2 types of data files.

* song files - Contains song metadata
* event/log files - Contains event data from the company associated user actions in the Sparkify app.  
Events contain information on which user plays which songs.

## Architecture

The company has bought into AWS as their cloud provider and want to harness the power of Amazon Redshift to accomplish the analytics.
This is beneficial since the json files are alrady in s3.

Redshift will be used to stage and transform the data with the help of python scripting.
The basic pipeline follows...

1. Create the table schema for loading the json files into a staging area and the final star schema.
2. Run a load process to load the data into the staging area.
3. Run an etl process to extract the staging data, transform it into the correct format and load it into the fact and dimension tables.

## Prequisites

In order to run the code you will need an AWS account, and a fair bit of knowledge about setting up the resources that are used in AWS.

1. python - I ran everything using python3
2. Access to input and output s3 buckets

## Running the code

1. git clone this repo and cd into it
2. copy example_dl.cfg to dl.cfg and populate it with aws access information
3. run `python ./etl.py` to perform both staging and final load.

When running etl.py you can expect a long duration since there are many s3 files to load.  
You can limit the number of songs that are loaded by using a longer prefix.






      




