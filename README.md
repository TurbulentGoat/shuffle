# Shuffle

## Overview

**Shuffle** is a comprehensive Python program designed to implement various card shuffling algorithms, track detailed shuffle statistics, and generate insightful reports. Whether you're a card enthusiast, a statistician, or someone interested in the mechanics of shuffling, this tool provides valuable insights into the effectiveness and randomness of different shuffling techniques.

## Features

- **Multiple Shuffling Algorithms:**
  - Built-in Shuffle
  - Fisher-Yates Shuffle (Manual)
  - Riffle Shuffle
  - Sattolo's Shuffle
  - Random Block Shuffle
  - Pile Shuffle

- **Statistics Tracking:**
  - Total number of shuffles performed
  - Unique and duplicate shuffle counts
  - Perfect shuffles (returning to original order)
  - Total and average distance from original order
  - Longest streak without duplicates
  - Return to original position counts for each card
  - Suit clustering frequencies
  - Sequential patterns frequencies

- **Chi-Square Test:**
  - Evaluates the randomness of shuffles by testing the uniformity of card positions

- **Reports and Visualizations:**
  - Generates comprehensive PDF reports with explanations and embedded graphs
  - Creates various visualizations including bar charts, histograms, and heatmaps
  - Exports data to CSV files for further analysis

- **Interactive Menu:**
  - User-friendly interface to select algorithms, start shuffling, save/load history, generate reports, and exit

- **Real-Time Progress Feedback:**
  - Utilizes progress bars to indicate shuffling progress
  - Allows users to stop shuffling at any time

## Installation

### Prerequisites

- **Python 3.6** or higher
- [Get Python here](https://www.python.org/downloads/)

### Clone the Repository

`git clone https://github.com/TurbulentGoat/shuffle.git`

`cd shuffle`

### **Additional Tips**
To install the dependencies, run `pip install -r requirements.txt`
