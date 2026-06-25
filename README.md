# Portfolio Optimizer

A Python-based tool for optimizing asset allocation in investment portfolios using modern portfolio theory.

## Overview

Portfolio Optimizer helps determine the ideal weight distribution across a set of assets to maximize returns for a given level of risk (or minimize risk for a given level of return). It uses historical price data to compute expected returns, volatility, and correlations, then applies optimization techniques to construct an efficient portfolio.

## Features

- Fetches and processes historical stock price data
- Calculates expected returns, variance, and covariance across assets
- Computes the efficient frontier
- Optimizes for maximum Sharpe ratio or minimum volatility
- Visualizes portfolio allocation and risk-return tradeoffs

## Tech Stack

- Python
- NumPy / Pandas for data processing
- SciPy for optimization
- Matplotlib for visualization

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

## Project Structure
