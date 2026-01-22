---
title: YOLO Card Detection
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# YOLO Card Detection System

A YOLOv8-based object detection system for identifying letter, number, and symbol cards.

## Features
- Detects 30 different card types
- Real-time confidence adjustment
- Easy-to-use web interface

## Detectable Cards
- **Letters (17)**: A, B, C, D, E, F, G, H, S, T, U, V, W, X, Y, Z
- **Numbers (9)**: 1-9
- **Symbols (4)**: Bullseye, circle, arrow directions

## Usage
1. Upload an image with physical cards
2. Adjust confidence threshold if needed
3. Click "Detect" to see results
