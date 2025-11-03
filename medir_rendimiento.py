import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from FA import OptimizedAdaptiveFilter, ParallelPipelineDetector
from baseline_secuencial import SequentialBaseline
import json

class PerformanceAnalyzer: