# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time adaptive video filtering system using computer vision and multiprocessing. The system analyzes video frames to detect environmental conditions (low light, fog, rain) and applies appropriate enhancement filters.

**Key Technologies**: Python, OpenCV (cv2), NumPy, multiprocessing

## Core Architecture

### Three Main Components

1. **[FA.py](FA.py)** - Optimized parallel implementation (production)
   - Uses multiprocessing with 2-process pipeline to avoid Python's GIL
   - Process 1: Frame capture (I/O bound)
   - Process 2: Complete processing (analysis + filtering + temporal smoothing)
   - Communication via multiprocessing.Queue with maxsize=2

2. **[baseline_secuencial.py](baseline_secuencial.py)** - Sequential baseline (benchmarking only)
   - Single-threaded processing for performance comparison
   - Intentionally unoptimized (processes frames 3x, uses larger kernels)
   - Used to measure speedup and efficiency metrics

3. **[medir_rendimiento.py](medir_rendimiento.py)** - Performance analysis tool
   - Compares sequential vs parallel implementations
   - Calculates speedup, efficiency, latency reduction
   - Generates performance graphs and JSON results

### Adaptive Filter Pipeline (FA.py)

The system implements a 4-stage processing pipeline:

1. **Condition Analysis** ([FA.py:36-163](FA.py#L36-L163))
   - Processes frames at reduced resolution (320x240) for speed
   - Detects: low light, fog, rain using computer vision metrics
   - Uses brightness, histogram analysis, edge detection (Laplacian, Sobel), and Hough line detection

2. **Temporal Smoothing** ([FA.py:188-209](FA.py#L188-L209))
   - Uses deque with maxlen=3 for voting-based stabilization
   - Prevents filter oscillation between frames

3. **Filter Application** ([FA.py:297-320](FA.py#L297-L320))
   - **LIME** (Low-light Image Enhancement): Bilateral filtering + illumination map
   - **FVR** (Fog/Visibility Restoration): Dark channel prior + atmospheric light estimation
   - **ARR** (Automatic Rain Removal): Morphological operations + inpainting

4. **Display with Overlay** ([FA.py:422-459](FA.py#L422-L459))
   - Real-time FPS counter and filter status indicators

### Decision Logic Priority

The system uses a hierarchical decision model ([FA.py:165-186](FA.py#L165-L186)):

1. **Priority 1**: Low light (brightness < 85) → Apply LIME
2. **Priority 2**: Rain (high confidence + vertical lines) → Apply ARR
3. **Priority 3**: Fog (high confidence + low contrast) → Apply FVR
4. **Default**: No filter

### Performance Optimizations

- **Reduced resolution analysis**: 320x240 instead of 640x480 (FA.py:23)
- **Smaller kernels**: 3x3 Gaussian instead of 5x5 (FA.py:32)
- **Downscaled processing**: LIME at 50% resolution, FVR at 60% (FA.py:217, 251)
- **Multiprocessing**: Separate processes for capture and processing to avoid GIL
- **Queue-based communication**: Non-blocking with timeout=0.01s to drop frames under load

## Common Commands

### Running the System

```bash
# Run optimized parallel version (main system)
python FA.py

# Run sequential baseline (for comparison)
python baseline_secuencial.py

# Run performance benchmark (100 frames)
python medir_rendimiento.py
```

### Interactive Controls (when running FA.py or baseline_secuencial.py)

- `q` - Quit the application
- `s` - Save current frame to disk
- `p` - Print statistics (baseline only)

### Performance Measurement

The benchmark script measures:
- **Speedup**: T_sequential / T_parallel
- **Efficiency**: Speedup / num_processes
- **Latency reduction**: Percentage improvement in frame processing time
- **Throughput**: Frames per second (target: ≥15 FPS)

Results are saved to:
- `performance_results.json` - Numerical data
- `performance_results.png` - Visualization graphs

## Code Structure Notes

### Multiprocessing Pattern

The parallel implementation uses `multiprocessing.Value('i', 0)` for shared running state between processes. Clean shutdown requires:
1. Set `running.value = 0`
2. Wait 0.5s for processes to finish current work
3. Terminate and join processes with timeout

### Queue Management

Both frame_queue and result_queue use:
- `maxsize=2` to prevent memory buildup
- Non-blocking `put(timeout=0.01)` to drop frames when queue is full
- This ensures real-time performance under CPU load

### Filter Implementations

All three filters (LIME, FVR, ARR) follow the pattern:
1. Optional downscaling for performance
2. Algorithm-specific processing
3. Upscaling back to original resolution

This is intentional - quality is preserved while processing cost is reduced.

### Camera Configuration

Standard setup is 640x480 @ 30 FPS with buffer size 1:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

Buffer size of 1 ensures minimal latency by preventing frame accumulation.

## Dependencies

Required Python packages:
- opencv-python (cv2)
- numpy
- matplotlib (for performance graphs only)

Install with: `pip install opencv-python numpy matplotlib`

## Known Issues from Performance Data

Based on [performance_results.json](performance_results.json):
- Current parallel implementation shows 0.58x speedup (slower than sequential)
- This indicates the overhead of multiprocessing exceeds the benefits
- Possible causes: Queue communication overhead, process spawning cost, GIL not being the bottleneck
- The sequential version with 3x redundant processing still outperforms parallel version

Consider investigating if threading (instead of multiprocessing) might be more efficient for I/O-bound camera capture, or if the processing workload needs to be more CPU-intensive to justify multiprocessing overhead.
