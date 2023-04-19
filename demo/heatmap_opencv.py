import heatmap
import numpy as np

# Generate random segmentation results (e.g., predicted class labels)
seg_results = np.random.randint(0, 10, size=(100, 100))

# Create a color map for the segmentation classes
colors = heatmap.colorgradient.ColorGradient()
colors.addColorStop(0, (0, 0, 0))  # black for background
for i in range(1, 11):
    colors.addColorStop(i/10, heatmap.colorgen.hsv2rgb((i*36)%360, 1, 1))

# Create a heatmap object and add the segmentation results
hm = heatmap.Heatmap()
hm.heatmap(seg_results, colors=colors)

# Save the heatmap as an image file
hm.saveMap("seg_results_heatmap.png")
