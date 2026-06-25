# Infrasound waveguides on Venus could enable the detection of explosive volcanism using balloons

## Summary
The following codes allow the computation of explosive volcanism detection probability for balloon missions on Venus. 

## Installation
- conda create -n venus_detectability python=3.10
- conda activate venus_detectability
- pip install -r requirements.txt

## Usage
- Compute detection probabilities for a given trajectory: "compute_proba_detection_volcano_venus.ipynb"
- Compute detection probabilities for range of trajectory parameters: "compute_proba_detection_volcano_venus.py"
- Plot detection probabilities for a given trajectory (Figure 8): "plot_one_trajectory_Figure.ipynb"
- Plot summary detection probabilities for all trajectories (Figure 9): "plot_summary_Figure.ipynb"

## Paper abstract
The prevalence of explosive volcanism on Venus is poorly constrained due to a lack of relevant in-situ and remote sensing data. Balloon‑borne pressure sensors in the middle atmosphere have the potential to detect infrasound from eruptions and thereby probe volcanic activity. 
In the current work, long-range ground-to-balloon infrasound propagation is modeled using global wind and temperature fields from the Venus Climate Database, combined with a novel extension to the established tau-p geometric ray-theoretical approach, as well as single-frequency parabolic-equation transmission loss simulations. We discover that strong zonal superrotation and subsolar‑to‑antisolar circulation generate persistent middle‑atmospheric acoustic waveguides that efficiently duct infrasound in both zonal and meridional directions, substantially enhancing detectability at balloon altitudes. To assess signal detectability, we couple global transmission loss estimates to a source model, relating Volcanic Explosivity Index to erupted mass, mass-flux histories, and source pressure spectra, and assumed Venus eruption statistics. Our statistical framework produces probability estimates of the detection of at least one volcano for any target Signal-to-Noise Ratio for different balloon flight trajectories and mission durations. The results indicate that, for realistic noise levels, 6-month long balloon missions at 60 km altitude can achieve up to 80% detection probability if only high-altitude volcanoes are considered. The detection rate varies depending on eruption duration, which drives the eruption pressure in our model. When considering a slow, long-duration eruptive process, the detection probability drops below 30%, but it can reach 99% if all large volcanoes are considered active. The findings demonstrate that future long‑duration balloon missions equipped with infrasound pressure sensors have great potential to provide novel data to constrain the rate and spatial distribution of explosive volcanism on Venus.

## Citation
Gullbekk, S. B., et al. (2026). Infrasound waveguides on Venus could enable the detection of explosive volcanism using balloons. submitted to JGR: Space Physics
```
@article{brissaud2026aerial,
  title={Infrasound waveguides on Venus could enable the detection of explosive volcanism using balloons},
  author={Gullbekk, Sophus Bredesen et al},
  journal={JGR: Space Physics},
  year={2026},
}
```